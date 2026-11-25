"""ROCm agent-loop support for collocated and external vLLM topologies.

The rollout server is a regular top-level process on a dedicated node.  Ray is
used only for small HTTP proxy actors and the FSDP trainers, so vLLM never has
to start its EngineCore below a Ray GPU actor on ROCm.
"""

from __future__ import annotations

import base64
import asyncio
from concurrent.futures import ThreadPoolExecutor
import io
import json
import os
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import ray
import requests
import torch
import torch.distributed
from PIL import Image
from torch.distributed.fsdp import FullStateDictConfig, FullyShardedDataParallel as FSDP, StateDictType

from verl.experimental.agent_loop.agent_loop import AgentLoopManager, AgentLoopWorker
from verl.protocol import DataProto
from verl.utils.device import get_device_id
from verl.utils.memory_utils import trim_host_memory
from verl.utils.model import convert_weight_keys
from verl.workers.fsdp_workers import AsyncActorRolloutRefWorker
from verl.workers.rollout.replica import TokenOutput


def _image_to_base64(image: Any) -> str:
    if isinstance(image, str):
        path = Path(image)
        if not path.is_file():
            return image
        image = Image.open(path)
    elif isinstance(image, dict):
        if image.get("bytes") is not None:
            raw = image["bytes"]
            if isinstance(raw, str):
                return raw
            return base64.b64encode(raw).decode("ascii")
        image = Image.open(image["path"])
    elif isinstance(image, bytes):
        return base64.b64encode(image).decode("ascii")

    if not isinstance(image, Image.Image):
        raise TypeError(f"unsupported rollout image type: {type(image)}")
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="JPEG", quality=95)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


class ExternalVllmServerProxy:
    """Ray-compatible token-in/token-out proxy for TRL's external server."""

    def __init__(self, base_url: str, timeout: float = 7200.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    async def generate(
        self,
        prompt_ids: list[int],
        sampling_params: dict[str, Any],
        request_id: str,
        image_data: list[Any] | None = None,
        video_data: list[Any] | None = None,
    ) -> TokenOutput:
        if video_data:
            raise ValueError("the Vision-OPD split server currently accepts images, not video")

        import aiohttp

        params = dict(sampling_params)
        max_tokens = int(params.pop("max_tokens", params.pop("max_new_tokens", 1024)))
        want_logprobs = bool(params.pop("logprobs", False))
        payload = {
            "prompts": [prompt_ids],
            "images": [[_image_to_base64(image) for image in image_data]] if image_data else [None],
            "n": 1,
            "temperature": float(params.pop("temperature", 1.0)),
            "top_p": float(params.pop("top_p", 1.0)),
            "top_k": int(params.pop("top_k", -1)),
            "repetition_penalty": float(params.pop("repetition_penalty", 1.0)),
            "max_tokens": max_tokens,
            "logprobs": 0 if want_logprobs else None,
            "generation_kwargs": {
                **params,
                "bad_words": ["<|vision_start|>", "<|vision_end|>", "<|image_pad|>", "<|video_pad|>"],
            },
        }
        timeout = aiohttp.ClientTimeout(total=self.timeout)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(f"{self.base_url}/generate/", json=payload) as response:
                body = await response.text()
                if response.status != 200:
                    raise RuntimeError(f"external vLLM generate failed ({response.status}): {body[-2000:]}")
                result = await response.json()

        token_ids = result["completion_ids"][0]
        sampled_logprobs = None
        if want_logprobs:
            sampled_logprobs = []
            for token_id, token_logprobs, logprob_ids in zip(
                token_ids,
                result["logprobs"][0],
                result["logprob_token_ids"][0],
                strict=True,
            ):
                try:
                    sampled_logprobs.append(token_logprobs[logprob_ids.index(token_id)])
                except ValueError:
                    sampled_logprobs.append(token_logprobs[0])
        return TokenOutput(token_ids=token_ids, log_probs=sampled_logprobs, stop_reason="completed")


class ExternalVllmWeightClient:
    """Minimal NCCL/RCCL client for TRL's weight-sync worker extension."""

    def __init__(self, base_url: str, group_port: int, timeout: float = 1800.0):
        self.base_url = base_url.rstrip("/")
        parsed = urlparse(self.base_url)
        if not parsed.hostname:
            raise ValueError(f"invalid external vLLM URL: {base_url}")
        self.host = parsed.hostname
        self.group_port = group_port
        self.timeout = timeout
        self.session = requests.Session()
        self.communicator = None
        self.client_rank = None

    def _json(self, method: str, path: str, **kwargs) -> dict:
        response = self.session.request(method, f"{self.base_url}{path}", timeout=self.timeout, **kwargs)
        if response.status_code != 200:
            raise RuntimeError(f"external vLLM {path} failed ({response.status_code}): {response.text[-2000:]}")
        return response.json()

    def initialize(self, device: int | str) -> None:
        from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
        from vllm.distributed.utils import StatelessProcessGroup

        world_size = int(self._json("GET", "/get_world_size/")["world_size"])
        self.client_rank = world_size
        try:
            device_uuid = str(torch.cuda.get_device_properties(device).uuid)
        except Exception:
            device_uuid = "external-trainer"
        self._json(
            "POST",
            "/init_communicator/",
            json={
                "host": "0.0.0.0",
                "port": self.group_port,
                "world_size": world_size + 1,
                "client_device_uuid": device_uuid,
            },
        )
        time.sleep(1)
        process_group = StatelessProcessGroup.create(
            host=self.host,
            port=self.group_port,
            rank=self.client_rank,
            world_size=world_size + 1,
        )
        self.communicator = PyNcclCommunicator(process_group, device=device)

    def update_named_param(self, name: str, weight: torch.Tensor) -> None:
        if self.communicator is None or self.client_rank is None:
            raise RuntimeError("external vLLM communicator is not initialized")
        self._json(
            "POST",
            "/update_named_param/",
            json={"name": name, "dtype": str(weight.dtype), "shape": list(weight.shape)},
        )
        self.communicator.broadcast(weight, src=self.client_rank, stream=torch.cuda.current_stream())
        torch.cuda.synchronize()
        self.communicator.group.barrier()

    def reset_prefix_cache(self) -> None:
        self._json("POST", "/reset_prefix_cache/")


class ExternalVllmAsyncActorRolloutRefWorker(AsyncActorRolloutRefWorker):
    """FSDP actor that synchronizes full policy weights to an external server."""

    def _build_rollout(self, trust_remote_code: bool = False):
        del trust_remote_code
        self.base_sync_done = True
        self.layered_summon = self.config.rollout.get("layered_summon", False)
        self._external_vllm_clients = None

    async def rollout_mode(self):
        from verl.utils.fsdp_utils import fsdp_version, load_fsdp_model_to_gpu, offload_fsdp_model_to_cpu

        if fsdp_version(self.actor_module_fsdp) != 1:
            raise RuntimeError("external vLLM weight sync currently requires the configured FSDP1 actor")
        if self._is_offload_param:
            load_fsdp_model_to_gpu(self.actor_module_fsdp)

        full_state_config = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with FSDP.state_dict_type(self.actor_module_fsdp, StateDictType.FULL_STATE_DICT, full_state_config):
            state_dict = self.actor_module_fsdp.state_dict()

        if self.rank == 0:
            model = getattr(self.actor_module_fsdp, "_fsdp_wrapped_module", self.actor_module_fsdp)
            state_dict = convert_weight_keys(state_dict, model)
            if self._external_vllm_clients is None:
                urls = json.loads(os.environ["EXTERNAL_VLLM_URLS"])
                group_ports = json.loads(os.environ["EXTERNAL_VLLM_GROUP_PORTS"])
                self._external_vllm_clients = [
                    ExternalVllmWeightClient(url, int(group_port))
                    for url, group_port in zip(urls, group_ports, strict=True)
                ]
                for client in self._external_vllm_clients:
                    client.initialize(get_device_id())
            sync_started = time.monotonic()
            total_tensors = len(state_dict)
            for index, (name, cpu_weight) in enumerate(state_dict.items(), start=1):
                device_weight = cpu_weight.to(get_device_id(), non_blocking=False)
                for client in self._external_vllm_clients:
                    client.update_named_param(name, device_weight)
                del device_weight
                if index % 100 == 0 or index == total_tensors:
                    print(
                        f"[external-vllm] weight_sync={index}/{total_tensors} "
                        f"elapsed_s={time.monotonic() - sync_started:.1f}",
                        flush=True,
                    )
            for client in self._external_vllm_clients:
                client.reset_prefix_cache()
            print(
                f"[external-vllm] synchronized {total_tensors} tensors "
                f"to {len(self._external_vllm_clients)} replicas",
                flush=True,
            )
        del state_dict

        if self._is_offload_param:
            offload_fsdp_model_to_cpu(self.actor_module_fsdp)
        torch.cuda.empty_cache()

    async def trainer_mode(self):
        self.actor_module_fsdp.train()
        torch.cuda.empty_cache()


class RocmLocalAgentLoopManager(AgentLoopManager):
    """Use in-process processors with the regular collocated vLLM servers."""

    def _init_agent_loop_workers(self):
        # ROCm Ray subprocesses exited natively while constructing the Qwen
        # processor, even with one readiness-checked actor. The manager already
        # runs in a CPU TaskRunner actor, so keep preprocessing there and retain
        # Ray only for the lightweight HTTP proxy and FSDP workers.
        num_workers = int(self.config.actor_rollout_ref.rollout.agent.num_workers)
        self.agent_loop_workers = [
            AgentLoopWorker(self.config, self.server_handles, self.reward_router_address)
            for _ in range(num_workers)
        ]
        print(f"{type(self).__name__}: local_agent_workers={num_workers}", flush=True)

    def generate_sequences(self, prompts: DataProto) -> DataProto:
        trim_host = os.environ.get("VERL_TRIM_HOST_MEMORY", "0") == "1"
        if trim_host:
            trim_host_memory()

        timing = {}
        phase_start = time.perf_counter()
        self.wake_up()
        if self.reward_model_manager:
            self.reward_model_manager.wake_up()

        timing["rollout/wake_and_sync"] = time.perf_counter() - phase_start
        phase_start = time.perf_counter()
        chunks = prompts.chunk(len(self.agent_loop_workers))
        if len(self.agent_loop_workers) == 1:
            outputs = [asyncio.run(self.agent_loop_workers[0].generate_sequences(chunks[0]))]
        else:
            # Separate workers own separate processors/tokenizers. Running their
            # event loops in threads parallelizes multimodal preprocessing while
            # each loop still dispatches its requests asynchronously.
            with ThreadPoolExecutor(max_workers=len(self.agent_loop_workers)) as executor:
                futures = [
                    executor.submit(asyncio.run, worker.generate_sequences(chunk))
                    for worker, chunk in zip(self.agent_loop_workers, chunks, strict=True)
                ]
                outputs = [future.result() for future in futures]
            del futures
        timing["rollout/workers"] = time.perf_counter() - phase_start
        phase_start = time.perf_counter()
        counts = [item.meta_info.pop("preprocess_counts", {}) for item in outputs]
        outputs = self._align_output_prompt_length(outputs)
        output = DataProto.concat(outputs)

        timing["rollout/concat"] = time.perf_counter() - phase_start
        phase_start = time.perf_counter()
        self.sleep()
        if self.reward_model_manager:
            self.reward_model_manager.sleep()

        timing["rollout/sleep"] = time.perf_counter() - phase_start
        performance = {}
        samples = sum(item.get("samples", 0) for item in counts)
        if samples:
            unique = sum(item["unique_prompts"] for item in counts)
            performance = {"perf/rollout_preprocess_cache_hit_fraction": 1 - unique / samples,
                           "perf/rollout_unique_prompts": unique}
        metrics = [item.meta_info.pop("metrics") for item in outputs]
        output.meta_info = {"timing": {**self._performance_metrics(metrics, output), **timing},
                            "performance": performance, **outputs[0].meta_info}
        del chunks, outputs
        if trim_host:
            trim_host_memory()
        return output


class ExternalVllmAgentLoopManager(RocmLocalAgentLoopManager):
    """Agent-loop manager backed by independent external HTTP replicas."""

    def _initialize_llm_servers(self):
        base_urls = [url.rstrip("/") for url in json.loads(os.environ["EXTERNAL_VLLM_URLS"])]
        proxy_cls = ray.remote(max_concurrency=512)(ExternalVllmServerProxy)
        self.rollout_replicas = []
        self.server_handles = [proxy_cls.remote(base_url) for base_url in base_urls]
        self.server_addresses = base_urls
        print(f"ExternalVllmAgentLoopManager: {self.server_addresses}", flush=True)

    def wake_up(self):
        ray.get([worker.wake_up.remote() for worker in self.worker_group.workers])

    def sleep(self):
        ray.get([worker.sleep.remote() for worker in self.worker_group.workers])
