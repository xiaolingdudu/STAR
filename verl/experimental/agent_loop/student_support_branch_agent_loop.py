# Copyright 2024 Bytedance Ltd. and/or its affiliates
"""Shared-trunk local branch rollout for SSP-OPD v2."""

from __future__ import annotations

import asyncio
import time
from typing import Any
from uuid import uuid4

from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopOutput, register
from verl.tools.utils.tool_registry import initialize_tools_from_config


def _sample_seed(step: int, sample_index: object) -> int:
    """Return one deterministic seed shared by all siblings of a prompt."""

    encoded = str(sample_index).encode("utf-8")
    text_term = sum((position + 1) * byte for position, byte in enumerate(encoded))
    return int((int(step) * 1_000_003 + text_term + 17) % 2_147_483_647)


@register("student_support_branch")
class StudentSupportBranchAgentLoop(AgentLoopBase):
    """Generate one reproducible Student trunk and one sibling-specific branch."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        cfg = self.config.actor_rollout_ref.actor.self_distillation
        self.trunk_depths = [int(value) for value in cfg.get("ssp_trunk_depths", [16, 32, 48, 64])]
        self.branch_tokens = int(cfg.get("ssp_chunk_tokens", 32))
        tool_config_path = self.config.data.tool_config_path
        tool_list = initialize_tools_from_config(tool_config_path) if tool_config_path else []
        self.tool_schemas = [tool.tool_schema.model_dump(exclude_unset=True, exclude_none=True) for tool in tool_list]

    async def _generate(self, prompt_ids, sampling_params, images, videos, *, route_id: str | None = None):
        return await self.server_manager.generate(
            request_id=route_id or uuid4().hex,
            prompt_ids=prompt_ids,
            sampling_params=sampling_params,
            image_data=images,
            video_data=videos,
        )

    async def _shared_trunk(self, key, route_id, prompt_ids, sampling_params, images, videos):
        # AgentLoop instances are per trajectory, while server_manager is shared
        # by the worker. With contiguous rollout siblings this executes the
        # common trunk once and lets all K branches await the same result.
        cache = getattr(self.server_manager, "_ssp_v2_trunk_tasks", None)
        if cache is None:
            cache = {}
            self.server_manager._ssp_v2_trunk_tasks = cache
        task = cache.get(key)
        if task is None:
            task = asyncio.create_task(
                self._generate(prompt_ids, sampling_params, images, videos, route_id=route_id)
            )
            cache[key] = task
        return await task

    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        messages = list(kwargs["raw_prompt"])
        multi_modal_data = await self.process_vision_info(messages)
        images = multi_modal_data.get("images")
        videos = multi_modal_data.get("videos")
        prompt_ids = await self.apply_chat_template(
            messages,
            tools=self.tool_schemas,
            images=images,
            videos=videos,
        )
        start = time.perf_counter()

        # Validation measures the normal deployable Student, not a short branch.
        if self.validate:
            output = await self._generate(prompt_ids, dict(sampling_params), images, videos)
            response_ids = output.token_ids[: self._get_response_length()]
            return AgentLoopOutput(
                prompt_ids=prompt_ids,
                response_ids=response_ids,
                response_mask=[1] * len(response_ids),
                response_logprobs=output.log_probs[: len(response_ids)] if output.log_probs else None,
                multi_modal_data=multi_modal_data,
                num_turns=2,
                metrics={"generate_sequences": time.perf_counter() - start},
            )

        trajectory = kwargs["__trajectory__"]
        shared_seed = _sample_seed(trajectory["step"], trajectory["sample_index"])
        trunk_tokens = self.trunk_depths[shared_seed % len(self.trunk_depths)]
        trunk_params = dict(sampling_params)
        trunk_params.update(
            max_tokens=trunk_tokens,
            min_tokens=trunk_tokens,
            ignore_eos=True,
            seed=shared_seed,
        )
        trunk_key = (int(trajectory["step"]), str(trajectory["sample_index"]))
        route_id = f"ssp-v2:{trunk_key[0]}:{trunk_key[1]}"
        trunk_output = await self._shared_trunk(
            trunk_key,
            route_id,
            prompt_ids,
            trunk_params,
            images,
            videos,
        )
        trunk_ids = trunk_output.token_ids[:trunk_tokens]

        branch_params = dict(sampling_params)
        branch_params.update(
            max_tokens=self.branch_tokens,
            min_tokens=self.branch_tokens,
            ignore_eos=True,
            seed=(shared_seed + 7_919 * (int(trajectory["rollout_n"]) + 1)) % 2_147_483_647,
        )
        branch_output = await self._generate(
            prompt_ids + trunk_ids,
            branch_params,
            images,
            videos,
            route_id=route_id,
        )
        branch_ids = branch_output.token_ids[: self.branch_tokens]
        response_ids = trunk_ids + branch_ids

        response_logprobs = None
        if trunk_output.log_probs is not None and branch_output.log_probs is not None:
            response_logprobs = trunk_output.log_probs[: len(trunk_ids)] + branch_output.log_probs[: len(branch_ids)]

        return AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=response_ids,
            response_mask=[1] * len(response_ids),
            response_logprobs=response_logprobs,
            multi_modal_data=multi_modal_data,
            num_turns=2,
            metrics={"generate_sequences": time.perf_counter() - start},
            extra_fields={
                "ssp_v2_branch_mask": [0] * len(trunk_ids) + [1] * len(branch_ids),
                "ssp_v2_trunk_length": len(trunk_ids),
                "ssp_v2_branch_length": len(branch_ids),
            },
        )
