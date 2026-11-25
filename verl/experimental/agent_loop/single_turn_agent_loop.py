# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import asyncio
import logging
import os
from typing import Any
from uuid import uuid4

from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopOutput, register
from verl.tools.utils.tool_registry import initialize_tools_from_config
from verl.utils.profiler import simple_timer

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


@register("single_turn_agent")
class SingleTurnAgentLoop(AgentLoopBase):
    """Naive agent loop that only do single turn chat completion."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.prompt_length = self.config.actor_rollout_ref.rollout.prompt_length

        tool_config_path = self.config.data.tool_config_path
        tool_list = initialize_tools_from_config(tool_config_path) if tool_config_path else []
        self.tool_schemas = [tool.tool_schema.model_dump(exclude_unset=True, exclude_none=True) for tool in tool_list]

    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        messages = list(kwargs["raw_prompt"])

        async def prepare_prompt():
            multi_modal_data = await self.process_vision_info(messages)
            prompt_ids = await self.apply_chat_template(
                messages, tools=self.tool_schemas,
                images=multi_modal_data.get("images"), videos=multi_modal_data.get("videos"),
            )
            return prompt_ids, multi_modal_data

        cache = kwargs.get("__prompt_cache__")
        if cache is None:
            prompt_ids, multi_modal_data = await prepare_prompt()
        else:
            uid = str(kwargs["uid"])
            # Store the pending task before awaiting it: concurrent siblings
            # share the first computation as well as its completed result.
            if uid not in cache:
                cache[uid] = asyncio.create_task(prepare_prompt())
            prompt_ids, multi_modal_data = await cache[uid]
        images = multi_modal_data.get("images")
        videos = multi_modal_data.get("videos")

        # Only generation sees the augmented image. Postprocessing below must
        # receive the clean data, so actor pixels, grids and positions stay clean.
        generation_images = images
        rollout_fields = {}
        if "rollout_view" in kwargs and not kwargs.get("__validate__", False):
            from verl.trainer.ppo.noisy_rollout import degrade_images

            rollout_fields = {
                "rollout_view": str(kwargs["rollout_view"]),
                "rollout_degrade_scale": float(kwargs["rollout_degrade_scale"]),
                "rollout_sample_index": int(kwargs["rollout_sample_index"]),
                "rollout_preserve_evidence": bool(kwargs.get("rollout_preserve_evidence", False)),
            }
            if rollout_fields["rollout_view"] == "noisy":
                if not images or videos:
                    raise ValueError("Noisy rollout currently supports image questions only")
                evidence_bbox = (tuple(kwargs["evidence_bbox"])
                                 if rollout_fields["rollout_preserve_evidence"] else None)
                noise_key = ("degraded_images", str(kwargs["uid"]),
                             float(kwargs["rollout_degrade_scale"]), evidence_bbox)
                noise_cache = kwargs.get("__noise_cache__")
                if noise_cache is not None and noise_key in noise_cache:
                    generation_images = noise_cache[noise_key]
                else:
                    generation_images = degrade_images(images, float(kwargs["rollout_degrade_scale"]),
                                                       evidence_bbox=evidence_bbox)
                    if noise_cache is not None:
                        noise_cache[noise_key] = generation_images

        prefix = list(kwargs.get("visual_repair_prefix_ids", []))
        response_length = self._get_response_length()
        remaining = response_length - len(prefix)
        # Keep the original prompt boundary: accepted and repaired text are all
        # response tokens, supervised alongside the new Student continuation.
        metrics = {"generate_sequences": 0.0}
        terminated = bool(prefix) and prefix[-1] in set(self.tokenizer.all_special_ids)
        if remaining <= 0 or terminated:
            generated, log_probs, routed = [], None, None
        else:
            with simple_timer("generate_sequences", metrics):
                result = await self.server_manager.generate(
                    request_id=uuid4().hex,
                    prompt_ids=prompt_ids + prefix,
                    sampling_params={**sampling_params, "max_tokens": remaining},
                    image_data=generation_images, video_data=videos,
                )
            generated = result.token_ids[:remaining]
            log_probs = result.log_probs[:remaining] if result.log_probs else None
            routed = result.routed_experts
        response = (prefix + generated)[:response_length]
        return AgentLoopOutput(
            prompt_ids=prompt_ids, response_ids=response,
            response_mask=[1] * len(response),
            response_logprobs=None if prefix else log_probs,
            routed_experts=None if prefix else routed,
            multi_modal_data=multi_modal_data, num_turns=2, metrics=metrics,
            extra_fields=rollout_fields,
        )
