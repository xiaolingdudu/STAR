# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
"""Student-authored rollout with a private visual observation relayed mid-response."""

from __future__ import annotations

import logging
import os
import time
from typing import Any
from uuid import uuid4

from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopOutput, register
from verl.tools.utils.tool_registry import initialize_tools_from_config
from verl.utils.silent_visual_relay import (
    append_private_images,
    build_suffix_mask,
    coerce_image_list,
    fit_relay_generation_budget,
    split_response_for_relay,
)

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


@register("silent_visual_relay")
class SilentVisualRelayAgentLoop(AgentLoopBase):
    """Generate a full-image prefix, privately add a crop, then let the student recover."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_model_length = int(self.config.actor_rollout_ref.rollout.max_model_len)
        self.prefix_ratio = float(self.config.actor_rollout_ref.actor.self_distillation.svr_prefix_ratio)
        self.crop_image_key = self.config.actor_rollout_ref.actor.self_distillation.teacher_image_key

        tool_config_path = self.config.data.tool_config_path
        tool_list = initialize_tools_from_config(tool_config_path) if tool_config_path else []
        self.tool_schemas = [tool.tool_schema.model_dump(exclude_unset=True, exclude_none=True) for tool in tool_list]

    async def _generate(self, prompt_ids, sampling_params, images, videos):
        return await self.server_manager.generate(
            request_id=uuid4().hex,
            prompt_ids=prompt_ids,
            sampling_params=sampling_params,
            image_data=images,
            video_data=videos,
        )

    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        messages = list(kwargs["raw_prompt"])
        original_mm = await self.process_vision_info(messages)
        original_images = original_mm.get("images")
        original_videos = original_mm.get("videos")
        prompt_ids = await self.apply_chat_template(
            messages,
            tools=self.tool_schemas,
            images=original_images,
            videos=original_videos,
        )

        start = time.perf_counter()
        base_output = await self._generate(
            prompt_ids,
            dict(sampling_params),
            original_images,
            original_videos,
        )
        response_length = self._get_response_length()
        base_ids = base_output.token_ids[:response_length]

        # Validation measures the deployable no-crop student. Training uses the
        # upstream evidence crop only as a private observation for suffix generation.
        if self.validate:
            return AgentLoopOutput(
                prompt_ids=prompt_ids,
                response_ids=base_ids,
                response_mask=[1] * len(base_ids),
                response_logprobs=base_output.log_probs[: len(base_ids)] if base_output.log_probs else None,
                multi_modal_data=original_mm,
                num_turns=2,
                metrics={"generate_sequences": time.perf_counter() - start},
            )

        prefix_ids, _ = split_response_for_relay(
            base_ids,
            self.prefix_ratio,
            getattr(self.tokenizer, "eos_token_id", None),
        )
        private_images = coerce_image_list(kwargs.get(self.crop_image_key))
        if not prefix_ids or not private_images:
            return AgentLoopOutput(
                prompt_ids=prompt_ids,
                response_ids=base_ids,
                response_mask=[1] * len(base_ids),
                response_logprobs=base_output.log_probs[: len(base_ids)] if base_output.log_probs else None,
                multi_modal_data=original_mm,
                num_turns=2,
                metrics={"generate_sequences": time.perf_counter() - start},
                extra_fields={
                    "svr_suffix_mask": [0] * len(base_ids),
                    "svr_prefix_length": len(base_ids),
                    "svr_suffix_length": 0,
                },
            )

        relay_messages = append_private_images(messages, private_images)
        relay_mm = await self.process_vision_info(relay_messages)
        relay_images = relay_mm.get("images")
        relay_videos = relay_mm.get("videos")
        relay_prompt_ids = await self.apply_chat_template(
            relay_messages,
            tools=self.tool_schemas,
            images=relay_images,
            videos=relay_videos,
            truncate_prompt_ids=False,
        )
        prefix_ids, remaining = fit_relay_generation_budget(
            prefix_ids,
            len(relay_prompt_ids),
            self.max_model_length,
            response_length,
        )
        suffix_params = dict(sampling_params)
        suffix_params["max_tokens"] = remaining
        suffix_output = await self._generate(
            relay_prompt_ids + prefix_ids,
            suffix_params,
            relay_images,
            relay_videos,
        )
        suffix_ids = suffix_output.token_ids[:remaining]
        response_ids = prefix_ids + suffix_ids

        response_logprobs = None
        if base_output.log_probs is not None and suffix_output.log_probs is not None:
            response_logprobs = base_output.log_probs[: len(prefix_ids)] + suffix_output.log_probs[: len(suffix_ids)]

        return AgentLoopOutput(
            # The trainable student is replayed without the crop on the fixed text trajectory.
            prompt_ids=prompt_ids,
            response_ids=response_ids,
            response_mask=[1] * len(response_ids),
            response_logprobs=response_logprobs,
            multi_modal_data=original_mm,
            num_turns=2,
            metrics={"generate_sequences": time.perf_counter() - start},
            extra_fields={
                "svr_suffix_mask": build_suffix_mask(len(prefix_ids), len(suffix_ids)),
                "svr_prefix_length": len(prefix_ids),
                "svr_suffix_length": len(suffix_ids),
            },
        )
