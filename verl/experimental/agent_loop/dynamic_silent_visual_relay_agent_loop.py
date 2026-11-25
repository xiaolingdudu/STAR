# Copyright 2024 Bytedance Ltd. and/or its affiliates
"""Dynamic Silent Visual Relay with a private 9B visual planner."""

from __future__ import annotations

import base64
from io import BytesIO
import json
import logging
import os
import time
from typing import Any
from uuid import uuid4

import aiohttp

from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopOutput, register
from verl.tools.utils.tool_registry import initialize_tools_from_config
from verl.utils.silent_visual_relay import (
    append_private_images,
    build_suffix_mask,
    crop_normalized_bbox,
    fit_relay_generation_budget,
    parse_dynamic_relay_plan,
    response_semantic_steps,
)

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


def _question_text(messages: list[dict[str, Any]]) -> str:
    chunks = []
    for message in messages:
        if message.get("role") != "user":
            continue
        content = message.get("content", "")
        if isinstance(content, str):
            chunks.append(content)
        else:
            chunks.extend(
                str(item.get("text", ""))
                for item in content
                if isinstance(item, dict) and item.get("type") == "text"
            )
    return "\n".join(chunk for chunk in chunks if chunk).strip()


def _image_data_url(image: Any, max_side: int = 1280) -> str:
    image = image.convert("RGB")
    if max(image.size) > max_side:
        scale = max_side / max(image.size)
        image = image.resize((round(image.width * scale), round(image.height * scale)))
    output = BytesIO()
    image.save(output, format="JPEG", quality=88)
    return "data:image/jpeg;base64," + base64.b64encode(output.getvalue()).decode("ascii")


@register("dynamic_silent_visual_relay")
class DynamicSilentVisualRelayAgentLoop(AgentLoopBase):
    """Audit a student trajectory once, relay real visual evidence, then rewrite."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_model_length = int(self.config.actor_rollout_ref.rollout.max_model_len)
        raw_planner_urls = self.config.actor_rollout_ref.actor.self_distillation.dsvr_planner_urls
        self.planner_urls = list(raw_planner_urls)
        self.planner_model = "dsvr-planner"
        self.planner_timeout = float(os.environ.get("DSVR_PLANNER_TIMEOUT", "180"))
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

    async def _plan(self, image: Any, question: str, steps: list[dict[str, Any]], route_key: str) -> dict[str, Any]:
        numbered_steps = "\n".join(f"{index + 1}. {step['text']}" for index, step in enumerate(steps))
        instruction = (
            "You are a private visual evidence planner. Inspect the original image, question, and the student's "
            "completed trajectory. Decide whether a visual relay can improve or verify the reasoning. If so, locate "
            "the earliest visually grounded commitment that should be rewritten and select one evidence region from "
            "the original image. Coordinates are normalized to [0,1]. The rollback step is the first numbered step "
            "to regenerate; it may be earlier than the detected error. Return JSON only with keys relay, "
            "detected_at_step, rollback_to_step, bbox, confidence. Use relay=false only when the trajectory is already "
            "fully correct and visually supported.\n\n"
            f"Question:\n{question}\n\nStudent trajectory:\n{numbered_steps}"
        )
        endpoint = self.planner_urls[hash(route_key) % len(self.planner_urls)]
        payload = {
            "model": self.planner_model,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": _image_data_url(image)}},
                    {"type": "text", "text": instruction},
                ],
            }],
            "temperature": 0.0,
            "max_tokens": 96,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        timeout = aiohttp.ClientTimeout(total=self.planner_timeout)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(f"{endpoint}/chat/completions", json=payload) as response:
                response.raise_for_status()
                result = await response.json()
        text = result["choices"][0]["message"]["content"]
        return parse_dynamic_relay_plan(text, len(steps))

    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        messages = list(kwargs["raw_prompt"])
        original_mm = await self.process_vision_info(messages)
        original_images = original_mm.get("images") or []
        original_videos = original_mm.get("videos")
        prompt_ids = await self.apply_chat_template(
            messages,
            tools=self.tool_schemas,
            images=original_images,
            videos=original_videos,
        )
        start = time.perf_counter()
        response_length = self._get_response_length()
        base_output = await self._generate(prompt_ids, dict(sampling_params), original_images, original_videos)
        base_ids = base_output.token_ids[:response_length]
        base_fields = {
            "dsvr_private_images": [],
            "dsvr_suffix_mask": [0] * len(base_ids),
            "dsvr_relay": 0,
            "dsvr_detected_step": 0,
            "dsvr_rollback_step": 0,
            "dsvr_prefix_length": len(base_ids),
            "dsvr_suffix_length": 0,
            "dsvr_bbox_area": 0.0,
            "dsvr_planner_confidence": 0.0,
        }
        if self.validate or not original_images:
            return AgentLoopOutput(
                prompt_ids=prompt_ids,
                response_ids=base_ids,
                response_mask=[1] * len(base_ids),
                response_logprobs=base_output.log_probs[: len(base_ids)] if base_output.log_probs else None,
                multi_modal_data=original_mm,
                num_turns=2,
                metrics={"generate_sequences": time.perf_counter() - start},
                extra_fields=base_fields,
            )

        steps = response_semantic_steps(self.tokenizer, base_ids)
        try:
            plan = await self._plan(original_images[0], _question_text(messages), steps, str(base_ids))
        except (
            aiohttp.ClientError,
            TimeoutError,
            TypeError,
            ValueError,
            KeyError,
            json.JSONDecodeError,
        ) as error:
            logger.warning("dynamic relay planner fallback to ordinary OPD: %s", error)
            plan = {"relay": False}
        if not plan.get("relay"):
            return AgentLoopOutput(
                prompt_ids=prompt_ids,
                response_ids=base_ids,
                response_mask=[1] * len(base_ids),
                response_logprobs=base_output.log_probs[: len(base_ids)] if base_output.log_probs else None,
                multi_modal_data=original_mm,
                num_turns=2,
                metrics={"generate_sequences": time.perf_counter() - start},
                extra_fields=base_fields,
            )

        rollback_step = int(plan["rollback_to_step"])
        prefix_length = steps[rollback_step - 1]["start"]
        private_crop = crop_normalized_bbox(original_images[0], plan["bbox"])
        relay_messages = append_private_images(messages, [private_crop])
        relay_mm = await self.process_vision_info(relay_messages)
        relay_prompt_ids = await self.apply_chat_template(
            relay_messages,
            tools=self.tool_schemas,
            images=relay_mm.get("images"),
            videos=relay_mm.get("videos"),
            truncate_prompt_ids=False,
        )
        prefix_ids, remaining = fit_relay_generation_budget(
            base_ids[:prefix_length],
            len(relay_prompt_ids),
            self.max_model_length,
            response_length,
        )
        suffix_params = dict(sampling_params)
        suffix_params["max_tokens"] = remaining
        suffix_output = await self._generate(
            relay_prompt_ids + prefix_ids,
            suffix_params,
            relay_mm.get("images"),
            relay_mm.get("videos"),
        )
        suffix_ids = suffix_output.token_ids[:remaining]
        response_ids = prefix_ids + suffix_ids
        response_logprobs = None
        if base_output.log_probs is not None and suffix_output.log_probs is not None:
            response_logprobs = base_output.log_probs[: len(prefix_ids)] + suffix_output.log_probs[: len(suffix_ids)]
        x1, y1, x2, y2 = plan["bbox"]
        return AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=response_ids,
            response_mask=[1] * len(response_ids),
            response_logprobs=response_logprobs,
            multi_modal_data=original_mm,
            num_turns=2,
            metrics={"generate_sequences": time.perf_counter() - start},
            extra_fields={
                "dsvr_private_images": [private_crop],
                "dsvr_suffix_mask": build_suffix_mask(len(prefix_ids), len(suffix_ids)),
                "dsvr_relay": 1,
                "dsvr_detected_step": int(plan["detected_at_step"]),
                "dsvr_rollback_step": rollback_step,
                "dsvr_prefix_length": len(prefix_ids),
                "dsvr_suffix_length": len(suffix_ids),
                "dsvr_bbox_area": (x2 - x1) * (y2 - y1),
                "dsvr_planner_confidence": float(plan["confidence"]),
            },
        )
