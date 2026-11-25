# Copyright 2024 Bytedance Ltd. and/or its affiliates
"""Online Student-Teacher-Student visual-grounding relay."""

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
    build_vgr_segment_masks,
    crop_normalized_bbox,
    parse_vgr_relay_plan,
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


@register("visual_grounding_relay")
class VisualGroundingRelayAgentLoop(AgentLoopBase):
    """Let a private 9B teacher relay evidence and text, then return control to the 4B student."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        cfg = self.config.actor_rollout_ref.actor.self_distillation
        self.max_model_length = int(self.config.actor_rollout_ref.rollout.max_model_len)
        self.planner_urls = list(cfg.dsvr_planner_urls)
        self.planner_model = "dsvr-planner"
        self.planner_timeout = float(os.environ.get("VGR_PLANNER_TIMEOUT", "180"))
        self.chunk_tokens = int(cfg.vgr_chunk_tokens)
        self.max_checkpoints = int(cfg.vgr_max_checkpoints)
        self.min_prefix_steps = int(cfg.vgr_min_prefix_steps)
        self.bridge_max_tokens = int(cfg.vgr_bridge_max_tokens)
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

    async def _complete_student(
        self,
        prompt_ids: list[int],
        response_ids: list[int],
        response_logprobs: list[float] | None,
        sampling_params: dict[str, Any],
        images: list[Any],
        videos: list[Any] | None,
        max_response_length: int,
    ) -> tuple[list[int], list[float] | None]:
        remaining = max_response_length - len(response_ids)
        if remaining <= 0 or self._has_eos(response_ids, self.tokenizer.eos_token_id):
            return response_ids, response_logprobs
        final_params = dict(sampling_params)
        final_params["max_tokens"] = remaining
        output = await self._generate(prompt_ids + response_ids, final_params, images, videos)
        final_ids = output.token_ids[:remaining]
        response_ids = response_ids + final_ids
        if response_logprobs is not None:
            if output.log_probs is None:
                response_logprobs = None
            else:
                response_logprobs = response_logprobs + output.log_probs[: len(final_ids)]
        return response_ids, response_logprobs

    def _endpoint(self, route_key: str) -> str:
        return self.planner_urls[hash(route_key) % len(self.planner_urls)]

    async def _chat(self, endpoint: str, content: list[dict[str, Any]], max_tokens: int) -> str:
        payload = {
            "model": self.planner_model,
            "messages": [{"role": "user", "content": content}],
            "temperature": 0.0,
            "max_tokens": max_tokens,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        timeout = aiohttp.ClientTimeout(total=self.planner_timeout)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(f"{endpoint}/chat/completions", json=payload) as response:
                response.raise_for_status()
                result = await response.json()
        return result["choices"][0]["message"]["content"]

    async def _plan(self, image: Any, question: str, prefix: str, route_key: str) -> dict[str, Any]:
        instruction = (
            "You are an online private visual-grounding controller. The student is still solving the problem; "
            "you see only its current prefix, not a completed answer. If the prefix has made a concrete visual "
            "commitment whose verification or correction would materially help the remaining reasoning, set "
            "relay=true and select the tightest useful evidence region in the original image. A relay may confirm "
            "a correct observation; it is not restricted to known errors. Otherwise set relay=false. Coordinates "
            "are normalized to [0,1]. Return JSON only with keys relay, bbox, confidence.\n\n"
            f"Question:\n{question}\n\nCurrent student prefix:\n{prefix}"
        )
        text = await self._chat(
            self._endpoint(route_key),
            [
                {"type": "image_url", "image_url": {"url": _image_data_url(image)}},
                {"type": "text", "text": instruction},
            ],
            max_tokens=80,
        )
        return parse_vgr_relay_plan(text)

    async def _bridge(
        self,
        image: Any,
        crop: Any,
        question: str,
        prefix: str,
        route_key: str,
    ) -> str:
        instruction = (
            "You are the private teacher briefly taking the baton from a student. Inspect the original image, the "
            "selected evidence crop, the question, and the student's exact prefix. Write one short natural-language "
            "bridge that states the decisive visual observation and corrects the prefix only if needed. Continue the "
            "reasoning directly: do not mention crops, tools, teachers, students, or this instruction; do not provide "
            "the complete final answer. Output the bridge text only.\n\n"
            f"Question:\n{question}\n\nStudent prefix:\n{prefix}"
        )
        return (
            await self._chat(
                self._endpoint(route_key),
                [
                    {"type": "image_url", "image_url": {"url": _image_data_url(image)}},
                    {"type": "image_url", "image_url": {"url": _image_data_url(crop)}},
                    {"type": "text", "text": instruction},
                ],
                max_tokens=self.bridge_max_tokens,
            )
        ).strip()

    @staticmethod
    def _has_eos(token_ids: list[int], eos_token_id: int | list[int] | tuple[int, ...] | None) -> bool:
        eos_ids = set(eos_token_id if isinstance(eos_token_id, (list, tuple)) else [eos_token_id])
        eos_ids.discard(None)
        return bool(token_ids and token_ids[-1] in eos_ids)

    def _base_fields(self, response_length: int) -> dict[str, Any]:
        return {
            "vgr_private_images": [],
            "vgr_bridge_mask": [0] * response_length,
            "vgr_suffix_mask": [0] * response_length,
            "vgr_relay": 0,
            "vgr_prefix_length": response_length,
            "vgr_bridge_length": 0,
            "vgr_suffix_length": 0,
            "vgr_checkpoint_count": 0,
            "vgr_bbox_area": 0.0,
            "vgr_planner_confidence": 0.0,
        }

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
        max_response_length = self._get_response_length()

        if self.validate or not original_images:
            output = await self._generate(prompt_ids, dict(sampling_params), original_images, original_videos)
            response_ids = output.token_ids[:max_response_length]
            return AgentLoopOutput(
                prompt_ids=prompt_ids,
                response_ids=response_ids,
                response_mask=[1] * len(response_ids),
                response_logprobs=output.log_probs[: len(response_ids)] if output.log_probs else None,
                multi_modal_data=original_mm,
                num_turns=2,
                metrics={"generate_sequences": time.perf_counter() - start},
                extra_fields=self._base_fields(len(response_ids)),
            )

        response_ids: list[int] = []
        response_logprobs: list[float] | None = []
        checked_steps = 0
        checkpoint_count = 0
        plan: dict[str, Any] = {"relay": False}
        prefix_length = 0
        question = _question_text(messages)

        while len(response_ids) < max_response_length and checkpoint_count < self.max_checkpoints:
            chunk_params = dict(sampling_params)
            chunk_params["max_tokens"] = min(self.chunk_tokens, max_response_length - len(response_ids))
            output = await self._generate(
                prompt_ids + response_ids,
                chunk_params,
                original_images,
                original_videos,
            )
            chunk_ids = output.token_ids[: chunk_params["max_tokens"]]
            if not chunk_ids:
                break
            response_ids.extend(chunk_ids)
            if response_logprobs is not None:
                if output.log_probs is None:
                    response_logprobs = None
                else:
                    response_logprobs.extend(output.log_probs[: len(chunk_ids)])
            if self._has_eos(response_ids, self.tokenizer.eos_token_id):
                break

            steps = response_semantic_steps(self.tokenizer, response_ids)
            complete_steps = [step for step in steps if step.get("complete", False)]
            if len(complete_steps) < self.min_prefix_steps or len(complete_steps) <= checked_steps:
                continue
            candidate = complete_steps[-1]
            prefix_length = int(candidate["end"])
            prefix_text = self.tokenizer.decode(response_ids[:prefix_length], skip_special_tokens=True).strip()
            checkpoint_count += 1
            checked_steps = len(complete_steps)
            try:
                plan = await self._plan(
                    original_images[0],
                    question,
                    prefix_text,
                    f"{kwargs.get('index', '')}:{prefix_text}",
                )
            except (aiohttp.ClientError, TimeoutError, TypeError, ValueError, KeyError, json.JSONDecodeError) as error:
                logger.warning("VGR planner checkpoint skipped: %s", error)
                plan = {"relay": False}
            if plan.get("relay"):
                response_ids = response_ids[:prefix_length]
                if response_logprobs is not None:
                    response_logprobs = response_logprobs[:prefix_length]
                break

        if not plan.get("relay"):
            response_ids, response_logprobs = await self._complete_student(
                prompt_ids,
                response_ids,
                response_logprobs,
                sampling_params,
                original_images,
                original_videos,
                max_response_length,
            )

        if not plan.get("relay"):
            fields = self._base_fields(len(response_ids))
            fields["vgr_checkpoint_count"] = checkpoint_count
            return AgentLoopOutput(
                prompt_ids=prompt_ids,
                response_ids=response_ids,
                response_mask=[1] * len(response_ids),
                response_logprobs=response_logprobs,
                multi_modal_data=original_mm,
                num_turns=2,
                metrics={"generate_sequences": time.perf_counter() - start},
                extra_fields=fields,
            )

        private_crop = crop_normalized_bbox(original_images[0], plan["bbox"])
        prefix_text = self.tokenizer.decode(response_ids, skip_special_tokens=True).strip()
        try:
            bridge_text = await self._bridge(
                original_images[0],
                private_crop,
                question,
                prefix_text,
                f"{kwargs.get('index', '')}:{prefix_text}",
            )
        except (aiohttp.ClientError, TimeoutError, TypeError, ValueError, KeyError) as error:
            logger.warning("VGR bridge generation failed; keeping ordinary OPD trajectory: %s", error)
            response_ids, response_logprobs = await self._complete_student(
                prompt_ids,
                response_ids,
                response_logprobs,
                sampling_params,
                original_images,
                original_videos,
                max_response_length,
            )
            fields = self._base_fields(len(response_ids))
            fields["vgr_checkpoint_count"] = checkpoint_count
            return AgentLoopOutput(
                prompt_ids=prompt_ids,
                response_ids=response_ids,
                response_mask=[1] * len(response_ids),
                response_logprobs=response_logprobs,
                multi_modal_data=original_mm,
                num_turns=2,
                metrics={"generate_sequences": time.perf_counter() - start},
                extra_fields=fields,
            )

        bridge_ids = self.tokenizer.encode("\n" + bridge_text, add_special_tokens=False)
        bridge_budget = min(
            self.bridge_max_tokens,
            max(0, max_response_length - len(response_ids) - 1),
            max(0, self.max_model_length - len(prompt_ids) - len(response_ids) - 1),
        )
        bridge_ids = bridge_ids[:bridge_budget]
        if not bridge_ids:
            response_ids, response_logprobs = await self._complete_student(
                prompt_ids,
                response_ids,
                response_logprobs,
                sampling_params,
                original_images,
                original_videos,
                max_response_length,
            )
            fields = self._base_fields(len(response_ids))
            fields["vgr_checkpoint_count"] = checkpoint_count
            return AgentLoopOutput(
                prompt_ids=prompt_ids,
                response_ids=response_ids,
                response_mask=[1] * len(response_ids),
                response_logprobs=response_logprobs,
                multi_modal_data=original_mm,
                num_turns=2,
                metrics={"generate_sequences": time.perf_counter() - start},
                extra_fields=fields,
            )

        suffix_budget = min(
            max_response_length - len(response_ids) - len(bridge_ids),
            self.max_model_length - len(prompt_ids) - len(response_ids) - len(bridge_ids),
        )
        suffix_params = dict(sampling_params)
        suffix_params["max_tokens"] = suffix_budget
        suffix_output = await self._generate(
            prompt_ids + response_ids + bridge_ids,
            suffix_params,
            original_images,
            original_videos,
        )
        suffix_ids = suffix_output.token_ids[:suffix_budget]
        prefix_ids = response_ids
        response_ids = prefix_ids + bridge_ids + suffix_ids
        bridge_mask, suffix_mask = build_vgr_segment_masks(
            len(prefix_ids), len(bridge_ids), len(suffix_ids)
        )
        x1, y1, x2, y2 = plan["bbox"]
        return AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=response_ids,
            response_mask=[1] * len(response_ids),
            # The bridge is teacher-authored, so central actor scoring must recompute old log-probs.
            response_logprobs=None,
            multi_modal_data=original_mm,
            num_turns=2,
            metrics={"generate_sequences": time.perf_counter() - start},
            extra_fields={
                "vgr_private_images": [private_crop],
                "vgr_bridge_mask": bridge_mask,
                "vgr_suffix_mask": suffix_mask,
                "vgr_relay": 1,
                "vgr_prefix_length": len(prefix_ids),
                "vgr_bridge_length": len(bridge_ids),
                "vgr_suffix_length": len(suffix_ids),
                "vgr_checkpoint_count": checkpoint_count,
                "vgr_bbox_area": (x2 - x1) * (y2 - y1),
                "vgr_planner_confidence": float(plan["confidence"]),
            },
        )
