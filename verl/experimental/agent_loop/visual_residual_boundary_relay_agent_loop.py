# Copyright 2024 Bytedance Ltd. and/or its affiliates
"""Visual-Residual Boundary Relay: Student draft, Teacher bridge, Student suffix."""

from __future__ import annotations

import asyncio
import json
import logging
import os
from pathlib import Path
import time
from typing import Any

import aiohttp
from PIL import Image

from verl.experimental.agent_loop.agent_loop import AgentLoopOutput, register
from verl.experimental.agent_loop.visual_grounding_relay_agent_loop import (
    VisualGroundingRelayAgentLoop,
    _image_data_url,
    _question_text,
)
from verl.utils.silent_visual_relay import build_vgr_segment_masks
from verl.utils.visual_residual_boundary import (
    extract_aligned_prompt_logprobs,
    infer_visual_residual_boundary,
)

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


def _load_prepared_crop(value: Any) -> Image.Image:
    reference = value[0] if isinstance(value, (list, tuple)) else value
    if isinstance(reference, Image.Image):
        return reference.convert("RGB")
    if not isinstance(reference, dict) or "path" not in reference:
        raise ValueError("VRBR-OPD expects bbox_images to contain one {'path': ...} reference")
    with Image.open(Path(reference["path"])) as image:
        return image.convert("RGB")


@register("visual_residual_boundary_relay")
class VisualResidualBoundaryRelayAgentLoop(VisualGroundingRelayAgentLoop):
    """Infer one handoff boundary from privileged visual likelihood residuals."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        cfg = self.config.actor_rollout_ref.actor.self_distillation
        self.chunk_tokens = int(cfg.vrbr_chunk_tokens)
        self.max_checkpoints = int(cfg.vrbr_max_checkpoints)
        self.minimum_prefix_tokens = int(cfg.vrbr_min_prefix_tokens)
        self.bridge_max_tokens = int(cfg.vrbr_bridge_max_tokens)
        self.boundary_window = int(cfg.vrbr_boundary_window)
        self.boundary_threshold = float(cfg.vrbr_boundary_threshold)

    def _base_fields(self, response_length: int) -> dict[str, Any]:
        fields = super()._base_fields(response_length)
        fields.update(
            {
                "vrbr_relay": 0,
                "vrbr_boundary": response_length,
                "vrbr_checked_tokens": 0,
                "vrbr_checkpoint_count": 0,
                "vrbr_boundary_score": 0.0,
                "vrbr_residual_mean": 0.0,
                "vrbr_positive_residual_fraction": 0.0,
                "vrbr_score_failure": 0,
            }
        )
        return fields

    async def _score_view(
        self,
        image: Any,
        crop: Any | None,
        question: str,
        response_ids: list[int],
        minimum_match: int,
        route_key: str,
    ) -> list[float]:
        content = [{"type": "image_url", "image_url": {"url": _image_data_url(image)}}]
        if crop is not None:
            content.append({"type": "image_url", "image_url": {"url": _image_data_url(crop)}})
        content.append({"type": "text", "text": question})
        # Keep whitespace exactly as decoded.  The scoring endpoint must end in
        # the same response token IDs so prompt-logprob alignment stays exact.
        draft = self.tokenizer.decode(
            response_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        payload = {
            "model": self.planner_model,
            "messages": [
                {"role": "user", "content": content},
                {"role": "assistant", "content": draft},
            ],
            "temperature": 0.0,
            "max_tokens": 1,
            "prompt_logprobs": 1,
            "return_token_ids": True,
            "continue_final_message": True,
            "add_generation_prompt": False,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        timeout = aiohttp.ClientTimeout(total=self.planner_timeout)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(
                f"{self._endpoint(route_key)}/chat/completions", json=payload
            ) as response:
                response.raise_for_status()
                result = await response.json()
        return extract_aligned_prompt_logprobs(
            list(result["prompt_token_ids"]),
            list(result["prompt_logprobs"]),
            response_ids,
            minimum_match=minimum_match,
        )

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

        private_crop = _load_prepared_crop(kwargs["bbox_images"])
        question = _question_text(messages)
        response_ids: list[int] = []
        response_logprobs: list[float] | None = []
        checkpoint_count = 0
        checked_tokens = 0
        score_failure = 0
        decision = None

        while len(response_ids) < max_response_length and checkpoint_count < self.max_checkpoints:
            chunk_start = len(response_ids)
            chunk_params = dict(sampling_params)
            chunk_params["max_tokens"] = min(self.chunk_tokens, max_response_length - chunk_start)
            output = await self._generate(prompt_ids + response_ids, chunk_params, original_images, original_videos)
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

            checkpoint_count += 1
            checked_tokens += len(chunk_ids)
            try:
                ordinary, privileged = await asyncio.gather(
                    self._score_view(
                        original_images[0], None, question, response_ids, len(chunk_ids),
                        f"{kwargs.get('index', '')}:ordinary:{checkpoint_count}",
                    ),
                    self._score_view(
                        original_images[0], private_crop, question, response_ids, len(chunk_ids),
                        f"{kwargs.get('index', '')}:privileged:{checkpoint_count}",
                    ),
                )
                matched = min(len(ordinary), len(privileged))
                response_start = len(response_ids) - matched
                decision = infer_visual_residual_boundary(
                    ordinary[-matched:],
                    privileged[-matched:],
                    response_start=response_start,
                    minimum_prefix_tokens=self.minimum_prefix_tokens,
                    window=self.boundary_window,
                    threshold=self.boundary_threshold,
                )
            except (aiohttp.ClientError, TimeoutError, TypeError, ValueError, KeyError, json.JSONDecodeError) as error:
                logger.warning("VRBR teacher scoring skipped: %s", error)
                score_failure = 1
                break
            if decision.boundary is not None:
                response_ids = response_ids[: decision.boundary]
                if response_logprobs is not None:
                    response_logprobs = response_logprobs[: decision.boundary]
                break

        if decision is None or decision.boundary is None:
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
            fields.update(
                {
                    "vrbr_checked_tokens": checked_tokens,
                    "vrbr_checkpoint_count": checkpoint_count,
                    "vrbr_boundary_score": 0.0 if decision is None else decision.score,
                    "vrbr_residual_mean": 0.0 if decision is None else decision.residual_mean,
                    "vrbr_positive_residual_fraction": 0.0 if decision is None else decision.positive_fraction,
                    "vrbr_score_failure": score_failure,
                    "vgr_checkpoint_count": checkpoint_count,
                }
            )
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

        prefix_ids = response_ids
        prefix_text = self.tokenizer.decode(prefix_ids, skip_special_tokens=True).strip()
        try:
            bridge_text = await self._bridge(
                original_images[0],
                private_crop,
                question,
                prefix_text,
                f"{kwargs.get('index', '')}:bridge:{decision.boundary}",
            )
        except (aiohttp.ClientError, TimeoutError, TypeError, ValueError, KeyError, json.JSONDecodeError) as error:
            logger.warning("VRBR teacher bridge skipped: %s", error)
            bridge_text = ""
        bridge_ids = self.tokenizer.encode("\n" + bridge_text, add_special_tokens=False)
        bridge_budget = min(
            self.bridge_max_tokens,
            max(0, max_response_length - len(prefix_ids) - 1),
            max(0, self.max_model_length - len(prompt_ids) - len(prefix_ids) - 1),
        )
        bridge_ids = bridge_ids[:bridge_budget]
        if not bridge_text.strip() or not bridge_ids:
            response_ids, response_logprobs = await self._complete_student(
                prompt_ids,
                prefix_ids,
                response_logprobs,
                sampling_params,
                original_images,
                original_videos,
                max_response_length,
            )
            fields = self._base_fields(len(response_ids))
            fields.update(
                {
                    "vrbr_checked_tokens": checked_tokens,
                    "vrbr_checkpoint_count": checkpoint_count,
                    "vrbr_boundary_score": decision.score,
                    "vrbr_residual_mean": decision.residual_mean,
                    "vrbr_positive_residual_fraction": decision.positive_fraction,
                    "vrbr_score_failure": 1,
                    "vgr_checkpoint_count": checkpoint_count,
                }
            )
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
            max_response_length - len(prefix_ids) - len(bridge_ids),
            self.max_model_length - len(prompt_ids) - len(prefix_ids) - len(bridge_ids),
        )
        suffix_ids: list[int] = []
        if bridge_ids and suffix_budget > 0:
            suffix_params = dict(sampling_params)
            suffix_params["max_tokens"] = suffix_budget
            suffix_output = await self._generate(
                prompt_ids + prefix_ids + bridge_ids,
                suffix_params,
                original_images,
                original_videos,
            )
            suffix_ids = suffix_output.token_ids[:suffix_budget]
        response_ids = prefix_ids + bridge_ids + suffix_ids
        bridge_mask, suffix_mask = build_vgr_segment_masks(
            len(prefix_ids), len(bridge_ids), len(suffix_ids)
        )
        crop_area = (private_crop.width * private_crop.height) / (
            original_images[0].width * original_images[0].height
        )
        return AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=response_ids,
            response_mask=[1] * len(response_ids),
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
                "vgr_bbox_area": crop_area,
                "vgr_planner_confidence": 0.0,
                "vrbr_relay": 1,
                "vrbr_boundary": decision.boundary,
                "vrbr_checked_tokens": checked_tokens,
                "vrbr_checkpoint_count": checkpoint_count,
                "vrbr_boundary_score": decision.score,
                "vrbr_residual_mean": decision.residual_mean,
                "vrbr_positive_residual_fraction": decision.positive_fraction,
                "vrbr_score_failure": 0,
            },
        )
