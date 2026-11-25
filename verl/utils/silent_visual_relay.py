"""Small helpers shared by Silent Visual Relay rollout and teacher scoring."""

from __future__ import annotations

from copy import deepcopy
import json
import re
from typing import Any

import numpy as np


def response_semantic_steps(tokenizer: Any, token_ids: list[int]) -> list[dict[str, Any]]:
    """Split a response into coarse reasoning steps while retaining token boundaries."""

    steps = []
    pieces = []
    start = 0
    for index, token_id in enumerate(token_ids):
        piece = tokenizer.decode(
            [token_id],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        pieces.append(piece)
        text = "".join(pieces).strip()
        is_boundary = bool(re.search(r"(?:\n|[.!?;。！？；])\s*$", piece))
        if text and is_boundary:
            steps.append({"text": text, "start": start, "end": index + 1, "complete": True})
            pieces = []
            start = index + 1
    tail = "".join(pieces).strip()
    if tail:
        steps.append({"text": tail, "start": start, "end": len(token_ids), "complete": False})
    return steps


def parse_dynamic_relay_plan(text: str, step_count: int) -> dict[str, Any]:
    """Parse the planner's compact JSON contract and clamp it to the trajectory."""

    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match is None:
        raise ValueError("planner did not return a JSON object")
    plan = json.loads(match.group(0))
    relay = bool(plan.get("relay", False))
    if not relay:
        return {"relay": False, "detected_at_step": 0, "rollback_to_step": 0, "bbox": None, "confidence": 0.0}
    if step_count <= 0:
        raise ValueError("cannot relay an empty response")
    bbox = [float(value) for value in plan["bbox"]]
    if len(bbox) != 4:
        raise ValueError("bbox must contain four normalized coordinates")
    x1, y1, x2, y2 = [min(1.0, max(0.0, value)) for value in bbox]
    if x2 - x1 < 0.02 or y2 - y1 < 0.02:
        raise ValueError("planner bbox is empty")
    detected_value = plan.get("detected_at_step")
    rollback_value = plan.get("rollback_to_step")
    if rollback_value is None:
        rollback_value = detected_value
    if rollback_value is None:
        raise ValueError("relay plan requires rollback_to_step or detected_at_step")
    rollback = min(step_count, max(1, int(rollback_value)))
    detected = min(
        step_count,
        max(rollback, int(detected_value if detected_value is not None else rollback)),
    )
    return {
        "relay": True,
        "detected_at_step": detected,
        "rollback_to_step": rollback,
        "bbox": [x1, y1, x2, y2],
        "confidence": min(1.0, max(0.0, float(plan.get("confidence", 0.0)))),
    }


def parse_vgr_relay_plan(text: str) -> dict[str, Any]:
    """Parse the online visual-grounding relay decision."""

    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match is None:
        raise ValueError("planner did not return a JSON object")
    plan = json.loads(match.group(0))
    if not bool(plan.get("relay", False)):
        return {"relay": False, "bbox": None, "confidence": 0.0}
    bbox = [float(value) for value in plan["bbox"]]
    if len(bbox) != 4:
        raise ValueError("bbox must contain four normalized coordinates")
    x1, y1, x2, y2 = [min(1.0, max(0.0, value)) for value in bbox]
    if x2 - x1 < 0.02 or y2 - y1 < 0.02:
        raise ValueError("planner bbox is empty")
    return {
        "relay": True,
        "bbox": [x1, y1, x2, y2],
        "confidence": min(1.0, max(0.0, float(plan.get("confidence", 0.0)))),
    }


def crop_normalized_bbox(image: Any, bbox: list[float]) -> Any:
    """Crop real pixels from a PIL-like image using normalized coordinates."""

    width, height = image.size
    x1, y1, x2, y2 = bbox
    box = (
        max(0, min(width - 1, round(x1 * width))),
        max(0, min(height - 1, round(y1 * height))),
        max(1, min(width, round(x2 * width))),
        max(1, min(height, round(y2 * height))),
    )
    if box[2] <= box[0] or box[3] <= box[1]:
        raise ValueError(f"invalid crop box {box} for image size {(width, height)}")
    return image.crop(box).convert("RGB")


def coerce_image_list(images: Any) -> list[Any]:
    if isinstance(images, np.ndarray):
        return images.tolist()
    if images is None:
        return []
    if isinstance(images, (list, tuple)):
        return list(images)
    return [images]


def image_message_item(image: Any) -> dict[str, Any]:
    if isinstance(image, dict):
        item = {"type": "image"}
        if "image" in image:
            item["image"] = image["image"]
        elif "path" in image:
            item["image"] = image["path"]
        if "path" in image:
            item["path"] = image["path"]
        if "bytes" in image:
            item["bytes"] = image["bytes"]
        if len(item) > 1:
            return item
    return {"type": "image", "image": image}


def append_private_images(messages: list[dict], images: Any) -> list[dict]:
    """Insert private observations after the original user image placeholders."""

    private_images = coerce_image_list(images)
    if not private_images:
        return deepcopy(messages)

    result = deepcopy(messages)
    user_index = next(
        (index for index in range(len(result) - 1, -1, -1) if result[index].get("role") == "user"),
        len(result) - 1,
    )
    content = result[user_index].get("content")
    if not isinstance(content, list):
        raise TypeError("Silent Visual Relay requires list-form multimodal user content.")

    insert_at = max(
        (index + 1 for index, item in enumerate(content) if isinstance(item, dict) and item.get("type") == "image"),
        default=0,
    )
    result[user_index]["content"] = (
        list(content[:insert_at])
        + [image_message_item(image) for image in private_images]
        + list(content[insert_at:])
    )
    return result


def split_response_for_relay(
    token_ids: list[int],
    prefix_ratio: float,
    eos_token_id: int | list[int] | tuple[int, ...] | None,
) -> tuple[list[int], list[int]]:
    """Split a completed student response at a response-relative token boundary."""

    if not 0.0 < prefix_ratio < 1.0:
        raise ValueError(f"prefix_ratio must be in (0, 1), got {prefix_ratio}")
    eos_ids = set(eos_token_id if isinstance(eos_token_id, (list, tuple)) else [eos_token_id])
    eos_ids.discard(None)
    active = list(token_ids)
    if active and active[-1] in eos_ids:
        active = active[:-1]
    if len(active) < 2:
        return active, []
    prefix_length = min(len(active) - 1, max(1, round(len(active) * prefix_ratio)))
    return active[:prefix_length], active[prefix_length:]


def build_suffix_mask(prefix_length: int, suffix_length: int) -> list[int]:
    return [0] * prefix_length + [1] * suffix_length


def build_vgr_segment_masks(prefix_length: int, bridge_length: int, suffix_length: int) -> tuple[list[int], list[int]]:
    """Return response-relative masks for the teacher bridge and student suffix."""

    bridge_mask = [0] * prefix_length + [1] * bridge_length + [0] * suffix_length
    suffix_mask = [0] * (prefix_length + bridge_length) + [1] * suffix_length
    return bridge_mask, suffix_mask


def fit_relay_generation_budget(
    prefix_ids: list[int],
    relay_prompt_length: int,
    max_model_length: int,
    max_response_length: int,
) -> tuple[list[int], int]:
    """Fit the student prefix and recovery suffix into the rollout context."""

    available_response_length = max_model_length - relay_prompt_length
    if available_response_length < 1 or max_response_length < 1:
        raise ValueError("Silent Visual Relay crop prompt leaves no room for a recovery suffix.")
    max_prefix_length = max(0, min(len(prefix_ids), max_response_length - 1, available_response_length - 1))
    fitted_prefix = prefix_ids[:max_prefix_length]
    suffix_length = min(
        max_response_length - len(fitted_prefix),
        available_response_length - len(fitted_prefix),
    )
    return fitted_prefix, suffix_length
