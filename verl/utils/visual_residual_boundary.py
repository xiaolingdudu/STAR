"""Visual-residual change-point helpers for VRBR-OPD."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class VisualResidualBoundary:
    boundary: int | None
    score: float
    residual_mean: float
    positive_fraction: float


def extract_aligned_prompt_logprobs(
    prompt_token_ids: list[int],
    prompt_logprobs: list[dict[str, Any] | dict[int, Any] | None],
    response_ids: list[int],
    *,
    minimum_match: int,
) -> list[float]:
    """Extract chosen-token prompt log-probs for a response suffix."""

    if len(prompt_token_ids) != len(prompt_logprobs):
        raise ValueError("prompt token IDs and prompt log-probs have different lengths")
    maximum = min(len(prompt_token_ids), len(response_ids))
    matched = 0
    for length in range(maximum, minimum_match - 1, -1):
        if prompt_token_ids[-length:] == response_ids[-length:]:
            matched = length
            break
    if matched < minimum_match:
        raise ValueError(
            f"teacher prompt does not end with the Student response: matched={matched}, required={minimum_match}"
        )

    values = []
    offset = len(prompt_token_ids) - matched
    for position, token_id in enumerate(prompt_token_ids[offset:], start=offset):
        item = prompt_logprobs[position]
        if item is None:
            raise ValueError(f"missing teacher prompt log-prob at position {position}")
        token_value = item.get(token_id)
        if token_value is None:
            token_value = item.get(str(token_id))
        if token_value is None:
            raise ValueError(f"teacher prompt log-probs omit chosen token {token_id} at position {position}")
        logprob = token_value.get("logprob") if isinstance(token_value, dict) else getattr(token_value, "logprob", None)
        if logprob is None:
            raise ValueError(f"invalid teacher prompt log-prob payload at position {position}")
        values.append(float(logprob))
    return values


def infer_visual_residual_boundary(
    original_logprobs: list[float],
    privileged_logprobs: list[float],
    *,
    response_start: int,
    minimum_prefix_tokens: int,
    window: int,
    threshold: float,
) -> VisualResidualBoundary:
    """Infer one relay boundary from crop-induced suppression of draft tokens."""

    if len(original_logprobs) != len(privileged_logprobs):
        raise ValueError("ordinary and privileged teacher scores must align")
    if window <= 0:
        raise ValueError("boundary window must be positive")

    residuals = [ordinary - privileged for ordinary, privileged in zip(original_logprobs, privileged_logprobs)]
    start = max(0, minimum_prefix_tokens - response_start)
    if start >= len(residuals):
        return VisualResidualBoundary(None, 0.0, 0.0, 0.0)

    positive = [max(0.0, value) for value in residuals]
    best_index = start
    best_score = -1.0
    for index in range(start, len(positive)):
        local = positive[index : index + window]
        score = sum(local) / len(local)
        if score > best_score:
            best_index = index
            best_score = score

    valid = residuals[start:]
    positive_fraction = sum(value > 0.0 for value in valid) / len(valid)
    boundary = response_start + best_index if best_score >= threshold else None
    return VisualResidualBoundary(
        boundary=boundary,
        score=best_score,
        residual_mean=sum(valid) / len(valid),
        positive_fraction=positive_fraction,
    )
