"""Visual Evidence Relay OPD utilities.

VER-OPD-v1 is the branch-free version of counterfactual visual rescue.  It
selects failed on-policy rollouts only when a sibling rollout from the same
prompt succeeds, then tilts the student distribution along the crop-vs-full
teacher direction from the earliest visually sensitive response position.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch


def compute_rescue_opportunity_weights(
    *,
    sequence_scores: torch.Tensor,
    uids: Sequence[Any],
    success_threshold: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Weight failed rollouts that have a successful sibling continuation."""

    scores = sequence_scores.float().reshape(-1)
    if len(uids) != scores.numel():
        raise ValueError(f"uid count {len(uids)} does not match score count {scores.numel()}")

    group_max = torch.empty_like(scores)
    normalized_uids = [str(uid) for uid in uids]
    for uid in dict.fromkeys(normalized_uids):
        indices = [index for index, candidate in enumerate(normalized_uids) if candidate == uid]
        index_tensor = torch.as_tensor(indices, device=scores.device, dtype=torch.long)
        group_max[index_tensor] = scores[index_tensor].max()

    failed = scores < float(success_threshold)
    group_solved = group_max >= float(success_threshold)
    opportunity = failed & group_solved
    weights = torch.where(opportunity, torch.relu(group_max - scores), torch.zeros_like(scores)).detach()

    active_scores = weights[opportunity]
    metrics = {
        "self_distillation/ver_failure_fraction": failed.float().mean().detach().item(),
        "self_distillation/ver_group_solved_fraction": group_solved.float().mean().detach().item(),
        "self_distillation/ver_rescue_opportunity_fraction": opportunity.float().mean().detach().item(),
        "self_distillation/ver_outcome_gap_mean": (
            active_scores.mean().detach().item() if active_scores.numel() else 0.0
        ),
    }
    return weights.to(dtype=torch.float32), metrics


def compute_active_token_normalization(
    *,
    response_mask: torch.Tensor,
    rollout_weights: torch.Tensor,
    teacher_present_mask: torch.Tensor,
) -> torch.Tensor:
    """Scale sparse VER tokens to the token mean of the available batch."""

    active_tokens = (
        response_mask * (rollout_weights > 0).to(response_mask.dtype).unsqueeze(1)
    ).sum()
    if active_tokens <= 0:
        return torch.zeros((), dtype=torch.float32, device=response_mask.device)
    available_tokens = (response_mask * teacher_present_mask.unsqueeze(1)).sum()
    return (available_tokens / active_tokens).to(dtype=torch.float32)


def build_visual_relay_target(
    *,
    student_log_probs: torch.Tensor,
    crop_teacher_log_probs: torch.Tensor,
    full_teacher_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    rollout_weights: torch.Tensor,
    beta: float,
    min_evidence: float,
    relay_span: int,
    support_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Construct q*(v) proportional to p_S(v) exp(beta * rho * r_visual(v))."""

    if student_log_probs.shape != crop_teacher_log_probs.shape:
        raise ValueError("student and crop-teacher log-probs must share shape")
    if student_log_probs.shape != full_teacher_log_probs.shape:
        raise ValueError("student and full-teacher log-probs must share shape")
    if response_mask.shape != student_log_probs.shape[:2]:
        raise ValueError("response_mask must match the batch and response dimensions")
    if rollout_weights.reshape(-1).shape[0] != student_log_probs.shape[0]:
        raise ValueError("rollout_weights must have one value per rollout")
    if relay_span <= 0:
        raise ValueError("relay_span must be positive")

    valid_support = (
        torch.ones_like(student_log_probs, dtype=torch.bool)
        if support_mask is None
        else support_mask.to(device=student_log_probs.device, dtype=torch.bool)
    )
    student_base = student_log_probs.detach().masked_fill(~valid_support, -1e9)
    crop = crop_teacher_log_probs.detach().masked_fill(~valid_support, -1e9)
    full = full_teacher_log_probs.detach().masked_fill(~valid_support, -1e9)

    student_probs = torch.softmax(student_base, dim=-1)
    raw_direction = (crop - full).masked_fill(~valid_support, 0.0)
    direction_mean = (student_probs * raw_direction).sum(dim=-1, keepdim=True)
    visual_direction = (raw_direction - direction_mean).masked_fill(~valid_support, 0.0)
    evidence = torch.sqrt((student_probs * visual_direction.square()).sum(dim=-1).clamp_min(0.0))

    weights = rollout_weights.to(device=student_log_probs.device, dtype=student_log_probs.dtype).reshape(-1)
    response_valid = response_mask.to(device=student_log_probs.device, dtype=torch.bool)
    candidate = response_valid & (weights[:, None] > 0) & (evidence >= float(min_evidence))

    positions = torch.arange(student_log_probs.shape[1], device=student_log_probs.device)[None, :]
    sentinel = torch.full_like(positions.expand_as(candidate), student_log_probs.shape[1])
    first_position = torch.where(candidate, positions.expand_as(candidate), sentinel).min(dim=-1).values
    relay_mask = (
        response_valid
        & (positions >= first_position[:, None])
        & (positions < (first_position + int(relay_span))[:, None])
        & (first_position[:, None] < student_log_probs.shape[1])
    )

    shift = float(beta) * weights[:, None, None] * relay_mask[..., None].to(student_log_probs.dtype)
    shifted_logits = student_base + shift * visual_direction
    target_log_probs = torch.log_softmax(shifted_logits, dim=-1).masked_fill(~valid_support, -1e9).detach()

    valid_token_count = response_valid.sum().clamp_min(1)
    active_rollout = relay_mask.any(dim=-1)
    active_evidence = evidence[relay_mask]
    active_start = first_position[active_rollout].float()
    shift_norm = torch.linalg.vector_norm(shift * visual_direction, dim=-1)
    metrics = {
        "self_distillation/ver_candidate_token_fraction": (
            candidate.sum().float() / valid_token_count
        ).detach().item(),
        "self_distillation/ver_relay_token_fraction": (
            relay_mask.sum().float() / valid_token_count
        ).detach().item(),
        "self_distillation/ver_active_rollout_fraction": active_rollout.float().mean().detach().item(),
        "self_distillation/ver_start_position_mean": (
            active_start.mean().detach().item() if active_start.numel() else 0.0
        ),
        "self_distillation/ver_visual_evidence_mean": (
            active_evidence.mean().detach().item() if active_evidence.numel() else 0.0
        ),
        "self_distillation/ver_visual_shift_norm_mean": (
            (shift_norm * response_mask).sum() / valid_token_count
        ).detach().item(),
    }
    return target_log_probs, metrics
