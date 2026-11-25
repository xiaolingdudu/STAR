"""Outcome-balancing helpers adapted from the public Uni-OPD release."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Sequence

import torch


def build_group_balanced_keep_mask(
    response_correct: torch.Tensor,
    group_ids: Sequence[Any],
    ratio: float = 1.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Keep a per-prompt correct:incorrect ratio by masking the majority class.

    Uni-OPD samples the excess class randomly.  The unified implementation keeps
    the earliest rollout indices instead so repeated runs with the same rollout
    batch are deterministic.
    """

    if response_correct.dim() != 1 or len(group_ids) != response_correct.numel():
        raise ValueError("response_correct and group_ids must describe the same flat rollout batch")
    if ratio <= 0:
        raise ValueError(f"ratio must be positive, got {ratio}")

    correct = response_correct.bool().cpu()
    keep = torch.ones_like(correct, dtype=torch.bool)
    groups: dict[Any, list[int]] = defaultdict(list)
    for index, group_id in enumerate(group_ids):
        groups[group_id].append(index)

    for indices in groups.values():
        correct_indices = [index for index in indices if bool(correct[index])]
        incorrect_indices = [index for index in indices if not bool(correct[index])]
        if not correct_indices or not incorrect_indices:
            continue
        desired_correct = max(1, int(len(incorrect_indices) * ratio))
        desired_incorrect = max(1, int(len(correct_indices) / ratio))
        for index in correct_indices[desired_correct:]:
            keep[index] = False
        for index in incorrect_indices[desired_incorrect:]:
            keep[index] = False

    kept_correct = (keep & correct).sum().item()
    kept_incorrect = (keep & ~correct).sum().item()
    metrics = {
        "self_distillation/uni_correct_fraction_before": correct.float().mean().item(),
        "self_distillation/uni_rollout_keep_fraction": keep.float().mean().item(),
        "self_distillation/uni_correct_kept": float(kept_correct),
        "self_distillation/uni_incorrect_kept": float(kept_incorrect),
    }
    return keep.to(response_correct.device), metrics


def apply_outcome_margin_shift(
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    response_correct: torch.Tensor,
    sample_keep_mask: torch.Tensor,
    group_index: torch.Tensor,
    delta: float = 0.4,
    direction: str = "both",
) -> tuple[torch.Tensor, dict[str, float]]:
    """Shift sampled-token OPD advantages using Uni-OPD's mean-margin rule."""

    if advantages.shape != response_mask.shape:
        raise ValueError("advantages and response_mask must have the same shape")
    batch_size = advantages.shape[0]
    if any(item.shape != (batch_size,) for item in (response_correct, sample_keep_mask, group_index)):
        raise ValueError("response_correct, sample_keep_mask, and group_index must have shape (batch_size,)")
    if direction not in {"correct_up", "incorrect_down", "both"}:
        raise ValueError(f"unsupported margin direction: {direction}")

    token_count = response_mask.sum(dim=-1).clamp(min=1.0)
    sample_mean = (advantages * response_mask).sum(dim=-1) / token_count
    calibrated = advantages
    gaps = []
    shifts = []
    for group_id in torch.unique(group_index):
        in_group = group_index == group_id
        correct = response_correct.bool() & sample_keep_mask.bool() & in_group
        incorrect = ~response_correct.bool() & sample_keep_mask.bool() & in_group
        if not correct.any() or not incorrect.any():
            continue
        gap = sample_mean[correct].mean() - sample_mean[incorrect].mean()
        shift = torch.clamp(torch.as_tensor(delta, device=gap.device, dtype=gap.dtype) - gap, min=0.0)
        gaps.append(gap)
        shifts.append(shift)
        if direction == "correct_up":
            calibrated = calibrated + correct.to(advantages.dtype).unsqueeze(-1) * shift
        elif direction == "incorrect_down":
            calibrated = calibrated - incorrect.to(advantages.dtype).unsqueeze(-1) * shift
        else:
            half = shift / 2.0
            calibrated = calibrated + correct.to(advantages.dtype).unsqueeze(-1) * half
            calibrated = calibrated - incorrect.to(advantages.dtype).unsqueeze(-1) * half

    if not gaps:
        return advantages, {
            "self_distillation/uni_margin_gap_before": 0.0,
            "self_distillation/uni_margin_shift": 0.0,
            "self_distillation/uni_margin_active": 0.0,
            "self_distillation/uni_margin_group_count": 0.0,
        }

    mean_gap = torch.stack(gaps).mean()
    mean_shift = torch.stack(shifts).mean()

    return calibrated, {
        "self_distillation/uni_margin_gap_before": mean_gap.detach().item(),
        "self_distillation/uni_margin_shift": mean_shift.detach().item(),
        "self_distillation/uni_margin_active": float(any(item.detach().item() > 0 for item in shifts)),
        "self_distillation/uni_margin_group_count": float(len(gaps)),
    }
