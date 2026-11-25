"""V-Zero contrastive evidence gates for sampled-token OPD.

Adapted to the legacy VAD actor interface from the official V-Zero
``compute_relative_evidence_opd_weights`` implementation.
"""

from collections.abc import Sequence
from typing import Any

import torch


def _sampled_logprobs(name: str, values: torch.Tensor) -> torch.Tensor:
    if values.dim() == 2:
        return values.float()
    if values.dim() == 3 and values.shape[-1] == 1:
        return values.squeeze(-1).float()
    raise ValueError(
        f"{name} must have shape (batch, response) or (batch, response, 1), "
        f"got {tuple(values.shape)}"
    )


def compute_vzero_trajectory_scores(
    *,
    teacher_logprobs: torch.Tensor,
    teacher_neg_logprobs: torch.Tensor,
    response_mask: torch.Tensor,
    gamma: float,
    eps: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Return the official V-Zero positive-vs-negative evidence score."""

    plus = _sampled_logprobs("teacher_logprobs", teacher_logprobs)
    minus = _sampled_logprobs("teacher_neg_logprobs", teacher_neg_logprobs)
    if plus.shape != minus.shape or plus.shape != response_mask.shape:
        raise ValueError(
            "teacher logprobs and response_mask must share shape, got "
            f"{tuple(plus.shape)}, {tuple(minus.shape)}, {tuple(response_mask.shape)}"
        )

    mask = response_mask.to(device=plus.device, dtype=plus.dtype)
    delta = plus - minus
    token_scores = torch.relu(delta) - float(gamma) * torch.relu(-delta)
    scores = (token_scores * mask).sum(dim=-1) / mask.sum(dim=-1).clamp_min(float(eps))

    valid_delta = delta[mask.bool()]
    metrics = {
        "rew_opd/score_mean": scores.mean().detach().item(),
        "rew_opd/score_std": scores.std(unbiased=False).detach().item(),
        "rew_opd/delta_mean": valid_delta.mean().detach().item() if valid_delta.numel() else 0.0,
        "rew_opd/delta_pos_frac": (
            (valid_delta > 0).float().mean().detach().item() if valid_delta.numel() else 0.0
        ),
    }
    return scores.to(dtype=torch.float32), metrics


def compute_vzero_evidence_weights(
    *,
    teacher_logprobs: torch.Tensor,
    teacher_neg_logprobs: torch.Tensor,
    response_mask: torch.Tensor,
    uids: Sequence[Any],
    gamma: float,
    alpha: float,
    w_min: float,
    w_max: float,
    eps: float,
    active: torch.Tensor | None = None,
    global_uids: Sequence[Any] | None = None,
    global_scores: Sequence[float] | torch.Tensor | None = None,
    global_active: Sequence[float] | torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, float], dict[str, torch.Tensor]]:
    """Normalize evidence scores across sibling rollouts and build gates."""

    scores, metrics = compute_vzero_trajectory_scores(
        teacher_logprobs=teacher_logprobs,
        teacher_neg_logprobs=teacher_neg_logprobs,
        response_mask=response_mask,
        gamma=gamma,
        eps=eps,
    )
    if len(uids) != scores.shape[0]:
        raise ValueError(f"uid count {len(uids)} does not match batch size {scores.shape[0]}")

    device = scores.device
    active_local = (
        torch.ones_like(scores)
        if active is None
        else torch.as_tensor(active, device=device, dtype=scores.dtype).reshape(-1)
    )
    sibling_uids = [str(uid) for uid in (global_uids if global_uids is not None else uids)]
    sibling_scores = torch.as_tensor(
        global_scores if global_scores is not None else scores,
        device=device,
        dtype=scores.dtype,
    ).reshape(-1)
    sibling_active = torch.as_tensor(
        global_active if global_active is not None else active_local,
        device=device,
        dtype=scores.dtype,
    ).reshape(-1)
    if len(sibling_uids) != sibling_scores.numel() or sibling_scores.shape != sibling_active.shape:
        raise ValueError("global_uids, global_scores, and global_active must describe the same rollouts")

    local_uids = [str(uid) for uid in uids]
    advantages = torch.zeros_like(scores)
    for uid in set(local_uids):
        group_idx = [
            idx for idx, sibling_uid in enumerate(sibling_uids) if sibling_uid == uid and sibling_active[idx] > 0
        ]
        if len(group_idx) <= 1:
            continue
        group = sibling_scores[torch.as_tensor(group_idx, device=device, dtype=torch.long)]
        group_mean = group.mean()
        group_std = torch.sqrt(torch.mean((group - group_mean).square()))
        for idx, local_uid in enumerate(local_uids):
            if local_uid == uid and active_local[idx] > 0:
                advantages[idx] = (scores[idx] - group_mean) / (group_std + float(eps))

    weights = torch.clamp(1.0 + float(alpha) * advantages, min=float(w_min), max=float(w_max)).detach()
    metrics.update(
        {
            "rew_opd/adv_mean": advantages.mean().detach().item(),
            "rew_opd/adv_std": advantages.std(unbiased=False).detach().item(),
            "rew_opd/weight_mean": weights.mean().detach().item(),
            "rew_opd/weight_min": weights.min().detach().item(),
            "rew_opd/weight_max": weights.max().detach().item(),
            "rew_opd/weight_zero_frac": (
                (weights <= float(w_min) + float(eps)).float().mean().detach().item()
            ),
        }
    )
    diagnostics = {
        "scores": scores.detach(),
        "advantages": advantages.detach(),
        "weights": weights.to(dtype=torch.float32),
    }
    return weights.to(dtype=torch.float32), metrics, diagnostics
