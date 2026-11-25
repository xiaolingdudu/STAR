"""Spurious-signal-aware token filtering for on-policy distillation.

This is the small algorithmic part of SA-OPD that is independent of its
separate teacher-serving stack.  The real-input and residual-input k1 signals
are produced by the actor worker; this module selects high-impact tokens whose
distillation signal changes little after removing the input.
"""

from __future__ import annotations

import torch


def _row_fraction_mask(
    score: torch.Tensor,
    valid_mask: torch.Tensor,
    fraction: float,
    *,
    largest: bool,
) -> torch.Tensor:
    valid_mask = valid_mask.bool()
    if fraction <= 0:
        return torch.zeros_like(valid_mask)
    counts = valid_mask.sum(dim=-1)
    selected = torch.ceil(counts.float() * min(float(fraction), 1.0)).long()
    selected = torch.where(counts > 0, selected.clamp_min(1), torch.zeros_like(selected))
    fill = float("-inf") if largest else float("inf")
    order = torch.argsort(score.masked_fill(~valid_mask, fill), dim=-1, descending=largest)
    ranks = torch.empty_like(order)
    rank_values = torch.arange(score.shape[-1], device=score.device).unsqueeze(0).expand_as(order)
    ranks.scatter_(dim=-1, index=order, src=rank_values)
    return (ranks < selected.unsqueeze(-1)) & valid_mask


def compute_sa_filter_mask(
    real_k1: torch.Tensor,
    residual_k1: torch.Tensor,
    response_mask: torch.Tensor,
    *,
    low_sensitivity_fraction: float = 0.2,
    extreme_loss_fraction: float = 0.3,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Return SA-OPD's high-impact, low-input-sensitivity token mask.

    ``real_k1`` is the sampled-token teacher/student log-probability gap under
    the original input and ``residual_k1`` is the same gap under a one-token
    residual context.  Tokens in both tails of the real loss are considered
    high impact, matching the public SA-OPD VLM launcher.
    """

    if real_k1.shape != residual_k1.shape or real_k1.shape != response_mask.shape:
        raise ValueError(
            "SA-OPD tensors must share shape, got "
            f"real={tuple(real_k1.shape)}, residual={tuple(residual_k1.shape)}, "
            f"mask={tuple(response_mask.shape)}"
        )

    valid = response_mask.bool()
    sensitivity = (real_k1.detach().float() - residual_k1.detach().float()).abs()
    low_sensitivity = _row_fraction_mask(
        sensitivity,
        valid,
        low_sensitivity_fraction,
        largest=False,
    )
    extreme_high = _row_fraction_mask(
        real_k1.detach().float(),
        valid,
        extreme_loss_fraction,
        largest=True,
    )
    extreme_low = _row_fraction_mask(
        real_k1.detach().float(),
        valid,
        extreme_loss_fraction,
        largest=False,
    )
    extreme = extreme_high | extreme_low
    filtered = low_sensitivity & extreme & valid

    valid_count = valid.sum().clamp_min(1)
    filtered_count = filtered.sum()
    valid_abs_mass = real_k1.detach().float().abs()[valid].sum().clamp_min(1e-12)
    filtered_abs_mass = real_k1.detach().float().abs()[filtered].sum()
    metrics = {
        "self_distillation/sa_filtered_token_fraction": (filtered_count / valid_count).item(),
        "self_distillation/sa_low_sensitivity_fraction": (low_sensitivity.sum() / valid_count).item(),
        "self_distillation/sa_extreme_loss_fraction": (extreme.sum() / valid_count).item(),
        "self_distillation/sa_filtered_abs_loss_mass_fraction": (filtered_abs_mass / valid_abs_mass).item(),
        "self_distillation/sa_input_sensitivity_mean": sensitivity[valid].mean().item() if valid.any() else 0.0,
    }
    return filtered, metrics
