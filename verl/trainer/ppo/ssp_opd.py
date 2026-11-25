"""Student-Support Posterior OPD objectives."""

from __future__ import annotations

from collections.abc import Sequence

import torch


def _mean_prefix_log_prob(
    log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    chunk_tokens: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    positions = torch.arange(response_mask.shape[-1], device=response_mask.device)
    chunk_mask = response_mask.to(log_probs.dtype) * (positions < chunk_tokens).to(log_probs.dtype)
    token_count = chunk_mask.sum(dim=-1).clamp(min=1.0)
    return (log_probs * chunk_mask).sum(dim=-1) / token_count, chunk_mask


def compute_ssp_listwise_loss(
    *,
    student_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    teacher_crop_log_probs: torch.Tensor,
    teacher_full_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    group_ids: Sequence[object],
    candidate_count: int,
    chunk_tokens: int,
    beta: float,
    reference_lambda: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute exact candidate-set KL for complete, rank-local rollout groups."""

    if student_log_probs.ndim != 2:
        raise ValueError("SSP-OPD expects [batch, response] sampled-token log-probabilities.")
    if len(group_ids) != student_log_probs.shape[0]:
        raise ValueError("SSP-OPD group_ids must align with the local candidate batch.")

    student_score, chunk_mask = _mean_prefix_log_prob(student_log_probs, response_mask, chunk_tokens)
    old_score, _ = _mean_prefix_log_prob(old_log_probs.detach(), response_mask, chunk_tokens)
    teacher_crop_score, _ = _mean_prefix_log_prob(
        teacher_crop_log_probs.detach(), response_mask, chunk_tokens
    )
    teacher_full_score, _ = _mean_prefix_log_prob(
        teacher_full_log_probs.detach(), response_mask, chunk_tokens
    )

    ordered_groups: dict[object, list[int]] = {}
    for index, group_id in enumerate(group_ids):
        ordered_groups.setdefault(group_id, []).append(index)
    incomplete = {key: len(indices) for key, indices in ordered_groups.items() if len(indices) != candidate_count}
    if incomplete:
        raise ValueError(
            f"SSP-OPD requires complete groups of {candidate_count} candidates in every micro-batch; "
            f"found {incomplete}."
        )

    group_losses = []
    prior_entropies = []
    posterior_entropies = []
    posterior_kls = []
    visual_residuals = []
    best_changed = []
    posterior_maxima = []
    prior_maxima = []
    for indices in ordered_groups.values():
        index = torch.as_tensor(indices, dtype=torch.long, device=student_log_probs.device)
        prior = torch.softmax(old_score[index], dim=0)
        visual_residual = teacher_crop_score[index] - teacher_full_score[index]
        posterior = torch.softmax(torch.log(prior.clamp_min(1e-8)) + beta * visual_residual, dim=0)
        current_log_p = torch.log_softmax(student_score[index], dim=0)
        group_losses.append((posterior * (torch.log(posterior.clamp_min(1e-8)) - current_log_p)).sum())
        prior_entropies.append(-(prior * torch.log(prior.clamp_min(1e-8))).sum())
        posterior_entropies.append(-(posterior * torch.log(posterior.clamp_min(1e-8))).sum())
        posterior_kls.append(
            (posterior * (torch.log(posterior.clamp_min(1e-8)) - torch.log(prior.clamp_min(1e-8)))).sum()
        )
        visual_residuals.append(visual_residual)
        best_changed.append((posterior.argmax() != prior.argmax()).to(student_log_probs.dtype))
        posterior_maxima.append(posterior.max())
        prior_maxima.append(prior.max())

    listwise_loss = torch.stack(group_losses).mean()
    unaudited_mask = response_mask.to(student_log_probs.dtype) * (1.0 - chunk_mask)
    unaudited_count = unaudited_mask.sum()
    if unaudited_count.item() > 0:
        reference_loss = ((old_log_probs.detach() - student_log_probs) * unaudited_mask).sum() / unaudited_count
    else:
        reference_loss = student_log_probs.sum() * 0.0
    total_loss = listwise_loss + reference_lambda * reference_loss

    residual = torch.cat(visual_residuals)
    metrics = {
        "self_distillation/ssp_listwise_kl": float(listwise_loss.detach().item()),
        "self_distillation/ssp_reference_loss": float(reference_loss.detach().item()),
        "self_distillation/ssp_visual_residual_mean": float(residual.mean().detach().item()),
        "self_distillation/ssp_visual_residual_std": float(residual.std(unbiased=False).detach().item()),
        "self_distillation/ssp_positive_residual_fraction": float((residual > 0).float().mean().item()),
        "self_distillation/ssp_prior_entropy": float(torch.stack(prior_entropies).mean().detach().item()),
        "self_distillation/ssp_posterior_entropy": float(
            torch.stack(posterior_entropies).mean().detach().item()
        ),
        "self_distillation/ssp_posterior_kl_from_prior": float(
            torch.stack(posterior_kls).mean().detach().item()
        ),
        "self_distillation/ssp_best_branch_changed_fraction": float(
            torch.stack(best_changed).mean().detach().item()
        ),
        "self_distillation/ssp_prior_max_probability": float(
            torch.stack(prior_maxima).mean().detach().item()
        ),
        "self_distillation/ssp_posterior_max_probability": float(
            torch.stack(posterior_maxima).mean().detach().item()
        ),
        "self_distillation/ssp_candidate_groups": float(len(ordered_groups)),
        "self_distillation/ssp_candidate_count": float(candidate_count),
        "self_distillation/ssp_chunk_token_fraction": float(
            (chunk_mask.sum() / response_mask.to(chunk_mask.dtype).sum().clamp(min=1.0)).detach().item()
        ),
        "self_distillation/ssp_beta": float(beta),
        "self_distillation/ssp_reference_lambda": float(reference_lambda),
    }
    return total_loss, metrics


def compute_ssp_v2_listwise_loss(
    *,
    student_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    teacher_crop_log_probs: torch.Tensor,
    teacher_full_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    branch_mask: torch.Tensor,
    group_ids: Sequence[object],
    candidate_count: int,
    beta: float,
    trust_lambda: float,
    candidate_token_ids: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Ratio-listwise update over fixed local branches from one shared state."""

    if student_log_probs.ndim != 2:
        raise ValueError("SSP-OPD v2 expects [batch, response] sampled-token log-probabilities.")
    if len(group_ids) != student_log_probs.shape[0]:
        raise ValueError("SSP-OPD v2 group_ids must align with the local candidate batch.")

    mask = response_mask.to(student_log_probs.dtype) * branch_mask.to(student_log_probs.dtype)
    if not torch.all(mask.sum(dim=-1) > 0):
        raise ValueError("SSP-OPD v2 requires a non-empty local branch for every candidate.")

    old = old_log_probs.detach()
    log_ratio = student_log_probs - old
    ratio_scores = (log_ratio * mask).sum(dim=-1)
    visual_residual = ((teacher_crop_log_probs.detach() - teacher_full_log_probs.detach()) * mask).sum(dim=-1)

    ordered_groups: dict[object, list[int]] = {}
    for index, group_id in enumerate(group_ids):
        ordered_groups.setdefault(group_id, []).append(index)
    incomplete = {key: len(indices) for key, indices in ordered_groups.items() if len(indices) != candidate_count}
    if incomplete:
        raise ValueError(
            f"SSP-OPD v2 requires complete groups of {candidate_count} candidates in every micro-batch; "
            f"found {incomplete}."
        )

    group_losses = []
    target_entropies = []
    policy_entropies = []
    target_maxima = []
    residual_ranges = []
    residual_stds = []
    best_changed = []
    effective_sample_sizes = []
    duplicate_fractions = []
    for indices in ordered_groups.values():
        index = torch.as_tensor(indices, dtype=torch.long, device=student_log_probs.device)
        target = torch.softmax(beta * visual_residual[index], dim=0)
        current_log_p = torch.log_softmax(ratio_scores[index], dim=0)
        current_p = current_log_p.exp()
        group_losses.append((target * (torch.log(target.clamp_min(1e-8)) - current_log_p)).sum())
        target_entropies.append(-(target * torch.log(target.clamp_min(1e-8))).sum())
        policy_entropies.append(-(current_p * current_log_p).sum())
        target_maxima.append(target.max())
        residual_ranges.append(visual_residual[index].max() - visual_residual[index].min())
        residual_stds.append(visual_residual[index].std(unbiased=False))
        best_changed.append((target.argmax() != current_p.argmax()).to(student_log_probs.dtype))
        effective_sample_sizes.append(1.0 / target.square().sum().clamp_min(1e-8))
        if candidate_token_ids is not None:
            branches = [
                tuple(candidate_token_ids[row][mask[row].bool()].detach().cpu().tolist()) for row in indices
            ]
            duplicate_fractions.append(1.0 - len(set(branches)) / len(branches))

    listwise_loss = torch.stack(group_losses).mean()
    # K3 is a non-negative sampled estimator of KL(old || current) on the
    # exact candidate states. Its gradient is zero at the old policy.
    trust_terms = torch.exp(log_ratio) - 1.0 - log_ratio
    trust_loss = (trust_terms * mask).sum() / mask.sum().clamp(min=1.0)
    total_loss = listwise_loss + trust_lambda * trust_loss

    metrics = {
        "self_distillation/ssp_v2_listwise_kl": float(listwise_loss.detach().item()),
        "self_distillation/ssp_v2_trust_k3": float(trust_loss.detach().item()),
        "self_distillation/ssp_v2_visual_residual_mean": float(visual_residual.mean().detach().item()),
        "self_distillation/ssp_v2_visual_residual_std": float(
            torch.stack(residual_stds).mean().detach().item()
        ),
        "self_distillation/ssp_v2_visual_residual_range": float(
            torch.stack(residual_ranges).mean().detach().item()
        ),
        "self_distillation/ssp_v2_positive_residual_fraction": float(
            (visual_residual > 0).float().mean().detach().item()
        ),
        "self_distillation/ssp_v2_target_entropy": float(
            torch.stack(target_entropies).mean().detach().item()
        ),
        "self_distillation/ssp_v2_policy_ratio_entropy": float(
            torch.stack(policy_entropies).mean().detach().item()
        ),
        "self_distillation/ssp_v2_target_max_probability": float(
            torch.stack(target_maxima).mean().detach().item()
        ),
        "self_distillation/ssp_v2_effective_sample_size": float(
            torch.stack(effective_sample_sizes).mean().detach().item()
        ),
        "self_distillation/ssp_v2_best_branch_changed_fraction": float(
            torch.stack(best_changed).mean().detach().item()
        ),
        "self_distillation/ssp_v2_branch_token_fraction": float(
            (mask.sum() / response_mask.to(mask.dtype).sum().clamp(min=1.0)).detach().item()
        ),
        "self_distillation/ssp_v2_candidate_groups": float(len(ordered_groups)),
        "self_distillation/ssp_v2_candidate_count": float(candidate_count),
        "self_distillation/ssp_v2_beta": float(beta),
        "self_distillation/ssp_v2_trust_lambda": float(trust_lambda),
    }
    if duplicate_fractions:
        metrics["self_distillation/ssp_v2_exact_duplicate_fraction"] = float(
            sum(duplicate_fractions) / len(duplicate_fractions)
        )
    return total_loss, metrics
