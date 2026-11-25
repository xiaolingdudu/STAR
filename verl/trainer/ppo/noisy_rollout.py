"""Mixed-view sampling with clean-input distributional distillation."""

import math

import numpy as np
from PIL import Image


def validate_config(config):
    """Check the input/probability contract for the first, single-turn version."""
    cfg = config.algorithm.noisy_rollout
    rollout = config.actor_rollout_ref.rollout
    actor = config.actor_rollout_ref.actor
    if not 0 <= cfg.noisy_count <= rollout.n:
        raise ValueError("noisy_count must be in [0, rollout.n]; zero is the matched clean control")
    if not 0 < cfg.initial_scale <= 1:
        raise ValueError("initial_scale must be in (0, 1], applied to each image dimension")
    if rollout.mode != "async" or rollout.agent.default_agent_loop != "single_turn_agent":
        raise ValueError("Noisy rollout requires the async single_turn_agent input path")
    if actor.policy_loss.loss_mode != "vopd" or not actor.self_distillation.full_logit_distillation:
        raise ValueError("Noisy rollout requires distributional vopd loss")
    correction = config.algorithm.get("rollout_correction", {}) or {}
    if correction.get("bypass_mode", False) or correction.get("rollout_is") is not None:
        raise ValueError("Disable rollout IS/bypass: noisy-input probabilities are not clean-input anchors")
    if rollout.calculate_log_probs:
        raise ValueError("Set calculate_log_probs=False; recompute old-policy probabilities on clean inputs")


def noisy_count_for_epoch(config, n, epoch):
    """Switch to all degraded samples from the configured one-based epoch."""
    all_from = config.get("all_noisy_from_epoch")
    return n if all_from is not None and epoch >= all_from else config.noisy_count


def can_skip_old_log_probs(config, batch_size, reward_free_teacher_vopd):
    """The single-update Actor already anchors to its current clean forward.

    Multiple optimizer updates need a fixed pre-update reference, so retain the
    pre-pass for those configurations. This does not reuse degraded-view logits.
    The optimization also applies when all rollouts use the clean image.
    """
    cfg = config.algorithm.get("noisy_rollout", {})
    actor = config.actor_rollout_ref.actor
    correction = config.algorithm.get("rollout_correction", {}) or {}
    return (
        cfg.get("skip_old_log_prob", False)
        and reward_free_teacher_vopd
        and actor.self_distillation.get("target_mode", "full_teacher") == "full_teacher"
        and actor.self_distillation.full_logit_distillation
        and actor.ppo_epochs == 1
        and batch_size == actor.ppo_mini_batch_size * config.actor_rollout_ref.rollout.n
        and not actor.get("use_rollout_log_probs", False)
        and not actor.get("use_kl_loss", False)
        and not config.algorithm.get("use_kl_in_reward", False)
        and not correction.get("bypass_mode", False)
        and correction.get("rollout_is") is None
        and correction.get("rollout_rs") is None
    )


def rollout_metadata(question_count, n, noisy_count, initial_scale, step, total_steps, preserve_evidence=False):
    """Interleaved groups; restore resolution linearly over training."""
    progress = (step - 1) / max(total_steps - 1, 1)
    scale = float(initial_scale) + (1.0 - float(initial_scale)) * min(max(progress, 0.0), 1.0)
    noisy = np.tile(np.arange(n) >= n - noisy_count, question_count)
    return {
        "rollout_view": np.where(noisy, "noisy", "clean").astype(object),
        "rollout_degrade_scale": np.where(noisy, scale, 1.0),
        "rollout_sample_index": np.tile(np.arange(n), question_count),
        "rollout_preserve_evidence": noisy & bool(preserve_evidence),
    }


def degrade_images(images, scale, evidence_bbox=None):
    """VAD degradation, optionally restoring a normalized evidence box verbatim.

    The evidence-preserving variant is for our single-image grounding data.
    Outward rounding protects every pixel intersecting the annotation box.
    """
    if evidence_bbox is not None:
        if len(images) != 1:
            raise ValueError("Evidence-preserving rollout requires one image per question")
        x1, y1, x2, y2 = evidence_bbox
        if not (0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1):
            raise ValueError("Evidence bbox must be normalized xyxy")
    if scale == 1:
        return images
    result = []
    for image in images:
        low_size = (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
        low = image.convert("RGB").resize(low_size, Image.Resampling.BILINEAR)
        degraded = low.resize(image.size, Image.Resampling.NEAREST)
        if evidence_bbox is not None:
            box = (math.floor(x1 * image.width), math.floor(y1 * image.height),
                   math.ceil(x2 * image.width), math.ceil(y2 * image.height))
            degraded.paste(image.crop(box).convert("RGB"), box)
        result.append(degraded)
    return result


def group_metrics(batch):
    """Track branch lengths and token weights; neither is a correctness score."""
    views = np.asarray(batch.non_tensor_batch["rollout_view"])
    lengths = batch.batch["response_mask"].sum(-1).detach().cpu().numpy()
    width = batch.batch["responses"].shape[-1]
    result = {}
    for view in ("clean", "noisy"):
        selected = views == view
        result[f"noisy_rollout/{view}_count"] = int(selected.sum())
        if selected.any():
            result[f"noisy_rollout/{view}_length_mean"] = float(lengths[selected].mean())
            result[f"noisy_rollout/{view}_cap_hit_fraction"] = float((lengths[selected] >= width).mean())
    result["noisy_rollout/noisy_token_fraction"] = float(
        lengths[views == "noisy"].sum() / max(lengths.sum(), 1)
    )
    scales = np.asarray(batch.non_tensor_batch["rollout_degrade_scale"], dtype=float)
    result["noisy_rollout/degrade_scale"] = float(scales.min())
    return result
