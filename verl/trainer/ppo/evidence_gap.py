"""Same-prefix original/box-erased probes; weights never replace the crop target."""
from collections import defaultdict
import time

import numpy as np
from PIL import Image, ImageDraw
import torch
import torch.distributed as dist
import torch.nn.functional as F


def erase_evidence(image, bbox):
    """Fill the registered crop footprint with mean RGB without resizing."""
    pixels = np.array(image.convert("RGB"), copy=True)
    height, width = pixels.shape[:2]
    x1, y1, x2, y2 = np.rint(np.asarray(bbox) * [width, height, width, height]).astype(int)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(width, x2), min(height, y2)
    if x2 <= x1 or y2 <= y1:
        raise ValueError("Evidence crop has no pixels to erase")
    region = pixels[y1:y2, x1:x2]
    region[:] = np.rint(region.mean(axis=(0, 1))).astype(np.uint8)
    return Image.fromarray(pixels)


def probe_views(image, geometry):
    original = image.convert("RGB").copy()
    erased = erase_evidence(original, geometry["bbox"])
    # Keep the question's localization cue in both probes, while removing all
    # scene detail inside the registered (potentially expanded) Teacher crop.
    w, h = original.size
    box = np.rint(np.asarray(geometry["annotation_bbox"]) * [w, h, w, h]).astype(int).tolist()
    for view in (original, erased):
        ImageDraw.Draw(view).rectangle(box, outline=(255, 0, 0), width=1)
    return original, erased


def build_probe_inputs(trainer, batch, responses, response_mask, config):
    started = time.perf_counter()
    prefixes, columns = {}, {view: [] for view in ("original", "erased")}
    for i, uid in enumerate(batch.non_tensor_batch["uid"]):
        uid = str(uid)
        if uid not in prefixes:
            messages = list(batch.non_tensor_batch["raw_prompt"][i])
            images = trainer._extract_images_from_messages(messages)
            if len(images) != 1:
                raise ValueError("Evidence-gap weighting currently uses the single-image grounding dataset")
            original, erased = probe_views(images[0], batch.non_tensor_batch["extra_info"][i]["attention_gap"])
            prefixes[uid] = {}
            for view, view_images in (("original", [original]), ("erased", [erased])):
                view_messages = trainer._prepare_teacher_messages(messages, view_images)
                prefixes[uid][view] = trainer._build_teacher_prompt_prefix_inputs(
                    view_messages, config.max_reprompt_len)
            for index in (0, 1, 2):
                if not torch.equal(prefixes[uid]["original"][index], prefixes[uid]["erased"][index]):
                    raise ValueError("Evidence erasure changed text/position alignment")
        for view in columns:
            columns[view].append(trainer._append_teacher_response_to_prompt(
                *prefixes[uid][view], responses[i], response_mask[i]))
    tensors, non_tensors = {}, {}
    for view, values in columns.items():
        ids, attention, positions, starts, multimodal = map(list, zip(*values))
        ids, attention, positions = trainer._pad_teacher_prompt_input_lists(
            input_ids_list=ids, attention_mask_list=attention, position_ids_list=positions,
            device=responses.device)
        prefix = f"teacher_eg_{view}"
        tensors.update({f"{prefix}_input_ids": ids, f"{prefix}_attention_mask": attention,
                        f"{prefix}_position_ids": positions,
                        f"{prefix}_response_start_idx": torch.stack(starts).to(responses.device)})
        non_tensors[f"{prefix}_multi_modal_inputs"] = multimodal
    tensors["eg_is_degraded"] = torch.as_tensor(
        np.asarray(batch.non_tensor_batch["rollout_view"]) == "noisy", device=responses.device)
    return tensors, non_tensors, {"timing_s/evidence_gap/build_inputs": time.perf_counter() - started}


@torch.no_grad()
def token_weights(delta, mask, beta):
    score = delta.detach().float().abs().tanh().masked_fill(~mask.bool(), 0)
    mean = (score * mask).sum(-1) / mask.sum(-1).clamp_min(1)
    weights = (1 + beta * score) / (1 + beta * mean[:, None])
    return weights, mean


def trajectory_weights(uids, means, global_rows, beta):
    """Group means are collected across all DP ranks, before microbatching."""
    groups = defaultdict(list)
    for uid, value in global_rows:
        groups[str(uid)].append(value)
    denominators = {uid: 1 + beta * sum(values) / len(values) for uid, values in groups.items()}
    return means.new_tensor([(1 + beta * value) / denominators[str(uid)]
                             for uid, value in zip(uids, means.tolist())])


@torch.no_grad()
def precompute_weights(actor, data, device, pad_token_id):
    from verl.utils.seqlen_balancing import prepare_dynamic_batch, restore_dynamic_batch

    cfg = actor.config.self_distillation
    if (actor.teacher_module is None or actor.use_remove_padding or actor.use_ulysses_sp
            or actor.use_prefix_grouper or actor.config.ppo_epochs != 1
            or actor.config.loss_agg_mode != "token-mean"):
        raise ValueError("Evidence-gap probes require a separate Teacher and one padded, non-SP token-mean update")
    started = time.perf_counter()
    if actor.config.use_dynamic_bsz:
        micros, order = prepare_dynamic_batch(data, max_token_len=actor.config.ppo_max_token_len_per_gpu)
    else:
        micros, order = data.split(actor.config.ppo_micro_batch_size_per_gpu), None
    teacher, deltas = actor.teacher_module, []
    was_training = teacher.training
    teacher.eval()
    try:
        for micro in micros:
            micro = micro.to(device)
            logps = {}
            for view in ("original", "erased"):
                prefix = f"teacher_eg_{view}"
                inputs = {name: micro.batch[f"{prefix}_{name}"] for name in
                          ("input_ids", "attention_mask", "position_ids", "response_start_idx")}
                inputs.update(responses=micro.batch["responses"], pad_token_id=pad_token_id,
                              multi_modal_inputs=micro.non_tensor_batch[f"{prefix}_multi_modal_inputs"])
                # Actual sampled-token log-probs only; no retained vocabulary logits.
                logps[view] = actor._forward_micro_batch(
                    inputs, temperature=1.0, calculate_entropy=False, module=teacher)["log_probs"].float().cpu()
            delta = logps["original"] - logps["erased"]
            deltas.append(F.pad(delta, (0, data.batch["responses"].shape[1] - delta.shape[1])))
    finally:
        teacher.train(was_training)
    delta = torch.cat(deltas)
    if order is not None:
        delta = restore_dynamic_batch(delta, order)
    mask = data.batch["response_mask"].float().cpu() * data.batch["self_distillation_mask"].float().cpu()[:, None]
    if not torch.isfinite(delta[mask.bool()]).all():
        raise ValueError("Non-finite evidence-gap log-probability difference")
    v, means = token_weights(delta, mask, float(cfg.evidence_gap_token_beta))
    uids = [str(uid) for uid in data.non_tensor_batch["uid"]]
    rows = [(uid, value) for uid, value, active in zip(uids, means.tolist(), mask.sum(-1).tolist()) if active]
    if dist.is_initialized():
        parts = [None] * dist.get_world_size()
        dist.all_gather_object(parts, rows)
        rows = [row for part in parts for row in part]
    # The configured profile uses an always-on crop Teacher for every rollout.
    u = trajectory_weights(uids, means, rows, float(cfg.evidence_gap_trajectory_beta))
    weights = u[:, None] * v
    data.batch["evidence_gap_weights"] = weights.to(data.batch.device)
    metrics = {"timing_s/evidence_gap/teacher_probes": time.perf_counter() - started}
    valid_delta = delta[mask.bool()]
    metrics.update({"evidence_gap/delta_mean": valid_delta.mean().item(),
                    "evidence_gap/positive_fraction": (valid_delta > 0).float().mean().item(),
                    "evidence_gap/negative_fraction": (valid_delta < 0).float().mean().item(),
                    "evidence_gap/absolute_p50": valid_delta.abs().quantile(.5).item(),
                    "evidence_gap/absolute_p90": valid_delta.abs().quantile(.9).item(),
                    "evidence_gap/weight_min": weights[mask.bool()].min().item(),
                    "evidence_gap/weight_max": weights[mask.bool()].max().item()})
    for name, selected in (("clean", ~data.batch["eg_is_degraded"].bool().cpu()),
                           ("degraded", data.batch["eg_is_degraded"].bool().cpu())):
        selected_mask = mask * selected[:, None]
        metrics[f"evidence_gap/{name}_weight_mean"] = (
            (weights * selected_mask).sum() / selected_mask.sum().clamp_min(1)).item()
    return metrics
