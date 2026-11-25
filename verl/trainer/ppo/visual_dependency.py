"""Teacher visual-dependency filtering on mixed-view Student trajectories."""

import time

import numpy as np
import torch
import torch.nn.functional as F


@torch.no_grad()
def full_vocabulary_kl(original_logits, degraded_logits, chunk_tokens=64):
    """KL(T(original) || T(degraded)), FP32 reduction over the entire vocabulary.

    Inputs are response-position logits at temperature 1. Chunking bounds the
    temporary FP32 distributions; it does not truncate the vocabulary.
    """
    if original_logits.shape != degraded_logits.shape:
        raise ValueError("Visual-dependency probes must have matching response logits")
    shape = original_logits.shape[:-1]
    original = original_logits.reshape(-1, original_logits.shape[-1])
    degraded = degraded_logits.reshape_as(original)
    result = torch.empty(len(original), device=original.device, dtype=torch.float32)
    for start in range(0, len(original), chunk_tokens):
        stop = start + chunk_tokens
        logp = original[start:stop].float().log_softmax(-1)
        logq = degraded[start:stop].float().log_softmax(-1)
        result[start:stop] = (logp.exp() * (logp - logq)).sum(-1).clamp_min(0)
    return result.reshape(shape)


def top_fraction_mask(scores, valid_mask, fraction):
    """Select ceil(fraction * valid length) positions independently per rollout."""
    valid = valid_mask.bool()
    order = scores.masked_fill(~valid, -torch.inf).argsort(dim=-1, descending=True, stable=True)
    ranks = torch.empty_like(order)
    ranks.scatter_(1, order, torch.arange(scores.shape[1], device=scores.device).expand_as(order))
    count = (valid.sum(-1, keepdim=True) * fraction).ceil().long()
    return ((ranks < count) & valid).to(valid_mask.dtype)


def build_probe_inputs(trainer, batch, responses, response_mask, config):
    """Encode both full-image probes once per prompt and append each fixed response.

    Both conditions use the identical prompt builder (no answer/crop hints).
    The fixed degradation is independent of the rollout annealing schedule.
    """
    from verl.trainer.ppo.noisy_rollout import degrade_images

    started = time.perf_counter()
    prefixes = {}
    columns = {view: [] for view in ("original", "degraded")}
    for i in range(len(responses)):
        uid = str(batch.non_tensor_batch["uid"][i])
        if uid not in prefixes:
            messages = list(batch.non_tensor_batch["raw_prompt"][i])
            images = trainer._extract_images_from_messages(messages)
            degraded = degrade_images(images, float(config.visual_dependency_probe_scale))
            prefixes[uid] = {}
            for view, view_images in (("original", images), ("degraded", degraded)):
                view_messages = trainer._prepare_teacher_messages(messages, view_images)
                prefixes[uid][view] = trainer._build_teacher_prompt_prefix_inputs(
                    view_messages, config.max_reprompt_len
                )
            # Same text, image dimensions, visual token count and positional ids.
            for index in (0, 1, 2):
                if not torch.equal(prefixes[uid]["original"][index], prefixes[uid]["degraded"][index]):
                    raise ValueError("Original/degraded Teacher probes changed token or position alignment")
        for view in columns:
            columns[view].append(trainer._append_teacher_response_to_prompt(
                *prefixes[uid][view], responses[i], response_mask[i]
            ))
    tensors, non_tensors = {}, {}
    for view, values in columns.items():
        ids, attention, positions, starts, multimodal = map(list, zip(*values))
        ids, attention, positions = trainer._pad_teacher_prompt_input_lists(
            input_ids_list=ids, attention_mask_list=attention, position_ids_list=positions,
            device=responses.device,
        )
        prefix = f"teacher_vd_{view}"
        tensors.update({f"{prefix}_input_ids": ids, f"{prefix}_attention_mask": attention,
                        f"{prefix}_position_ids": positions,
                        f"{prefix}_response_start_idx": torch.stack(starts).to(responses.device)})
        non_tensors[f"{prefix}_multi_modal_inputs"] = multimodal
    tensors["vd_is_degraded"] = torch.as_tensor(
        np.asarray(batch.non_tensor_batch["rollout_view"]) == "noisy", device=responses.device
    )
    return tensors, non_tensors, {"timing_s/visual_dependency/build_inputs": time.perf_counter() - started}


@torch.no_grad()
def precompute_scores(actor, data, device, pad_token_id):
    """Freeze one mask per rollout batch before the Student optimizer step."""
    from verl.utils.seqlen_balancing import prepare_dynamic_batch, restore_dynamic_batch

    cfg = actor.config.self_distillation
    teacher = actor.teacher_module
    if teacher is None or actor.use_remove_padding or actor.use_ulysses_sp or actor.use_prefix_grouper:
        raise ValueError("Visual dependency requires a separate Teacher and padded non-SP forwards")
    if actor.config.ppo_epochs != 1 or actor.config.loss_agg_mode != "token-mean":
        raise ValueError("Visual dependency currently requires one token-mean update per rollout batch")
    started = time.perf_counter()
    if actor.config.use_dynamic_bsz:
        micros, indices = prepare_dynamic_batch(data, max_token_len=actor.config.ppo_max_token_len_per_gpu)
    else:
        micros, indices = data.split(actor.config.ppo_micro_batch_size_per_gpu), None
    scores = []
    was_training = teacher.training
    teacher.eval()
    try:
        for micro in micros:
            micro = micro.to(device)
            outputs = {}
            for view in ("original", "degraded"):
                prefix = f"teacher_vd_{view}"
                inputs = {name: micro.batch[f"{prefix}_{name}"]
                          for name in ("input_ids", "attention_mask", "position_ids", "response_start_idx")}
                inputs.update(responses=micro.batch["responses"], pad_token_id=pad_token_id,
                              multi_modal_inputs=micro.non_tensor_batch[f"{prefix}_multi_modal_inputs"])
                outputs[view] = actor._forward_micro_batch(
                    inputs, temperature=1.0, return_response_logits=True,
                    pad_response_outputs=False, module=teacher,
                )["response_logits"]
            kl = full_vocabulary_kl(outputs["original"], outputs["degraded"],
                                    int(cfg.visual_dependency_chunk_tokens))
            scores.append(F.pad(kl.cpu(), (0, data.batch["responses"].shape[1] - kl.shape[1])))
            del outputs, kl
    finally:
        teacher.train(was_training)
    scores = torch.cat(scores)
    if indices is not None:
        scores = restore_dynamic_batch(scores, indices)
    valid = data.batch["response_mask"].float().cpu() * data.batch["self_distillation_mask"].float().cpu()[:, None]
    if not torch.isfinite(scores[valid.bool()]).all():
        raise ValueError("Non-finite Teacher visual-dependency KL")
    mask = top_fraction_mask(scores, valid, float(cfg.visual_dependency_keep_fraction))
    data.batch["distillation_token_mask"] = mask.to(data.batch.device)
    degraded = data.batch["vd_is_degraded"].bool().cpu()
    metrics = {"timing_s/visual_dependency/teacher_probes": time.perf_counter() - started,
               "visual_dependency/keep_fraction": (mask.sum() / valid.sum().clamp_min(1)).item(),
               "visual_dependency/kl_max": scores[valid.bool()].max().item()}
    for name, rows in (("clean", ~degraded), ("degraded", degraded)):
        whole, chosen = valid * rows[:, None], mask * rows[:, None]
        dropped = whole - chosen
        for group, weights in (("all", whole), ("selected", chosen), ("dropped", dropped)):
            metrics[f"visual_dependency/{name}_{group}_kl"] = (
                (scores * weights).sum() / weights.sum().clamp_min(1)
            ).item()
        metrics[f"visual_dependency/{name}_keep_fraction"] = (chosen.sum() / whole.sum().clamp_min(1)).item()
    return metrics
