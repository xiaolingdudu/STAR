"""Two independent visual worlds, one Student and a fixed per-pair rollout budget.

This module selects rollout inputs. It adds no loss and does not change the
existing Gaussian-gated Teacher target. Labels are used only by weak_side.
"""
from copy import deepcopy
import time

import numpy as np
import torch

from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto


def object_array(items):
    result = np.empty(len(items), dtype=object)
    result[:] = items
    return result


def allocation(paired, mode="uniform", probabilities=None, gap=0.2):
    """probabilities contains each world's own correct-option probability."""
    paired = np.asarray(paired, dtype=bool)
    counts = np.tile([8, 8], (len(paired), 1))
    if mode == "weak_side":
        p = np.asarray(probabilities, dtype=float)
        if p.shape != (len(paired), 2) or not np.isfinite(p).all():
            raise ValueError("Need finite current-Student probabilities for both worlds")
        delta = p[:, 0] - p[:, 1]
        counts[delta > gap] = [4, 12]
        counts[delta < -gap] = [12, 4]
    elif mode != "uniform":
        raise ValueError(f"Unknown world allocation: {mode}")
    counts[~paired] = [16, 0]
    return counts


def make_worlds(batch):
    """2N prompt records; matching Teacher, answer and cache UID travel together."""
    worlds = batch.repeat(repeat_times=2, interleave=True)
    pairs = batch.non_tensor_batch["world_pair"]
    from evaluation.routing_scoring import extract_routing_option
    side_names, parent_ids, labels = [], [], []
    for i, pair in enumerate(pairs):
        parent = str(batch.non_tensor_batch["uid"][i])
        for side in range(2):
            j = 2 * i + side
            name = "edited" if side else "original"
            worlds.non_tensor_batch["uid"][j] = f"{parent}:{name}"
            side_names.append(name)
            parent_ids.append(parent)
            labels.append(pair[f"{name}_answer"] if pair else
                          extract_routing_option(batch.non_tensor_batch["reward_model"][i]["ground_truth"])[0])
        if not pair:
            continue  # Dummy second world is never selected for long rollout.
        j = 2*i + 1
        messages = deepcopy(list(batch.non_tensor_batch["raw_prompt"][i]))
        images = [c for m in messages for c in m["content"] if c["type"] == "image"]
        if len(images) != 1:
            raise ValueError("Two-world grounding currently requires one Student image")
        images[0].clear()
        images[0].update(type="image", image=pair["edited_image"])
        worlds.non_tensor_batch["raw_prompt"][j] = messages
        worlds.non_tensor_batch["bbox_images"][j] = [{"path": pair["edited_teacher_image"]}]
        reward = deepcopy(batch.non_tensor_batch["reward_model"][i])
        reward["ground_truth"] = pair["edited_answer"]
        worlds.non_tensor_batch["reward_model"][j] = reward
        extra = deepcopy(batch.non_tensor_batch["extra_info"][i])
        extra["answer"] = pair["edited_answer"]
        extra["attention_gap"]["original_path"] = pair["edited_image"]
        worlds.non_tensor_batch["extra_info"][j] = extra
    worlds.non_tensor_batch["world_side"] = object_array(side_names)
    worlds.non_tensor_batch["world_parent_uid"] = object_array(parent_ids)
    worlds.non_tensor_batch["world_correct_option"] = object_array(labels)
    return worlds


def neutral_probe_messages(messages):
    result = deepcopy(list(messages))
    user = next(m for m in reversed(result) if m["role"] == "user")
    user["content"].append({"type": "text", "text": "\nAnswer with exactly one option letter (A, B, C, or D)."})
    return result


def build_probe_batch(trainer, worlds):
    """Score all four next-letter tokens, without EOS or answer-dependent text.

    Reuse the multimodal prefix encoder, but call the *current actor* below,
    not the frozen Teacher. Left-padding keeps the sampled-token scorer aligned.
    """
    token_ids = [trainer.tokenizer.encode(x, add_special_tokens=False) for x in "ABCD"]
    if any(len(t) != 1 for t in token_ids):
        raise ValueError("The Qwen short-option probe expects single-token A/B/C/D")
    prefixes = [trainer._build_teacher_prompt_prefix_inputs(
        neutral_probe_messages(m), int(trainer.config.data.max_prompt_length))
        for m in worlds.non_tensor_batch["raw_prompt"]]
    width = max(len(p[0]) for p in prefixes)
    n = 4 * len(prefixes)
    pad = trainer.tokenizer.pad_token_id or 0
    ids = torch.full((n, width + 1), pad, dtype=torch.long)
    mask = torch.zeros_like(ids)
    pos_shape = prefixes[0][2].shape[:-1]
    positions = torch.zeros((n, *pos_shape, width + 1), dtype=torch.long)
    mm = []
    for i, (prefix, attention, pos, visual) in enumerate(prefixes):
        start = width - len(prefix)
        for k, token in enumerate(token_ids):
            j = 4*i + k
            ids[j, start:width] = prefix
            ids[j, width] = token[0]
            mask[j, start:width] = attention
            mask[j, width] = 1
            positions[j, ..., start:width] = pos
            positions[j, ..., width] = pos[..., -1] + 1
            mm.append(visual)
    return DataProto.from_dict(tensors={"input_ids": ids, "attention_mask": mask,
        "position_ids": positions, "responses": ids[:, -1:], "response_mask": mask[:, -1:]},
        non_tensors={"multi_modal_inputs": object_array(mm)},
        meta_info={"world_probe": True, "temperature": 1.0, "pad_token_id": pad})


def correct_probabilities(log_probs, labels):
    scores = log_probs.detach().float().reshape(-1, 4)
    distribution = torch.softmax(scores, dim=-1)
    indices = torch.tensor(["ABCD".index(x) for x in labels], device=distribution.device)
    return distribution.gather(1, indices[:, None]).squeeze(1).cpu().numpy().reshape(-1, 2)


def prepare_rollouts(trainer, batch):
    cfg = trainer.config.algorithm.counterfactual_world
    sd = trainer.config.actor_rollout_ref.actor.self_distillation
    if int(trainer.config.actor_rollout_ref.rollout.n) != 16:
        raise ValueError("The paired experiment fixes 16 rollouts per source question")
    if (float(sd.get("pair_aux_weight", 0)) != 0 or sd.get("teacher_image_key") != "bbox_images"
            or sd.get("teacher_prompt_mode") not in (None, "null")):
        raise ValueError("Two-world OPD needs matching crop Teacher, no answer hints and no pair auxiliary loss")
    if trainer.use_prefix_grouper or trainer.use_grouped_candidate_rollouts:
        raise ValueError("Use the existing ungrouped Gaussian OPD actor for variable per-world counts")
    if trainer.use_legacy_worker_impl == "disable":
        raise ValueError("This runtime uses the existing FSDP DataProto worker")
    noisy = trainer.config.algorithm.get("noisy_rollout", {})
    if noisy.get("noisy_count", 0) or noisy.get("all_noisy_from_epoch") is not None:
        raise ValueError("Both worlds must generate on the same clean images used for Student re-scoring")
    # Teacher updates are applied by the actor after optimization; they do not
    # change the independent rollout inputs or the per-world sampling budget.
    if float(sd.get("caption_kl_lambda", 0)):
        raise ValueError("Keep the caption auxiliary loss disabled")
    worlds = make_worlds(batch)
    paired = [p is not None for p in batch.non_tensor_batch["world_pair"]]
    metrics = {"world/paired_fraction": float(np.mean(paired))}
    probabilities = None
    if cfg.mode == "weak_side":
        start = time.monotonic()
        probe = build_probe_batch(trainer, worlds)
        metrics["world/probe_input_tokens"] = int(probe.batch["attention_mask"].sum())
        probe, padding = pad_dataproto_to_divisor(probe, trainer._get_dp_size(trainer.actor_rollout_wg, "actor"))
        scores = unpad_dataproto(trainer.actor_rollout_wg.compute_log_prob(probe), padding)
        probabilities = correct_probabilities(scores.batch["old_log_probs"], worlds.non_tensor_batch["world_correct_option"])
        metrics["world/probe_seconds"] = time.monotonic() - start
        for s, name in enumerate(("original", "edited")):
            values = probabilities[np.asarray(paired), s]
            if len(values):
                metrics[f"world/{name}_correct_option_probability"] = float(values.mean())
    counts = allocation(paired, cfg.mode, probabilities, float(cfg.gap))
    metrics.update({"world/original_rollouts": int(counts[:, 0].sum()),
                    "world/edited_rollouts": int(counts[:, 1].sum()),
                    "world/total_rollouts": int(counts.sum()),
                    "world/original_priority_pairs": int(np.sum(counts[:, 0] == 12)),
                    "world/edited_priority_pairs": int(np.sum(counts[:, 1] == 12))})
    indices = np.repeat(np.arange(len(worlds)), counts.reshape(-1))
    result = worlds.select_idxs(indices)
    # Labels/options have served allocation. Do not pass them to generation or
    # Teacher construction; the normal reward_model field remains metadata only.
    for key in ("world_pair", "world_correct_option"):
        result.non_tensor_batch.pop(key)
    return result, metrics


def rollout_metrics(batch):
    metrics = {}
    sides = batch.non_tensor_batch["world_side"]
    lengths = batch.batch["response_mask"].sum(-1).float().cpu().numpy()
    prompts = batch.batch["attention_mask"].sum(-1).float().cpu().numpy() - lengths
    for name in ("original", "edited"):
        selected = sides == name
        if selected.any():
            metrics[f"world/{name}_response_tokens"] = int(lengths[selected].sum())
            metrics[f"world/{name}_mean_response_length"] = float(lengths[selected].mean())
            metrics[f"world/{name}_prompt_tokens"] = int(prompts[selected].sum())
    return metrics
