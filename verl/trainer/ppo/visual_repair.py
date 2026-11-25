"""Visual counterfactual gating and bounded, prefix-causal teacher repair."""
import os
import re
import time
from contextlib import contextmanager

import numpy as np
import torch
import torch.distributed as dist

from verl import DataProto


@contextmanager
def teacher_repair_attention(teacher):
    """Avoid the ROCm FlashAttention fault in the repair teacher's forward."""
    if not torch.version.hip:
        yield
        return
    from torch.nn.attention import SDPBackend, sdpa_kernel

    config = getattr(teacher, "module", teacher).config.text_config
    previous = config._attn_implementation
    config._attn_implementation = "sdpa"
    if not dist.is_initialized() or dist.get_rank() == 0:
        print("[visual-repair] teacher text attention=sdpa/math (ROCm)", flush=True)
    try:
        # The text model also uses this config to build the SDPA causal mask.
        # Limit this backend to teacher scoring/repair, retaining the fast
        # training, vision, and vLLM paths outside this context.
        with sdpa_kernel(SDPBackend.MATH):
            yield
    finally:
        config._attn_implementation = previous


@contextmanager
def trace_teacher_kernels(teacher):
    """Synchronize only a diagnostic run to locate asynchronous GPU faults."""
    if os.environ.get("VISUAL_REPAIR_DEBUG_SYNC") != "1":
        yield
        return
    rank = dist.get_rank() if dist.is_initialized() else 0
    handles = []

    def report(name, boundary):
        print(f"[visual-repair-debug] rank={rank} {boundary} {name}", flush=True)
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    for name, module in teacher.named_modules():
        if type(module).__name__ in {"Qwen3_5GatedDeltaNet", "Qwen3_5Attention", "Qwen3_5VisionModel"}:
            handles.append(module.register_forward_pre_hook(
                lambda module, args, name=name: report(name, "begin")))
            handles.append(module.register_forward_hook(
                lambda module, args, output, name=name: report(name, "end")))
    try:
        yield
    finally:
        for handle in handles:
            handle.remove()


def trace_repair_stage(stage):
    if os.environ.get("VISUAL_REPAIR_DEBUG_SYNC") == "1":
        rank = dist.get_rank() if dist.is_initialized() else 0
        print(f"[visual-repair-debug] rank={rank} stage={stage}", flush=True)
        if torch.cuda.is_available():
            torch.cuda.synchronize()


def first_conflict(plus_a, plus_b, minus_a, minus_b, a, b, mask,
                   tokenizer, preference_margin, visual_margin):
    preference = plus_b - plus_a
    visual = preference - (minus_b - minus_a)
    eligible = mask.bool() & a.ne(b) & (preference >= preference_margin) & (visual >= visual_margin)
    special = set(tokenizer.all_special_ids)
    for t in eligible.nonzero().flatten().tolist():
        if int(a[t]) in special or int(b[t]) in special:
            continue
        # Formatting-only disagreements are not useful factual repair points.
        if not any(c.isalnum() for c in tokenizer.decode([int(b[t])])):
            continue
        if not any(c.isalnum() for c in tokenizer.decode([int(a[t])])):
            continue
        return t, float(preference[t]), float(visual[t])
    return -1, 0.0, 0.0


def erased_view(multi_modal_inputs):
    """Erase normalized visual content; preserve the token grid and positions."""
    return np.array([
        {k: torch.zeros_like(v) if k == "pixel_values" else v for k, v in mm.items()}
        for mm in multi_modal_inputs
    ], dtype=object)


def next_token_inputs(row, prefix, pad_id):
    """Retain only the teacher prompt and accepted response prefix, no old suffix."""
    start = int(row["response_start_idx"][0])
    device = row["input_ids"].device
    suffix = torch.tensor(prefix + [pad_id], dtype=torch.long, device=device).unsqueeze(0)
    prompt_ids = row["input_ids"][:, :start]
    prompt_pos = row["position_ids"][..., :start]
    extension = torch.arange(1, suffix.shape[1] + 1, device=device) + prompt_pos[..., -1:]
    ids = torch.cat([prompt_ids, suffix], dim=-1)
    return {
        "input_ids": ids,
        "attention_mask": torch.cat([row["attention_mask"][:, :start], torch.ones_like(suffix)], dim=-1),
        "position_ids": torch.cat([prompt_pos, extension], dim=-1),
        "responses": suffix[:, -1:], "response_mask": torch.ones_like(suffix[:, -1:]),
        "response_start_idx": torch.tensor([ids.shape[1] - 1], device=device),
        "multi_modal_inputs": row["multi_modal_inputs"], "pad_token_id": pad_id,
    }


def clause_finished(tokens, tokenizer):
    if tokens[-1] in set(tokenizer.all_special_ids):
        return True
    return bool(re.search(r"[,，;；。!?！？\n.]\s*$", tokenizer.decode(tokens)))


def collective_max(value, device):
    tensor = torch.tensor(value, dtype=torch.long, device=device)
    if dist.is_initialized():
        dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
    return int(tensor)


class TeacherRepairDecoder:
    """Reuse the teacher KV/recurrent cache across the short repair segment."""
    def __init__(self, actor, row, prefix, pad):
        from verl.utils.model import extract_multi_modal_inputs
        self.actor = actor
        inputs = next_token_inputs(row, prefix, pad)
        self.ids = inputs["input_ids"][:, :-1]
        self.mask = inputs["attention_mask"][:, :-1]
        self.positions = inputs["position_ids"][..., :-1]
        self.visual = {k: v.to(self.ids.device) for k, v in
                       extract_multi_modal_inputs(row["multi_modal_inputs"]).items()
                       if k in {"pixel_values", "image_grid_thw", "pixel_values_videos", "video_grid_thw"}}
        self.cache = None
        self.blocked_tokens = [i for i in actor.tokenizer.all_special_ids
                               if i != actor.tokenizer.eos_token_id]

    @torch.no_grad()
    def step(self):
        positions = self.positions.transpose(0, 1) if self.positions.ndim == 3 else self.positions
        with torch.autocast(device_type=self.actor.device_name, dtype=self.actor.param_dtype):
            output = self.actor.teacher_module(
                input_ids=self.ids, attention_mask=self.mask, position_ids=positions,
                past_key_values=self.cache, use_cache=True, logits_to_keep=1,
                return_dict=True, **self.visual)
        self.cache = output.past_key_values
        if self.cache is None:
            raise RuntimeError("Teacher repair requires a working generation cache")
        logits = output.logits[:, -1].clone()
        logits[:, self.blocked_tokens] = -torch.inf
        token = logits.argmax(-1, keepdim=True)
        self.ids = token
        self.mask = torch.cat([self.mask, torch.ones_like(token)], dim=-1)
        self.positions = self.positions[..., -1:] + 1
        self.visual = {}
        return int(token[0, 0])


@torch.no_grad()
def repair_visual_prefixes(actor, data, device):
    with teacher_repair_attention(actor.teacher_module), trace_teacher_kernels(actor.teacher_module):
        return _repair_visual_prefixes(actor, data, device)


@torch.no_grad()
def _repair_visual_prefixes(actor, data, device):
    started = time.perf_counter()
    cfg = actor.config.self_distillation
    teacher = actor.teacher_module
    if teacher is None:
        raise ValueError("Visual repair requires the EMA teacher module")
    teacher.eval()
    temperature = float(data.meta_info["temperature"])
    pad = int(data.meta_info["pad_token_id"])
    max_patch = int(cfg.visual_repair_max_tokens)
    count = len(data)
    positions = torch.full((count,), -1, dtype=torch.long)
    patches = torch.full((count, max_patch), pad, dtype=torch.long)
    lengths = torch.zeros(count, dtype=torch.long)
    preferences = torch.zeros(count)
    visual_margins = torch.zeros(count)
    # Equal row counts on all ranks preserve FSDP collective ordering.
    probe_batch_size = int(cfg.get("visual_repair_probe_batch_size", 1))
    for i in range(0, count, probe_batch_size):
        item = data[i:i + probe_batch_size].to(device)
        row = {k: item.batch["teacher_" + k] for k in
               ("input_ids", "attention_mask", "position_ids", "response_start_idx")}
        row.update(responses=item.batch["responses"], response_mask=item.batch["response_mask"],
                   multi_modal_inputs=item.non_tensor_batch["teacher_multi_modal_inputs"], pad_token_id=pad)
        trace_repair_stage(f"row={i}/positive")
        positive = actor._forward_micro_batch(row, temperature, distill_topk=1, module=teacher)
        trace_repair_stage(f"row={i}/negative")
        negative = actor._forward_micro_batch(
            {**row, "multi_modal_inputs": erased_view(row["multi_modal_inputs"])},
            temperature, topk_indices=positive["topk_indices"], module=teacher)
        trace_repair_stage(f"row={i}/conflict")
        for j in range(len(item)):
            t, pref, visual = first_conflict(
                positive["log_probs"][j], positive["topk_logps"][j, :, 0],
                negative["log_probs"][j], negative["topk_logps"][j, :, 0],
                row["responses"][j], positive["topk_indices"][j, :, 0], row["response_mask"][j],
                actor.tokenizer, cfg.visual_repair_preference_margin, cfg.visual_repair_visual_margin)
            positions[i+j], preferences[i+j], visual_margins[i+j] = t, pref, visual
            if t >= 0:
                patches[i+j, 0] = positive["topk_indices"][j, t, 0].cpu()
                lengths[i+j] = 1
        del positive, negative, row, item

    active_rows = positions.ge(0).nonzero().flatten().tolist()
    slots = collective_max(len(active_rows), device)
    for slot in range(slots):
        active = slot < len(active_rows)
        i = active_rows[slot] if active else 0
        item = data[i:i + 1].to(device)
        row = {k: item.batch["teacher_" + k] for k in
               ("input_ids", "attention_mask", "position_ids", "response_start_idx")}
        row["multi_modal_inputs"] = item.non_tensor_batch["teacher_multi_modal_inputs"]
        t = max(0, int(positions[i]))
        prefix = item.batch["responses"][0, :t].tolist()
        patch = patches[i, :int(lengths[i])].tolist() if active else [pad]
        budget = min(max_patch, item.batch["responses"].shape[-1] - t)
        unfinished = active and len(patch) < budget and not clause_finished(patch, actor.tokenizer)
        decoder = TeacherRepairDecoder(actor, row, prefix + patch, pad)
        decode_step = 0
        # Ranks with fewer repairs still join each FSDP forward with an inert prompt.
        while collective_max(int(unfinished), device):
            trace_repair_stage(f"slot={slot}/decode={decode_step}/cached={decoder.cache is not None}")
            token = decoder.step()
            decode_step += 1
            if unfinished:
                patch.append(token)
                unfinished = len(patch) < budget and not clause_finished(patch, actor.tokenizer)
        if active:
            patches[i, :len(patch)] = torch.tensor(patch)
            lengths[i] = len(patch)
        del row, item, decoder
    return DataProto.from_dict({
        "repair_position": positions, "repair_tokens": patches, "repair_length": lengths,
        "repair_preference": preferences, "repair_visual_margin": visual_margins,
        "repair_worker_seconds": torch.full((count,), time.perf_counter() - started),
    })


def repaired_prefix(response, position, patch, patch_length):
    return response[:position].tolist() + patch[:patch_length].tolist()


def repair_span_mask(response_mask, positions, lengths):
    """Mark only the Teacher-written span in the final repaired trajectory."""
    t = torch.arange(response_mask.shape[-1], device=response_mask.device)[None, :]
    starts = positions.to(response_mask.device)[:, None]
    ends = starts + lengths.to(response_mask.device)[:, None]
    return ((starts >= 0) & (t >= starts) & (t < ends)) & response_mask.bool()


def repair_weight_denominator(response_mask, patch_mask, weight, device):
    """Global weighted-token denominator per DP rank, before microbatching."""
    weights = 1.0 + (weight - 1.0) * patch_mask.float()
    denominator = (weights * response_mask).sum().to(device)
    if dist.is_initialized():
        dist.all_reduce(denominator)
        denominator /= dist.get_world_size()
    return denominator.clamp(min=1.0).item()


def can_skip_repair_old_log_probs(config):
    """Direct crop JSD needs no old-policy anchor when both IS paths are off."""
    actor = config.actor_rollout_ref.actor
    sd = actor.get("self_distillation", {})
    correction = config.algorithm.get("rollout_correction", {}) or {}
    return (
        (sd.get("visual_repair_enabled", False) or sd.get("correctness_entropy_enabled", False))
        and actor.policy_loss.get("loss_mode") == "vopd"
        and sd.get("target_mode", "full_teacher") == "full_teacher"
        and sd.get("full_logit_distillation", False)
        and sd.get("teacher_always_on", False)
        and sd.get("teacher_image_key") is not None
        and sd.get("teacher_regularization", "ema") == "ema"
        and not sd.get("fallback_to_policy_loss_on_missing_teacher", False)
        and sd.get("is_clip") is None
        and not correction.get("rollout_is")
        and not correction.get("rollout_rs")
        and not correction.get("bypass_mode", False)
        and not actor.get("use_kl_loss", False)
        and not config.algorithm.get("use_kl_in_reward", False)
    )


def include_response_in_support(indices, logps, responses, response_logps):
    """Keep the actual token explicit instead of hiding a repair in the tail."""
    present = indices.eq(responses.unsqueeze(-1)).any(-1, keepdim=True)
    last_index = torch.where(present, indices[..., -1:], responses.unsqueeze(-1))
    last_logp = torch.where(present, logps[..., -1:], response_logps.unsqueeze(-1))
    return (torch.cat([indices[..., :-1], last_index], -1),
            torch.cat([logps[..., :-1], last_logp], -1))


def replace_generation_rows(batch, generated, indices):
    """Replace complete trajectories and their multimodal positions together."""
    for key in ("responses", "response_mask", "prompts", "input_ids", "attention_mask", "position_ids"):
        batch.batch[key][indices] = generated.batch[key]
    if "multi_modal_inputs" in generated.non_tensor_batch:
        batch.non_tensor_batch["multi_modal_inputs"][indices] = generated.non_tensor_batch["multi_modal_inputs"]
    # A teacher-written token has no Student rollout propensity. The direct JSD
    # update computes current Student probabilities in its differentiable pass.
    if "rollout_log_probs" in batch.batch:
        del batch.batch["rollout_log_probs"]
    return batch
