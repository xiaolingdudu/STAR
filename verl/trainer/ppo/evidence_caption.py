"""Box attention supervision and a separate, text-only caption Teacher.

The crop Teacher and Student forward are unchanged. Attention is conditional
on visual keys, at the causal rows predicting the sampled response tokens.
All trajectories participate; this module does not inspect correctness/reward.
"""
from dataclasses import dataclass

import torch
from torch.utils.checkpoint import checkpoint

from verl.trainer.ppo.grounded_opd import GroundedAttentionCapture


def attention_weight_at_step(config, step):
    """Keep the initial weight through start, then linearly reach the end weight."""
    initial = float(config.get("evidence_attention_lambda", 0.0))
    start = config.get("evidence_attention_decay_start_step")
    if start is None:
        return initial
    end = int(config.evidence_attention_decay_end_step)
    final = float(config.evidence_attention_final_lambda)
    progress = min(max((int(step) - start) / (end - start), 0.0), 1.0)
    return initial + progress * (final - initial)


def gaussian_target(height, width, bbox, *, alpha=0.1, device=None, anchor_centers=None):
    """Normalized xyxy boxes -> equally mixed Gaussians on native token centers."""
    boxes = torch.as_tensor(bbox, dtype=torch.float32, device=device).detach().reshape(-1, 4)
    yy, xx = torch.meshgrid(
        (torch.arange(height, device=device, dtype=torch.float32) + 0.5) / height,
        (torch.arange(width, device=device, dtype=torch.float32) + 0.5) / width,
        indexing="ij",
    )
    centers = (boxes[:, :2] + boxes[:, 2:]) / 2
    sigma = (boxes[:, 2:] - boxes[:, :2]) / 4
    if anchor_centers is not None:
        if len(boxes) != 1:
            raise ValueError("Refined centers require the original single evidence box")
        centers = torch.as_tensor(anchor_centers, dtype=torch.float32, device=device).detach().reshape(-1, 2)
        if len(centers) == 0 or not bool(torch.isfinite(centers).all()) or not bool(((centers >= 0) & (centers <= 1)).all()):
            raise ValueError("Refined centers must be nonempty normalized image coordinates")
    if not bool((sigma > 0).all()):
        raise ValueError("Evidence boxes must have positive area")
    log_weights = -0.5 * (
        ((xx.flatten()[None] - centers[:, 0:1]) / sigma[:, 0:1]).square()
        + ((yy.flatten()[None] - centers[:, 1:2]) / sigma[:, 1:2]).square()
    )
    evidence = log_weights.softmax(-1).mean(0)
    return (1 - alpha) * evidence + alpha / (height * width)


def pack_evidence_centers(rows, device=None):
    """Preserve multiple centers through tensor batching, repetition and reordering."""
    count = max(len(row) for row in rows)
    centers = torch.zeros(len(rows), count, 2, dtype=torch.float32, device=device)
    mask = torch.zeros(len(rows), count, dtype=torch.bool, device=device)
    for i, row in enumerate(rows):
        centers[i, :len(row)] = torch.tensor([list(point) for point in row], dtype=torch.float32, device=device)
        mask[i, :len(row)] = True
    return {"evidence_centers": centers, "evidence_center_mask": mask}


@torch.no_grad()
def region_projection_target(probs, inside, minimum_mass=0.9):
    """Raise evidence mass only when insufficient; preserve within-region ratios."""
    probs = probs.detach().float()
    mass = probs[:, inside].sum(-1, keepdim=True)
    outside_mass = probs[:, ~inside].sum(-1, keepdim=True)
    scale = torch.where(inside[None], minimum_mass / mass.clamp_min(1e-30),
                        (1 - minimum_mass) / outside_mass.clamp_min(1e-30))
    return torch.where(mass >= minimum_mass, probs, probs * scale)


def evidence_attention_layer_types(config):
    """Qwen3-VL is full attention throughout; Qwen3.5 declares its hybrid layers."""
    if config.model_type == "qwen3_vl":
        return ["full_attention"] * config.text_config.num_hidden_layers
    return config.text_config.layer_types


def select_evidence_attention_layers(layer_types, layer):
    """A positive layer is one-based; zero selects every full-attention layer."""
    indices = ([i for i, kind in enumerate(layer_types) if kind == "full_attention"]
               if int(layer) == 0 else [int(layer) - 1])
    if not indices or any(i < 0 or i >= len(layer_types) or layer_types[i] != "full_attention" for i in indices):
        raise ValueError(f"Evidence attention requires full-attention layers, got selection={layer}")
    return tuple(indices)


def selected_response_offsets(mask, max_tokens=64):
    """Zero means all valid tokens; padding is never selected."""
    offsets = mask.nonzero(as_tuple=False).flatten()
    if max_tokens > 0 and offsets.numel() > max_tokens:
        take = torch.linspace(0, offsets.numel() - 1, max_tokens, device=mask.device).round().long()
        offsets = offsets[take]
    return offsets


def _layer_visual_probabilities(query, key, scale, softcap):
    query, key = query.float(), key.float()
    if query.shape[0] != key.shape[0]:
        key = key.repeat_interleave(query.shape[0] // key.shape[0], dim=0)
    scores = torch.matmul(query, key.transpose(-1, -2)) * scale
    if softcap is not None:
        scores = torch.tanh(scores / float(softcap)) * float(softcap)
    return scores.softmax(-1).mean(0)


@dataclass
class EvidenceAttentionCapture(GroundedAttentionCapture):
    response_mask: torch.Tensor = None
    max_tokens: int = 64
    background_alpha: float = 0.1
    anchor_centers: torch.Tensor = None
    anchor_mask: torch.Tensor = None
    expected_layer_indices: tuple = ()
    response_chunk_size: int = 128
    detached_scores: bool = False
    token_kl: torch.Tensor = None
    token_selected: torch.Tensor = None
    local_crop_footprints: torch.Tensor = None
    local_attention: list = None
    collect_competitor_attention: bool = False
    competitor_attention: list = None
    center_attention: list = None
    center_attention_requires_grad: bool = False
    target_map: str = "gaussian"
    region_minimum_mass: float = 0.9

    def finish(self):
        if not self.raw_qk or (self.expected_layer_indices and tuple(self.layer_indices) != self.expected_layer_indices):
            raise RuntimeError(f"Evidence attention captured {self.layer_indices}; expected {self.expected_layer_indices}")
        with torch.autocast(device_type=self.raw_qk[0][1].device.type, enabled=False):
            if self.detached_scores and not self.center_attention_requires_grad:
                with torch.no_grad():
                    return self._finish_float()
            return self._finish_float()

    def _finish_float(self):
        query = self.raw_qk[0][1]
        losses, masses, selected_counts, valid_counts = [], [], [], []
        gaussian_gate_sums, region_satisfied_counts = [], []
        if self.detached_scores:
            self.token_kl = torch.zeros_like(self.response_mask, dtype=torch.float32)
            self.token_selected = torch.zeros_like(self.response_mask, dtype=torch.bool)
        if self.local_crop_footprints is not None:
            self.local_attention = []
        if self.collect_competitor_attention:
            self.competitor_attention = []
        if self.target_map == "teacher_center":
            self.center_attention = []
        for b in range(query.shape[0]):
            offsets = selected_response_offsets(self.response_mask[b], self.max_tokens)
            valid_counts.append(int(self.response_mask[b].count_nonzero()))
            positions = self.response_positions[b, offsets]
            visual = self.input_ids[b].eq(self.image_token_id).nonzero(as_tuple=False).flatten()
            temporal, raw_h, raw_w = self.image_grid_thw[b].tolist()
            height, width = int(raw_h) // self.spatial_merge_size, int(raw_w) // self.spatial_merge_size
            if temporal != 1 or visual.numel() != height * width:
                raise ValueError("Evidence attention requires one image with a matching native visual grid")
            if offsets.numel() == 0:
                if self.center_attention is not None:
                    self.center_attention.append((offsets, None, height, width))
                if self.competitor_attention is not None:
                    self.competitor_attention.append(None)
                if self.local_attention is not None:
                    self.local_attention.append((offsets, None, None))
                losses.append(query[b].sum() * 0)
                masses.append(query.new_zeros(()))
                selected_counts.append(0)
                continue
            if not bool((visual.max() < positions.min()).item()):
                raise ValueError("Visual keys must precede response prediction rows")
            layers = [(q[b, :, positions, :], k[b, :, visual, :],
                       float(scale if scale is not None else q.shape[-1] ** -0.5), softcap)
                      for _, q, k, scale, softcap in self.raw_qk]
            centers = None if self.anchor_centers is None else self.anchor_centers[b, self.anchor_mask[b]]
            target = gaussian_target(height, width, self.bboxes[b], alpha=self.background_alpha, device=query.device,
                                     anchor_centers=centers)
            yy, xx = torch.meshgrid(
                (torch.arange(height, device=query.device) + 0.5) / height,
                (torch.arange(width, device=query.device) + 0.5) / width, indexing="ij")
            x1, y1, x2, y2 = self.bboxes[b]
            inside = ((xx >= x1) & (xx <= x2) & (yy >= y1) & (yy <= y2)).flatten()
            if self.target_map == "region_projection":
                # A visual patch belongs to the evidence region when their
                # spatial supports have positive overlap, including small boxes.
                inside = ((xx + 0.5 / width > x1) & (xx - 0.5 / width < x2)
                          & (yy + 0.5 / height > y1) & (yy - 0.5 / height < y2)).flatten()
                if not bool(inside.any()):
                    raise ValueError("Evidence box does not overlap the visual grid")
            token_losses, token_masses, spatial_probs = [], [], []
            competitor_sum = torch.zeros_like(target) if self.collect_competitor_attention else None
            for start in range(0, offsets.numel(), self.response_chunk_size):
                probs = None
                for q, k, scale, softcap in layers:
                    args = (q[:, start:start + self.response_chunk_size], k, scale, softcap)
                    # Recompute only the small response-to-image attention branch
                    # on backward, not an entire additional model forward.
                    if torch.is_grad_enabled() and (len(layers) > 1 or offsets.numel() > self.response_chunk_size):
                        layer_probs = checkpoint(_layer_visual_probabilities, *args,
                                                 use_reentrant=False, preserve_rng_state=False)
                    else:
                        layer_probs = _layer_visual_probabilities(*args)
                    probs = layer_probs if probs is None else probs + layer_probs
                probs = probs / len(layers)
                if competitor_sum is not None:
                    competitor_sum.add_(probs.detach().sum(0))
                if self.local_attention is not None or self.center_attention is not None:
                    spatial_probs.append(probs if self.center_attention_requires_grad else probs.detach())
                # Average distributions BEFORE KL; averaging per-layer KL is a
                # different objective. Chunks cover every selected response row.
                gaussian_kl = (probs * (probs.clamp_min(1e-30).log() - target.log())).sum(-1)
                mass = probs.detach()[:, inside].sum(-1)
                if self.target_map == "region_projection":
                    projected = region_projection_target(probs, inside, self.region_minimum_mass)
                    region_kl = (probs * (probs.clamp_min(1e-30).log()
                                         - projected.clamp_min(1e-30).log())).sum(-1).clamp_min(0)
                    token_losses.append(region_kl)
                    gaussian_gate_sums.append((gaussian_kl.detach() / (1 + gaussian_kl.detach())).sum())
                    region_satisfied_counts.append((mass >= self.region_minimum_mass).sum())
                else:
                    token_losses.append(gaussian_kl)
                token_masses.append(mass)
            if competitor_sum is not None:
                self.competitor_attention.append((competitor_sum / offsets.numel()).reshape(height, width))
            if self.detached_scores:
                self.token_kl[b, offsets] = torch.cat(token_losses).detach()
                self.token_selected[b, offsets] = True
            if self.center_attention is not None:
                self.center_attention.append((offsets, torch.cat(spatial_probs), height, width))
            if self.local_attention is not None:
                from verl.trainer.ppo.local_evidence_target import quadrant_basis
                basis = quadrant_basis(height, width, self.local_crop_footprints[b])
                self.local_attention.append((offsets, torch.cat(spatial_probs), basis))
            losses.append(torch.cat(token_losses).mean())
            masses.append(torch.cat(token_masses).mean())
            selected_counts.append(offsets.numel())
        loss = torch.stack(losses).mean()
        metrics = {
            "evidence/attention_kl": loss.detach().item(),
            "evidence/box_conditional_mass": torch.stack(masses).mean().item(),
            "evidence/selected_tokens_mean": sum(selected_counts) / len(selected_counts),
            "evidence/selected_token_fraction": sum(selected_counts) / max(sum(valid_counts), 1),
            "evidence/attention_layer_count": len(self.raw_qk),
            "evidence/trajectory_fraction": sum(n > 0 for n in selected_counts) / len(selected_counts),
            "evidence/reward_gate": 0.0,
        }
        if self.target_map == "region_projection" and gaussian_gate_sums:
            count = sum(selected_counts)
            metrics.update({
                "evidence/region_minimum_mass": self.region_minimum_mass,
                "evidence/region_satisfied_fraction": (torch.stack(region_satisfied_counts).sum() / count).item(),
                "evidence/gaussian_reference_gate_mean": (torch.stack(gaussian_gate_sums).sum() / count).item(),
            })
        return loss, metrics


def caption_forward_kl(student_logps, caption_logps):
    """Teacher -> Student KL on shared top-k vocabulary support plus one tail."""
    def with_tail(logps):
        logps = logps.float()
        mass = torch.logsumexp(logps, -1, keepdim=True).clamp(max=-1e-7)
        return torch.cat((logps, (-torch.expm1(mass)).log()), dim=-1)
    p, q = with_tail(student_logps), with_tail(caption_logps.detach())
    return (q.exp() * (q - p)).sum(-1)


def build_caption_batch(trainer, batch, max_prompt_len):
    """Use production prompt cache/position handling, with no visual payload."""
    ids, masks, positions, starts = [], [], [], []
    cache = {}
    for i, messages in enumerate(batch.non_tensor_batch["caption_prompt"]):
        inp, mask, pos, start, mm, _ = trainer._build_teacher_prompt_inputs(
            list(messages), batch.batch["responses"][i], batch.batch["response_mask"][i],
            max_prompt_len=max_prompt_len, prompt_cache=cache)
        if mm and any(key in mm for key in ("pixel_values", "image_grid_thw", "pixel_values_videos")):
            raise ValueError("Caption Teacher must receive text only")
        ids.append(inp); masks.append(mask); positions.append(pos); starts.append(start)
    device = batch.batch["responses"].device
    input_ids, attention_mask, position_ids = trainer._pad_teacher_prompt_input_lists(
        input_ids_list=ids, attention_mask_list=masks, position_ids_list=positions, device=device)
    return {
        "caption_input_ids": input_ids, "caption_attention_mask": attention_mask,
        "caption_position_ids": position_ids,
        "caption_response_start_idx": torch.stack(starts).to(device),
    }
