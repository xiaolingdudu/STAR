"""Head-wise crop routing on full-image Values, before Qwen3.5's output gate."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch


def axis_overlap(crop_cells: int, full_cells: int, lo: float, hi: float) -> torch.Tensor:
    """Each row distributes one crop cell's mass over full-image cells."""
    source = torch.linspace(lo, hi, crop_cells + 1, dtype=torch.float32)
    target = torch.linspace(0, 1, full_cells + 1, dtype=torch.float32)
    overlap = (torch.minimum(source[1:, None], target[None, 1:])
               - torch.maximum(source[:-1, None], target[None, :-1])).clamp_min(0)
    return overlap / overlap.sum(-1, keepdim=True)


class SpatialMap:
    def __init__(self, crop_hw, full_hw, bbox, device="cpu"):
        self.crop_hw, self.full_hw = tuple(crop_hw), tuple(full_hw)
        x1, y1, x2, y2 = bbox
        self.y = axis_overlap(crop_hw[0], full_hw[0], y1, y2).to(device)
        self.x = axis_overlap(crop_hw[1], full_hw[1], x1, x2).to(device)
        self.support = (self.y.sum(0)[:, None] * self.x.sum(0)[None, :]).flatten() > 0

    def __call__(self, attention):
        # Two axis products avoid materializing an N_crop by N_full matrix.
        values = attention.float().reshape(*attention.shape[:-1], *self.crop_hw)
        values = values @ self.x
        values = (values.transpose(-1, -2) @ self.y).transpose(-1, -2)
        return values.flatten(-2)


def query_rows(query_length, key_length, suffix_start, prompt_length, scope, device):
    positions = torch.arange(key_length - query_length, key_length, device=device)
    if scope == "prompt":
        selected = (positions >= suffix_start) & (positions < prompt_length - 1)
    elif scope == "response":
        # The last prompt row predicts the FIRST response token.
        selected = positions >= prompt_length - 1
    else:
        raise ValueError(scope)
    return selected.nonzero().flatten(), positions[selected] - suffix_start


def selected_attention(query, key, attention_mask, rows, image_positions, scaling):
    """Exact total image mass and conditional image distribution for selected rows."""
    scores = query[:, :, rows].float() @ key.float().transpose(-1, -2) * scaling
    q_abs = key.shape[-2] - query.shape[-2] + rows
    causal = torch.arange(key.shape[-2], device=query.device)[None, :] <= q_abs[:, None]
    scores = scores.masked_fill(~causal[None, None], -torch.inf)
    if attention_mask is not None:
        mask = attention_mask if attention_mask.shape[-2] == 1 else attention_mask[:, :, rows]
        if mask.dtype == torch.bool:
            scores = scores.masked_fill(~mask, -torch.inf)
        else:
            scores = scores + mask
    visual = scores[..., image_positions]
    mass = (visual.logsumexp(-1) - scores.logsumexp(-1)).exp().unsqueeze(-1)
    return mass, visual.softmax(-1)


@dataclass
class RoutingContext:
    mode: str = "off"
    layer: int = -1
    scope: str = "response"
    image_positions: torch.Tensor | None = None
    suffix_start: int = 0
    prompt_length: int = 0
    spatial_map: SpatialMap | None = None
    donor: dict = field(default_factory=dict)
    changed_rows: int = 0
    mass_sum: float = 0.0
    route_l1_sum: float = 0.0
    measured_heads: int = 0

    def set_view(self, view, mode):
        self.mode = mode
        self.image_positions = view.image_positions
        self.suffix_start = view.suffix_start
        self.prompt_length = view.prompt_length

    def reset_measurements(self):
        self.changed_rows = 0
        self.mass_sum = self.route_l1_sum = 0.0
        self.measured_heads = 0

    def measurements(self):
        n = max(1, self.measured_heads)
        return {"intervened_query_rows": self.changed_rows,
                "mean_image_mass": self.mass_sum / n,
                "mean_route_l1": self.route_l1_sum / n}


class RoutingAttention:
    """Wrap SDPA only; Q/K/V projection, RoPE, cache, gate and W_O stay upstream."""

    def __init__(self, context, original):
        self.context, self.original = context, original

    def __call__(self, module, query, key, value, attention_mask, scaling=None, **kwargs):
        output, weights = self.original(module, query, key, value, attention_mask,
                                        scaling=scaling, **kwargs)
        ctx = self.context
        if ctx.mode == "off" or getattr(module, "layer_idx", None) != ctx.layer:
            return output, weights
        if query.shape[0] != 1:
            raise ValueError("Routing diagnostic uses one unpadded example per GPU")
        rows, logical = query_rows(query.shape[-2], key.shape[-2], ctx.suffix_start,
                                   ctx.prompt_length, ctx.scope, query.device)
        if not len(rows):
            return output, weights
        repeats = query.shape[1] // key.shape[1]
        key = key.repeat_interleave(repeats, dim=1)
        visual_values = value[:, :, ctx.image_positions].repeat_interleave(repeats, dim=1)
        if ctx.mode != "capture":
            output = output.clone()
        for start in range(0, len(rows), 32):
            chunk, labels = rows[start:start + 32], logical[start:start + 32].tolist()
            mass, own = selected_attention(query, key, attention_mask, chunk,
                                           ctx.image_positions, scaling or module.scaling)
            if ctx.mode == "capture":
                mapped = ctx.spatial_map(own[0])
                for i, label in enumerate(labels):
                    ctx.donor[label] = mapped[:, i].detach()
                continue
            support = ctx.spatial_map.support
            if ctx.mode == "crop_route":
                route = torch.stack([ctx.donor[label] for label in labels], dim=1)[None]
            elif ctx.mode == "bbox_uniform":
                route = support.float()[None, None, None].expand_as(own) / support.sum()
            elif ctx.mode == "bbox_self":
                # Recompute the within-bbox softmax to avoid underflow in tiny masses.
                visual_scores = query[:, :, chunk].float() @ key[:, :, ctx.image_positions].float().transpose(-1, -2)
                visual_scores = visual_scores * (scaling or module.scaling)
                route = visual_scores.masked_fill(~support, -torch.inf).softmax(-1)
            elif ctx.mode == "identity":
                route = own
            else:
                raise ValueError(ctx.mode)
            delta = (mass * (route - own)) @ visual_values.float()
            output[:, chunk] += delta.transpose(1, 2).to(output.dtype)
            ctx.changed_rows += len(chunk)
            ctx.mass_sum += mass.sum().item()
            ctx.route_l1_sum += (route - own).abs().sum(-1).sum().item()
            ctx.measured_heads += mass.numel()
        return output, weights


def install_routing(context):
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    original = ALL_ATTENTION_FUNCTIONS["sdpa"]
    ALL_ATTENTION_FUNCTIONS.register("sdpa", RoutingAttention(context, original))
    return original
