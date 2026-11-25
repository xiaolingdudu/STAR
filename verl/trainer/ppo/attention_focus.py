"""Crop-guided Teacher routing on original-image Values, without Student hooks."""
from contextlib import contextmanager
from contextvars import ContextVar

import torch

from evaluation.routing_intervention import SpatialMap


_ACTIVE = ContextVar("attention_focus", default=None)
_PATCHED = False


class AttentionFocus:
    def __init__(self, layer, rho, bbox, chunk_size=16):
        self.layer, self.rho = int(layer), float(rho)
        self.bboxes = torch.as_tensor(bbox).reshape(-1, 4).tolist()
        self.chunk_size = chunk_size
        self.donor = None
        self.changed = 0
        self.mass_sum = self.l1_sum = self.delta_sum = 0.0
        self.mode = "capture"

    def configure(self, input_ids, attention_mask, rows, grid_thw, image_token_id, merge_size, mode):
        self.mode = mode
        self.positions = [(ids == image_token_id).nonzero().flatten() for ids in input_ids]
        self.valid_keys = attention_mask.bool()
        self.rows = rows
        self.valid_rows = self.valid_keys.gather(1, rows)
        if len(grid_thw) != len(input_ids) or not bool((grid_thw[:, 0] == 1).all()):
            raise ValueError("Attention-focus v1 requires one image per view")
        grids = [tuple(int(x) // merge_size for x in g[1:]) for g in grid_thw]
        if any(g[0] * g[1] != len(p) for g, p in zip(grids, self.positions)):
            raise ValueError("Image token count does not match the spatial grid")
        if mode == "capture":
            self.crop_grids = grids
            self.donor = None
        else:
            if self.donor is None or len(self.donor) != len(rows) or any(d.shape[1] != rows.shape[1] for d in self.donor):
                raise ValueError("Crop/full response-prediction rows are not aligned")
            self.spatial_maps = [SpatialMap(c, f, b, input_ids.device)
                                 for c, f, b in zip(self.crop_grids, grids, self.bboxes)]

    def observe_or_route(self, query, key, value, output, scaling):
        if query.shape[0] != len(self.valid_keys) or query.shape[2] != self.valid_keys.shape[1]:
            raise ValueError("Attention-focus requires an unpartitioned padded Teacher forward")
        repeats = query.shape[1] // key.shape[1]
        captured = []
        result = output if self.mode == "capture" else output.clone()
        for b, positions in enumerate(self.positions):
            keys = key[b].repeat_interleave(repeats, dim=0).float()
            visual_values = value[b, :, positions].repeat_interleave(repeats, dim=0).float()
            chunks = []
            for start in range(0, self.rows.shape[1], self.chunk_size):
                rows = self.rows[b, start:start+self.chunk_size]
                scores = (query[b, :, rows].float() @ keys.transpose(-1, -2)) * scaling
                causal = torch.arange(keys.shape[1], device=key.device)[None] <= rows[:, None]
                scores.masked_fill_(~(causal & self.valid_keys[b, None])[None], -torch.inf)
                visual_scores = scores[..., positions]
                own = visual_scores.softmax(-1)
                if self.mode == "capture":
                    chunks.append(own.detach())
                    continue
                mass = (visual_scores.logsumexp(-1) - scores.logsumexp(-1)).exp()[..., None]
                mapping = self.spatial_maps[b]
                if self.mode == "bbox_self":
                    route = visual_scores.masked_fill(~mapping.support, -torch.inf).softmax(-1)
                else:
                    route = mapping(self.donor[b][:, start:start+len(rows)])
                shift = self.rho * (route - own)
                valid = self.valid_rows[b, start:start+len(rows)][None, :, None]
                shift = shift * valid
                delta = (mass * shift) @ visual_values
                result[b, rows] += delta.transpose(0, 1).to(output.dtype)
                self.changed += valid.sum() * query.shape[1]
                self.mass_sum += (mass * valid).sum()
                self.l1_sum += shift.abs().sum()
                self.delta_sum += delta.square().sum(-1).sqrt().sum()
            if self.mode == "capture":
                captured.append(torch.cat(chunks, dim=1))
        if self.mode == "capture":
            self.donor = captured
        return result

    def metrics(self):
        n = torch.as_tensor(self.changed).clamp_min(1)
        return {"attention_focus/image_mass": float(self.mass_sum/n),
                "attention_focus/route_l1": float(self.l1_sum/n),
                "attention_focus/readout_delta_norm": float(self.delta_sum/n),
                "attention_focus/layer": self.layer+1,
                "attention_focus/rho": self.rho}


def _wrap(original):
    def forward(module, query, key, value, attention_mask, *args, **kwargs):
        output, weights = original(module, query, key, value, attention_mask, *args, **kwargs)
        ctx = _ACTIVE.get()
        if ctx is not None and getattr(module, "layer_idx", None) == ctx.layer:
            scale = kwargs.get("scaling")
            output = ctx.observe_or_route(query, key, value, output,
                                          float(scale if scale is not None else query.shape[-1] ** -0.5))
        return output, weights
    return forward


def install_attention_focus():
    global _PATCHED
    if not _PATCHED:
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
        for name in ("flash_attention_2", "flash_attention_3", "sdpa"):
            if name in ALL_ATTENTION_FUNCTIONS:
                ALL_ATTENTION_FUNCTIONS[name] = _wrap(ALL_ATTENTION_FUNCTIONS[name])
        _PATCHED = True


@contextmanager
def use_attention_focus(context):
    token = _ACTIVE.set(context)
    try:
        yield
    finally:
        _ACTIVE.reset(token)
