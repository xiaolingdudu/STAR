"""Grounded OPD response-to-evidence attention distillation.

The module captures the attention distribution that is already used by the
Qwen3.5 language model.  It does not add a parallel heatmap head: gradients
therefore update the same Q/K attention path that produces the model's visual
evidence readout.
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn.functional as F


_ACTIVE_CAPTURE: ContextVar[Optional["GroundedAttentionCapture"]] = ContextVar(
    "grounded_opd_attention_capture", default=None
)
_PATCHED = False


def select_grounded_attention_layer(
    layer_types: list[str],
    layer_start_fraction: float,
    layer_end_fraction: float,
) -> int:
    """Select one representative full-attention layer from the configured range."""

    total_layers = len(layer_types)
    start = int(total_layers * layer_start_fraction)
    end = max(start + 1, int(total_layers * layer_end_fraction))
    candidates = [
        index
        for index in range(start, min(total_layers, end))
        if "full" in str(layer_types[index]).lower()
    ]
    if not candidates:
        raise ValueError(
            "Grounded OPD found no full-attention layer in the configured range: "
            f"start={start}, end={end}, layer_types={layer_types}"
        )
    return candidates[len(candidates) // 2]


def _visual_runs(mask: torch.Tensor) -> list[torch.Tensor]:
    positions = mask.nonzero(as_tuple=False).flatten()
    if positions.numel() == 0:
        return []
    split_points = (positions[1:] != positions[:-1] + 1).nonzero(as_tuple=False).flatten() + 1
    return list(torch.tensor_split(positions, split_points.tolist()))


def _resize_mass(mass: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """Resize a spatial probability map and preserve its total mass."""

    source_mass = mass.sum(dim=(-2, -1), keepdim=True)
    resized = F.interpolate(mass.unsqueeze(1), size=(height, width), mode="bilinear", align_corners=False).squeeze(1)
    return resized * (source_mass / resized.sum(dim=(-2, -1), keepdim=True).clamp_min(1e-12))


def _canonicalize_visual_attention(
    attention: torch.Tensor,
    input_ids: torch.Tensor,
    image_token_id: int,
    image_grid_thw: torch.Tensor,
    spatial_merge_size: int,
    grid_size: int,
    view: str,
    bboxes: Optional[torch.Tensor],
) -> torch.Tensor:
    """Map per-token visual attention to a common full-image coordinate grid."""

    batch_size, response_length, _ = attention.shape
    result = attention.new_zeros(batch_size, response_length, 1 + grid_size * grid_size)
    grid_cursor = 0
    for sample_idx in range(batch_size):
        visual_mask = input_ids[sample_idx].eq(image_token_id)
        result[sample_idx, :, 0] = attention[sample_idx, :, ~visual_mask].sum(dim=-1)
        canvas = attention.new_zeros(response_length, grid_size, grid_size)
        runs = _visual_runs(visual_mask)
        for image_idx, positions in enumerate(runs):
            if grid_cursor >= image_grid_thw.shape[0]:
                raise ValueError("image_grid_thw has fewer entries than image token runs")
            temporal, raw_height, raw_width = [int(value) for value in image_grid_thw[grid_cursor].tolist()]
            grid_cursor += 1
            height = raw_height // spatial_merge_size
            width = raw_width // spatial_merge_size
            expected = temporal * height * width
            if temporal != 1 or positions.numel() != expected:
                raise ValueError(
                    f"Grounded OPD expected one-frame grid with {expected} tokens, got {positions.numel()}"
                )
            image_mass = attention[sample_idx, :, positions].reshape(response_length, height, width)
            if view == "teacher" and image_idx > 0:
                if bboxes is None:
                    raise ValueError("Teacher grounding requires normalized crop bboxes")
                x1, y1, x2, y2 = bboxes[sample_idx].clamp(0, 1).tolist()
                left = min(grid_size - 1, max(0, int(round(x1 * grid_size))))
                top = min(grid_size - 1, max(0, int(round(y1 * grid_size))))
                right = min(grid_size, max(left + 1, int(round(x2 * grid_size))))
                bottom = min(grid_size, max(top + 1, int(round(y2 * grid_size))))
                canvas[:, top:bottom, left:right] += _resize_mass(image_mass, bottom - top, right - left)
            else:
                canvas += _resize_mass(image_mass, grid_size, grid_size)
        result[sample_idx, :, 1:] = canvas.flatten(1)
    if grid_cursor != image_grid_thw.shape[0]:
        raise ValueError(
            f"image_grid_thw has {image_grid_thw.shape[0]} entries but consumed {grid_cursor} image runs"
        )
    return result / result.sum(dim=-1, keepdim=True).clamp_min(1e-12)


@dataclass
class GroundedAttentionCapture:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    response_positions: torch.Tensor
    image_grid_thw: torch.Tensor
    image_token_id: int
    spatial_merge_size: int
    total_layers: int
    grid_size: int = 24
    layer_start_fraction: float = 0.3
    layer_end_fraction: float = 0.6
    response_chunk_size: int = 16
    target_layer_idx: Optional[int] = None
    view: str = "student"
    bboxes: Optional[torch.Tensor] = None
    raw_qk: list[tuple[object, torch.Tensor, torch.Tensor, Optional[float], Optional[float]]] = field(
        default_factory=list
    )
    layer_indices: list[int] = field(default_factory=list)

    def should_capture(self, module, query: torch.Tensor, key: torch.Tensor) -> bool:
        layer_idx = getattr(module, "layer_idx", None)
        if layer_idx is None or query.ndim != 4 or key.ndim != 4:
            return False
        if query.shape[0] != self.input_ids.shape[0] or query.shape[2] != self.input_ids.shape[1]:
            return False
        if self.target_layer_idx is not None:
            return int(layer_idx) == self.target_layer_idx
        start = int(self.total_layers * self.layer_start_fraction)
        end = max(start + 1, int(self.total_layers * self.layer_end_fraction))
        return start <= int(layer_idx) < min(self.total_layers, end)

    def add(self, module, query: torch.Tensor, key: torch.Tensor, scaling: Optional[float], softcap: Optional[float]) -> None:
        if not self.should_capture(module, query, key):
            return
        # Keep capture itself free of tensor operations.  Qwen decoder layers are
        # activation-checkpointed, so any extra tensor graph created as a side
        # effect inside the layer would make the original forward differ from
        # checkpoint recomputation.  Hold the selected layer's native Q/K tensors
        # and build the grounding graph in finish(), after model.forward returns.
        self.raw_qk.append((module, query, key, scaling, softcap))
        self.layer_indices.append(int(module.layer_idx))

    def _build_map(
        self,
        module,
        query: torch.Tensor,
        key: torch.Tensor,
        scaling: Optional[float],
        softcap: Optional[float],
    ) -> torch.Tensor:
        if query.shape[1] != key.shape[1]:
            if query.shape[1] % key.shape[1] != 0:
                raise ValueError("Grounded OPD cannot align query and KV heads")
            key = key.repeat_interleave(query.shape[1] // key.shape[1], dim=1)
        scale = float(scaling if scaling is not None else getattr(module, "scaling", query.shape[-1] ** -0.5))
        key_positions = torch.arange(key.shape[2], device=query.device).view(1, 1, 1, -1)
        sample_maps = []
        for start in range(0, self.response_positions.shape[1], self.response_chunk_size):
            positions = self.response_positions[:, start : start + self.response_chunk_size]
            gather_index = positions[:, None, :, None].expand(-1, query.shape[1], -1, query.shape[-1])
            response_query = torch.gather(query, dim=2, index=gather_index)
            scores = torch.matmul(response_query.float(), key.float().transpose(-1, -2)) * scale
            if softcap is not None:
                scores = torch.tanh(scores / float(softcap)) * float(softcap)
            valid_keys = self.attention_mask[:, None, None, :].bool()
            causal = key_positions <= positions[:, None, :, None]
            scores = scores.masked_fill(~(valid_keys & causal), torch.finfo(scores.dtype).min)
            sample_maps.append(torch.softmax(scores, dim=-1).mean(dim=1))
        attention = torch.cat(sample_maps, dim=1)
        return _canonicalize_visual_attention(
            attention=attention,
            input_ids=self.input_ids,
            image_token_id=self.image_token_id,
            image_grid_thw=self.image_grid_thw,
            spatial_merge_size=self.spatial_merge_size,
            grid_size=self.grid_size,
            view=self.view,
            bboxes=self.bboxes,
        )

    def finish(self) -> tuple[torch.Tensor, list[int]]:
        if not self.raw_qk:
            raise RuntimeError(
                "Grounded OPD captured no full-attention layer; check the Qwen3.5 attention backend and layer range"
            )
        maps = [self._build_map(*capture) for capture in self.raw_qk]
        return torch.stack(maps, dim=0).mean(dim=0), self.layer_indices


def _grounded_wrapper(original_fn):
    def wrapped(module, query, key, value, attention_mask, *args, **kwargs):
        output = original_fn(module, query, key, value, attention_mask, *args, **kwargs)
        capture = _ACTIVE_CAPTURE.get()
        if capture is not None:
            capture.add(
                module,
                query,
                key,
                scaling=kwargs.get("scaling"),
                softcap=kwargs.get("softcap"),
            )
        return output

    return wrapped


def apply_grounded_attention_patch() -> list[str]:
    global _PATCHED
    if _PATCHED:
        return []
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    patched = []
    for name in ("flash_attention_2", "flash_attention_3", "sdpa", "flex_attention"):
        if name in ALL_ATTENTION_FUNCTIONS:
            ALL_ATTENTION_FUNCTIONS[name] = _grounded_wrapper(ALL_ATTENTION_FUNCTIONS[name])
            patched.append(name)
    _PATCHED = True
    return patched


@contextmanager
def capture_grounding(capture: GroundedAttentionCapture):
    token = _ACTIVE_CAPTURE.set(capture)
    try:
        yield capture
    finally:
        _ACTIVE_CAPTURE.reset(token)


def compute_grounding_loss(
    student_probs: torch.Tensor,
    teacher_probs: torch.Tensor,
    response_mask: torch.Tensor,
    bboxes: torch.Tensor,
    grid_size: int,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Forward KL over the joint null/full-image evidence grid."""

    eps = 1e-8
    teacher = teacher_probs.detach().float().clamp_min(eps)
    student = student_probs.float().clamp_min(eps)
    token_kl = (teacher * (teacher.log() - student.log())).sum(dim=-1)
    mask = response_mask.to(token_kl.dtype)
    count = mask.sum().clamp_min(1.0)
    loss = (token_kl * mask).sum() / count

    teacher_entropy = -(teacher * teacher.log()).sum(dim=-1)
    student_entropy = -(student * student.log()).sum(dim=-1)
    cosine = F.cosine_similarity(student, teacher, dim=-1)
    teacher_spatial = teacher[..., 1:].reshape(*teacher.shape[:2], grid_size, grid_size)
    student_spatial = student[..., 1:].reshape(*student.shape[:2], grid_size, grid_size)
    teacher_bbox_mass = token_kl.new_zeros(token_kl.shape)
    student_bbox_mass = token_kl.new_zeros(token_kl.shape)
    for sample_idx, bbox in enumerate(bboxes):
        x1, y1, x2, y2 = bbox.clamp(0, 1).tolist()
        left = min(grid_size - 1, max(0, int(round(x1 * grid_size))))
        top = min(grid_size - 1, max(0, int(round(y1 * grid_size))))
        right = min(grid_size, max(left + 1, int(round(x2 * grid_size))))
        bottom = min(grid_size, max(top + 1, int(round(y2 * grid_size))))
        teacher_bbox_mass[sample_idx] = teacher_spatial[sample_idx, :, top:bottom, left:right].sum((-2, -1))
        student_bbox_mass[sample_idx] = student_spatial[sample_idx, :, top:bottom, left:right].sum((-2, -1))

    def mean(value: torch.Tensor) -> float:
        return ((value * mask).sum() / count).detach().item()

    metrics = {
        "self_distillation/grounded_kl_token_mean": loss.detach().item(),
        "self_distillation/grounded_response_cosine": mean(cosine),
        "self_distillation/grounded_teacher_entropy": mean(teacher_entropy),
        "self_distillation/grounded_student_entropy": mean(student_entropy),
        "self_distillation/grounded_teacher_visual_mass": mean(1.0 - teacher[..., 0]),
        "self_distillation/grounded_student_visual_mass": mean(1.0 - student[..., 0]),
        "self_distillation/grounded_teacher_null_mass": mean(teacher[..., 0]),
        "self_distillation/grounded_student_null_mass": mean(student[..., 0]),
        "self_distillation/grounded_bbox_mass_teacher": mean(teacher_bbox_mass),
        "self_distillation/grounded_bbox_mass_student": mean(student_bbox_mass),
        "self_distillation/grounded_bbox_mass_gap": mean((teacher_bbox_mass - student_bbox_mass).abs()),
        "self_distillation/grounded_top1_region_agreement": mean(
            student[..., 1:].argmax(dim=-1).eq(teacher[..., 1:].argmax(dim=-1)).to(mask.dtype)
        ),
    }
    return loss, metrics
