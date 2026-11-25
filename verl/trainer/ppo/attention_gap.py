"""Online Teacher-minus-Student attention crops for an auxiliary Student view."""
from __future__ import annotations

from dataclasses import dataclass, field
import math

import numpy as np
from PIL import Image
import torch


@dataclass
class PromptAttentionCapture:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    query_positions: torch.Tensor
    image_grid_thw: torch.Tensor
    image_token_id: int
    merge_size: int
    layers: tuple[int, ...]
    maps: dict = field(default_factory=dict)

    @torch.no_grad()
    def add(self, module, query, key, scaling=None, softcap=None):
        layer = getattr(module, "layer_idx", None)
        if layer not in self.layers or query.shape[0] != self.input_ids.shape[0] or query.shape[2] != self.input_ids.shape[1]:
            return
        # One pre-answer query per row; never materialize sequence-squared attention.
        index = self.query_positions[:, None, None, None].expand(-1, query.shape[1], 1, query.shape[-1])
        q = query.detach().gather(2, index).float()
        groups = query.shape[1] // key.shape[1]
        q = q.reshape(query.shape[0], key.shape[1], groups, query.shape[-1])
        scores = torch.einsum("bkgd,bknd->bkgn", q, key.detach().float()).flatten(1, 2)
        scores *= scaling if scaling is not None else module.scaling
        if softcap is not None:
            scores = scores.div(float(softcap)).tanh().mul(float(softcap))
        valid = self.attention_mask.bool() & (torch.arange(key.shape[2], device=key.device)[None] <= self.query_positions[:, None])
        attention = scores.masked_fill(~valid[:, None], -torch.inf).softmax(-1).mean(1)
        self.maps[layer] = [attention[i, self.input_ids[i].eq(self.image_token_id)].detach() for i in range(len(attention))]

    def finish(self):
        if set(self.maps) != set(self.layers):
            raise RuntimeError(f"Attention-gap captured layers {sorted(self.maps)}, expected {self.layers}")
        result = []
        for i, grid in enumerate(self.image_grid_thw):
            t, h, w = map(int, grid.tolist())
            heat = torch.stack([self.maps[layer][i] for layer in self.layers]).sum(0)
            if t != 1 or heat.numel() != h * w // self.merge_size ** 2:
                raise ValueError("Attention-gap expects exactly one image per original/Teacher prompt")
            heat = heat / heat.sum().clamp_min(1e-12)
            result.append(heat.reshape(h // self.merge_size, w // self.merge_size))
        return result


def axis_map(source_cells, target_cells, lo, hi, device):
    source = torch.linspace(float(lo), float(hi), source_cells + 1, device=device)
    target = torch.linspace(0, 1, target_cells + 1, device=device)
    overlap = (torch.minimum(source[1:, None], target[None, 1:]) - torch.maximum(source[:-1, None], target[None, :-1])).clamp_min(0)
    return overlap / overlap.sum(-1, keepdim=True).clamp_min(1e-12)


@torch.no_grad()
def attention_difference(student, teacher, bbox):
    """Conserve crop attention mass when projecting into full-image coordinates."""
    x1, y1, x2, y2 = bbox
    y = axis_map(teacher.shape[0], student.shape[0], y1, y2, student.device)
    x = axis_map(teacher.shape[1], student.shape[1], x1, x2, student.device)
    mapped = y.T @ teacher.float() @ x
    mapped = mapped / mapped.sum().clamp_min(1e-12)
    full = student.float() / student.sum().clamp_min(1e-12)
    return (mapped - full).clamp_min(0)


def select_gap_box(gap, image_size, scale=.25):
    width, height = image_size
    side = max(1, round(min(width, height) * scale))
    gh, gw = gap.shape
    wh, ww = max(1, round(side / height * gh)), max(1, round(side / width * gw))
    integral = np.pad(np.asarray(gap, dtype=np.float64), ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    scores = integral[wh:, ww:] - integral[:-wh, ww:] - integral[wh:, :-ww] + integral[:-wh, :-ww]
    iy, ix = np.unravel_index(scores.argmax(), scores.shape)
    x = min(max(0, round((ix + ww / 2) / gw * width - side / 2)), width - side)
    y = min(max(0, round((iy + wh / 2) / gh * height - side / 2)), height - side)
    return [x, y, x + side, y + side], float(scores[iy, ix])


def make_augmented_prefix(processor, prompt_ids, full_mm, metadata, gap, *, crop_size=1024, crop_scale=.25, max_prompt=10240):
    """Reuse original encoded pixels and prompt, inserting one supplemental image."""
    with Image.open(metadata["original_path"]) as image:
        image = image.convert("RGB")
        box, mass = select_gap_box(gap.cpu().numpy(), image.size, crop_scale)
        crop = image.crop(box).resize((crop_size, crop_size), Image.Resampling.BICUBIC)
    crop_inputs = processor.image_processor(images=[crop], do_resize=False, return_tensors="pt")
    full_grid = full_mm["image_grid_thw"].cpu().reshape(-1, 3)
    crop_grid = crop_inputs["image_grid_thw"]
    merge = processor.image_processor.merge_size
    crop_tokens = int(crop_grid.prod(-1).sum()) // merge ** 2
    token_id = int(processor.image_token_id)
    end_id = processor.tokenizer.convert_tokens_to_ids("<|vision_end|>")
    start_id = processor.tokenizer.convert_tokens_to_ids("<|vision_start|>")
    prompt_ids = prompt_ids.cpu()
    insertion = int((prompt_ids == end_id).nonzero()[0]) + 1
    note = processor.tokenizer.encode("\nImage 1 is the original image. Image 2 is a magnified detail from the same image.\n", add_special_tokens=False)
    added = torch.tensor([start_id] + [token_id] * crop_tokens + [end_id] + note, dtype=prompt_ids.dtype)
    ids = torch.cat([prompt_ids[:insertion], added, prompt_ids[insertion:]])
    if len(ids) > max_prompt:
        raise ValueError(f"Supplemental prompt {len(ids)} exceeds {max_prompt}; no visual tokens are truncated")
    mm = {"pixel_values": torch.cat([full_mm["pixel_values"].detach().cpu(), crop_inputs["pixel_values"]]),
          "image_grid_thw": torch.cat([full_grid, crop_grid])}
    expected = int(mm["image_grid_thw"].prod(-1).sum()) // merge ** 2
    if int(ids.eq(token_id).sum()) != expected:
        raise ValueError("Supplemental image tokens do not match reused full pixels and crop pixels")
    mask = torch.ones_like(ids)
    types = ids.eq(token_id).long().unsqueeze(0)
    positions = processor.get_rope_index(input_ids=ids[None], mm_token_type_ids=types,
        image_grid_thw=mm["image_grid_thw"], attention_mask=mask[None])
    if isinstance(positions, tuple):
        positions = positions[0]
    positions = positions.squeeze(1)
    if positions.shape[0] == 3:
        positions = torch.cat([torch.arange(len(ids))[None], positions])
    return ids, positions, mm, box, mass


def build_auxiliary_batch(processor, batch, student_maps, teacher_maps, cache, *, crop_size=1024, crop_scale=.25):
    device = batch["input_ids"].device
    ids_list, pos_list, masks, mms, starts, active, statistics = [], [], [], [], [], [], []
    response_width = batch["responses"].shape[1]
    full_start = batch.get("response_start_idx")
    if full_start is None:
        full_start = batch["input_ids"].new_full((len(student_maps),), batch["input_ids"].shape[1] - response_width)
    for i, (smap, tmap, meta) in enumerate(zip(student_maps, teacher_maps, batch["gap_metadata"])):
        gap = attention_difference(smap, tmap, meta["bbox"])
        before = batch["input_ids"][i, :int(full_start[i])]
        before = before[batch["attention_mask"][i, :int(full_start[i])].bool()]
        # Duplicate rollouts of one question have the same pre-answer attention.
        key = (meta["original_path"], tuple(before.cpu().tolist()))
        if key not in cache:
            cache[key] = make_augmented_prefix(processor, before, batch["multi_modal_inputs"][i], meta, gap,
                crop_size=crop_size, crop_scale=crop_scale)
        prefix, positions, mm, box, selected_mass = cache[key]
        responses = batch["responses"][i].cpu()
        ids_list.append(torch.cat([prefix, responses]))
        suffix_pos = positions[:, -1:] + 1 + torch.arange(response_width)[None]
        pos_list.append(torch.cat([positions, suffix_pos], -1))
        masks.append(torch.cat([torch.ones_like(prefix), batch["response_mask"][i].cpu()]))
        mms.append(mm); starts.append(len(prefix)); active.append(float(gap.sum() > 1e-8))
        statistics.append([float(gap.sum()), selected_mass, len(prefix), (box[2]-box[0])])
    width = max(map(len, ids_list)); bsz = len(ids_list)
    ids = torch.full((bsz, width), int(processor.tokenizer.pad_token_id), dtype=torch.long)
    attention = torch.zeros_like(ids); pos = torch.zeros(bsz, pos_list[0].shape[0], width, dtype=torch.long)
    for i, (seq, mask, p) in enumerate(zip(ids_list, masks, pos_list)):
        ids[i, :len(seq)] = seq; attention[i, :len(seq)] = mask; pos[i, :, :len(seq)] = p
    output = {"input_ids": ids.to(device), "attention_mask": attention.to(device), "position_ids": pos.to(device),
        "response_start_idx": torch.tensor(starts, device=device), "responses": batch["responses"],
        "response_mask": batch["response_mask"], "multi_modal_inputs": [{k: v.to(device) for k,v in mm.items()} for mm in mms],
        "pad_token_id": int(processor.tokenizer.pad_token_id)}
    values = np.asarray(statistics)
    metrics = {"gap/positive_mass": float(values[:,0].mean()), "gap/selected_mass": float(values[:,1].mean()),
        "gap/aux_prompt_tokens": float(values[:,2].mean()), "gap/source_crop_side": float(values[:,3].mean()),
        "gap/active_fraction": float(np.mean(active))}
    return output, torch.tensor(active, device=device), metrics


def auxiliary_jsd(student_logps, teacher_logps):
    """Same original-Student top-k support plus a tail bucket for both views."""
    def tail(logp):
        logp = logp.float()
        mass = torch.logsumexp(logp, -1, keepdim=True).clamp(max=-1e-7)
        return torch.cat([logp, torch.log(-torch.expm1(mass))], -1)
    lp, lq = tail(student_logps), tail(teacher_logps.detach())
    lm = torch.logaddexp(lp, lq) - math.log(2)
    return .5 * ((lp.exp() * (lp-lm)).sum(-1) + (lq.exp() * (lq-lm)).sum(-1))
