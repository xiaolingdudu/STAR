"""Use the positive crop Teacher's per-token attention centroid as Gaussian center."""
from dataclasses import dataclass

import torch

from verl.trainer.ppo.evidence_caption import _layer_visual_probabilities, selected_response_offsets
from verl.trainer.ppo.grounded_opd import GroundedAttentionCapture


def visual_coordinates(height, width, device):
    yy, xx = torch.meshgrid(
        (torch.arange(height, device=device, dtype=torch.float32) + .5) / height,
        (torch.arange(width, device=device, dtype=torch.float32) + .5) / width,
        indexing="ij")
    return torch.stack((xx.flatten(), yy.flatten()), -1)


def map_centroids(probabilities, height, width, footprint):
    """Crop-token centers -> registered full-image coordinates, including crop padding."""
    local = probabilities @ visual_coordinates(height, width, probabilities.device)
    return footprint[:2] + local * (footprint[2:] - footprint[:2])


def centered_gaussians(height, width, bbox, centers, alpha=.2):
    coordinates = visual_coordinates(height, width, centers.device)
    sigma = (bbox[2:] - bbox[:2]) / 4
    logits = -.5 * ((coordinates[None] - centers[:, None]) / sigma).square().sum(-1)
    return (1 - alpha) * logits.softmax(-1) + alpha / (height * width)


@dataclass
class TeacherCenterCapture(GroundedAttentionCapture):
    response_mask: torch.Tensor = None
    crop_footprints: torch.Tensor = None
    max_tokens: int = 0
    expected_layer_indices: tuple = ()
    response_chunk_size: int = 128

    @torch.no_grad()
    def finish(self):
        if tuple(self.layer_indices) != self.expected_layer_indices:
            raise RuntimeError(f"Teacher center captured {self.layer_indices}; expected {self.expected_layer_indices}")
        device = self.input_ids.device
        centers = torch.zeros((*self.response_mask.shape, 2), device=device, dtype=torch.float32)
        selected = torch.zeros_like(self.response_mask, dtype=torch.bool)
        with torch.autocast(device_type=device.type, enabled=False):
            for b in range(self.input_ids.shape[0]):
                offsets = selected_response_offsets(self.response_mask[b], self.max_tokens)
                if offsets.numel() == 0:
                    continue
                positions = self.response_positions[b, offsets]
                visual = self.input_ids[b].eq(self.image_token_id).nonzero(as_tuple=False).flatten()
                temporal, raw_h, raw_w = self.image_grid_thw[b].tolist()
                height, width = int(raw_h) // self.spatial_merge_size, int(raw_w) // self.spatial_merge_size
                if temporal != 1 or visual.numel() != height * width:
                    raise ValueError("Teacher center requires one crop with its native visual grid")
                if not bool(visual.max() < positions.min()):
                    raise ValueError("Crop keys must precede Teacher response prediction rows")
                layers = [(q[b, :, positions], k[b, :, visual],
                           float(scale if scale is not None else q.shape[-1] ** -.5), softcap)
                          for _, q, k, scale, softcap in self.raw_qk]
                for start in range(0, offsets.numel(), self.response_chunk_size):
                    stop = start + self.response_chunk_size
                    probabilities = sum(_layer_visual_probabilities(q[:, start:stop], k, scale, softcap)
                                        for q, k, scale, softcap in layers) / len(layers)
                    centers[b, offsets[start:stop]] = map_centroids(
                        probabilities, height, width, self.crop_footprints[b].float())
                selected[b, offsets] = True
        return centers, selected


def replace_attention_distance(outputs, teacher_outputs, bboxes, response_mask, alpha=.2,
                               *, compute_auxiliary_loss=False):
    """Reuse one Teacher-center reference for detached gates and optional Student supervision."""
    records = outputs.pop("center_attention")
    centers = teacher_outputs.pop("teacher_attention_centers").detach()
    teacher_selected = teacher_outputs.pop("teacher_center_selected")
    distance = torch.zeros_like(response_mask, dtype=torch.float32)
    total = distance.new_zeros(())
    fixed_total, shift_total, fixed_gate_total, gate_total = (total.clone() for _ in range(4))
    count = 0
    sequence_losses = []
    for b, (offsets, attention, height, width) in enumerate(records):
        if offsets.numel() == 0:
            if compute_auxiliary_loss:
                sequence_losses.append(distance.new_zeros(()))
            continue
        if not bool(teacher_selected[b, offsets].all()):
            raise ValueError("Teacher and Student center-gate response positions do not align")
        box = bboxes[b].detach().float()
        token_losses = []
        for start in range(0, offsets.numel(), 128):
            ids = offsets[start:start + 128]
            probs = attention[start:start + 128].float()
            if not compute_auxiliary_loss:
                probs = probs.detach()
            mu = centers[b, ids].float()
            target = centered_gaussians(height, width, box, mu, alpha)
            values = (probs * (probs.clamp_min(1e-30).log() - target.log())).sum(-1).clamp_min(0)
            distance[b, ids] = values.detach()
            if compute_auxiliary_loss:
                token_losses.append(values)
            fixed = outputs["attention_token_kl"][b, ids].clamp_min(0)
            total += values.detach().sum()
            fixed_total += fixed.sum()
            shift_total += (((mu - (box[:2] + box[2:]) / 2) / (box[2:] - box[:2]))
                            .square().sum(-1).sqrt()).sum()
            gate_total += (values.detach() / (1 + values.detach())).sum()
            fixed_gate_total += (fixed / (1 + fixed)).sum()
        count += offsets.numel()
        if compute_auxiliary_loss:
            sequence_losses.append(torch.cat(token_losses).mean())
    outputs["attention_token_kl"] = distance
    denom = max(count, 1)
    outputs["evidence_attention_metrics"]["evidence/attention_kl"] = float(total / denom)
    metrics = {
        "teacher_center/attention_kl": float(total / denom),
        "teacher_center/fixed_center_kl_same_alpha": float(fixed_total / denom),
        "teacher_center/shift_box_units": float(shift_total / denom),
        "teacher_center/gate_mean": float(gate_total / denom),
        "teacher_center/fixed_center_gate_same_alpha": float(fixed_gate_total / denom),
        "teacher_center/background_alpha": alpha,
        "teacher_center/extra_model_forwards": 0.0,
    }
    if compute_auxiliary_loss:
        # Keep the existing auxiliary's mean per trajectory, then microbatch mean.
        # G is detached above; only the Student visual probabilities carry gradients.
        loss = torch.stack(sequence_losses).mean()
        outputs["evidence_attention_loss"] = loss
        metrics["teacher_center/auxiliary_loss"] = loss.detach().item()
        metrics["teacher_center/auxiliary_student_grad"] = float(loss.requires_grad)
    return metrics
