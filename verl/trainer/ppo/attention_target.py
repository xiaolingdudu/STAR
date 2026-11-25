"""Detached spatial mismatch gates a crop/erased-crop probability contrast.

The vocabulary support is shared Student top-k plus one aggregate tail bin.
This is a target on that compressed distribution, not a full-vocabulary target.
"""
import torch


@torch.no_grad()
def external_attention_weights(token_kl, selected, loss_mask):
    """Mean-one weights under the existing microbatch token-mean reduction.

    The outer sequence-count and FSDP reductions are unchanged. This is an
    external weighting control: no gradient enters the attention gate.
    """
    active = selected.bool() & loss_mask.bool()
    distance = token_kl.detach().float().clamp_min(0)
    gate = distance / (1 + distance) * active
    raw = 1 + gate
    count = loss_mask.sum().clamp_min(1)
    normalizer = (raw * loss_mask).sum().clamp_min(1) / count
    weights = raw / normalizer
    return weights, {
        "attention_external/raw_weight_mean": normalizer.item(),
        "attention_external/weight_mean": ((weights * loss_mask).sum() / count).item(),
        "attention_external/gate_mean": ((gate * loss_mask).sum() / count).item(),
    }


def mean_erased_crop(image):
    from PIL import Image, ImageStat
    image = image.convert("RGB")
    color = tuple(round(value) for value in ImageStat.Stat(image).mean)
    return Image.new("RGB", image.size, color)


@torch.no_grad()
def reconstruct_target(positive_logps, negative_logps, token_kl, selected, response_mask, beta,
                       gate_mode="adaptive"):
    positive, negative = positive_logps.detach().float(), negative_logps.detach().float()
    active = selected.bool() & response_mask.bool()
    distance = token_kl.detach().float().clamp_min(0)
    raw_gate = distance / (1 + distance) * active
    if gate_mode == "adaptive":
        gate = raw_gate
    elif gate_mode == "zero":
        gate = torch.zeros_like(raw_gate)
    elif gate_mode == "one":
        # Fixed-strength control: reconstruct every valid response token.
        gate = response_mask.bool().to(raw_gate.dtype)
    elif gate_mode == "shuffle_response":
        # Preserve each response's gate multiset, including any zero gates.
        # Never exchange gates across responses/worlds or into padding.
        gate = raw_gate.clone()
        for row in range(gate.shape[0]):
            positions = response_mask[row].bool().nonzero(as_tuple=True)[0]
            order = torch.randperm(positions.numel(), device=gate.device)
            gate[row, positions] = raw_gate[row, positions[order]]
    else:
        raise ValueError(f"Unknown attention target gate mode: {gate_mode}")
    target = (positive + float(beta) * gate.unsqueeze(-1) * (positive - negative)).log_softmax(-1)
    # Preserve the crop target exactly wherever the applied gate is zero.
    target = torch.where((gate > 0).unsqueeze(-1), target, positive)
    count = response_mask.sum().clamp_min(1)
    change = (target.exp() * (target - positive)).sum(-1)
    return target, {
        "attention_target/gate_mean": (gate.sum() / count).item(),
        "attention_target/raw_gate_mean": (raw_gate.sum() / count).item(),
        "attention_target/applied_lambda_mean": (float(beta) * gate.sum() / count).item(),
        "attention_target/gate_reassignment_abs": (((gate - raw_gate).abs() * response_mask).sum() / count).item(),
        "attention_target/selected_fraction": (active.sum() / count).item(),
        "attention_target/kl_from_crop": ((change * response_mask).sum() / count).item(),
        "attention_target/entropy": ((-(target.exp() * target).sum(-1) * response_mask).sum() / count).item(),
        "attention_target/tail_mass": ((target[..., -1].exp() * response_mask).sum() / count).item(),
        "attention_target/beta": float(beta),
    }
