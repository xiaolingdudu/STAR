"""Build detached, token-specific spatial targets from four crop occlusions."""
import math

import torch


QUADRANTS = ((0, 0, .5, .5), (.5, 0, 1, .5), (0, .5, .5, 1), (.5, .5, 1, 1))


def quadrant_views(image):
    from PIL import ImageStat
    image = image.convert("RGB")
    color = tuple(round(value) for value in ImageStat.Stat(image).mean)
    views = []
    for box in QUADRANTS:
        view = image.copy()
        x1, y1, x2, y2 = box
        view.paste(color, (round(x1 * image.width), round(y1 * image.height),
                           round(x2 * image.width), round(y2 * image.height)))
        views.append(view)
    return views


def quadrant_basis(height, width, footprint):
    """Map crop quadrants to the original image by exact patch overlap."""
    footprint = footprint.float()
    x1, y1, x2, y2 = footprint.unbind()
    xx = torch.arange(width, device=footprint.device, dtype=torch.float32) / width
    yy = torch.arange(height, device=footprint.device, dtype=torch.float32) / height
    maps = []
    for a, b, c, d in QUADRANTS:
        left, right = x1 + a * (x2 - x1), x1 + c * (x2 - x1)
        top, bottom = y1 + b * (y2 - y1), y1 + d * (y2 - y1)
        dx = (torch.minimum(xx + 1 / width, right) - torch.maximum(xx, left)).clamp_min(0)
        dy = (torch.minimum(yy + 1 / height, bottom) - torch.maximum(yy, top)).clamp_min(0)
        mass = (dy[:, None] * dx[None, :]).flatten()
        maps.append(mass / mass.sum())
    return torch.stack(maps)


def with_tail(logps):
    logps = logps.float()
    mass = logps.logsumexp(-1, keepdim=True).clamp(max=-1e-7)
    return torch.cat((logps, (-torch.expm1(mass)).log()), -1)


def sensitivity(positive, negative):
    mid = torch.logaddexp(positive, negative) - math.log(2)
    return (.5 * ((positive.exp() * (positive - mid)).sum(-1)
                  + (negative.exp() * (negative - mid)).sum(-1))).clamp_min(0)


def spatial_target(scores, basis, alpha=.1):
    total = scores.sum(-1, keepdim=True)
    informative = total.squeeze(-1) > 1e-6
    weights = torch.where(informative[:, None], scores / total.clamp_min(1e-30),
                          torch.full_like(scores, .25))
    target = (1 - alpha) * (weights @ basis) + alpha / basis.shape[-1]
    return target, weights, informative


@torch.no_grad()
def replace_attention_distance(outputs, positive_topk, quadrant_topks, response_mask, alpha=.1):
    """Replace Gaussian distances; leave reconstruction and selected rows intact.

    Attention records stay local to the actor microbatch. Teacher distributions
    use the Student's exact top-k indices, plus a shared aggregate tail bin.
    """
    records = outputs.pop("local_evidence_attention")
    distance = torch.zeros_like(response_mask, dtype=torch.float32)
    counts, informative_counts, score_sums, entropies, distances, gaussian = [], [], [], [], [], []
    assert len(quadrant_topks) == 4
    for b, record in enumerate(records):
        offsets, attention, basis = record
        if offsets.numel() == 0:
            continue
        positive = with_tail(positive_topk[b, offsets].detach())
        scores = torch.stack([sensitivity(positive, with_tail(view[b, offsets].detach()))
                              for view in quadrant_topks], -1)
        target, weights, informative = spatial_target(scores, basis, alpha)
        values = (attention * (attention.clamp_min(1e-30).log() - target.log())).sum(-1).clamp_min(0)
        distance[b, offsets] = values
        counts.append(offsets.numel())
        informative_counts.append(informative.sum())
        score_sums.append(scores.sum())
        entropies.append(-(weights * weights.clamp_min(1e-30).log()).sum())
        distances.append(values.sum())
        gaussian.append(outputs["attention_token_kl"][b, offsets].sum())
    count = max(sum(counts), 1)
    outputs["attention_token_kl"] = distance
    metrics = {
        "local_evidence/informative_fraction": float(sum(informative_counts) / count),
        "local_evidence/mean_occlusion_jsd": float(sum(score_sums) / (4 * count)),
        "local_evidence/region_weight_entropy": float(sum(entropies) / count),
        "local_evidence/attention_kl": float(sum(distances) / count),
        "local_evidence/gaussian_attention_kl": float(sum(gaussian) / count),
        "local_evidence/teacher_probe_views": 4.0,
    }
    return metrics
