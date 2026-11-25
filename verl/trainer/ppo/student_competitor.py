"""Select a real competing crop from detached Student response attention."""
import numpy as np
from PIL import Image
import torch


def select_competitor_box(attention, image_size, crop_size, evidence_bbox):
    """Maximize attention mass over equal-size windows outside the evidence.

    Search at native visual-grid resolution, including image/evidence edges.
    Patch mass is integrated by fractional area, with no candidate-size tuning.
    """
    width, height = image_size
    cw, ch = map(int, crop_size)
    gh, gw = attention.shape
    x1, y1, x2, y2 = np.asarray(evidence_bbox) * [width, height, width, height]
    if cw > width or ch > height:
        return None, 0.0, 0

    def positions(length, span, cells, lo, hi):
        starts = np.r_[np.rint(np.linspace(0, length - span, cells + 1)),
                       np.floor(lo) - span, np.ceil(hi)].astype(int)
        return np.unique(starts[(starts >= 0) & (starts <= length - span)])

    xs = positions(width, cw, gw, x1, x2)
    ys = positions(height, ch, gh, y1, y2)
    valid = ((xs[None] + cw <= x1) | (xs[None] >= x2)
             | (ys[:, None] + ch <= y1) | (ys[:, None] >= y2))
    count = int(valid.sum())
    if count == 0:
        return None, 0.0, 0

    def coverage(starts, span, length, cells):
        edges = np.linspace(0, length, cells + 1)
        overlap = (np.minimum(starts[:, None] + span, edges[None, 1:])
                   - np.maximum(starts[:, None], edges[None, :-1])).clip(min=0)
        return overlap / (length / cells)

    x_weights = coverage(xs, cw, width, gw)
    y_weights = coverage(ys, ch, height, gh)
    scores = y_weights @ attention @ x_weights.T
    scores[~valid] = -np.inf
    iy, ix = np.unravel_index(scores.argmax(), scores.shape)
    x, y = int(xs[ix]), int(ys[iy])
    return [x, y, x + cw, y + ch], float(scores[iy, ix]), count


@torch.no_grad()
def build_competitor_inputs(processor, positive_inputs, metadata, attention_maps,
                            evidence_boxes, image_cache, *, device):
    """One crop per trajectory; retain the positive Teacher's exact token grid."""
    negative_inputs, available, masses, counts = [], [], [], []
    boxes = evidence_boxes.detach().cpu().numpy()
    for positive, meta, heat, evidence in zip(positive_inputs, metadata, attention_maps, boxes):
        if heat is None:
            negative_inputs.append(positive)
            available.append(False)
            counts.append(0)
            continue
        path = meta["original_path"]
        if path not in image_cache:
            with Image.open(path) as original:
                image_cache[path] = original.convert("RGB")
        original = image_cache[path]
        box, mass, count = select_competitor_box(
            heat.detach().float().cpu().numpy(), original.size, meta["source_crop_size"], evidence)
        counts.append(count)
        available.append(box is not None)
        if box is None:
            # The Gaussian correction is disabled for this trajectory, giving
            # ordinary positive-crop distillation without a fabricated negative.
            negative_inputs.append(positive)
            continue
        # Dataset Teacher images are enlarged source crops. Match that exact
        # output size, so negative text tokens and mRoPE positions stay valid.
        crop = original.crop(box).resize(tuple(meta["teacher_crop_size"]), Image.Resampling.BICUBIC)
        encoded = processor.image_processor(images=[crop], return_tensors="pt")
        if not torch.equal(encoded["image_grid_thw"].cpu(), positive["image_grid_thw"].cpu()):
            raise ValueError("Competing crop must preserve the positive Teacher image grid")
        negative = dict(positive)
        negative.update({key: value.to(positive[key].device) for key, value in encoded.items()})
        negative_inputs.append(negative)
        masses.append(mass)
    return negative_inputs, torch.tensor(available, dtype=torch.bool, device=device), {
        "competitor/selected_fraction": float(np.mean(available)),
        "competitor/attention_mass": float(np.mean(masses)) if masses else 0.0,
        "competitor/candidates_mean": float(np.mean(counts)),
    }
