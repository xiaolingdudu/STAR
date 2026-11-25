"""Visual-condition utilities for the OPD-V reproduction."""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw


def random_mask_image(image: Image.Image) -> Image.Image:
    """Apply the deterministic one-third-area mask from the public OPD-V code."""
    output = image.convert("RGB").copy()
    rng = np.random.default_rng(0)
    width, height = output.size
    target_area = max(1, int(width * height / 3.0))
    aspect = float(rng.uniform(0.5, 2.0))
    patch_width = min(width, max(1, int((target_area * aspect) ** 0.5)))
    patch_height = min(height, max(1, int(target_area / patch_width)))
    x0 = int(rng.integers(0, max(1, width - patch_width + 1)))
    y0 = int(rng.integers(0, max(1, height - patch_height + 1)))
    ImageDraw.Draw(output).rectangle(
        (x0, y0, x0 + patch_width, y0 + patch_height),
        fill=(0, 0, 0),
    )
    return output
