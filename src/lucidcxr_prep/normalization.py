"""Step 6: intensity normalization constants, computed from training-split images only.

Stats are over pixel values scaled to [0, 1], computed on the stored 512 PNGs *including* the
zero padding, because that is exactly what the dataloader will feed the network. The 256 stats
are computed on the 512 PNG downsampled in memory with Pillow bicubic, mirroring how the
dataloader produces 256 (CLAUDE.md decision #7) -- if the dataloader uses a different resize,
recompute.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


def _accumulate(paths: list[Path], size: int | None) -> dict:
    total = 0.0
    total_sq = 0.0
    n = 0
    for p in paths:
        with Image.open(p) as im:
            im = im.convert("L")
            if size is not None and im.size != (size, size):
                im = im.resize((size, size), Image.Resampling.BICUBIC)
            arr = np.asarray(im, dtype=np.float64) / 255.0
        total += arr.sum()
        total_sq += np.square(arr).sum()
        n += arr.size
    mean = total / n
    std = float(np.sqrt(total_sq / n - mean**2))
    return {"mean": float(mean), "std": std, "n_images": len(paths), "n_pixels": int(n)}


def compute_stats(image_paths: list[Path]) -> dict:
    if not image_paths:
        raise ValueError("No training images given -- refusing to compute stats from nothing")
    return {
        "512": _accumulate(image_paths, size=None),
        "256_from_512_bicubic": _accumulate(image_paths, size=256),
    }
