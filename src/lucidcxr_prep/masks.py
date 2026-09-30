"""Mask unioning: Montgomery left/right lung masks, Shenzhen per-abnormality lesion masks.

Both sources need the same operation -- logical OR of one or more binary mask files into a
single mask -- so it's implemented once here and specialized per source below.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from .io_utils import load_mask


def _binarize(im: Image.Image) -> np.ndarray:
    return np.array(im.convert("L")) > 0


def union_masks(paths: list[Path], size: tuple[int, int]) -> Image.Image:
    """Logical OR of one or more binary mask PNGs, all expected to be `size` (width, height).

    Returns an all-zero mask (never None, never skipped) if `paths` is empty -- this is the
    "empty reference mask" case the proposal requires be preserved and counted, not imputed.
    """
    w, h = size
    acc = np.zeros((h, w), dtype=bool)
    for p in paths:
        im = load_mask(p)
        if im.size != size:
            raise ValueError(f"Mask size mismatch: {p} is {im.size}, expected {size}")
        acc |= _binarize(im)
    return Image.fromarray((acc.astype(np.uint8) * 255), mode="L")


def montgomery_lung_masks(left_mask_path: Path, right_mask_path: Path) -> dict[str, Image.Image]:
    """Return the three lung masks Montgomery contributes, patient-side named.

    Verified by visual QC (IMPLEMENTATION_PLAN.md Step 2): `leftMask` sits under the image's "R"
    marker, i.e. it is the patient's RIGHT lung. `rightMask` is the patient's LEFT lung.
    """
    left_im = load_mask(left_mask_path)  # patient's right lung
    right_im = load_mask(right_mask_path)  # patient's left lung
    size = left_im.size
    if right_im.size != size:
        raise ValueError(
            f"Montgomery left/right mask size mismatch: {left_mask_path} is {size}, "
            f"{right_mask_path} is {right_im.size}"
        )

    patient_right = Image.fromarray((_binarize(left_im).astype(np.uint8) * 255), mode="L")
    patient_left = Image.fromarray((_binarize(right_im).astype(np.uint8) * 255), mode="L")
    field = union_masks([left_mask_path, right_mask_path], size)

    return {
        "lung_patient_right": patient_right,
        "lung_patient_left": patient_left,
        "lung_field": field,
    }


def shenzhen_lesion_mask(image_id: str, masks_dir: Path, size: tuple[int, int]) -> Image.Image:
    """Union every per-abnormality mask file for `image_id` in `masks_dir`.

    `image_id` is e.g. 'CHNCXR_0327_1'. Returns an all-zero mask if the image has none (either a
    normal image, or one of the ~6 TB-positive images with no visible lesion per Yang et al., 2022)
    -- caller is responsible for deciding whether to write/flag that as an empty reference mask.
    """
    paths = sorted(masks_dir.glob(f"{image_id}_*.png"))
    return union_masks(paths, size)
