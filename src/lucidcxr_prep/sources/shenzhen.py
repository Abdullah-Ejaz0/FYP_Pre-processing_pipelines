"""Manifest builder for the Shenzhen Hospital CXR Set.

Verified on-disk layout (CLAUDE.md Section 2.2):
- CXR_png/CHNCXR_{patient_id}_{label}.png -- 662 files, label 0=normal, 1=TB.
    Image mode is NOT uniform: 635 files are palette-mode ('P'), 27 are 'RGB' (verified R==G==B
    on every sampled RGB file -- genuinely grayscale content, just stored as RGB). Both need
    `.convert('L')`.
- ClinicalReadings/CHNCXR_{patient_id}_{label}.txt
- Annotations/masks/CHNCXR_{patient_id}_{label}_{AbnormalityType}_{n}.png
    -- one file per abnormality instance, covering 330 of the 336 TB images (primary ground truth)
- Annotations-2/masks/...    -- second, independent annotation set, covering 323 images
  (not identical image coverage to Annotations/ -- verified via diff)
- mask/mask/CHNCXR_{patient_id}_{label}_mask.png -- 566 files (279 normal + 287 TB, verified),
  a single combined lung-FIELD mask per image (both lungs together, binary 0/255, 'L' mode,
  pixel-size-matched to its CXR). This is the Rajaraman et al. (2023) lung-boundary ground truth
  the proposal cites and that was originally missing from this download -- added later, once
  located. Unlike the lesion masks, there's nothing to union here: one file per image already.

This module only builds the flat per-image manifest (counts, booleans). Unioning the
per-abnormality masks into a single binary lesion mask is a separate step (masks.py) -- this
manifest just records how many mask files exist per image per annotation set, and whether a lung
mask exists.
"""

from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from ..io_utils import read_raw_image_info

FILENAME_RE = re.compile(r"^CHNCXR_(\d+)_([01])\.png$")
MASK_ID_RE = re.compile(r"^(CHNCXR_\d+_\d)_")
LABEL_MAP = {"0": "normal", "1": "TB"}


def _mask_counts_by_image(masks_dir: Path) -> dict[str, int]:
    """Map 'CHNCXR_0327_1' -> number of per-abnormality mask files for that image."""
    counts: dict[str, int] = defaultdict(int)
    if not masks_dir.exists():
        return counts
    for mask_path in masks_dir.glob("*.png"):
        m = MASK_ID_RE.match(mask_path.name)
        if not m:
            raise ValueError(f"Unexpected Shenzhen mask filename format: {mask_path.name}")
        counts[m.group(1)] += 1
    return dict(counts)


def _rgb_channels_equal(path: Path) -> bool:
    """For RGB-mode files, confirm R==G==B exactly so .convert('L') is lossless, not assumed."""
    with Image.open(path) as im:
        arr = np.array(im)
    return bool(np.array_equal(arr[..., 0], arr[..., 1]) and np.array_equal(arr[..., 1], arr[..., 2]))


def build_manifest(raw_root: Path) -> pd.DataFrame:
    """raw_root is the Shenzhen-Hospital-CXR-Set/ directory (see configs/paths.yaml)."""
    cxr_dir = raw_root / "CXR_png"
    readings_dir = raw_root / "ClinicalReadings"

    masks_primary = _mask_counts_by_image(raw_root / "Annotations" / "masks")
    masks_secondary = _mask_counts_by_image(raw_root / "Annotations-2" / "masks")
    lung_mask_dir = raw_root / "mask" / "mask"

    rows = []
    for img_path in sorted(cxr_dir.glob("*.png")):
        m = FILENAME_RE.match(img_path.name)
        if not m:
            raise ValueError(f"Unexpected Shenzhen filename format: {img_path.name}")
        patient_id, label_code = m.group(1), m.group(2)
        image_id = f"CHNCXR_{patient_id}_{label_code}"

        info = read_raw_image_info(img_path)

        reading_path = readings_dir / f"{img_path.stem}.txt"
        clinical_text = None
        if reading_path.exists():
            clinical_text = reading_path.read_text(encoding="utf-8", errors="replace").strip()

        n_masks_primary = masks_primary.get(image_id, 0)
        n_masks_secondary = masks_secondary.get(image_id, 0)

        rgb_channels_equal = _rgb_channels_equal(img_path) if info["mode"] == "RGB" else None

        lung_mask_path = lung_mask_dir / f"{image_id}_mask.png"
        has_lung_mask = lung_mask_path.exists()

        rows.append(
            {
                "source": "shenzhen",
                "patient_id": patient_id,
                "filename": img_path.name,
                "label": LABEL_MAP[label_code],
                "image_path": str(img_path),
                "width": info["width"],
                "height": info["height"],
                "mode": info["mode"],
                "rgb_channels_equal": rgb_channels_equal,
                "has_clinical_reading": reading_path.exists(),
                "clinical_text": clinical_text,
                "n_lesion_masks_primary": n_masks_primary,
                "has_lesion_mask_primary": n_masks_primary > 0,
                "n_lesion_masks_secondary": n_masks_secondary,
                "has_lesion_mask_secondary": n_masks_secondary > 0,
                "has_lung_mask": has_lung_mask,
                "lung_mask_path": str(lung_mask_path) if has_lung_mask else None,
            }
        )

    df = pd.DataFrame(rows)

    bad_mode = df[~df["mode"].isin(["P", "RGB"])]
    if not bad_mode.empty:
        raise AssertionError(
            f"Shenzhen images with an unexpected mode (expected 'P' or 'RGB'): "
            f"{bad_mode[['filename', 'mode']].to_dict('records')}"
        )

    rgb_rows = df[df["mode"] == "RGB"]
    not_equal = rgb_rows[~rgb_rows["rgb_channels_equal"].astype(bool)]
    if not not_equal.empty:
        raise AssertionError(
            "Shenzhen RGB images where R/G/B channels are NOT equal -- .convert('L') would "
            f"lose information for: {not_equal['filename'].tolist()}"
        )

    lung_mask_counts = df.groupby("label")["has_lung_mask"].sum().to_dict()
    expected = {"normal": 279, "TB": 287}
    if lung_mask_counts != expected:
        raise AssertionError(
            f"Shenzhen lung-mask count mismatch: got {lung_mask_counts}, expected {expected}"
        )

    missing_readings = df[~df["has_clinical_reading"]]
    if not missing_readings.empty:
        raise AssertionError(
            f"Shenzhen images missing a ClinicalReadings file: "
            f"{missing_readings['filename'].tolist()}"
        )

    # TB-positive images should have a lesion mask in at least one annotation set, except for
    # the handful with no visible TB sign (Yang et al., 2022) -- don't assert, just surface the
    # count so it can be checked against the expected ~6 empty cases.
    tb_no_mask = df[
        (df["label"] == "TB")
        & ~df["has_lesion_mask_primary"]
        & ~df["has_lesion_mask_secondary"]
    ]
    if len(tb_no_mask) > 10:
        raise AssertionError(
            "Unexpectedly many TB-positive Shenzhen images with no lesion mask in either "
            f"annotation set ({len(tb_no_mask)}, expected ~6): {tb_no_mask['filename'].tolist()}"
        )

    return df
