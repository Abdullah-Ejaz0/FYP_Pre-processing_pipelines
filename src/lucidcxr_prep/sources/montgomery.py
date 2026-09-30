"""Manifest builder for the Montgomery County CXR Set.

Verified on-disk layout (CLAUDE.md Section 2.1):
- MontgomerySet/CXR_png/MCUCXR_{patient_id}_{label}.png   -- 138 files, label 0=normal, 1=TB
- MontgomerySet/ClinicalReadings/MCUCXR_{patient_id}_{label}.txt
- MontgomerySet/ManualMask/leftMask/...   -- present for all 138 images
- MontgomerySet/ManualMask/rightMask/...  -- present for all 138 images
- Exactly two image sizes exist, (4020,4892) and (4892,4020), an exact transpose of each other
  for a 41-image subset -- flagged here as `is_landscape`, corrected (or not) in a later step.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from ..io_utils import read_raw_image_info

FILENAME_RE = re.compile(r"^MCUCXR_(\d+)_([01])\.png$")
LABEL_MAP = {"0": "normal", "1": "TB"}


def build_manifest(raw_root: Path) -> pd.DataFrame:
    """raw_root is the MontgomerySet/ directory (see configs/paths.yaml)."""
    cxr_dir = raw_root / "CXR_png"
    readings_dir = raw_root / "ClinicalReadings"
    left_mask_dir = raw_root / "ManualMask" / "leftMask"
    right_mask_dir = raw_root / "ManualMask" / "rightMask"

    rows = []
    for img_path in sorted(cxr_dir.glob("*.png")):
        m = FILENAME_RE.match(img_path.name)
        if not m:
            raise ValueError(f"Unexpected Montgomery filename format: {img_path.name}")
        patient_id, label_code = m.group(1), m.group(2)

        info = read_raw_image_info(img_path)

        reading_path = readings_dir / f"{img_path.stem}.txt"
        left_mask_path = left_mask_dir / img_path.name
        right_mask_path = right_mask_dir / img_path.name

        clinical_text = None
        if reading_path.exists():
            clinical_text = reading_path.read_text(encoding="utf-8", errors="replace").strip()

        rows.append(
            {
                "source": "montgomery",
                "patient_id": patient_id,
                "filename": img_path.name,
                "label": LABEL_MAP[label_code],
                "image_path": str(img_path),
                "width": info["width"],
                "height": info["height"],
                "mode": info["mode"],
                "is_landscape": info["width"] > info["height"],
                "has_clinical_reading": reading_path.exists(),
                "clinical_text": clinical_text,
                "has_left_mask": left_mask_path.exists(),
                "has_right_mask": right_mask_path.exists(),
                "left_mask_path": str(left_mask_path) if left_mask_path.exists() else None,
                "right_mask_path": str(right_mask_path) if right_mask_path.exists() else None,
            }
        )

    df = pd.DataFrame(rows)

    missing_masks = df[~(df["has_left_mask"] & df["has_right_mask"])]
    if not missing_masks.empty:
        raise AssertionError(
            "Montgomery images missing a lung mask (expected all 138 to have both): "
            f"{missing_masks['filename'].tolist()}"
        )
    missing_readings = df[~df["has_clinical_reading"]]
    if not missing_readings.empty:
        raise AssertionError(
            f"Montgomery images missing a ClinicalReadings file: "
            f"{missing_readings['filename'].tolist()}"
        )

    return df
