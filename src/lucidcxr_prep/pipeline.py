"""Step 4: canonical image/mask processing, one row (one image) at a time.

Each function processes a single manifest row: load -> pad -> resize -> save, for the image and
every mask it owns, at every configured resolution. Returns the list of files it wrote (for
hash-logging by the caller) and a dict of processed-manifest fields.

No orientation correction is applied (Step 2 found none needed -- see CLAUDE.md Section 2.1).
No content-cropping is applied (decided against -- proposal doesn't call for it and a
brightness-threshold crop can misfire on dark lung fields).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from PIL import Image

from .io_utils import load_grayscale, sha256_file
from .masks import montgomery_lung_masks, shenzhen_lesion_mask, union_masks
from .transforms import pad_and_resize_image, pad_and_resize_mask

# MCUCXR_0251_1 is an apical lordotic view, not a standard PA chest X-ray (visually confirmed,
# IMPLEMENTATION_PLAN.md Step 2). Montgomery is an external test set only (never used for
# training), so keeping the file can't contaminate training -- but scoring the model against a
# projection it was never meant to handle would distort the *measured* sensitivity/specificity
# with a data point that isn't a genuine model failure. Processed and kept, just excluded from
# the headline external-test metrics.
NON_PA_PATIENT_IDS = {"0251"}


def _save_png(im: Image.Image, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    im.save(path, format="PNG")
    return path


def process_montgomery_row(row: pd.Series, resolutions: tuple[int, ...], output_root: Path) -> dict:
    written: list[Path] = []

    image = load_grayscale(Path(row["image_path"]), "montgomery")
    lung_masks = montgomery_lung_masks(Path(row["left_mask_path"]), Path(row["right_mask_path"]))

    for res in resolutions:
        img_out = output_root / "images" / str(res) / "montgomery" / f"{row['patient_id']}.png"
        written.append(_save_png(pad_and_resize_image(image, res), img_out))

        for mask_name, mask_im in lung_masks.items():
            mask_out = (
                output_root / "masks" / mask_name / str(res) / "montgomery" / f"{row['patient_id']}.png"
            )
            written.append(_save_png(pad_and_resize_mask(mask_im, res), mask_out))

    return {
        "source": "montgomery",
        "patient_id": row["patient_id"],
        "label": row["label"],
        "exclude_from_external_test": row["patient_id"] in NON_PA_PATIENT_IDS,
        "written_paths": written,
    }


def process_shenzhen_row(
    row: pd.Series,
    resolutions: tuple[int, ...],
    output_root: Path,
    masks_primary_dir: Path,
    masks_secondary_dir: Path,
) -> dict:
    written: list[Path] = []

    image = load_grayscale(Path(row["image_path"]), "shenzhen")
    original_size = (int(row["width"]), int(row["height"]))
    image_id = f"CHNCXR_{row['patient_id']}_{'1' if row['label'] == 'TB' else '0'}"

    is_tb = row["label"] == "TB"
    primary_mask = shenzhen_lesion_mask(image_id, masks_primary_dir, original_size) if is_tb else None
    secondary_mask = (
        shenzhen_lesion_mask(image_id, masks_secondary_dir, original_size) if is_tb else None
    )
    # Lung-FIELD mask (both lungs, one file) -- independent of TB status, per Rajaraman et al.
    # (2023). Present for 279 normal + 287 TB images (not all 662), so handle it as optional.
    lung_mask = None
    if pd.notna(row.get("lung_mask_path")):
        lung_mask = union_masks([Path(row["lung_mask_path"])], original_size)

    for res in resolutions:
        img_out = output_root / "images" / str(res) / "shenzhen" / f"{row['patient_id']}.png"
        written.append(_save_png(pad_and_resize_image(image, res), img_out))

        if primary_mask is not None:
            out = output_root / "masks" / "tb_lesion" / str(res) / "shenzhen" / f"{row['patient_id']}.png"
            written.append(_save_png(pad_and_resize_mask(primary_mask, res), out))
        if secondary_mask is not None:
            out = (
                output_root
                / "masks"
                / "tb_lesion_annot2"
                / str(res)
                / "shenzhen"
                / f"{row['patient_id']}.png"
            )
            written.append(_save_png(pad_and_resize_mask(secondary_mask, res), out))
        if lung_mask is not None:
            out = output_root / "masks" / "lung_field" / str(res) / "shenzhen" / f"{row['patient_id']}.png"
            written.append(_save_png(pad_and_resize_mask(lung_mask, res), out))

    return {
        "source": "shenzhen",
        "patient_id": row["patient_id"],
        "label": row["label"],
        "has_lesion_mask_primary": bool(row["has_lesion_mask_primary"]),
        "has_lesion_mask_secondary": bool(row["has_lesion_mask_secondary"]),
        "empty_reference_mask_primary": is_tb and not bool(row["has_lesion_mask_primary"]),
        "empty_reference_mask_secondary": is_tb and not bool(row["has_lesion_mask_secondary"]),
        "has_lung_mask": lung_mask is not None,
        "written_paths": written,
    }


def hash_written_files(written_paths: list[Path], output_root: Path) -> list[dict]:
    """One row per file: relative path (from output_root) + sha256, for the hash manifest."""
    rows = []
    for p in written_paths:
        # as_posix: forward slashes, so the committed manifest also works on Linux (e.g. a GPU cluster)
        rows.append({"relative_path": p.relative_to(output_root).as_posix(), "sha256": sha256_file(p)})
    return rows
