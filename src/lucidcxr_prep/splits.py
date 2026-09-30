"""Step 5: patient-level splits for Montgomery and Shenzhen.

Montgomery is not split -- it's held out entirely as an external test set (proposal Section
4.3.1: "deliberately excluded from all training and validation"), a decision already locked in
(CLAUDE.md #2), not something this module re-derives.

Shenzhen TB images (336) get the proposal's fixed segmentation split, 235/34/67
train/val/test, drawn once with a fixed seed. Shenzhen normal images (326) have no
proposal-specified ratio, so they're split at the *same proportions* the TB split already
uses (235/336, 34/336, 67/336 ~= 69.9%/10.1%/19.9%) rather than an arbitrary new number --
scaled to 326 images this is 228/33/65.

Both sources have exactly one image per patient (verified, CLAUDE.md Section 2), so "patient-level
split" and "image-level split" coincide here -- the split still keys off patient_id, not row
index, so the same code doesn't silently break on a future source with repeat patients.

Once written, a split CSV is meant to be pinned: re-running scripts/build_splits.py without
--force will refuse to overwrite an existing split file, so a later code change (e.g. a numpy
version bump) can't silently redraw an already-used partition.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

SHENZHEN_TB_COUNTS = {"train": 235, "val": 34, "test": 67}  # fixed by the proposal
SHENZHEN_NORMAL_COUNTS = {"train": 228, "val": 33, "test": 65}  # same ratio, scaled to 326


def _seeded_split(patient_ids: list[str], counts: dict[str, int], seed: int) -> dict[str, str]:
    total = sum(counts.values())
    if len(patient_ids) != total:
        raise ValueError(
            f"Expected {total} patient IDs to split ({counts}), got {len(patient_ids)}"
        )
    shuffled = list(patient_ids)
    np.random.RandomState(seed).shuffle(shuffled)

    assignment: dict[str, str] = {}
    start = 0
    for split_name, n in counts.items():
        for pid in shuffled[start : start + n]:
            assignment[pid] = split_name
        start += n
    return assignment


def montgomery_split(processed_df: pd.DataFrame) -> pd.DataFrame:
    """Every Montgomery image is `external_test`; carries over the lordotic-view exclusion flag."""
    out = processed_df[["patient_id", "label", "exclude_from_external_test"]].copy()
    out["split"] = "external_test"
    return out[["patient_id", "label", "split", "exclude_from_external_test"]]


def shenzhen_split(raw_df: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Fixed 235/34/67 for TB images, 228/33/65 (same ratio) for normal images."""
    tb_ids = sorted(raw_df.loc[raw_df["label"] == "TB", "patient_id"])
    normal_ids = sorted(raw_df.loc[raw_df["label"] == "normal", "patient_id"])

    tb_assignment = _seeded_split(tb_ids, SHENZHEN_TB_COUNTS, seed)
    normal_assignment = _seeded_split(normal_ids, SHENZHEN_NORMAL_COUNTS, seed)

    rows = [
        {"patient_id": pid, "label": "TB", "split": split}
        for pid, split in tb_assignment.items()
    ] + [
        {"patient_id": pid, "label": "normal", "split": split}
        for pid, split in normal_assignment.items()
    ]
    return pd.DataFrame(rows).sort_values("patient_id").reset_index(drop=True)
