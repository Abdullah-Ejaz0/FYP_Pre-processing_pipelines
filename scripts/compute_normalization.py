"""Step 6a: compute mean/std from the Shenzhen TRAIN split only.

Montgomery is external_test and never contributes. Val/test never contribute.
These are provisional: once other sources join training, recompute over the pooled train set.

Usage:
    python scripts/compute_normalization.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lucidcxr_prep.config import load_paths  # noqa: E402
from lucidcxr_prep.normalization import compute_stats  # noqa: E402


def main() -> None:
    paths = load_paths()
    split = pd.read_csv(paths.manifests_dir / "shenzhen_split.csv", dtype={"patient_id": str})
    train_ids = split.loc[split["split"] == "train", "patient_id"].tolist()

    image_paths = [
        paths.output_root / "images" / "512" / "shenzhen" / f"{pid}.png" for pid in train_ids
    ]
    missing = [p for p in image_paths if not p.exists()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} train images missing, e.g. {missing[:3]}")

    stats = {
        "computed_from": "shenzhen train split only (montgomery excluded: external_test)",
        "pixel_scale": "values divided by 255 before computing",
        "includes_zero_padding": True,
        "provisional": "recompute over the pooled train set once other sources are added",
        "shenzhen_train": compute_stats(image_paths),
    }
    out = paths.manifests_dir / "normalization_stats.json"
    out.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(json.dumps(stats, indent=2))
    print(f"\nwritten -> {out}")


if __name__ == "__main__":
    main()
