"""Step 5: build and pin the Montgomery + Shenzhen splits.

Montgomery -> entirely 'external_test' (proposal decision, not derived here).
Shenzhen   -> TB fixed 235/34/67, normals scaled to the same ratio, 228/33/65, both seeded.

Refuses to overwrite an existing split CSV unless --force is passed, so the partition is pinned
once drawn (CLAUDE.md Section 3 "split rule").

Usage:
    python scripts/build_splits.py [--force]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lucidcxr_prep.config import load_paths  # noqa: E402
from lucidcxr_prep.splits import montgomery_split, shenzhen_split  # noqa: E402


def _write_if_absent(df: pd.DataFrame, path: Path, force: bool) -> None:
    if path.exists() and not force:
        print(f"SKIP (already pinned, use --force to redraw) -> {path}")
        return
    df.to_csv(path, index=False)
    print(f"written -> {path}  ({len(df)} rows)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="overwrite an existing pinned split")
    args = parser.parse_args()

    paths = load_paths()

    mont_processed = pd.read_csv(
        paths.manifests_dir / "montgomery_processed.csv", dtype={"patient_id": str}
    )
    shen_raw = pd.read_csv(paths.manifests_dir / "shenzhen_raw.csv", dtype={"patient_id": str})

    mont_split = montgomery_split(mont_processed)
    _write_if_absent(mont_split, paths.manifests_dir / "montgomery_split.csv", args.force)

    shen_split_df = shenzhen_split(shen_raw, seed=paths.seed)
    _write_if_absent(shen_split_df, paths.manifests_dir / "shenzhen_split.csv", args.force)

    print("\nMontgomery split counts:")
    print(mont_split["split"].value_counts().to_string())
    print("\nShenzhen split counts (by label):")
    print(shen_split_df.groupby(["label", "split"]).size().to_string())


if __name__ == "__main__":
    main()
