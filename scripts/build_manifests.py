"""Step 1: build raw manifests for Montgomery and Shenzhen.

Reads directly from the raw dataset folders (never modifies them) and writes
manifests/montgomery_raw.csv and manifests/shenzhen_raw.csv, plus a short console summary
so counts can be eyeballed against CLAUDE.md's verified numbers immediately.

Usage:
    python scripts/build_manifests.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lucidcxr_prep.config import load_paths  # noqa: E402
from lucidcxr_prep.sources import montgomery, shenzhen  # noqa: E402


def summarize(name: str, df, label_col: str = "label") -> None:
    print(f"\n=== {name} ===")
    print(f"total images: {len(df)}")
    print(df[label_col].value_counts().to_string())


def main() -> None:
    paths = load_paths()
    paths.manifests_dir.mkdir(parents=True, exist_ok=True)

    mont_df = montgomery.build_manifest(paths.montgomery_raw)
    mont_out = paths.manifests_dir / "montgomery_raw.csv"
    mont_df.to_csv(mont_out, index=False)
    summarize("Montgomery", mont_df)
    print(f"has_left_mask & has_right_mask: {(mont_df['has_left_mask'] & mont_df['has_right_mask']).all()}")
    print(f"landscape (verified rotation quirk) count: {mont_df['is_landscape'].sum()}")
    print(f"written -> {mont_out}")

    shen_df = shenzhen.build_manifest(paths.shenzhen_raw)
    shen_out = paths.manifests_dir / "shenzhen_raw.csv"
    shen_df.to_csv(shen_out, index=False)
    summarize("Shenzhen", shen_df)
    print(f"TB images with primary lesion mask: {shen_df['has_lesion_mask_primary'].sum()}")
    print(f"TB images with secondary lesion mask: {shen_df['has_lesion_mask_secondary'].sum()}")
    print(f"image mode distribution: {shen_df['mode'].value_counts().to_dict()}")
    print(f"written -> {shen_out}")


if __name__ == "__main__":
    main()
