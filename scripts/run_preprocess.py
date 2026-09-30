"""Step 4: run the canonical image/mask processing pipeline on Montgomery + Shenzhen.

Reads manifests/montgomery_raw.csv and manifests/shenzhen_raw.csv (built by
scripts/build_manifests.py). Writes processed images/masks under <output_root>, and two new
manifests: montgomery_processed.csv / shenzhen_processed.csv, plus a combined file_hashes.csv.

Never touches the raw dataset folders -- only reads from them.

Usage:
    python scripts/run_preprocess.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lucidcxr_prep.config import load_paths  # noqa: E402
from lucidcxr_prep.pipeline import (  # noqa: E402
    hash_written_files,
    process_montgomery_row,
    process_shenzhen_row,
)


def main() -> None:
    paths = load_paths()
    mont_manifest = paths.manifests_dir / "montgomery_raw.csv"
    shen_manifest = paths.manifests_dir / "shenzhen_raw.csv"
    for m in (mont_manifest, shen_manifest):
        if not m.exists():
            raise FileNotFoundError(f"{m} missing -- run scripts/build_manifests.py first")

    mont_df = pd.read_csv(mont_manifest, dtype={"patient_id": str})
    shen_df = pd.read_csv(shen_manifest, dtype={"patient_id": str})

    hash_rows: list[dict] = []
    processed_rows: list[dict] = []

    print(f"Processing Montgomery ({len(mont_df)} images) at resolutions {paths.resolutions} ...")
    for _, row in tqdm(mont_df.iterrows(), total=len(mont_df)):
        result = process_montgomery_row(row, paths.resolutions, paths.output_root)
        written = result.pop("written_paths")
        hash_rows.extend(hash_written_files(written, paths.output_root))
        processed_rows.append(result)
    mont_processed = pd.DataFrame(processed_rows)
    mont_out = paths.manifests_dir / "montgomery_processed.csv"
    mont_processed.to_csv(mont_out, index=False)
    print(f"written -> {mont_out}  ({len(mont_processed)} rows)")
    print(
        f"excluded_from_external_test: "
        f"{mont_processed['exclude_from_external_test'].sum()} "
        f"({mont_processed.loc[mont_processed['exclude_from_external_test'], 'patient_id'].tolist()})"
    )

    processed_rows = []
    print(f"\nProcessing Shenzhen ({len(shen_df)} images) at resolutions {paths.resolutions} ...")
    masks_primary_dir = paths.shenzhen_raw / "Annotations" / "masks"
    masks_secondary_dir = paths.shenzhen_raw / "Annotations-2" / "masks"
    for _, row in tqdm(shen_df.iterrows(), total=len(shen_df)):
        result = process_shenzhen_row(
            row, paths.resolutions, paths.output_root, masks_primary_dir, masks_secondary_dir
        )
        written = result.pop("written_paths")
        hash_rows.extend(hash_written_files(written, paths.output_root))
        processed_rows.append(result)
    shen_processed = pd.DataFrame(processed_rows)
    shen_out = paths.manifests_dir / "shenzhen_processed.csv"
    shen_processed.to_csv(shen_out, index=False)
    print(f"written -> {shen_out}  ({len(shen_processed)} rows)")
    tb_rows = shen_processed[shen_processed["label"] == "TB"]
    print(
        f"TB rows with empty primary reference mask: "
        f"{tb_rows['empty_reference_mask_primary'].sum()} (expected ~6)"
    )

    hash_df = pd.DataFrame(hash_rows)
    hash_out = paths.manifests_dir / "file_hashes.csv"
    hash_df.to_csv(hash_out, index=False)
    print(f"\nwritten -> {hash_out}  ({len(hash_df)} files hashed)")
    dup = hash_df["relative_path"].duplicated().sum()
    if dup:
        raise AssertionError(f"{dup} duplicate relative_path entries in hash manifest")


if __name__ == "__main__":
    main()
