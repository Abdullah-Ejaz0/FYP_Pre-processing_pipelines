"""Extract structured (sex, age) + raw diagnosis text from ClinicalReadings, per patient.

Written to <output_root>/metadata/{montgomery,shenzhen}_metadata.csv -- alongside the images and
masks, since this is patient-level data needed at training/eval time (e.g. for subgroup checks),
not a build artifact. Join to the split manifests (manifests/*_split.csv, in this repo) by
patient_id when needed; split is intentionally not duplicated into this file to avoid the two
drifting out of sync if a split is ever redrawn.

Usage:
    python scripts/build_metadata.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lucidcxr_prep.config import load_paths  # noqa: E402
from lucidcxr_prep.metadata import parse_montgomery, parse_shenzhen  # noqa: E402


def build(raw_csv: Path, parser) -> pd.DataFrame:
    df = pd.read_csv(raw_csv, dtype={"patient_id": str})
    parsed = df["clinical_text"].apply(parser).apply(pd.Series)
    return pd.concat([df[["patient_id", "source", "label"]], parsed], axis=1)


def main() -> None:
    paths = load_paths()
    out_dir = paths.output_root / "metadata"
    out_dir.mkdir(parents=True, exist_ok=True)

    mont = build(paths.manifests_dir / "montgomery_raw.csv", parse_montgomery)
    shen = build(paths.manifests_dir / "shenzhen_raw.csv", parse_shenzhen)

    # Sanity check, not a tautology: catches a unit-conversion bug that previously produced NaN
    # for 658/662 rows silently (no exception, just a wrong number). age_years should be known
    # for all but the single row with no unit at all.
    n_nan = shen["age_years"].isna().sum()
    if n_nan > 1:
        raise AssertionError(
            f"{n_nan} Shenzhen rows have unresolved age_years (expected exactly 1, the row with "
            f"no unit at all): {shen.loc[shen['age_years'].isna(), 'patient_id'].tolist()}"
        )

    mont.to_csv(out_dir / "montgomery_metadata.csv", index=False)
    shen.to_csv(out_dir / "shenzhen_metadata.csv", index=False)

    print(f"Montgomery: {len(mont)} rows, age {mont.age_years.min()}-{mont.age_years.max()}, "
          f"sex counts {mont.sex.value_counts().to_dict()}")
    print(f"Shenzhen:   {len(shen)} rows, age {shen.age_years.min()}-{shen.age_years.max()}, "
          f"sex counts {shen.sex.value_counts().to_dict()}")
    print(f"\nwritten -> {out_dir / 'montgomery_metadata.csv'}")
    print(f"written -> {out_dir / 'shenzhen_metadata.csv'}")


if __name__ == "__main__":
    main()
