"""Regression tests for the Step 1 manifest builders.

Skip cleanly if the raw dataset folders aren't present (e.g. repo cloned without the data
siblings) rather than failing -- these tests describe verified properties of the specific raw
folders documented in CLAUDE.md, not properties the code can guarantee in general.
"""

from pathlib import Path

import pytest

from lucidcxr_prep.config import load_paths
from lucidcxr_prep.sources import montgomery, shenzhen

PATHS = load_paths()


def _require(path: Path) -> None:
    if not path.exists():
        pytest.skip(f"raw data not found at {path} -- skipping (see CLAUDE.md Section 2)")


def test_montgomery_manifest_counts():
    _require(PATHS.montgomery_raw)
    df = montgomery.build_manifest(PATHS.montgomery_raw)

    assert len(df) == 138
    counts = df["label"].value_counts().to_dict()
    assert counts == {"normal": 80, "TB": 58}
    assert (df["has_left_mask"] & df["has_right_mask"]).all()
    assert df["patient_id"].is_unique


def test_montgomery_orientation_quirk():
    _require(PATHS.montgomery_raw)
    df = montgomery.build_manifest(PATHS.montgomery_raw)

    # Verified: exactly two sizes on disk, an exact transpose pair, 41 "landscape" images.
    sizes = set(zip(df["width"], df["height"]))
    assert sizes == {(4020, 4892), (4892, 4020)}
    assert df["is_landscape"].sum() == 41


def test_shenzhen_manifest_counts():
    _require(PATHS.shenzhen_raw)
    df = shenzhen.build_manifest(PATHS.shenzhen_raw)

    assert len(df) == 662
    counts = df["label"].value_counts().to_dict()
    assert counts == {"TB": 336, "normal": 326}
    assert df["patient_id"].is_unique


def test_shenzhen_lesion_mask_coverage():
    _require(PATHS.shenzhen_raw)
    df = shenzhen.build_manifest(PATHS.shenzhen_raw)

    assert df["has_lesion_mask_primary"].sum() == 330
    assert df["has_lesion_mask_secondary"].sum() == 323

    # Only TB-positive images should ever have a lesion mask.
    normals = df[df["label"] == "normal"]
    assert not normals["has_lesion_mask_primary"].any()
    assert not normals["has_lesion_mask_secondary"].any()


def test_shenzhen_mode_distribution_and_rgb_safety():
    _require(PATHS.shenzhen_raw)
    df = shenzhen.build_manifest(PATHS.shenzhen_raw)

    # Verified on the full 662-file set (CLAUDE.md Section 2.2): NOT uniformly palette-mode --
    # 635 are 'P', 27 are 'RGB'. Both need .convert('L'); this is the regression guard against
    # re-assuming "Shenzhen is all palette-mode" the way the code first (wrongly) did.
    counts = df["mode"].value_counts().to_dict()
    assert counts == {"P": 635, "RGB": 27}

    # Every RGB file must have R==G==B exactly, or .convert('L') would silently discard data.
    rgb_rows = df[df["mode"] == "RGB"]
    assert rgb_rows["rgb_channels_equal"].all()
