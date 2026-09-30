"""Regression tests for Step 5 split logic -- synthetic IDs, no raw data needed."""

import pandas as pd

from lucidcxr_prep.splits import _seeded_split, montgomery_split, shenzhen_split


def test_seeded_split_exact_counts_and_no_overlap():
    ids = [f"p{i:03d}" for i in range(20)]
    counts = {"train": 12, "val": 3, "test": 5}
    assignment = _seeded_split(ids, counts, seed=42)

    assert len(assignment) == 20
    from collections import Counter

    assert Counter(assignment.values()) == counts


def test_seeded_split_is_deterministic():
    ids = [f"p{i:03d}" for i in range(20)]
    counts = {"train": 12, "val": 3, "test": 5}
    a = _seeded_split(ids, counts, seed=42)
    b = _seeded_split(ids, counts, seed=42)
    assert a == b


def test_seeded_split_wrong_count_raises():
    import pytest

    with pytest.raises(ValueError):
        _seeded_split(["a", "b"], {"train": 1, "val": 1, "test": 1}, seed=0)


def test_shenzhen_split_matches_proposal_counts():
    tb_ids = [f"t{i:03d}" for i in range(336)]
    normal_ids = [f"n{i:03d}" for i in range(326)]
    raw_df = pd.DataFrame(
        {
            "patient_id": tb_ids + normal_ids,
            "label": ["TB"] * 336 + ["normal"] * 326,
        }
    )
    out = shenzhen_split(raw_df, seed=42)

    assert len(out) == 662
    assert out["patient_id"].is_unique
    counts = out.groupby(["label", "split"]).size().to_dict()
    assert counts[("TB", "train")] == 235
    assert counts[("TB", "val")] == 34
    assert counts[("TB", "test")] == 67
    assert counts[("normal", "train")] == 228
    assert counts[("normal", "val")] == 33
    assert counts[("normal", "test")] == 65


def test_montgomery_split_is_all_external_test():
    processed_df = pd.DataFrame(
        {
            "patient_id": ["0001", "0002", "0251"],
            "label": ["normal", "normal", "TB"],
            "exclude_from_external_test": [False, False, True],
        }
    )
    out = montgomery_split(processed_df)
    assert (out["split"] == "external_test").all()
    assert out.loc[out["patient_id"] == "0251", "exclude_from_external_test"].item() is True
