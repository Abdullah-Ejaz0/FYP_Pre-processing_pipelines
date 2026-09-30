"""Step 6b: independently re-check every processed output against the manifests.

Checks (each prints PASS/FAIL; exits non-zero if any FAIL):
  1. File counts per image/mask folder match what the raw manifests say they should be.
  2. Every row in file_hashes.csv exists on disk and its sha256 still matches.
  3. No PNG under images/ or masks/ is missing from file_hashes.csv (no untracked outputs).
  4. Every image/mask is mode 'L', 512x512; every mask contains only {0, 255}.
  5. Splits: unique patient IDs, every split patient exists in the processed manifest,
     expected counts, Montgomery all external_test.
  6. Shenzhen primary lesion masks: exactly the 6 flagged rows are empty; any other TB mask
     that came out empty after resize is a lesion lost to downsampling (reported, not hidden).
  7. Every lung_field mask is non-empty.

Usage:
    python scripts/verify_outputs.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lucidcxr_prep.config import load_paths  # noqa: E402
from lucidcxr_prep.io_utils import sha256_file  # noqa: E402

RES = 512
failures: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" -- {detail}" if detail else ""))
    if not ok:
        failures.append(name)


def main() -> None:
    paths = load_paths()
    root = paths.output_root
    man = paths.manifests_dir

    def read(name: str) -> pd.DataFrame:
        return pd.read_csv(man / name, dtype={"patient_id": str})

    mont_raw, shen_raw = read("montgomery_raw.csv"), read("shenzhen_raw.csv")
    mont_proc, shen_proc = read("montgomery_processed.csv"), read("shenzhen_processed.csv")
    mont_split, shen_split = read("montgomery_split.csv"), read("shenzhen_split.csv")
    hashes = pd.read_csv(man / "file_hashes.csv")

    # 1. counts
    n_shen_tb = int((shen_raw["label"] == "TB").sum())
    expected = {
        f"images/{RES}/montgomery": len(mont_raw),
        f"images/{RES}/shenzhen": len(shen_raw),
        f"masks/lung_field/{RES}/montgomery": len(mont_raw),
        f"masks/lung_patient_right/{RES}/montgomery": len(mont_raw),
        f"masks/lung_patient_left/{RES}/montgomery": len(mont_raw),
        f"masks/lung_field/{RES}/shenzhen": int(shen_raw["has_lung_mask"].sum()),
        f"masks/tb_lesion/{RES}/shenzhen": n_shen_tb,
        f"masks/tb_lesion_annot2/{RES}/shenzhen": n_shen_tb,
    }
    for rel, n in expected.items():
        got = len(list((root / rel).glob("*.png")))
        check(f"count {rel}", got == n, f"expected {n}, got {got}")

    # 2. hashes still match
    bad_hash, missing = [], []
    for rel, digest in tqdm(zip(hashes["relative_path"], hashes["sha256"]), total=len(hashes),
                            desc="re-hashing"):
        p = root / rel
        if not p.exists():
            missing.append(rel)
        elif sha256_file(p) != digest:
            bad_hash.append(rel)
    check("all hashed files exist", not missing, f"{len(missing)} missing")
    check("all hashes match", not bad_hash, f"{len(bad_hash)} changed")

    # 3. no untracked outputs
    on_disk = {
        p.relative_to(root).as_posix()
        for sub in ("images", "masks")
        for p in (root / sub).rglob("*.png")
    }
    untracked = on_disk - set(hashes["relative_path"])
    check("no untracked outputs", not untracked, f"{len(untracked)} untracked")

    # 4. format
    bad_format, bad_values = [], []
    mask_sums: dict[str, int] = {}
    for rel in tqdm(sorted(on_disk), desc="format check"):
        with Image.open(root / rel) as im:
            if im.mode != "L" or im.size != (RES, RES):
                bad_format.append(rel)
                continue
            if rel.startswith("masks"):
                arr = np.asarray(im)
                if not set(np.unique(arr)).issubset({0, 255}):
                    bad_values.append(rel)
                mask_sums[rel] = int((arr > 0).sum())
    check(f"all outputs mode L, {RES}x{RES}", not bad_format, f"{len(bad_format)} bad")
    check("all masks binary {0,255}", not bad_values, f"{len(bad_values)} non-binary")

    # 5. splits
    check("montgomery split all external_test", (mont_split["split"] == "external_test").all())
    check("montgomery split covers all images",
          set(mont_split["patient_id"]) == set(mont_proc["patient_id"]))
    check("shenzhen split unique patients", shen_split["patient_id"].is_unique)
    check("shenzhen split covers all images",
          set(shen_split["patient_id"]) == set(shen_proc["patient_id"]))
    counts = shen_split.groupby(["label", "split"]).size().to_dict()
    want = {("TB", "train"): 235, ("TB", "val"): 34, ("TB", "test"): 67,
            ("normal", "train"): 228, ("normal", "val"): 33, ("normal", "test"): 65}
    check("shenzhen split counts", counts == want, str(counts))

    # 6. lesion masks: flagged-empty vs lost-to-downsampling
    tb = shen_proc[shen_proc["label"] == "TB"]
    flagged = set(tb.loc[tb["empty_reference_mask_primary"], "patient_id"])
    empty_now = {
        pid for pid in tb["patient_id"]
        if mask_sums.get(f"masks/tb_lesion/{RES}/shenzhen/{pid}.png", 0) == 0
    }
    lesion_found = sum(1 for r in mask_sums if r.startswith(f"masks/tb_lesion/{RES}/"))
    check("lesion masks were actually found", lesion_found == len(tb), f"{lesion_found}")
    check("flagged-empty lesion masks are empty", flagged <= empty_now,
          f"{len(flagged)} flagged")
    lost = sorted(empty_now - flagged)
    check("no lesion masks lost to downsampling", not lost,
          f"{len(lost)} lost: {lost}" if lost else "")

    # 7. lung fields non-empty
    lung_keys = [r for r in mask_sums if r.startswith("masks/lung_")]
    empty_lung = [r for r in lung_keys if mask_sums[r] == 0]
    # Guard against this check passing vacuously (it once did, on Windows path separators).
    check("lung masks were actually found", len(lung_keys) == 138 * 3 + 566, f"{len(lung_keys)}")
    check("all lung masks non-empty", not empty_lung, f"{len(empty_lung)} empty")

    print(f"\n{'ALL CHECKS PASSED' if not failures else f'{len(failures)} CHECK(S) FAILED'}")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
