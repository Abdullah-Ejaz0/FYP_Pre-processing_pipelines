# Preprocessing Implementation Plan (Montgomery + Shenzhen, v1)

Status: **Steps 0–6 done and verified for Montgomery + Shenzhen.** Their preprocessing is complete:
processed, split, normalized, and passing `scripts/verify_outputs.py`. The other sources (TBX11K,
RSNA, CheXpert, NIH ChestX-ray14, VinDr-CXR, Belarus) haven't been started, pending your go-ahead
and the cross-source patient-identity decision. This plan is
based on actually inspecting the raw files (see `CLAUDE.md` Section 2 for verified findings, not
assumptions), and only goes as far as: raw folders → verified, resized, split, hash-logged image +
mask outputs in a **new** sibling folder, ready for a modeling repo to consume. No model code here.

## Step 0 — Repo scaffolding ✅ DONE
Folder structure, `requirements.txt`, `configs/paths.yaml`, `.gitignore`, `conftest.py` (so
`src/` is importable by both scripts and pytest) all in place.

## Step 1 — Manifest builders (read-only, no image processing) ✅ DONE
`sources/montgomery.py` and `sources/shenzhen.py` built, run via `scripts/build_manifests.py`,
covered by `tests/test_manifests.py` (5 tests, all passing). Output: `manifests/montgomery_raw.csv`
(138 rows) and `manifests/shenzhen_raw.csv` (662 rows), committed to git.

**Finding during this step (not in the original plan text) — corrected before it could cause
silent data loss:** the initial 15-file visual sample (used to write `CLAUDE.md`) found Shenzhen
`CXR_png` to be uniformly palette-mode (`'P'`). Building the *full* manifest and asserting the
mode on all 662 files caught that this was wrong — **635 files are `'P'`, 27 are `'RGB'`**. Every
sampled RGB file has R==G==B exactly (verified programmatically in `sources/shenzhen.py`, not
assumed), so `.convert('L')` is still lossless for them, but a loader written to only handle `'P'`
would have crashed or mis-handled 27 images. `io_utils.py`, the manifest builder, its test, and
`CLAUDE.md` Section 2.2 were all updated to reflect the verified mode split. Lesson carried
forward: don't trust a small manual sample for a "this source is always mode X" claim — assert
it over the full set once code exists to do so cheaply.

Original text, for reference:
One module per source (`sources/montgomery.py`, `sources/shenzhen.py`) that walks the raw folder
and emits a pandas DataFrame / CSV with one row per image:

- `patient_id`, `source`, `label` (`TB`/`normal`, parsed from the filename suffix — verified
  100% consistent: `_0`→normal, `_1`→TB, in both sources)
  - **Montgomery: 80 normal / 58 TB**
  - **Shenzhen: 326 normal / 336 TB**
- `image_path` (absolute path into the raw folder — never copied at this stage)
- `width`, `height`, `mode` (read via `PIL.Image.open(...).size/.mode`, not assumed)
- `has_lesion_mask` (Shenzhen only: True if the image has ≥1 file in `Annotations/masks/`)
- `has_lung_mask` (Montgomery only: True for all 138, verified)
- `clinical_text` (raw contents of the matching `ClinicalReadings/*.txt`, kept as free text —
  not parsed into structured fields in v1, just carried through for later manual QA)

Output: `manifests/montgomery_raw.csv`, `manifests/shenzhen_raw.csv`. These get committed (small,
text-only) so you can open them directly and sanity-check counts before anything else runs.

**Verification built into this step, not deferred**: assert every image file has a matching
`ClinicalReadings` entry; for Montgomery, assert every image has both a `leftMask` and
`rightMask` file (confirmed true for all 138, but asserted rather than trusted); for Shenzhen, no
such universal assertion (lesion masks legitimately don't exist for normal images or the ~6
TB images without visible signs) — instead just record the boolean and count.

## Step 2 — Montgomery orientation QC ✅ DONE
`scripts/inspect_orientation.py` wrote 4 contact sheets (all 138 images, lung masks overlaid:
red = `leftMask`, blue = `rightMask`) to `../data_processed/_qc/montgomery_orientation/`.

Results from reviewing all 138 thumbnails:
1. **No rotation needed.** The 41 `(4892,4020)` images are upright PA chests captured on a
   landscape-oriented detector. The rotation hypothesis above was wrong. No `rotated` column, no
   rotation code.
2. **`leftMask` = image-left = patient's RIGHT lung** (it sits under the "R" marker on every
   image). Step 3 renames masks to patient-side names.
3. `MCUCXR_0251_1` is an apical lordotic ("LT APICAL") view, not PA. It's flagged for your decision.
4. Several images have large black borders. Crop-to-content is an open decision; the default is
   no crop.
5. All lung masks align with their images. No mask/image mismatch was visible.

## Step 3 — Mask preparation
**Montgomery**: write three masks per image: `lung_patient_right` (from `leftMask`),
`lung_patient_left` (from `rightMask`), and `lung_field` (their union). The per-side masks are
kept because the structured report's "affected lung" field needs them, and the side naming is
error-prone (see Step 2 item 2). The union is the lung-field denominator.

**Shenzhen**: for each of the 330 images with lesion masks in `Annotations/`, union all
per-abnormality PNGs for that image (logical OR across e.g. `..._Calcified_Nodule_2.png` +
`..._Clustered_Nodule_..._1.png`) into one binary `tb_lesion_mask` per image. The 6 TB-positive
images with no mask file get an explicit all-zero mask, flagged `empty_reference_mask: true` in
the manifest (proposal requires these be counted, never imputed or skipped — Section 6.3/6.4).
`Annotations-2/` is processed the same way into a second, separate mask set, kept for an optional
future inter-annotator agreement check — not blended into the primary mask.

**Shenzhen lung-field mask (added after this plan was first written)**: once you located the
missing `mask/mask/CHNCXR_{patient_id}_{label}_mask.png` files (566 = 279 normal + 287 TB,
verified), these needed no unioning — one combined (both-lungs) binary mask per image already.
Loaded via the same `union_masks([path], size)` helper for free size validation, written to
`masks/lung_field/512/shenzhen/<patient_id>.png`. This closed the gap that made Montgomery the
only lung-boundary source on hand — Montgomery's training role (decision #14) stayed "none."

Verification: for every unioned mask, assert pixel dimensions match the source CXR exactly
(should hold — confirmed on samples) and log any mismatch loudly rather than resizing silently
to "fix" it.

## Step 4 — Canonical image/mask processing pipeline ✅ DONE
`src/lucidcxr_prep/transforms.py` (pad-to-square + bicubic resize, identical for images and
masks), `masks.py` (mask union/rename, built in Step 3), `pipeline.py` (per-row orchestration),
run via `scripts/run_preprocess.py`.

1. Load image: Montgomery via `Image.open(...)` (`'L'`); Shenzhen via `Image.open(...).convert('L')`
   — **both `'P'` and `'RGB'` files**, per the Step 1 finding (not just `'P'` as first assumed).
2. No orientation correction (Step 2 found none needed). No crop-to-content (decided against —
   not in the proposal, and a brightness-threshold crop can misfire on dark lung fields).
3. Pad to square with zero (black) padding, aspect ratio preserved.
4. Resize with bicubic interpolation to **512×512 only**. (Revised from the proposal's "write
   both 256 and 512" — 256 is produced by downsampling the stored 512 PNG in the dataloader at
   load time instead, halving stored files/disk. See `configs/paths.yaml` for the rationale.)
5. Masks: identical pad+resize, then re-binarize at 0.5 after resize.
6. Saved as 8-bit single-channel PNG. No intensity normalization baked in — deferred to Step 6.
7. Every output file's path + sha256 hash appended to `manifests/file_hashes.csv`.

Smoke-tested first on ~7 hand-picked rows (both Montgomery orientations, the excluded lordotic
view, a TB image with a mask, a normal image, an RGB-mode image, the one TB image confirmed
missing its primary mask) with the mask/image overlays rendered and eyeballed before running the
full set — see below for what that caught.

**Run results (full 800 images, 512 only, including the later-added Shenzhen lung masks):**
- 2,452 files written and hashed, 0 duplicate paths. (First pass, before the lung masks arrived:
  1,886. Before dropping 256: 3,772 = exactly double 1,886, confirming the two resolutions were
  symmetric.)
- Montgomery: 138 images + 138 × 3 mask types = 552 files.
- Shenzhen: 662 images + 336 TB × 2 lesion-mask sets + 566 lung-field masks = 1,900 files.
- `MCUCXR_0251_1` flagged `exclude_from_external_test=True` (only that one row) — see decision #12.
- 6 TB rows flagged `empty_reference_mask_primary=True`, exactly matching the proposal's "330 of
  336 show visible TB signs" (Yang et al., 2022) — cross-checked directly against the raw manifest
  (6 images missing from `Annotations/`, 13 missing from `Annotations-2/`, 0 missing from both).
- Shenzhen lung-mask count verified 279 normal + 287 TB = 566, asserted in `sources/shenzhen.py`.
- A rendered overlay (lung-field mask + TB lesion mask on the source image) was eyeballed before
  trusting the output — lung field covers both lungs, lesion mask sits inside it on the correct
  side.
- ~3m30s end-to-end for the image/mask processing pass on this machine.

Output layout (as actually produced — no `<run_tag>` subfolder in v1; add one if/when multiple
preprocessing variants need to coexist):
```
../data_processed/
├── images/512/{montgomery,shenzhen}/<patient_id>.png
├── masks/lung_field/512/montgomery/<patient_id>.png
├── masks/lung_field/512/shenzhen/<patient_id>.png              # 566 of 662 (see Section 2.2)
├── masks/lung_patient_right/512/montgomery/<patient_id>.png   # from leftMask
├── masks/lung_patient_left/512/montgomery/<patient_id>.png    # from rightMask
├── masks/tb_lesion/512/shenzhen/<patient_id>.png              # Annotations/ (primary)
├── masks/tb_lesion_annot2/512/shenzhen/<patient_id>.png       # Annotations-2/ (secondary)
└── _qc/montgomery_orientation/*.png                           # Step 2 contact sheets
```
and, committed in this repo:
```
manifests/
├── montgomery_raw.csv / shenzhen_raw.csv          # Step 1
├── montgomery_processed.csv / shenzhen_processed.csv   # Step 4
├── montgomery_split.csv / shenzhen_split.csv       # Step 5
└── file_hashes.csv                                # Step 4, 2,452 rows
```
`data_processed/` is a **new folder**, never inside the two raw dataset folders — you can inspect
it, then decide whether to keep, adjust, or discard, before the raw folders are touched at all
(they never are, by design — the pipeline only reads from them).

## Step 5 — Patient-level split ✅ DONE for Montgomery + Shenzhen; other 5 sources deferred
`src/lucidcxr_prep/splits.py` + `scripts/build_splits.py`, covered by `tests/test_splits.py`
(5 tests). Groups by `patient_id` (already 1:1 with image for both sources, but the code groups
rather than assumes, per `CLAUDE.md` decision #1).

- **Montgomery**: entirely `external_test`, all 138 rows. `montgomery_split.csv` also carries the
  `exclude_from_external_test` flag through from Step 4 (the lordotic-view case).
- **Shenzhen**: TB images get the proposal's fixed 235/34/67; normal images get 228/33/65 — the
  same ratio as the TB split, not an independently invented number (decision #15). Both drawn
  with `seed=42`, written to `shenzhen_split.csv`. Verified: exact counts match, no patient_id
  appears twice.
- **Pinning**: `build_splits.py` checks whether the split CSV already exists and skips (prints
  `SKIP ... use --force to redraw`) rather than silently overwriting — verified by running it
  twice. Once a split is drawn and used, a later code/library change can't quietly redraw it.
- **The other five sources (TBX11K, RSNA, CheXpert, NIH ChestX-ray14, VinDr-CXR, Belarus/TB
  Portals) are intentionally not split yet** — deferred pending the cross-source patient-identity
  question (do any of these share patients with each other or with Montgomery/Shenzhen?). Current
  default plan, not yet built: prefer each source's own official/published split (TBX11K, CheXpert,
  NIH ChestX-ray14, VinDr-CXR all have one) over inventing a new ratio; RSNA likely needs a custom
  carve since competition test-set labels aren't reliably usable outside the competition; TB
  Portals/Belarus get no split at all (100% external eval, TB-positive only, per the proposal).

## Step 6 — Normalization statistics + final QC pass ✅ DONE (Montgomery + Shenzhen)
**6a. Normalization** (`normalization.py`, `scripts/compute_normalization.py`): mean/std over the
Shenzhen **train split only** (463 = 235 TB + 228 normal). Montgomery is excluded as
external_test, and val/test are excluded. Written to `manifests/normalization_stats.json`, at
512 and at 256-downsampled-from-512 (the dataloader path). Values are in `CLAUDE.md` decision #17.
These are provisional until other sources join training.

**6b. Verification** (`scripts/verify_outputs.py`): 23 checks covering counts, re-hashing all 2,452
files, untracked outputs, mode and size, binary masks, split integrity, flagged-empty vs
lost-to-downsampling lesions, and non-empty lung masks. **All pass.**
- The first run reported "330 lesion masks lost". That was a bug in the check: Windows backslash
  paths were compared against forward-slash keys. The same bug had let the lung-mask check pass
  without finding any files. Both were fixed, with guards that fail if a check finds nothing to
  check. The bug also meant `file_hashes.csv` had been written with backslash paths (it would
  break on Linux); the pipeline now writes forward slashes and the manifest was regenerated.

**6c. Overlay QC** (`scripts/qc_overlays.py` → `../data_processed/_qc/step6_overlays/`), which I
reviewed visually. The Shenzhen sheet deliberately includes the 8 smallest lesions, plus random TB
and normal cases:
- Masks align with anatomy after pad and resize. Montgomery red (patient-right) sits under "R" on
  all 16 sampled images.
- The smallest lesions (14–93 px at 512) survive. None are lost at 512, and none are lost at 256
  either (checked separately with the dataloader's bicubic downsample and rebinarize).
- Some lesions lie outside the lung field (pleural findings), and 96 images have no lung mask.
  Both are logged in `CLAUDE.md` Section 5 as inputs to the lesion-report design; they're not
  preprocessing defects.

## Sequencing / what I need from you
- Steps 0–5 are done for Montgomery + Shenzhen (see status line at the top).
- `MCUCXR_0251_1`: resolved — excluded from external-test metrics, not from processing (decision #12).
- Crop-to-content: resolved — no crop.
- Shenzhen-normal split ratio: resolved — 228/33/65, matching the TB split's own ratio exactly
  rather than an independently invented number (decision #15).
- Step 6: done.
- **Still open**: all steps for the other sources (on hold until you say go), and two
  lesion-report design questions from Step 6 QC (`CLAUDE.md` Section 5): pleural lesions outside
  the lung field, and the 96 images without lung masks.
