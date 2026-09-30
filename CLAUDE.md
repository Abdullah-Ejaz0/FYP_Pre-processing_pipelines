# LucidCXR — Data Preprocessing Pipeline

This repo is the **preprocessing stage** of LucidCXR, a Final Year Project (FAST-NUCES Lahore,
Fall 2026). It does not train models — it turns raw, heterogeneous chest X-ray releases into a
clean, patient-split, hash-logged set of images + masks + label tables that the modeling repo
consumes.

## 1. Proposal summary (why this pipeline exists)

LucidCXR is a semi-supervised, multi-task chest X-ray framework for **TB, pneumonia, and pleural
effusion**, built around one idea: match localization effort to what the ground truth actually
supports.

- **TB** → pixel-level U-Net segmentation, trained on radiologist-annotated Shenzhen lesion masks.
- **Pneumonia** → bounding-box regression, trained on RSNA radiologist boxes.
- **Pleural effusion** → classification only (out of scope for localization).
- Training uses **Mean Teacher semi-supervised learning** (confidence-gated, dual-level
  consistency) so the small number of labeled TB masks (336 in Shenzhen) can be supplemented with
  a large unlabeled pool (VinDr-CXR train split).
- TB explainability fuses Grad-CAM++ with the trained segmentation mask
  (`fused = min(seg, cam)`), plus an automatic structured lesion report (lung / zone / %
  involvement) derived from the segmentation mask's geometry.
- Evaluation targets the WHO Target Product Profile threshold (90% sensitivity, specificity as
  the discriminating metric), with calibration (ECE), an annotation-scarcity ablation
  (10/25/50/75/100% of TB masks), and a source-bias/shortcut audit across datasets.

Full detail lives in `FYP_Proposal_Detailed.docx.pdf` (not in this repo) — Chapters 4–6 are the
ones this pipeline directly serves. **This repo currently only has Montgomery and Shenzhen on
disk.** RSNA, CheXpert, NIH ChestX-ray14, VinDr-CXR, TB Portals SIFT, and Belarus are not yet
present; the pipeline is structured so adding a new source is a new `sources/<name>.py` module,
not a rewrite.

## 2. Dataset layout (as currently on disk)

Raw data lives **outside this repo**, as sibling folders to it:

```
Data Preprocessing/                          <- parent folder, NOT the repo
├── FYP_Pre-processing_pipelines/            <- THIS repo
├── Montgomery-County-CXR-Set/
└── Shenzhen-Hospital-CXR-Set/
```

Never move or copy the raw folders into the repo. The repo reads them via a relative path
(`../Montgomery-County-CXR-Set`, `../Shenzhen-Hospital-CXR-Set`) configured once in
`configs/paths.yaml`, and writes all output elsewhere (Section 5).

### 2.1 Montgomery County CXR Set

```
Montgomery-County-CXR-Set/
├── aria2_download_list.txt
└── MontgomerySet/
    ├── CXR_png/                  138 PNGs, 8-bit grayscale ('L' mode already)
    ├── ClinicalReadings/         138 .txt, free-text (sex, age, diagnosis)
    ├── ManualMask/
    │   ├── leftMask/             138 PNGs, 1-bit, left-lung field
    │   └── rightMask/            138 PNGs, 1-bit, right-lung field
    ├── montgomery_consensus_roi.csv   TB bounding boxes (109 rows, subset of TB-positive cases)
    └── NLM-MontgomeryCXRSet-ReadMe.pdf
```

- **Patient ID / label**: filename `MCUCXR_{patient_id}_{label}.png` — `label` is `0` = normal,
  `1` = active TB. Confirmed count: 80 normal, 58 TB (matches proposal, Section 4.3.1).
  **Correction: filename IDs are NOT all distinct real patients** — the free-text diagnosis
  explicitly cross-references repeat scans: `{0113, 0117}` are the same 85-year-old (2 scans, 6
  months apart) and `{0162, 0166, 0170}` are the same 49-year-old TB patient (3 serial
  treatment-monitoring scans). So 138 filenames = **135 distinct real patients**, and the 58
  "TB images" are 55 unique TB patients, not 58. Found by reading the diagnosis text for
  "same pt" cross-references (`clinical_text` column), not by ID collision — ID collision alone
  would have missed this, since every filename ID is still unique. Since Montgomery is never
  split (100% `external_test`, decision #2), this can't cause train/test leakage — but it does
  mean any confidence interval computed on Montgomery by treating each image as an independent
  sample (proposal Section 6.3) is slightly optimistic; a patient-clustered CI would be more
  honest. Not fixed in preprocessing; flagged for whoever writes the evaluation code.
- **Lung masks**: `leftMask`/`rightMask` cover **all 138 images** (not a subset) — this is the
  lung-field ground truth Montgomery contributes (classification + external calibration test
  set per proposal Section 6.3; Montgomery is *excluded* from segmentation training).
- **Two canvas sizes, NOT a rotation artifact (visually verified in Step 2)**: images come in
  exactly two sizes, `(4020, 4892)` (97 images) and `(4892, 4020)` (41 images). The contact sheets
  (`../data_processed/_qc/montgomery_orientation/`) show all 138 are **upright PA chests** — the 41
  wide ones were simply captured on a landscape-oriented detector, with the chest upright inside
  it. **No rotation is applied.** Pad-to-square handles both sizes. (The earlier hypothesis that
  these were rotated was wrong; don't reintroduce a rotation step.)
- **Mask naming is image-side, not patient-side (visually verified)**: `leftMask` is the lung on
  the **image's left**, which is under the "R" marker — i.e. the **patient's RIGHT lung**.
  `rightMask` is the **patient's LEFT lung**. Anything that reports "affected lung" (the
  structured lesion report, proposal Section 4.6) must use patient-side names:
  `leftMask → patient_right`, `rightMask → patient_left`. Getting this wrong silently flips every
  left/right field in the report.
- **Non-standard view**: `MCUCXR_0251_1` is labelled "LT APICAL" (an apical lordotic view), not a
  standard PA. Kept in the manifest but flagged; see Section 5.
- **Content doesn't always fill the canvas**: several images (e.g. `MCUCXR_0017_0`, `0042_0`,
  `0060_0`, `0061_0`, `0077_0`, `0080_0`) have large black borders, so after pad+resize to 256 the
  lungs occupy a small part of the frame. Whether to crop to content first is an open decision
  (Section 5).
- `montgomery_consensus_roi.csv` has bounding boxes for a subset of TB-positive images
  (`x_dis, y_dis, width_dis, height_dis, Labels=TB`). Not used by the proposal's Montgomery role
  (classification/calibration only), kept but not required for the current pipeline.

### 2.2 Shenzhen Hospital CXR Set

```
Shenzhen-Hospital-CXR-Set/
├── CXR_png/                       662 PNGs, 8-bit, mode NOT uniform — must .convert('L')
├── ClinicalReadings/               662 .txt, free-text (sex, age, diagnosis)
├── shenzhen_consensus_roi.csv       TB bounding boxes (250 rows, subset of TB-positive cases)
├── Annotations/
│   ├── Annotations_json/
│   │   ├── Annotations_AllinOne_json.json
│   │   └── SeparateFiles/*.json    336 files, one per TB-positive image, VIA polygon format
│   └── masks/                      1,153 per-abnormality PNG masks, 330 unique images
└── Annotations-2/
    ├── Annotations_json/...        (same structure)
    └── masks/                      2,177 per-abnormality PNG masks, 323 unique images
```

- **Patient ID / label**: filename `CHNCXR_{patient_id}_{label}.png`, same `0`/`1` convention.
  326 normal, 336 TB. One image per patient — checked for the same repeat-scan pattern found in
  Montgomery (Section 2.1) by searching all 662 `clinical_text` entries for "same pt"/"prior"/
  "previous" cross-references; found none. Shenzhen's clinical text is terse (sex/age/diagnosis
  codes only), so this is weaker evidence than Montgomery's narrative text gave for its own
  repeats — absence of a mention isn't proof of absence — but it's the best available signal.
- **Image mode is NOT uniform** — verified across the full 662-file set, not a sample: **635
  files are palette-mode (`'P'`)**, **27 are `'RGB'`** (an initial 15-file sample only caught
  `'P'` and missed this; the mode-based assertion in `sources/shenzhen.py` is what caught it).
  Neither is plain grayscale. Confirmed on every RGB file checked: R==G==B exactly, so
  `.convert('L')` is lossless for both variants — but the loader must accept and convert both
  modes explicitly, not assume `'P'` alone (Montgomery, by contrast, is uniformly `'L'`).
- **Resolution**: varies per image (not a fixed size), roughly square but not exactly
  (e.g. `2986×2992`, `2573×2917`, `1250×1136`) — genuine per-scan variation, not a rotation
  artifact (unlike Montgomery, width/height are always close, never an exact swap pair).
- **Lesion masks: `Annotations/` and `Annotations-2/`** — **not two independent reader sets**,
  contrary to what this doc originally said (corrected after checking the actual Yang et al.,
  2022 paper directly rather than guessing). The paper describes a single consensus workflow: a
  junior radiologist labels, a senior radiologist reviews, "consensus reached for all cases" —
  one final set, and its own stated count, "330 of 336 images show visible TB signs," matches
  `Annotations/` exactly. The official NLM page for this dataset lists only one `Annotations/`
  folder. So `Annotations-2/`'s provenance is *not* documented anywhere official; empirically
  (checked directly) it shares 317 images with `Annotations/` at a mean IoU of 0.36 (moderate
  overlap, never identical, never zero) — related to the same underlying work, most plausibly an
  earlier draft pass, but not a verified independent second reading. **Don't use it as an
  inter-rater-reliability signal** — that would overstate what's actually known about it.
  Each folder contains **one binary mask PNG per abnormality-type-per-image** (e.g.
  `CHNCXR_0327_1_Calcified_Nodule_2.png`), covering 330 images in `Annotations/` and 323 in
  `Annotations-2/`. Mask pixel dimensions match their source CXR exactly.
  The proposal's "336 radiologist-annotated pixel-level TB lesion masks" / "330 of 336 show
  visible TB signs" (Yang et al., 2022) means: **per image, union all abnormality masks from one
  annotation set into a single binary TB-lesion mask**; the 6 TB-positive images with no mask in
  a set have an empty (all-zero) reference mask, which is expected and must be preserved as
  "empty", never skipped or treated as missing data.
  Decision: use `Annotations/` (330 images, matching the paper's own published count) as the
  lesion-mask ground truth; `Annotations-2/` is kept on disk but not used for training or metrics.
- **Some abnormality types are pleural/mediastinal, not parenchymal** (found while investigating
  why TB lesion pixels sometimes fall outside the lung-field mask): Pleural_Effusion (74.5% of its
  own pixels outside lung field), Pleural_Thickening (56.9%), Apical_Thickening (49.0%), Adenopathy
  (32.7%) — vs. 0.4–11% for nodules/infiltrates/cavities, which are genuinely intra-parenchymal.
  This is anatomically expected, not a registration bug: the lung-field mask traces the aerated
  lung silhouette, and pleural/mediastinal findings sit outside it by definition, even though
  they're real TB manifestations the radiologist correctly flagged. Affects the structured
  report's "% lung field involved" field (Section 5), not the masks or training data themselves —
  those stay exactly as annotated; we don't get to overrule a radiologist's read.
- The JSON files are the polygon source-of-truth (VIA format: `all_points_x`/`all_points_y` per
  region, one region per abnormality instance) that the PNG masks were rasterized from. We work
  from the rasterized PNG masks, not the JSON, for v1 (simpler, already pixel-aligned); JSON is
  kept as a fallback if a mask/image size mismatch ever needs re-rasterizing.
- **Lung-field masks — added later, now present**: the proposal (Section 4.3.1) cites Shenzhen
  lung-field masks for 287 TB-positive + 279 normal images (Rajaraman et al., 2023). Originally
  absent from this download (flagged as a gap); since resolved — they now live at
  `mask/mask/CHNCXR_{patient_id}_{label}_mask.png`, one combined (both-lungs-together) binary
  mask per image, verified 279 normal + 287 TB = 566 files, pixel-size-matched to their CXR.
  Unlike Montgomery's masks, there's no left/right split and no union step needed — one file per
  image already. This closes the gap that previously made Montgomery the *only* lung-boundary
  source on hand (Section 4, decision revisited below): Montgomery stays external-test-only, and
  the auxiliary lung-field supervision the proposal wants now comes from Shenzhen itself, matching
  the proposal's original design.

### 2.2b Patient metadata (age, sex, diagnosis text)

`ClinicalReadings/*.txt` is free text, not structured — `src/lucidcxr_prep/metadata.py` extracts
only what's mechanically unambiguous (**sex**, **age**) and carries the **raw diagnosis narrative
forward untouched**. No clinical interpretation (e.g. "is this active or inactive TB") is derived
from it — see decision below. Output: `data_processed/metadata/{montgomery,shenzhen}_metadata.csv`
(patient-level data needed at train/eval time, so it lives next to images/masks, not in the
repo's `manifests/`; join by `patient_id` to the split manifests when needed, rather than
duplicating the split column, so the two can't drift out of sync).

Format is far less regular than it first looked — every one of the following was found by
actually running the parser over all 800 rows, not by inspecting a sample:
- **Montgomery sex is not always M/F**: one patient (`0080`, age 5, normal) is coded `'O'`.
  Passed through as-is; not guessed at or remapped.
- **Shenzhen age units vary**: 658 of 662 say "yrs", but 2 patients are given in **months** and 1
  in **days** (infants), and 1 has no unit at all. Treating the raw number as years by default
  would have mislabeled a 16-month-old as 16 years old. `age_value` + `age_unit` are stored
  separately; `age_years` is only populated when the unit is actually known (NaN for the 1
  unitless row, not guessed).
  **First bug caught by the sanity check, not by an exception**: an initial unit-normalization
  routine (`"yrs".rstrip("s")` → `"yr"`, which matched no dict key) silently turned 658 of 662
  rows' `age_years` into NaN with no error — caught only because the printed summary showed an
  implausible age range. Fixed by classifying units on their first letter instead of stripping
  suffixes, plus an explicit assertion that no more than 1 row may have unresolved `age_years`.
- **Shenzhen sex spelling**: `"femal"` (typo for "female") appears in the data, alongside
  no-space variants (`"female24yrs"`) and trailing commas (`"male ,"`). All handled.
- **Montgomery images aren't all distinct real patients** (found while reading diagnosis text,
  not from ID collisions): see the correction in Section 2.1. `diagnosis_text` is exactly where
  this surfaced (`"(same pt as MCUCXR_0162_1)"`), which is itself the argument for keeping the
  raw text rather than only structured fields.

**Decision: no derived "active/inactive TB" flag.** Montgomery's free text frequently mixes both
in one note (e.g. *"old inactive disease in RL and new active TB in LL"*) or hedges (*"?active"*).
Classifying that from keywords would be a clinical judgment call, not an engineering one — the
same principle already applied to the pleural-lesion masks above. `diagnosis_text` is preserved
verbatim for a human (or a later, deliberate NLP effort) to read directly.

### 2.3 Not yet present (future sources, referenced by the proposal)

RSNA Pneumonia (with DICOM boxes), CheXpert, NIH ChestX-ray14, TBX11K, VinDr-CXR, TB Portals
SIFT, Belarus. When one of these arrives, add `../<DatasetFolder>` next to the two existing raw
folders, add a `sources/<name>.py` module following the Montgomery/Shenzhen pattern, and extend
`configs/paths.yaml` — do not change the shared pipeline code for a new source unless the source
genuinely needs new logic (e.g. actual DICOM decoding, which neither current source needs).

## 3. Conventions

- **Python**: 3.12 (matches the installed interpreter; pin in `requirements.txt` /
  `pyproject.toml`). No notebooks committed with output cells — clear outputs before commit, or
  keep exploration in `.ipynb` under a gitignored `notebooks/scratch/`.
- **Core libraries**: `Pillow` (image I/O — bicubic resize, palette handling), `numpy`,
  `pandas` (label/manifest tables), `pydicom` (added once a DICOM source shows up — not needed
  yet), `pyyaml` (config), `tqdm`. Kept deliberately minimal; no torch/training deps in this repo.
- **Folder structure** (this repo):
  ```
  FYP_Pre-processing_pipelines/
  ├── CLAUDE.md
  ├── README.md
  ├── requirements.txt
  ├── configs/
  │   ├── paths.yaml          # raw-data locations, output root, resolutions
  │   └── sources.yaml        # per-source settings (label suffix map, mask rules)
  ├── src/lucidcxr_prep/
  │   ├── config.py           # loads configs/paths.yaml
  │   ├── io_utils.py         # mode-aware image loading, sha256
  │   ├── sources/
  │   │   ├── montgomery.py   # manifest builder for Montgomery
  │   │   └── shenzhen.py     # manifest builder for Shenzhen (lesion + lung masks)
  │   ├── masks.py            # mask union, patient-side renaming, empty-mask handling
  │   ├── transforms.py       # pad-to-square + bicubic resize (images and masks)
  │   ├── pipeline.py         # per-row orchestration: load -> pad -> resize -> save -> hash
  │   ├── splits.py           # seeded patient-level splits
  │   ├── normalization.py    # train-split mean/std
  │   └── metadata.py         # sex/age extraction from free-text ClinicalReadings
  ├── scripts/                # run in this order:
  │   ├── build_manifests.py       # Step 1
  │   ├── inspect_orientation.py   # Step 2 (Montgomery contact sheets)
  │   ├── run_preprocess.py        # Steps 3-4
  │   ├── build_splits.py          # Step 5 (pinned; --force to redraw)
  │   ├── compute_normalization.py # Step 6a
  │   ├── verify_outputs.py        # Step 6b -- acceptance check, run after any re-processing
  │   ├── qc_overlays.py           # Step 6c (overlay sheets for human review)
  │   └── build_metadata.py        # Step 7 -- writes to data_processed/metadata/, not manifests/
  └── tests/                  # pytest, 18 tests: manifests, transforms/masks, splits, stats
  ```
- **Output location**: processed data is **never** written into the raw folders or mixed with
  raw files. It goes into a sibling folder `../data_processed/<source>/<run_tag>/...` (outside
  git — large binaries), while manifests/split CSVs/hash logs (small, text) are committed under
  `manifests/` in this repo. This lets you `ls` and open sample PNGs in the new folder to eyeball
  quality before anything touches the original two folders.
- **Split rule (decided)**: all splits are **patient-level**, keyed on the numeric ID embedded in
  the filename (`MCUCXR_XXXX`, `CHNCXR_XXXX`). For Shenzhen this coincides with an image-level
  split (no repeat-patient evidence found — checked, see Section 2.2). **For Montgomery it does
  not**: 5 of the 138 filenames are repeat scans of 2 real patients (Section 2.1) — harmless only
  because Montgomery is never split (100% `external_test`). The split code groups by patient ID
  rather than assuming 1:1 regardless, so it doesn't silently break when a multi-image-per-patient
  source (e.g. VinDr-CXR) is added later, or if Montgomery's role ever changes.
- **Held-out-by-design, not by random split**: Montgomery is excluded from all TB segmentation
  training/validation and reserved as an external classification+calibration test set (per
  proposal Section 4.3.1/6.3) — this is a fixed role, not something the random split should
  reassign. Shenzhen's 336 TB images get their own internal train/val/test split for the
  segmentation head (235/34/67, per proposal) — the split code must expose a way to pin this
  exact partition (or regenerate it deterministically from a fixed seed) rather than re-drawing
  it every run.
- **Hashing/logging**: every file entering any split gets a content hash (sha256) logged before
  any training run, per proposal Section 5.3 — this repo produces that manifest, it doesn't wait
  for the modeling repo to do it.
- **Reproducibility**: all random operations (split seeding, scarcity-ablation subset selection)
  take an explicit `--seed`, default recorded in `configs/paths.yaml`, never a bare `random()` call.

## 4. Decisions already made

1. **Patient-level split**, keyed on filename-embedded numeric ID (Section 3).
2. **Montgomery is a fixed external test set** for classification/calibration — not part of the
   random train/val split, and never used for segmentation supervision (no lesion masks exist).
3. **Shenzhen `Annotations/` (330 images) is primary lesion-mask ground truth**; `Annotations-2/`
   is secondary/optional for an agreement check, not blocking v1.
4. **Work from rasterized mask PNGs, not the VIA JSON**, for the v1 pipeline — re-rasterizing
   from JSON is a documented fallback, not the default path.
5. **Per-image abnormality masks are unioned into one binary TB-lesion mask per image**, per
   annotation set — this union is computed once and cached, not redone per training run.
6. **Empty reference masks are preserved as empty**, never imputed, counted and reported
   separately (matches proposal's Dice-undefined handling, Section 6.3/6.4).
7. **8-bit single-channel PNG is the canonical stored format**, at **512×512 only**. 256×256 is
   *not* stored — the dataloader downsamples the 512 PNG at load time instead. This is a
   deliberate deviation from the proposal's literal "write both 256 and 512 from the original"
   (Section 5.2): a 512→256 bicubic downsample is a very close approximation of an
   original→256 resize, and halving stored files/disk matters more at this project's scale.
   If a training run ever shows the two differ meaningfully, revisit.
8. **Shenzhen `CXR_png` is a mix of palette-mode and RGB (635/27) and must be `.convert('L')`
   explicitly in both cases** — verified across the full file set (an early 15-file sample
   missed the RGB subset; don't trust small samples for mode assumptions again).
9. **No rotation for Montgomery.** The 97/41 size split was checked visually (Step 2) and all
   images are upright; the wide ones are landscape detector captures (Section 2.1).
10. Normalization statistics (mean/std) are computed from the **training split only**, applied in
    the data loader, not baked into stored files (proposal Section 5.2) — this repo computes and
    records those constants per source/split, it doesn't hardcode them.
11. **Montgomery lung masks are renamed to patient-side on output**: `leftMask` →
    `lung_patient_right`, `rightMask` → `lung_patient_left` (Section 2.1).
12. **`MCUCXR_0251_1` (apical lordotic view) is processed and kept, but flagged
    `exclude_from_external_test=True`** in `montgomery_processed.csv` — not silently dropped, just
    excluded from the headline external-test metrics since it's a different projection than every
    other image (Section 2.1, Plan Step 4).
13. **Gotcha, not a decision**: every manifest CSV's `patient_id` column must be read with
    `dtype={"patient_id": str}` — IDs are zero-padded (`"0001"`) and pandas will silently parse
    them as int (losing the padding) if you don't pin the dtype. Caught once already while
    manually spot-checking `montgomery_processed.csv`; the on-disk CSV was fine, the ad-hoc
    verification script wasn't.
14. **Montgomery is never used to train anything** — not TB, not lung-field segmentation. Once
    Shenzhen's own lung-field masks turned up (Section 2.2), the only reason under consideration
    for giving Montgomery a training role went away; it stays 100% `external_test`
    (`montgomery_split.csv`, all 138 rows).
15. **Shenzhen split, pinned**: TB images get the proposal's fixed 235/34/67 (train/val/test);
    normal images get 228/33/65 — the *same ratio* as the TB split (≈70/10/20), not an
    independently chosen number. Both drawn with `seed=42` via `splits.py`, written to
    `manifests/shenzhen_split.csv`. `scripts/build_splits.py` refuses to redraw an existing split
    file without `--force`, so this partition is fixed once made.
16. **Splits for the other five sources are deferred, and when built, default to each source's
    own official/published split rather than a hand-carved ratio** (TBX11K, CheXpert, NIH
    ChestX-ray14, VinDr-CXR already have one; RSNA likely needs a custom carve since its
    competition test labels aren't reliably usable outside the competition; TB Portals/Belarus get
    no split at all — 100% external eval, TB-positive only).
17. **Normalization stats** live in `manifests/normalization_stats.json`, computed from the
    **Shenzhen train split only** (463 images), pixel values in [0,1], zero padding included (the
    network sees the padding too). 512: mean 0.5809, std 0.2847; 256 (downsampled from the 512 PNG
    with Pillow bicubic, i.e. the dataloader path): mean 0.5810, std 0.2839. **Provisional**:
    recompute over the pooled train set once other sources join training.
18. **All paths in committed manifests use forward slashes** (`Path.as_posix()`), so they work on
    Linux too. The first version wrote Windows backslashes; that was caught by `verify_outputs.py`
    and fixed.
19. **`scripts/verify_outputs.py` is the acceptance check**: run it after any re-processing. It
    re-hashes every output, checks counts, formats, and splits, and checks that no lesion was lost
    to downsampling. It has guards so that no check can pass vacuously (one did at first).
20. **Patient metadata (sex, age) is extracted; clinical interpretation is not.** Only sex and age
    are pulled from `ClinicalReadings`; the diagnosis narrative is kept as raw text. No
    active/inactive-TB flag is derived — that's a clinical judgment call, matching the same
    principle already applied to the pleural lesion masks (Section 2.2). Output lives in
    `data_processed/metadata/`, not `manifests/`, since it's training/eval-time patient data, not
    a build artifact (Section 2.2b).
21. **Montgomery is 135 real patients, not 138** — 5 filenames are repeat scans of 2 patients,
    found via diagnosis-text cross-references while building the metadata extraction, not from
    any ID collision. Doesn't cause leakage (Montgomery is never split) but is corrected
    everywhere the old "one image per patient, no repeats" claim appeared (Section 2.1).

## 5. Open items / known gaps (for you, not silently worked around)

- ~~Shenzhen lung-field masks not on disk~~ — **resolved**, see Section 2.2.
- ~~`MCUCXR_0251_1` keep or exclude~~ — **resolved**, excluded from external-test metrics only
  (decision #12).
- ~~Crop-to-content~~ — **resolved**, no crop (decision, Plan Step 4).
- ~~Shenzhen lesion-mask side check~~ — **resolved (Step 6 QC)**: Shenzhen masks carry no
  left/right naming at all (they're polygons in image coordinates), so there's nothing to flip.
  Images follow the standard convention: "R" marker at image-left, "L" marker at image-right. So
  whoever builds the lesion report must map **image-left half → patient's RIGHT lung** for Shenzhen
  (the combined lung-field mask has to be split at the midline for this).
- **Lesions outside the lung field (Step 6 finding, affects the lesion report, not
  preprocessing)**: in the 282 TB images with both masks, on average 9.2% of lesion pixels fall
  *outside* the lung-field mask; 33 images have >25% outside, 10 have >50%. These look like pleural
  findings (effusion/thickening at the costophrenic angles, e.g. `0644`, `0440`, `0563`), which are
  outside the lung field by nature. The proposal's "% of lung field involved = lesion pixels / lung
  pixels" would silently miscount these. The report design needs to decide: intersect the lesion
  with the lung field, or report pleural involvement separately.
- **96 Shenzhen images have no lung-field mask** (49 TB, 47 normal — the source only covers 566 of
  662). They get no `lung_field` output. Lung-field supervision and the lesion report's
  denominator are unavailable for those images.
- Shenzhen includes some **AP** (not PA) views (e.g. normal `0248` is marked "ap"). Not flagged
  per-image; worth knowing if AP/PA shift ever shows up in results.
- `montgomery_consensus_roi.csv` / `shenzhen_consensus_roi.csv` (bounding boxes) exist but have no
  defined role in the current proposal scope (TB uses pixel segmentation, not boxes) — kept as-is,
  not processed, until/unless a future decision uses them.
- **Splits for the other five sources** (TBX11K, RSNA, CheXpert, NIH ChestX-ray14, VinDr-CXR,
  Belarus) are not yet decided — only Montgomery and Shenzhen are split so far (Section 4,
  decisions #14–16). The plan is to prefer each source's own official/published split where one
  exists, and only hand-carve a split where none does (see conversation record / Plan Step 5)
  — not yet implemented for those five.
