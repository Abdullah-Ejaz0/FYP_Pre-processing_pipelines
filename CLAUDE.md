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
  `1` = active TB. One image per patient (no repeats), so patient-level split == image-level
  split here. Confirmed count: 80 normal, 58 TB (matches proposal, Section 4.3.1).
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
  326 normal, 336 TB. One image per patient.
- **Image mode is NOT uniform** — verified across the full 662-file set, not a sample: **635
  files are palette-mode (`'P'`)**, **27 are `'RGB'`** (an initial 15-file sample only caught
  `'P'` and missed this; the mode-based assertion in `sources/shenzhen.py` is what caught it).
  Neither is plain grayscale. Confirmed on every RGB file checked: R==G==B exactly, so
  `.convert('L')` is lossless for both variants — but the loader must accept and convert both
  modes explicitly, not assume `'P'` alone (Montgomery, by contrast, is uniformly `'L'`).
- **Resolution**: varies per image (not a fixed size), roughly square but not exactly
  (e.g. `2986×2992`, `2573×2917`, `1250×1136`) — genuine per-scan variation, not a rotation
  artifact (unlike Montgomery, width/height are always close, never an exact swap pair).
- **Lesion masks = two independent annotation sets**, `Annotations/` and `Annotations-2/`
  (two annotators or two annotation passes). Each contains **one binary mask PNG per
  abnormality-type-per-image** (e.g. `CHNCXR_0327_1_Calcified_Nodule_2.png`,
  `CHNCXR_0327_1_Clustered_Nodule_(2mm-5mm_apart)_1.png`), covering 330 images in `Annotations/`
  and 323 in `Annotations-2/` (not identical image sets — verified via diff, ~13 images differ
  between the two). Mask pixel dimensions match their source CXR exactly.
  The proposal's "336 radiologist-annotated pixel-level TB lesion masks" / "330 of 336 show
  visible TB signs" (Yang et al., 2022) means: **per image, union all abnormality masks from one
  annotation set into a single binary TB-lesion mask**; the 6 TB-positive images with no mask in
  a set have an empty (all-zero) reference mask, which is expected and must be preserved as
  "empty", never skipped or treated as missing data.
  Decision: use `Annotations/` (330 images) as primary ground truth; `Annotations-2/` is
  available for an inter-annotator agreement check if time allows, not required for v1.
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
  │   └── normalization.py    # train-split mean/std
  ├── scripts/                # run in this order:
  │   ├── build_manifests.py       # Step 1
  │   ├── inspect_orientation.py   # Step 2 (Montgomery contact sheets)
  │   ├── run_preprocess.py        # Steps 3-4
  │   ├── build_splits.py          # Step 5 (pinned; --force to redraw)
  │   ├── compute_normalization.py # Step 6a
  │   ├── verify_outputs.py        # Step 6b -- acceptance check, run after any re-processing
  │   └── qc_overlays.py           # Step 6c (overlay sheets for human review)
  └── tests/                  # pytest, 18 tests: manifests, transforms/masks, splits, stats
  ```
- **Output location**: processed data is **never** written into the raw folders or mixed with
  raw files. It goes into a sibling folder `../data_processed/<source>/<run_tag>/...` (outside
  git — large binaries), while manifests/split CSVs/hash logs (small, text) are committed under
  `manifests/` in this repo. This lets you `ls` and open sample PNGs in the new folder to eyeball
  quality before anything touches the original two folders.
- **Split rule (decided)**: all splits are **patient-level**, keyed on the numeric ID embedded in
  the filename (`MCUCXR_XXXX`, `CHNCXR_XXXX`). For Montgomery and Shenzhen specifically this is
  equivalent to an image-level split (one image per patient, confirmed by ID uniqueness), but the
  split code always groups by patient ID rather than assuming 1:1, so it doesn't silently break
  when a multi-image-per-patient source (e.g. VinDr-CXR) is added later.
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
