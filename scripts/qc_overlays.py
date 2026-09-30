"""Step 6c: overlay contact sheets of the processed 512 outputs, for human review.

Shenzhen: green = lung_field, yellow = tb_lesion (primary). Sheet 1 is the 8 smallest non-empty
lesions (most at risk from downsampling) + 8 random TB + 8 random normal.
Montgomery: red = lung_patient_right, blue = lung_patient_left, 16 random. On a correct image
the red lung sits under the "R" marker.

Output: <output_root>/_qc/step6_overlays/*.png

Usage:
    python scripts/qc_overlays.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lucidcxr_prep.config import load_paths  # noqa: E402

THUMB, LABEL_H, COLS = 256, 18, 8


def overlay(img_path: Path, layers: list[tuple[Path, tuple[int, int, int], float]]) -> Image.Image:
    rgb = np.repeat(np.asarray(Image.open(img_path).convert("L"), dtype=np.float32)[..., None], 3, 2)
    for mask_path, color, alpha in layers:
        if mask_path.exists():
            m = np.asarray(Image.open(mask_path)) > 0
            rgb[m] = (1 - alpha) * rgb[m] + alpha * np.array(color, dtype=np.float32)
    return Image.fromarray(rgb.astype(np.uint8)).resize((THUMB, THUMB), Image.Resampling.BILINEAR)


def sheet(tiles: list[tuple[Image.Image, str]], out: Path) -> None:
    rows = -(-len(tiles) // COLS)
    canvas = Image.new("RGB", (COLS * THUMB, rows * (THUMB + LABEL_H)), (40, 40, 40))
    for i, (tile, label) in enumerate(tiles):
        x, y = (i % COLS) * THUMB, (i // COLS) * (THUMB + LABEL_H)
        canvas.paste(tile, (x, y))
        ImageDraw.Draw(canvas).text((x + 4, y + THUMB + 2), label, fill=(255, 255, 0))
    canvas.save(out)
    print(f"written -> {out}")


def main() -> None:
    paths = load_paths()
    root, res = paths.output_root, 512
    out_dir = root / "_qc" / "step6_overlays"
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(paths.seed)

    split = pd.read_csv(paths.manifests_dir / "shenzhen_split.csv", dtype={"patient_id": str})
    split_of = dict(zip(split["patient_id"], split["split"]))

    def lesion_px(pid: str) -> int:
        return int((np.asarray(Image.open(root / f"masks/tb_lesion/{res}/shenzhen/{pid}.png")) > 0).sum())

    tb_ids = split.loc[split["label"] == "TB", "patient_id"].tolist()
    sizes = {pid: lesion_px(pid) for pid in tb_ids}
    smallest = sorted((p for p in tb_ids if sizes[p] > 0), key=sizes.get)[:8]
    random_tb = list(rng.choice([p for p in tb_ids if p not in smallest], 8, replace=False))
    normals = list(rng.choice(split.loc[split["label"] == "normal", "patient_id"], 8, replace=False))

    tiles = []
    for group, ids in (("small", smallest), ("TB", random_tb), ("normal", normals)):
        for pid in ids:
            tile = overlay(
                root / f"images/{res}/shenzhen/{pid}.png",
                [(root / f"masks/lung_field/{res}/shenzhen/{pid}.png", (60, 220, 60), 0.25),
                 (root / f"masks/tb_lesion/{res}/shenzhen/{pid}.png", (255, 220, 0), 0.6)],
            )
            px = f" {sizes[pid]}px" if pid in sizes else ""
            tiles.append((tile, f"{group} {pid} {split_of[pid]}{px}"))
    sheet(tiles, out_dir / "shenzhen_overlays.png")

    mont = pd.read_csv(paths.manifests_dir / "montgomery_split.csv", dtype={"patient_id": str})
    tiles = []
    for pid in rng.choice(mont["patient_id"], 16, replace=False):
        tile = overlay(
            root / f"images/{res}/montgomery/{pid}.png",
            [(root / f"masks/lung_patient_right/{res}/montgomery/{pid}.png", (255, 60, 60), 0.35),
             (root / f"masks/lung_patient_left/{res}/montgomery/{pid}.png", (60, 120, 255), 0.35)],
        )
        tiles.append((tile, f"MCU {pid} red=pt-right"))
    sheet(tiles, out_dir / "montgomery_overlays.png")


if __name__ == "__main__":
    main()
