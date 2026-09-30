"""Step 2: contact sheets for the Montgomery orientation question.

Montgomery has 97 images at (4020, 4892) and 41 at (4892, 4020). This renders every image as a
thumbnail with its lung masks overlaid (leftMask = red, rightMask = blue), grouped into
"tall" and "wide" sheets, so a human can confirm whether the wide ones are rotated chests.

Read-only on the raw data. Output: <output_root>/_qc/montgomery_orientation/*.png

Usage:
    python scripts/inspect_orientation.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lucidcxr_prep.config import load_paths  # noqa: E402
from lucidcxr_prep.io_utils import load_grayscale, load_mask  # noqa: E402

THUMB = 256
LABEL_H = 18
COLS = 8
ROWS_PER_PAGE = 6
LEFT_COLOR = np.array([255, 60, 60], dtype=np.float32)
RIGHT_COLOR = np.array([60, 120, 255], dtype=np.float32)
ALPHA = 0.35


def make_thumb(row: pd.Series) -> Image.Image:
    img = load_grayscale(Path(row["image_path"]), "montgomery")
    left = load_mask(Path(row["left_mask_path"])).convert("L")
    right = load_mask(Path(row["right_mask_path"])).convert("L")

    scale = THUMB / max(img.size)
    size = (max(1, round(img.width * scale)), max(1, round(img.height * scale)))
    img = img.resize(size, Image.Resampling.BILINEAR)
    left = np.array(left.resize(size, Image.Resampling.NEAREST)) > 0
    right = np.array(right.resize(size, Image.Resampling.NEAREST)) > 0

    rgb = np.repeat(np.array(img, dtype=np.float32)[..., None], 3, axis=2)
    rgb[left] = (1 - ALPHA) * rgb[left] + ALPHA * LEFT_COLOR
    rgb[right] = (1 - ALPHA) * rgb[right] + ALPHA * RIGHT_COLOR

    tile = Image.new("RGB", (THUMB, THUMB + LABEL_H), (0, 0, 0))
    tile.paste(Image.fromarray(rgb.astype(np.uint8)), ((THUMB - size[0]) // 2, (THUMB - size[1]) // 2))
    draw = ImageDraw.Draw(tile)
    draw.text((4, THUMB + 2), f"{row['filename']} {row['width']}x{row['height']}", fill=(255, 255, 0))
    return tile


def write_sheets(df: pd.DataFrame, tag: str, out_dir: Path) -> list[Path]:
    per_page = COLS * ROWS_PER_PAGE
    written = []
    for page_idx, start in enumerate(range(0, len(df), per_page)):
        chunk = df.iloc[start : start + per_page]
        rows = -(-len(chunk) // COLS)
        sheet = Image.new("RGB", (COLS * THUMB, rows * (THUMB + LABEL_H)), (40, 40, 40))
        for i, (_, row) in enumerate(tqdm(chunk.iterrows(), total=len(chunk), desc=f"{tag} p{page_idx + 1}")):
            sheet.paste(make_thumb(row), ((i % COLS) * THUMB, (i // COLS) * (THUMB + LABEL_H)))
        path = out_dir / f"{tag}_page{page_idx + 1}.png"
        sheet.save(path)
        written.append(path)
    return written


def main() -> None:
    paths = load_paths()
    manifest = paths.manifests_dir / "montgomery_raw.csv"
    if not manifest.exists():
        raise FileNotFoundError(f"{manifest} missing -- run scripts/build_manifests.py first")
    df = pd.read_csv(manifest, dtype={"patient_id": str})

    out_dir = paths.output_root / "_qc" / "montgomery_orientation"
    out_dir.mkdir(parents=True, exist_ok=True)

    wide = df[df["is_landscape"]]
    tall = df[~df["is_landscape"]]
    written = write_sheets(wide, f"wide_{len(wide)}", out_dir)
    written += write_sheets(tall, f"tall_{len(tall)}", out_dir)

    print("\nLegend: red = leftMask, blue = rightMask (as named in the raw folders)")
    for p in written:
        print(f"written -> {p}")


if __name__ == "__main__":
    main()
