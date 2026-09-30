"""Source-aware image I/O.

Montgomery CXR_png files are already 'L' (8-bit grayscale). Shenzhen CXR_png files are NOT
uniformly one mode — verified on the full 662-file set, not a sample: 635 are palette-mode
('P') and 27 are 'RGB' (with R==G==B exactly on every sampled file, i.e. genuinely grayscale
content stored as RGB). Both must be explicitly `.convert('L')`'d; a loader that assumes only
'P' shows up will crash or silently mis-handle the 27 RGB files.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from PIL import Image


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    """Content hash of a file already written to disk (proposal Section 5.3: hash-log every
    split member before any training run)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()

# Raw, on-disk PIL modes per source, as verified by direct inspection (CLAUDE.md Section 2).
# Montgomery is uniformly 'L'. Shenzhen is a mix of 'P' (635) and 'RGB' (27) -- both are
# converted to 'L' the same way, so no per-mode branching is needed beyond the source switch.
EXPECTED_RAW_MODES = {
    "montgomery": {"L"},
    "shenzhen": {"P", "RGB"},
}


def read_raw_image_info(path: Path) -> dict:
    """Read only image header info (size, mode) without decoding full pixel data."""
    with Image.open(path) as im:
        return {"width": im.size[0], "height": im.size[1], "mode": im.mode}


def load_grayscale(path: Path, source: str) -> Image.Image:
    """Load an image and return it as a single-channel 'L' PIL Image.

    `source` selects the conversion rule so the loader never has to guess:
    - montgomery: already 'L', loaded as-is.
    - shenzhen: palette-mode, explicitly `.convert('L')`'d.
    """
    with Image.open(path) as im:
        im.load()
        if source == "montgomery":
            if im.mode != "L":
                raise ValueError(
                    f"Expected Montgomery image to be mode 'L', got {im.mode!r} for {path}"
                )
            return im.copy()
        if source == "shenzhen":
            if im.mode not in EXPECTED_RAW_MODES["shenzhen"]:
                raise ValueError(
                    f"Unexpected Shenzhen mode {im.mode!r} for {path} "
                    f"(expected one of {EXPECTED_RAW_MODES['shenzhen']})"
                )
            return im.convert("L")
        raise ValueError(f"Unknown source {source!r}")


def load_mask(path: Path) -> Image.Image:
    """Load a mask PNG as-is (masks are 1-bit or 8-bit grayscale, never palette)."""
    with Image.open(path) as im:
        im.load()
        return im.copy()
