"""Pad-to-square + resize, applied identically to images and their masks.

Proposal Section 5.2: pad to square preserving aspect ratio (zero padding), then resize with
antialiased bicubic interpolation to each target resolution. Masks get the same pad+resize, then
re-binarized at 0.5 (Rajaraman et al., 2023 convention).
"""

from __future__ import annotations

from PIL import Image


def pad_to_square(im: Image.Image, fill: int = 0) -> Image.Image:
    """Center the image on a square black canvas sized to its longer side.

    Deterministic given only (width, height), so calling this separately on an image and on its
    mask (same original size) always produces identically-offset padding -- no shared state
    needs to be threaded through.
    """
    w, h = im.size
    side = max(w, h)
    if w == h:
        return im.copy()
    canvas = Image.new(im.mode, (side, side), fill)
    canvas.paste(im, ((side - w) // 2, (side - h) // 2))
    return canvas


def resize_image(im: Image.Image, size: int) -> Image.Image:
    """Bicubic resize for a grayscale intensity image."""
    return im.resize((size, size), Image.Resampling.BICUBIC)


def resize_mask(mask: Image.Image, size: int, threshold: int = 128) -> Image.Image:
    """Bicubic resize a binary mask, then re-binarize at `threshold` (default 0.5 of 255)."""
    resized = mask.convert("L").resize((size, size), Image.Resampling.BICUBIC)
    return resized.point(lambda p: 255 if p >= threshold else 0)


def pad_and_resize_image(im: Image.Image, size: int) -> Image.Image:
    return resize_image(pad_to_square(im, fill=0), size)


def pad_and_resize_mask(mask: Image.Image, size: int) -> Image.Image:
    return resize_mask(pad_to_square(mask, fill=0), size)
