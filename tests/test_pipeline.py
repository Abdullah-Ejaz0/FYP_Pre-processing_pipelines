"""Unit tests for transforms.py and masks.py -- no raw data needed, synthetic images only."""

import numpy as np
import pytest
from PIL import Image

from lucidcxr_prep.masks import union_masks
from lucidcxr_prep.transforms import pad_and_resize_image, pad_and_resize_mask, pad_to_square


def test_pad_to_square_centers_content():
    im = Image.new("L", (100, 50), 0)
    im.putpixel((50, 25), 255)  # center of the original image
    padded = pad_to_square(im, fill=0)

    assert padded.size == (100, 100)
    arr = np.array(padded)
    ys, xs = np.nonzero(arr)
    # the bright pixel should land at the padded image's center too
    assert (xs[0], ys[0]) == (50, 50)


def test_pad_to_square_noop_on_already_square():
    im = Image.new("L", (64, 64), 128)
    assert pad_to_square(im).size == (64, 64)


def test_pad_and_resize_image_and_mask_agree_on_geometry():
    # A mask matching an asymmetric image should land in the same relative position after
    # identical pad+resize -- this is the alignment guarantee the whole pipeline depends on.
    img = Image.new("L", (200, 100), 50)
    mask = Image.new("L", (200, 100), 0)
    mask.paste(255, (150, 20, 200, 70))  # a block in the right portion of the image

    out_img = pad_and_resize_image(img, 64)
    out_mask = pad_and_resize_mask(mask, 64)

    assert out_img.size == (64, 64)
    assert out_mask.size == (64, 64)
    mask_arr = np.array(out_mask)
    ys, xs = np.nonzero(mask_arr)
    assert xs.mean() > 32  # still on the right half after pad+resize


def test_union_masks_empty_list_is_all_zero(tmp_path):
    out = union_masks([], size=(10, 20))
    assert out.size == (10, 20)
    assert np.array(out).sum() == 0


def test_union_masks_logical_or(tmp_path):
    m1 = Image.new("L", (10, 10), 0)
    m1.paste(255, (0, 0, 5, 5))
    m2 = Image.new("L", (10, 10), 0)
    m2.paste(255, (5, 5, 10, 10))
    p1, p2 = tmp_path / "m1.png", tmp_path / "m2.png"
    m1.save(p1)
    m2.save(p2)

    out = np.array(union_masks([p1, p2], size=(10, 10)))
    assert out[0:5, 0:5].all()
    assert out[5:10, 5:10].all()
    assert not out[0:5, 5:10].any()


def test_union_masks_size_mismatch_raises(tmp_path):
    m = Image.new("L", (10, 10), 255)
    p = tmp_path / "m.png"
    m.save(p)
    with pytest.raises(ValueError):
        union_masks([p], size=(20, 20))


def test_normalization_stats_known_values(tmp_path):
    from lucidcxr_prep.normalization import compute_stats

    # half black, half white -> mean 0.5, std 0.5, at both resolutions
    arr = np.zeros((512, 512), dtype=np.uint8)
    arr[:, 256:] = 255
    p = tmp_path / "img.png"
    Image.fromarray(arr).save(p)

    stats = compute_stats([p])
    assert stats["512"]["mean"] == pytest.approx(0.5)
    assert stats["512"]["std"] == pytest.approx(0.5)
    assert stats["256_from_512_bicubic"]["mean"] == pytest.approx(0.5, abs=1e-3)


def test_normalization_refuses_empty():
    from lucidcxr_prep.normalization import compute_stats

    with pytest.raises(ValueError):
        compute_stats([])
