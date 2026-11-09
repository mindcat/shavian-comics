"""A Region is one piece of text to replace, plus the masks the later stages need."""
from collections import Counter
from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage


@dataclass
class Region:
    kind: str              # "bubble" (enclosed) or "free" (text on artwork / panel background)
    box: tuple             # (x1, y1, x2, y2) of the text (or bubble, for YOLO)
    layout_mask: np.ndarray  # where the new text may be drawn
    erase_mask: np.ndarray   # what the flat "fill" eraser paints over
    text_mask: np.ndarray    # the original lettering's pixels (inpainting mask, size estimate)
    bg: tuple              # background colour (RGB)
    letter_height: float   # original lettering cap height in px (0 if unknown)
    score: float = 1.0
    note: str = ""
    # filled in by later stages
    ocr: dict = field(default_factory=dict)       # backend -> English text
    shavian: dict = field(default_factory=dict)   # backend -> Shavian text


def mask_bbox(mask):
    ys, xs = np.nonzero(mask)
    if not len(ys):
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def rect_mask(shape, box, pad=0):
    m = np.zeros(shape[:2], bool)
    x1, y1, x2, y2 = box
    m[max(0, int(y1) - pad):max(0, int(y2) + pad), max(0, int(x1) - pad):max(0, int(x2) + pad)] = True
    return m


def dominant_colour(pixels):
    """Most common colour, quantised a little so JPEG noise doesn't split it."""
    q = (pixels // 8) * 8 + 4
    return tuple(int(v) for v in Counter(map(tuple, q)).most_common(1)[0][0])


def colour_region(image_rgb, seed_mask, bg=None, tol=40, margin=None):
    """
    The connected region of bg-coloured pixels that overlaps seed_mask most,
    with its holes (the letters, if it's a bubble) filled in.

    Returns (filled_region, raw_bg_region, bg_colour, enclosed) where the regions
    are full-size bool arrays and `enclosed` says the region didn't run into the
    edge of the search window (i.e. it's closed off within `margin` px), or None
    if there's no such region.
    """
    if bg is None:
        bg = dominant_colour(image_rgb[seed_mask])
    bg_arr = np.array(bg, dtype=np.int16)

    box = mask_bbox(seed_mask)
    h, w = seed_mask.shape
    if margin is None:
        margin = max(h, w)  # whole image
    x1, y1 = max(0, box[0] - margin), max(0, box[1] - margin)
    x2, y2 = min(w, box[2] + margin), min(h, box[3] + margin)

    crop = image_rgb[y1:y2, x1:x2].astype(np.int16)
    is_bg = np.abs(crop - bg_arr).max(axis=2) <= tol
    labels, n = ndimage.label(is_bg)
    if not n:
        return None
    overlap = ndimage.sum(seed_mask[y1:y2, x1:x2], labels, index=range(1, n + 1))
    best = int(np.argmax(overlap)) + 1
    if overlap[best - 1] == 0:
        return None
    raw = labels == best
    filled = ndimage.binary_fill_holes(raw)

    full_raw = np.zeros_like(seed_mask)
    full_raw[y1:y2, x1:x2] = raw
    full_filled = np.zeros_like(seed_mask)
    full_filled[y1:y2, x1:x2] = filled
    # Touching the window edge (where that isn't the image edge) means it's not closed off
    enclosed = not (
        (y1 > 0 and raw[0].any()) or (y2 < h and raw[-1].any())
        or (x1 > 0 and raw[:, 0].any()) or (x2 < w and raw[:, -1].any())
    )
    return full_filled, full_raw, bg, enclosed


def ink_mask(image_rgb, area_mask, bg, tol=60):
    """Pixels inside area_mask that clearly differ from the background colour."""
    diff = np.abs(image_rgb.astype(np.int16) - np.array(bg, dtype=np.int16)).max(axis=2)
    return area_mask & (diff > tol)


def letter_height(text_mask):
    """Typical letter height of the lettering in text_mask (comics are mostly caps)."""
    labels, n = ndimage.label(text_mask)
    if not n:
        return 0.0
    heights = []
    for sl in ndimage.find_objects(labels):
        hh, ww = sl[0].stop - sl[0].start, sl[1].stop - sl[1].start
        if hh * ww >= 12 and hh >= 4:
            heights.append(hh)
    if not heights:
        return 0.0
    # Upper-middle of the distribution: letters, not punctuation; joined letters don't change height
    return float(np.percentile(heights, 70))
