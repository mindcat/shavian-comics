"""Typesetting Shavian text into a region."""
from functools import lru_cache

import numpy as np
from PIL import ImageDraw, ImageFont
from scipy import ndimage

from . import config

MIN_FONT_SIZE = 8
TEXT_PADDING = 3        # px between drawn text and the region edge
TALL_LETTERS = "𐑐𐑑𐑒𐑓𐑔𐑕𐑖𐑗𐑘𐑙"  # Shavian "tall" letters: their height plays the role of a capital's


def font_path():
    return str(config.FONT_DIR / config.FONT)


@lru_cache(maxsize=None)
def font(size):
    return ImageFont.truetype(font_path(), size)


@lru_cache(maxsize=None)
def tall_letter_ratio():
    """Height of a Shavian tall letter (baseline to top) as a fraction of font size."""
    f = font(100)
    tops = [f.getbbox(ch, anchor="ls")[1] for ch in TALL_LETTERS]
    return -float(np.median(tops)) / 100


def size_for_letter_height(letter_height):
    """Font size whose tall letters match the original lettering's cap height."""
    return max(MIN_FONT_SIZE, int(round(letter_height / tall_letter_ratio())))


def wrap_words(words, f, max_width, draw):
    lines, current = [], ""
    for word in words:
        test = f"{current} {word}" if current else word
        if draw.textlength(test, font=f) <= max_width:
            current = test
        elif not current or draw.textlength(word, font=f) > max_width:
            return None
        else:
            lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines


def _fits(draw, f, lines, center, fit_mask):
    box = draw.multiline_textbbox(center, "\n".join(lines), font=f, anchor="mm", align="center")
    x1, y1, x2, y2 = (int(round(v)) for v in box)
    h, w = fit_mask.shape
    if x1 < 0 or y1 < 0 or x2 > w or y2 > h:
        return False
    line_h = (y2 - y1) / len(lines)
    for i, line in enumerate(lines):
        lw = draw.textlength(line, font=f)
        lx1, lx2 = int(center[0] - lw / 2), int(np.ceil(center[0] + lw / 2))
        ly1, ly2 = int(y1 + i * line_h), int(np.ceil(y1 + (i + 1) * line_h))
        if not fit_mask[ly1:ly2, max(lx1, 0):lx2].all():
            return False
    return True


def fit_text(draw, text, area, max_size=None):
    """
    Largest font size <= max_size (and its wrapping) at which the text fits inside
    `area`. Returns (font, lines, center) or None.
    """
    words = text.split()
    fit_mask = ndimage.binary_erosion(area, iterations=TEXT_PADDING)
    if not words or not fit_mask.any():
        return None

    dist = ndimage.distance_transform_edt(fit_mask)
    deep_y, deep_x = np.unravel_index(np.argmax(dist), dist.shape)
    cy, cx = ndimage.center_of_mass(fit_mask)
    centers = [(float(deep_x), float(deep_y))]
    if fit_mask[int(cy), int(cx)]:
        centers.insert(0, (float(cx), float(cy)))

    ys, xs = np.nonzero(fit_mask)
    max_width = np.ptp(xs)
    hi = max(MIN_FONT_SIZE, int(np.ptp(ys) * 0.5))
    if max_size:
        hi = min(hi, max_size)

    def try_size(size):
        f = font(size)
        for frac in np.linspace(0.35, 1.0, 14):
            lines = wrap_words(words, f, max_width * frac, draw)
            if lines:
                for c in centers:
                    if _fits(draw, f, lines, c, fit_mask):
                        return f, lines, c
        return None

    # Try the cap first: with "match" sizing it usually fits outright
    best = try_size(hi)
    if best:
        return best
    lo, hi = MIN_FONT_SIZE, hi - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        found = try_size(mid)
        if found:
            best, lo = found, mid + 1
        else:
            hi = mid - 1
    return best


def draw_text(image, region, text, size_mode="match"):
    """Draws `text` into the region. size_mode: "fit" (as big as fits) or "match" (original size)."""
    draw = ImageDraw.Draw(image)
    max_size = None
    if size_mode == "match" and region.letter_height:
        max_size = size_for_letter_height(region.letter_height)
    fitted = fit_text(draw, text, region.layout_mask, max_size)
    if fitted is None:
        f = font(MIN_FONT_SIZE)
        cy, cx = ndimage.center_of_mass(region.layout_mask)
        lines, center = [text], (cx, cy)
        note = "didn't fit"
    else:
        f, lines, center = fitted
        note = f"{f.size}px" + (f" (target {max_size}px)" if max_size else "")
    draw.multiline_text(center, "\n".join(lines), font=f, fill=config.TEXT_COLOR, anchor="mm", align="center")
    return note
