"""
Pipeline B: comic-text-detector (CTD) finds the text itself rather than bubbles.

CTD gives a per-pixel text mask ("seg") and a text-line map ("det"). We:
  1. turn each line-map blob into a text line, using the text pixels it touches
     and splitting lines at wide horizontal gaps (two bubbles side by side);
  2. skip lettering that's much bigger than the page's normal text (logo, sound
     effects) or much smaller (signatures, copyright lines);
  3. find the bubble each line sits in, if any, by flood-filling its background
     colour; lines in the same bubble become one region, and lines that aren't
     in a closed-off bubble are grouped by proximity as "free" text.
"""
import cv2
import numpy as np
from scipy import ndimage

from . import models
from .regions import Region, colour_region, dominant_colour, letter_height, rect_mask

INPUT_SIZE = 1024
SEG_THRESHOLD = 0.4
DET_THRESHOLD = 0.3
SFX_RATIO = 2.2      # lines this many times taller than the page's typical line are sound effects / logos
SMALL_RATIO = 0.45   # ... and this much smaller are signatures, copyright lines
BUBBLE_SEARCH = 10   # look for a bubble's edge within this many line-heights
MAX_BUBBLE_AREA = 15 # a bubble is at most this many times its text's box area
MAX_NONTEXT_HOLES = 0.04  # a bubble encloses (almost) nothing but its text
MIN_SOLIDITY = 0.75  # bubbles are roughly convex (area / convex hull area); a panel background with a figure cut out isn't
MAX_HOLE_HEIGHT = 3.0  # nothing but lettering inside a bubble: no enclosed non-text shape taller than this many line heights
SNUG_MARGIN = 2.0      # a bubble hugs its text: beyond this many line heights around the text...
MAX_OUTSIDE = 0.4      # ...there's at most this fraction of the bubble (its tail); a whole panel has much more


def run_ctd(image_rgb):
    h, w = image_rgb.shape[:2]
    r = INPUT_SIZE / max(h, w)
    nh, nw = round(h * r), round(w * r)
    canvas = np.zeros((INPUT_SIZE, INPUT_SIZE, 3), np.uint8)
    canvas[:nh, :nw] = cv2.resize(image_rgb, (nw, nh), interpolation=cv2.INTER_AREA)
    x = canvas.transpose(2, 0, 1)[None].astype(np.float32) / 255
    _blk, seg, det = models.ctd().run(None, {"images": x})
    seg = cv2.resize(seg[0, 0, :nh, :nw], (w, h), interpolation=cv2.INTER_LINEAR)
    det = cv2.resize(det[0, 0, :nh, :nw], (w, h), interpolation=cv2.INTER_LINEAR)
    return seg, det


def find_lines(seg, det):
    """
    Text lines as (x1, y1, x2, y2) boxes. Each line-map blob is a thin core along
    a line; grow it up and down row by row while the rows still contain text
    pixels (stopping at the blank gap between lines), then split it where there's
    a wide horizontal gap (two bubbles' lines that the line map joined up).
    """
    ink = seg > SEG_THRESHOLD
    det_labels, _ = ndimage.label(det > DET_THRESHOLD)
    h, w = ink.shape
    lines = set()
    for sl in ndimage.find_objects(det_labels):
        cy1, cy2, cx1, cx2 = sl[0].start, sl[0].stop, sl[1].start, sl[1].stop
        ch = cy2 - cy1
        if ch < 2 or cx2 - cx1 < 3:
            continue
        grow = max(2, int(0.9 * ch))
        x1, x2 = max(0, cx1 - ch), min(w, cx2 + ch)
        y1, y2 = cy1, cy2
        for _ in range(grow):
            if y1 > 0 and ink[y1 - 1, x1:x2].any():
                y1 -= 1
            else:
                break
        for _ in range(grow):
            if y2 < h and ink[y2, x1:x2].any():
                y2 += 1
            else:
                break
        band = ink[y1:y2, x1:x2]
        cols = band.any(axis=0)
        if not cols.any():
            continue
        lh = y2 - y1
        # Split into runs of columns separated by gaps wider than ~1.2 line heights
        idx = np.nonzero(cols)[0]
        breaks = np.nonzero(np.diff(idx) > 1.2 * lh)[0]
        for run in np.split(idx, breaks + 1):
            rx1, rx2 = x1 + int(run[0]), x1 + int(run[-1]) + 1
            rows = np.nonzero(ink[y1:y2, rx1:rx2].any(axis=1))[0]
            if len(rows) and rx2 - rx1 >= 3:
                lines.add((rx1, y1 + int(rows[0]), rx2, y1 + int(rows[-1]) + 1))
    return sorted(lines, key=lambda b: (b[1], b[0]))


def solidity(mask):
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0.0
    hull = cv2.convexHull(np.vstack(contours))
    return float(mask.sum()) / max(cv2.contourArea(hull), 1.0)


def union_box(boxes):
    return (min(b[0] for b in boxes), min(b[1] for b in boxes), max(b[2] for b in boxes), max(b[3] for b in boxes))


def group_free_lines(lines):
    """Union lines that stack vertically, overlap horizontally and have similar heights."""
    parent = list(range(len(lines)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i, a in enumerate(lines):
        for j in range(i + 1, len(lines)):
            b = lines[j]
            ha, hb = a[3] - a[1], b[3] - b[1]
            vgap = max(a[1], b[1]) - min(a[3], b[3])
            hover = min(a[2], b[2]) - max(a[0], b[0])
            if vgap <= 0.8 * max(ha, hb) and hover > 0 and max(ha, hb) <= 1.6 * min(ha, hb):
                parent[find(i)] = find(j)
    groups = {}
    for i in range(len(lines)):
        groups.setdefault(find(i), []).append(lines[i])
    groups = list(groups.values())
    # Merge groups whose boxes overlap (the line map sometimes splits a line into staggered pieces)
    merged = True
    while merged:
        merged = False
        for i in range(len(groups)):
            for j in range(i + 1, len(groups)):
                a, b = union_box(groups[i]), union_box(groups[j])
                if min(a[2], b[2]) > max(a[0], b[0]) and min(a[3], b[3]) > max(a[1], b[1]):
                    groups[i] += groups.pop(j)
                    merged = True
                    break
            if merged:
                break
    return groups


def find_bubble(image_rgb, box, ink_wide, typical):
    """(bubble interior, bg colour) if the text in `box` sits in a closed-off bubble, else None."""
    lh = box[3] - box[1]
    seed = rect_mask(image_rgb.shape, box)
    between = seed & ~ink_wide  # background between the letters (bold text would skew the colour)
    bg = dominant_colour(image_rgb[between if between.sum() > 20 else seed])
    found = colour_region(image_rgb, seed, bg=bg, margin=int(BUBBLE_SEARCH * min(lh, 2 * typical)))
    if found is None:
        return None
    filled, raw, bg, enclosed = found
    holes = filled & ~raw & ~ink_wide
    if not (enclosed and (filled & seed).sum() >= 0.9 * seed.sum() and holes.sum() <= MAX_NONTEXT_HOLES * filled.sum()
            and solidity(filled) >= MIN_SOLIDITY):
        return None
    # Something tall enclosed with the text (a stick figure on a plain panel) means this
    # is a panel background, not a speech bubble
    labels, n = ndimage.label(holes)
    if n and max(sl[0].stop - sl[0].start for sl in ndimage.find_objects(labels)) > MAX_HOLE_HEIGHT * lh:
        return None
    return filled, bg


def hugs_text(interior, lines):
    """A speech bubble hugs its text; apart from its tail, little of it is far from the text."""
    lh = float(np.median([l[3] - l[1] for l in lines]))
    near = rect_mask(interior.shape, union_box(lines), pad=int(SNUG_MARGIN * lh))
    return (interior & ~near).sum() <= MAX_OUTSIDE * interior.sum()


def bubble_region(ink, interior, bg, lines, note=""):
    text = ink & np.logical_or.reduce([rect_mask(ink.shape, l, pad=2) for l in lines])
    return Region(
        kind="bubble", box=union_box(lines), layout_mask=interior,
        erase_mask=ndimage.binary_erosion(interior, iterations=1), text_mask=text, bg=bg,
        letter_height=float(np.median([l[3] - l[1] for l in lines])), note=note or f"{len(lines)} line(s) in bubble",
    )


def free_region(image_rgb, ink, lines, note=""):
    shape = ink.shape
    box = union_box(lines)
    lh = float(np.median([l[3] - l[1] for l in lines]))
    text = ink & np.logical_or.reduce([rect_mask(shape, l, pad=2) for l in lines])
    around = rect_mask(shape, box, pad=int(0.5 * lh)) & ~ndimage.binary_dilation(text, iterations=2)
    bg = dominant_colour(image_rgb[around]) if around.any() else (255, 255, 255)
    return Region(
        kind="free", box=box, layout_mask=rect_mask(shape, box, pad=int(0.3 * lh)),
        erase_mask=ndimage.binary_dilation(text, iterations=3), text_mask=text, bg=bg,
        letter_height=lh, note=note or f"{len(lines)} free line(s)",
    )


def region_from_box(ctx, box, note="added by hand"):
    """
    A region for an arbitrary box (a skipped detection, or one drawn by the user):
    the text pixels inside it, in its bubble if it's in one, otherwise as free text.
    """
    image_rgb, ink, ink_wide = ctx["image_rgb"], ctx["ink"], ctx["ink_wide"]
    h, w = ink.shape
    x1, y1, x2, y2 = (int(round(v)) for v in box)
    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
    inside = ink[y1:y2, x1:x2]
    if inside.any():  # tighten to the text inside the box
        ys, xs = np.nonzero(inside)
        x1, y1, x2, y2 = x1 + xs.min(), y1 + ys.min(), x1 + xs.max() + 1, y1 + ys.max() + 1
    tight = (int(x1), int(y1), int(x2), int(y2))
    found = find_bubble(image_rgb, tight, ink_wide, ctx["typical"])
    if found is not None:
        r = bubble_region(ink, found[0], found[1], [tight], note)
    else:
        r = free_region(image_rgb, ink, [tight], note)
    # One "line" spanning several lines over-estimates the letter size; use the pixels instead
    lh = letter_height(r.text_mask)
    if lh:
        r.letter_height = lh
    return r


def detect(image, ctx=None):
    """
    Returns (regions, skipped) where skipped is a list of (box, reason). If a dict
    is passed as ctx, it's filled with what region_from_box() needs later.
    """
    image_rgb = np.array(image.convert("RGB"))
    shape = image_rgb.shape[:2]
    seg, det = run_ctd(image_rgb)
    ink = seg > SEG_THRESHOLD
    ink_wide = ndimage.binary_dilation(ink, iterations=2)
    lines = find_lines(seg, det)
    if not lines:
        return [], []

    # "Typical" from the lower part of the distribution: sound-effect-heavy pages skew the median
    typical = float(np.percentile([b[3] - b[1] for b in lines], 35))

    if ctx is not None:
        ctx.update(image_rgb=image_rgb, ink=ink, ink_wide=ink_wide, typical=typical)

    skipped, normal, containers = [], [], []
    for b in lines:
        lh, lw = b[3] - b[1], b[2] - b[0]
        if lh < SMALL_RATIO * typical:
            skipped.append((b, "small print"))
            continue
        if lh > 2.5 * lw and lh > 1.5 * typical:
            skipped.append((b, "vertical text"))
            continue
        c = find_bubble(image_rgb, b, ink_wide, typical)
        if lh > SFX_RATIO * typical and c is None:
            skipped.append((b, "sound effect / logo"))  # big lettering is fine if it's in a bubble (shouting)
            continue
        normal.append(b)
        containers.append(c)

    # Lines sharing a bubble form one region
    bubble_groups = []  # [interior, bg, [lines]]
    free_lines = []
    for b, c in zip(normal, containers):
        if c is None:
            free_lines.append(b)
            continue
        for g in bubble_groups:
            inter = (g[0] & c[0]).sum()
            if inter >= 0.7 * min(g[0].sum(), c[0].sum()):
                g[2].append(b)
                break
        else:
            bubble_groups.append([c[0], c[1], [b]])

    regions = []
    for interior, bg, group in bubble_groups:
        box = union_box(group)
        if interior.sum() > MAX_BUBBLE_AREA * (box[2] - box[0]) * (box[3] - box[1]) or not hugs_text(interior, group):
            free_lines.extend(group)  # too roomy to be this text's bubble (e.g. a whole plain panel)
            continue
        regions.append(bubble_region(ink, interior, bg, group))
    for group in group_free_lines(free_lines):
        regions.append(free_region(image_rgb, ink, group))

    print(f"  [ctd] {len(lines)} lines -> {len(regions)} regions "
          f"({sum(r.kind == 'bubble' for r in regions)} bubbles), {len(skipped)} lines skipped")
    regions.sort(key=lambda r: (r.box[1], r.box[0]))
    return regions, skipped
