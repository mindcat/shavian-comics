"""Pipeline A: YOLO speech-bubble masks, confirmed by moondream's bubble boxes."""
import numpy as np
from PIL import Image
from scipy import ndimage

from . import config, models
from .regions import Region, colour_region, dominant_colour, ink_mask, letter_height, mask_bbox

YOLO_MIN_CONFIDENCE = 0.05
YOLO_NMS_IOU = 0.5
MATCH_IOU = 0.3             # moondream box <-> YOLO mask "same bubble" threshold
YOLO_ONLY_CONFIDENCE = 0.6  # keep YOLO masks moondream didn't see if at least this confident
DUPLICATE_OVERLAP = 0.5


def box_iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def moondream_boxes(image):
    w, h = image.size
    objects = models.moondream().detect(image.convert("RGB"), "speech bubble")["objects"]
    return [(o["x_min"] * w, o["y_min"] * h, o["x_max"] * w, o["y_max"] * h) for o in objects]


def yolo_candidates(image):
    res = models.yolo()(image.convert("RGB"), conf=YOLO_MIN_CONFIDENCE, iou=YOLO_NMS_IOU,
                        device=config.DEVICE, verbose=False)[0]
    if res.masks is None:
        return []
    w, h = image.size
    cands = []
    for m, conf in zip(res.masks.data.cpu().numpy(), res.boxes.conf.cpu().numpy()):
        pil = Image.fromarray((m > 0.5).astype(np.uint8) * 255)
        if pil.size != (w, h):
            pil = pil.resize((w, h), Image.NEAREST)
        mask = np.array(pil) > 0
        if mask.any():
            cands.append({"mask": mask, "bbox": mask_bbox(mask), "confidence": float(conf)})
    cands.sort(key=lambda c: c["confidence"], reverse=True)
    kept = []
    for c in cands:
        area = c["mask"].sum()
        if not any((c["mask"] & k["mask"]).sum() / min(area, k["mask"].sum()) > DUPLICATE_OVERLAP for k in kept):
            kept.append(c)
    return kept


def bubble_from_yolo(image_rgb, yolo_mask):
    """Refine YOLO's mask into the bubble's real interior, using the image's colours."""
    bg = dominant_colour(image_rgb[yolo_mask])
    ys, xs = np.nonzero(yolo_mask)
    margin = max(10, int(0.1 * max(np.ptp(ys), np.ptp(xs))))
    found = colour_region(image_rgb, yolo_mask, bg=bg, margin=margin)
    area = yolo_mask.sum()
    if found is not None:
        interior = found[0]
        if 0.6 * area <= interior.sum() <= 1.4 * area and (interior & yolo_mask).sum() >= 0.7 * area:
            return interior, bg, "colour-refined"
    return ndimage.binary_fill_holes(yolo_mask), bg, "yolo mask"


def detect(image):
    """Returns (regions, skipped): regions are all kind="bubble"; skipped is [(box, reason)]."""
    image_rgb = np.array(image.convert("RGB"))
    md = moondream_boxes(image)
    cands = yolo_candidates(image)

    pairs = sorted(((box_iou(m, c["bbox"]), mi, ci) for mi, m in enumerate(md) for ci, c in enumerate(cands)),
                   reverse=True)
    used_m, used_c = set(), set()
    for iou, mi, ci in pairs:
        if iou < MATCH_IOU:
            break
        if mi not in used_m and ci not in used_c:
            used_m.add(mi)
            used_c.add(ci)

    regions, skipped = [], []
    for ci, c in enumerate(cands):
        if ci in used_c:
            why = "matched moondream"
        elif c["confidence"] >= YOLO_ONLY_CONFIDENCE:
            why = "YOLO only, confident"
        else:
            skipped.append((c["bbox"], f"YOLO only, conf {c['confidence']:.2f}"))
            continue
        interior, bg, how = bubble_from_yolo(image_rgb, c["mask"])
        inner = ndimage.binary_erosion(interior, iterations=3)  # stay off the outline
        text = ink_mask(image_rgb, inner, bg)
        regions.append(Region(
            kind="bubble", box=mask_bbox(interior), layout_mask=interior,
            erase_mask=ndimage.binary_erosion(interior, iterations=1), text_mask=text, bg=bg,
            letter_height=letter_height(text), score=c["confidence"],
            note=f"conf {c['confidence']:.2f}, {why}, {how}",
        ))
    for mi, m in enumerate(md):
        if mi not in used_m:
            skipped.append((tuple(int(v) for v in m), "moondream only (no YOLO mask)"))
    print(f"  [yolo] moondream {len(md)} boxes, YOLO {len(cands)} masks -> {len(regions)} bubbles, "
          f"{len(skipped)} skipped")
    regions.sort(key=lambda r: (r.box[1], r.box[0]))
    return regions, skipped
