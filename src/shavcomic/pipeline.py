"""Runs the stages across a batch of images, one stage at a time (keeps GPU memory low)."""
import traceback

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

from . import detect_ctd, detect_yolo, erase, layout, models, ocr, translit

DETECTORS = {"yolo": detect_yolo.detect, "ctd": detect_ctd.detect}


def load_images(paths):
    return {p.stem: Image.open(p).convert("RGBA") for p in paths}


def detect_all(images, detector):
    out = {}
    for name, img in images.items():
        print(f"[detect:{detector}] {name}")
        out[name] = DETECTORS[detector](img)
    return out


def ocr_all(images, detections, backend):
    """Fills region.ocr[backend] for every region of every image."""
    for name, (regions, _) in detections.items():
        print(f"[ocr:{backend}] {name}: {len(regions)} regions")
        if backend in ocr.LINE_BACKENDS:
            crops = [ocr.line_crops(images[name], r) for r in regions]
        else:
            crops = [ocr.crop_for_ocr(images[name], r) for r in regions]
        for r, text in zip(regions, ocr.BACKENDS[backend](crops)):
            r.ocr[backend] = text


def translit_all(detections_list, backend, pos_tagging=True, key=None):
    """One shaw.py run for every region in every detection set. Fills region.shavian[key]."""
    key = key or backend
    regions = [r for dets in detections_list for regs, _ in dets.values() for r in regs]
    texts = [r.ocr.get(backend, "") for r in regions]
    print(f"[translit] {sum(bool(t) for t in texts)} texts (POS tagging {'on' if pos_tagging else 'off'})")
    for r, s in zip(regions, translit.transliterate(texts, pos_tagging)):
        r.shavian[key] = s


def _setting(r, name, default):
    value = getattr(r, "settings", {}).get(name)
    return default if value is None or value == "" else value


def erase_only(image, regions, eraser, shavian_key):
    out = image.copy()
    for r in regions:
        if r.shavian.get(shavian_key) and _setting(r, "enabled", True):
            try:
                erase.BACKENDS[_setting(r, "eraser", eraser)](out, r)
            except Exception:
                print(f"  erase failed for region {r.box}:\n{traceback.format_exc()}")
    return out


def render(image, regions, eraser, shavian_key, size_mode="match", draw=True):
    """
    Erase + typeset every region that has Shavian text. Regions without text are left
    alone. A region's optional `settings` dict ({"enabled", "eraser", "size"}) overrides
    the defaults for that region.
    """
    out = erase_only(image, regions, eraser, shavian_key)
    if not draw:
        return out
    for r in regions:
        text = r.shavian.get(shavian_key)
        if text and _setting(r, "enabled", True):
            try:
                r.note_layout, r.font_size = layout.draw_text(out, r, text, _setting(r, "size", size_mode))
            except Exception:
                print(f"  layout failed for region {r.box}:\n{traceback.format_exc()}")
    return out


def overlay(image, regions, skipped):
    """Detection debug view: green = bubble, orange = free text, red = text pixels, grey = skipped."""
    base = np.array(image.convert("RGB")).astype(np.float32)
    tint = base.copy()
    for r in regions:
        tint[r.layout_mask] = (60, 200, 90) if r.kind == "bubble" else (255, 160, 40)
    for r in regions:
        tint[r.text_mask] = (230, 30, 30)
    out = Image.fromarray((0.55 * base + 0.45 * tint).astype(np.uint8))
    d = ImageDraw.Draw(out)
    for i, r in enumerate(regions, 1):
        edge = r.layout_mask & ~ndimage.binary_erosion(r.layout_mask, iterations=2)
        out_np = np.array(out)
        out_np[edge] = (0, 120, 40) if r.kind == "bubble" else (200, 90, 0)
        out = Image.fromarray(out_np)
        d = ImageDraw.Draw(out)
        d.text((r.box[0], max(0, r.box[1] - 11)), str(i), fill=(0, 0, 255))
    for box, reason in skipped:
        d.rectangle(box, outline=(130, 130, 130), width=2)
    return out


def unload_ocr():
    models.unload("moondream", "qwen")
