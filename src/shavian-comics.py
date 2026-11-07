# Core libraries
import os
import re
import sys
import subprocess
from collections import Counter
from pathlib import Path
import shutil # For clearing temp directory if needed

# Image processing
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import numpy as np
from scipy import ndimage

# Object detection/segmentation
from ultralytics import YOLO # Preferred for running .pt files

# OCR
import torch
from transformers import AutoModelForCausalLM

# All relative paths below are relative to this script's directory
os.chdir(Path(__file__).resolve().parent)


def pick_device():
    """Use an NVIDIA GPU if present, then Apple Silicon, then fall back to CPU."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


DEVICE = os.environ.get("SHAVIAN_DEVICE") or pick_device()
print(f"Using device: {DEVICE}")

ocr_model = AutoModelForCausalLM.from_pretrained(
    "vikhyatk/moondream2",
    revision="2025-06-21",
    trust_remote_code=True,
    device_map={"": DEVICE},
)

# --- Directories ---
INPUT_DIR = Path("img/in")
TEMP_DIR = Path("img/tmp") # For intermediate bubble images (optional use)
OUTPUT_DIR = Path("img/out")
MODEL_DIR = Path("models")
FONT_DIR = Path("fonts") # Directory containing the .otf font file
DICT_DIR = Path("dict") # Directory containing the dictionary for shaw.py

# --- Model ---
MODEL_NAME = "model.pt"
MODEL_PATH = MODEL_DIR / MODEL_NAME

# --- Transliteration Script ---
SHAW_SCRIPT_PATH = "shaw.py"
SHAW_DICT_PATH = DICT_DIR / "amer.dict"

# --- Font ---
SHAVIAN_FONT_NAME = "backlogshavian.otf"
# SHAVIAN_FONT_NAME = "stampatello-faceto.otf"
# SHAVIAN_FONT_NAME = "bantershaviantriline.otf"
SHAVIAN_FONT_PATH = FONT_DIR / SHAVIAN_FONT_NAME
MIN_FONT_SIZE = 8 # Smallest font size the fitter will try

# --- Detection settings ---
# YOLO runs once at a low confidence; candidates are then filtered by the rules below.
YOLO_MIN_CONFIDENCE = 0.05
IOU_THRESH = 0.5 # YOLO's own NMS threshold
# A YOLO mask and a moondream box are "the same bubble" if their boxes overlap at least this much (IoU)
MATCH_IOU = 0.3
# A YOLO mask that moondream didn't see is still kept if YOLO is at least this confident
YOLO_ONLY_CONFIDENCE = 0.6
# Two YOLO masks overlapping more than this (as a fraction of the smaller one) are duplicates
DUPLICATE_OVERLAP = 0.5

# --- Fill / layout settings ---
TEXT_PADDING = 4 # Minimum gap (px) between drawn text and the bubble edge
TEXT_COLOR = (0, 0, 0, 255)
# Set to True to clear the temp directory each run
CLEAR_TEMP_DIR = False


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------

def box_iou(a, b):
    """IoU of two (x1, y1, x2, y2) boxes."""
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def moondream_bubble_boxes(image):
    """Ask moondream for speech-bubble boxes, returned in pixel (x1, y1, x2, y2)."""
    w, h = image.size
    objects = ocr_model.detect(image.convert("RGB"), "speech bubble")["objects"]
    return [
        (o["x_min"] * w, o["y_min"] * h, o["x_max"] * w, o["y_max"] * h)
        for o in objects
    ]


def yolo_candidates(image_path, model, img_shape):
    """Run YOLO once at a low confidence and return de-duplicated mask candidates."""
    results = model(str(image_path), conf=YOLO_MIN_CONFIDENCE, iou=IOU_THRESH, device=DEVICE, verbose=False)
    res = results[0]
    if res.masks is None:
        return []

    h, w = img_shape
    candidates = []
    for mask_np, conf in zip(res.masks.data.cpu().numpy(), res.boxes.conf.cpu().numpy()):
        mask_pil = Image.fromarray((mask_np > 0.5).astype(np.uint8) * 255, mode="L")
        if mask_pil.size != (w, h):
            mask_pil = mask_pil.resize((w, h), Image.NEAREST)
        mask = np.array(mask_pil) > 0
        bbox = mask_pil.getbbox()
        if bbox:
            candidates.append({"mask": mask, "bbox": bbox, "confidence": float(conf)})

    # Drop masks that mostly sit inside a more confident mask
    candidates.sort(key=lambda c: c["confidence"], reverse=True)
    kept = []
    for c in candidates:
        area = c["mask"].sum()
        duplicate = any(
            (c["mask"] & k["mask"]).sum() / min(area, k["mask"].sum()) > DUPLICATE_OVERLAP
            for k in kept
        )
        if not duplicate:
            kept.append(c)
    return kept


def detect_speech_bubbles(image_path, image, model):
    """
    Finds speech bubbles by combining YOLO (good masks) with moondream (good at
    knowing which things really are speech bubbles).

    Rules:
      * each moondream box is matched to the YOLO mask it overlaps most (IoU >= MATCH_IOU)
      * YOLO masks moondream didn't see are kept only if YOLO is confident (>= YOLO_ONLY_CONFIDENCE)
      * moondream boxes with no matching YOLO mask are reported and skipped

    Returns a list of dicts with 'mask' (bool ndarray), 'bbox' and 'confidence'.
    """
    md_boxes = moondream_bubble_boxes(image)
    candidates = yolo_candidates(image_path, model, (image.height, image.width))
    print(f"  moondream boxes: {len(md_boxes)}, YOLO candidates: {len(candidates)}")

    # Greedy one-to-one matching, best IoU first
    pairs = sorted(
        ((box_iou(md, c["bbox"]), mi, ci) for mi, md in enumerate(md_boxes) for ci, c in enumerate(candidates)),
        reverse=True,
    )
    matched_md, matched_yolo = set(), set()
    for iou, mi, ci in pairs:
        if iou < MATCH_IOU:
            break
        if mi in matched_md or ci in matched_yolo:
            continue
        matched_md.add(mi)
        matched_yolo.add(ci)

    bubbles = []
    for ci, c in enumerate(candidates):
        if ci in matched_yolo:
            reason = "matched"
        elif c["confidence"] >= YOLO_ONLY_CONFIDENCE:
            reason = "YOLO-only, confident"
        else:
            continue
        print(f"  keep bubble bbox={c['bbox']} conf={c['confidence']:.2f} ({reason})")
        bubbles.append(c)

    for mi, md in enumerate(md_boxes):
        if mi not in matched_md:
            print(f"  moondream box {tuple(round(v) for v in md)} has no YOLO mask; skipped")

    # Top-to-bottom, left-to-right
    bubbles.sort(key=lambda b: (b["bbox"][1], b["bbox"][0]))
    return bubbles


# ---------------------------------------------------------------------------
# Bubble interior: the region to erase and draw into
# ---------------------------------------------------------------------------

def bubble_interior(image_rgb, yolo_mask):
    """
    Works out the bubble's real interior from the image itself, so the erase
    covers all of the old text even where YOLO's mask is a little off.

    Bubble background pixels are found by colour (close to the most common
    colour under the mask); the connected region overlapping YOLO's mask most
    is taken, and its holes (the letters) are filled in.

    Returns (interior mask as bool ndarray, background colour as RGB tuple).
    """
    pixels = image_rgb[yolo_mask]
    # Quantise a little so JPEG/WebP noise doesn't split the mode
    bg = Counter(map(tuple, (pixels // 8) * 8 + 4)).most_common(1)[0][0]
    bg = np.array(bg, dtype=np.int16)

    # Work inside YOLO's box plus a margin
    ys, xs = np.nonzero(yolo_mask)
    pad = max(10, int(0.1 * max(np.ptp(ys), np.ptp(xs))))
    y1, y2 = max(0, ys.min() - pad), min(yolo_mask.shape[0], ys.max() + pad + 1)
    x1, x2 = max(0, xs.min() - pad), min(yolo_mask.shape[1], xs.max() + pad + 1)
    crop = image_rgb[y1:y2, x1:x2].astype(np.int16)
    crop_mask = yolo_mask[y1:y2, x1:x2]

    is_bg = np.abs(crop - bg).max(axis=2) <= 40
    labels, n = ndimage.label(is_bg)
    yolo_area = crop_mask.sum()
    interior = None
    if n:
        overlap = ndimage.sum(crop_mask, labels, index=range(1, n + 1))
        best = int(np.argmax(overlap)) + 1
        region = ndimage.binary_fill_holes(labels == best)
        # Bubbles whose outline has a gap can leak into the background; reject those
        if 0.6 * yolo_area <= region.sum() <= 1.4 * yolo_area and (region & crop_mask).sum() >= 0.7 * yolo_area:
            interior = region

    if interior is None:
        print("  Colour-based bubble region didn't look right; using YOLO's mask")
        interior = ndimage.binary_fill_holes(crop_mask)

    full = np.zeros_like(yolo_mask)
    full[y1:y2, x1:x2] = interior
    return full, tuple(int(v) for v in bg)


def extract_bubble_region(original_image, interior):
    """
    Crops the bubble (using its interior mask) onto a white background and
    upscales it 2x, which helps moondream read small lettering.
    """
    mask_pil = Image.fromarray(ndimage.binary_dilation(interior, iterations=2).astype(np.uint8) * 255, mode="L")
    bbox = mask_pil.getbbox()
    if not bbox:
        return None
    crop = original_image.crop(bbox).convert("RGB")
    canvas = Image.new("RGB", crop.size, (255, 255, 255))
    canvas.paste(crop, (0, 0), mask_pil.crop(bbox))
    upscaled = canvas.resize((canvas.width * 2, canvas.height * 2), Image.Resampling.LANCZOS)
    return upscaled.filter(ImageFilter.SHARPEN)


# ---------------------------------------------------------------------------
# OCR + transliteration
# ---------------------------------------------------------------------------

def perform_ocr(bubble_image):
    """Reads the bubble's text with moondream. Returns '' if there's no real text."""
    if bubble_image is None:
        return ""
    try:
        result = ocr_model.query(
            bubble_image,
            "Transcribe the text, all lowercase unless it is a proper name or place. "
            "If it isn't actual text, return an empty string.",
        )["answer"].strip()
        print(f"  Moondream OCR Result: '{result}'")
        # Treat answers with no letters (e.g. '""' or '...') as no text
        return result if any(ch.isalpha() for ch in result) else ""
    except Exception as e:
        print(f"  Error during OCR: {e}")
        return ""


def transliterate_to_shavian(text):
    """Transliterates English text to Shavian using shaw.py."""
    if not text:
        return ""
    command = [sys.executable, SHAW_SCRIPT_PATH, str(SHAW_DICT_PATH)]
    try:
        process = subprocess.run(command, input=text, capture_output=True, text=True, check=False)
        if process.returncode != 0:
            print(f"  Error running shaw.py (return code {process.returncode}):")
            print(f"  Stderr: {process.stderr.strip()}")
            return ""
        # Newer dechifro dictionaries tag some words with a leading '_' that this
        # (older) shaw.py passes through; it's never part of Shavian text
        shavian_text = re.sub(r"(?<![^\W_])_", "", process.stdout).strip()
        print(f"  Shavian Transliteration: '{shavian_text}'")
        return shavian_text
    except Exception as e:
        print(f"  Error during transliteration: {e}")
        return ""


# ---------------------------------------------------------------------------
# Text layout
# ---------------------------------------------------------------------------

def wrap_words(words, font, max_width, draw):
    """Greedy word wrap. Returns lines, or None if a single word is wider than max_width."""
    lines, current = [], ""
    for word in words:
        test = f"{current} {word}" if current else word
        if draw.textlength(test, font=font) <= max_width:
            current = test
        elif not current or draw.textlength(word, font=font) > max_width:
            return None
        else:
            lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines


def layout_fits(draw, font, lines, center, fit_mask):
    """True if every line's box, centred on `center`, lies inside fit_mask."""
    text = "\n".join(lines)
    box = draw.multiline_textbbox(center, text, font=font, anchor="mm", align="center")
    x1, y1, x2, y2 = (int(round(v)) for v in box)
    h, w = fit_mask.shape
    if x1 < 0 or y1 < 0 or x2 > w or y2 > h:
        return False
    # Check each line separately: bubbles are round, so short top/bottom lines can sit nearer the edge
    line_h = (y2 - y1) / len(lines)
    for i, line in enumerate(lines):
        lw = draw.textlength(line, font=font)
        lx1 = int(center[0] - lw / 2)
        lx2 = int(np.ceil(center[0] + lw / 2))
        ly1 = int(y1 + i * line_h)
        ly2 = int(np.ceil(y1 + (i + 1) * line_h))
        if not fit_mask[ly1:ly2, max(lx1, 0):lx2].all():
            return False
    return True


def fit_text(draw, text, font_path, interior):
    """
    Finds the largest font size (and line wrapping) whose text fits inside the
    bubble's actual shape, centred on the point deepest inside the bubble.

    Returns (font, lines, center) or None if nothing fits even at MIN_FONT_SIZE.
    """
    words = text.split()
    if not words:
        return None

    fit_mask = ndimage.binary_erosion(interior, iterations=TEXT_PADDING)
    if not fit_mask.any():
        return None

    # Candidate centres: deepest point inside the bubble, and the bubble's centroid
    dist = ndimage.distance_transform_edt(fit_mask)
    deep_y, deep_x = np.unravel_index(np.argmax(dist), dist.shape)
    cy, cx = ndimage.center_of_mass(fit_mask)
    centers = [(float(deep_x), float(deep_y))]
    if fit_mask[int(cy), int(cx)]:
        centers.insert(0, (float(cx), float(cy)))

    ys, xs = np.nonzero(fit_mask)
    max_width = np.ptp(xs)
    max_size = max(MIN_FONT_SIZE, int(np.ptp(ys) * 0.5))

    def try_size(size):
        font = ImageFont.truetype(font_path, size)
        # Try narrow to wide wrapping; narrower blocks suit round bubbles
        for frac in np.linspace(0.35, 1.0, 14):
            lines = wrap_words(words, font, max_width * frac, draw)
            if not lines:
                continue
            for center in centers:
                if layout_fits(draw, font, lines, center, fit_mask):
                    return font, lines, center
        return None

    best = None
    lo, hi = MIN_FONT_SIZE, max_size
    while lo <= hi:
        mid = (lo + hi) // 2
        found = try_size(mid)
        if found:
            best, lo = found, mid + 1
        else:
            hi = mid - 1
    return best


def draw_shavian_text_in_bubble(draw, interior, shavian_text, font_path):
    """Draws the Shavian text at the largest size that fits inside the bubble."""
    fitted = fit_text(draw, shavian_text, font_path, interior)
    if fitted is None:
        # Fall back to the smallest size centred in the bubble, and say so
        print(f"  Warning: text doesn't fit at {MIN_FONT_SIZE}px; drawing it anyway")
        font = ImageFont.truetype(font_path, MIN_FONT_SIZE)
        cy, cx = ndimage.center_of_mass(interior)
        lines, center = [shavian_text], (cx, cy)
    else:
        font, lines, center = fitted
    print(f"  Drawing at {font.size}px, {len(lines)} line(s):\n------\n" + "\n".join(lines) + "\n------")
    draw.multiline_text(center, "\n".join(lines), font=font, fill=TEXT_COLOR, anchor="mm", align="center")


# ---------------------------------------------------------------------------
# Per-bubble processing
# ---------------------------------------------------------------------------

def process_bubble(original_image, image_rgb, output_image, bubble, temp_stem):
    """Erases one bubble's text and draws the Shavian version. Raises on failure."""
    interior, bg = bubble_interior(image_rgb, bubble["mask"])

    bubble_image = extract_bubble_region(original_image, interior)
    if bubble_image is None:
        print("  Skipping bubble: empty interior")
        return
    bubble_image.save(TEMP_DIR / f"{temp_stem}.png")

    original_text = perform_ocr(bubble_image)
    if not original_text:
        print("  No text found by OCR; leaving this bubble untouched.")
        return

    shavian_text = transliterate_to_shavian(original_text)
    if not shavian_text:
        print("  Transliteration failed or returned empty; leaving this bubble untouched.")
        return

    # Erase: fill the interior (shrunk by 1px so the outline's anti-aliasing survives)
    erase = ndimage.binary_erosion(interior, iterations=1)
    erase_mask = Image.fromarray(erase.astype(np.uint8) * 255, mode="L")
    erase_mask = erase_mask.filter(ImageFilter.GaussianBlur(radius=0.7))
    fill_img = Image.new("RGBA", output_image.size, (*bg, 255))
    output_image.paste(fill_img, (0, 0), erase_mask)

    draw_shavian_text_in_bubble(ImageDraw.Draw(output_image), interior, shavian_text, str(SHAVIAN_FONT_PATH))


def main():
    for d in (INPUT_DIR, TEMP_DIR, OUTPUT_DIR, FONT_DIR, DICT_DIR, MODEL_DIR):
        d.mkdir(parents=True, exist_ok=True)

    if CLEAR_TEMP_DIR and TEMP_DIR.exists():
        print(f"Clearing temporary directory: {TEMP_DIR}")
        for item in TEMP_DIR.iterdir():
            if item.is_file():
                item.unlink()
            elif item.is_dir():
                shutil.rmtree(item)

    for path, what in ((MODEL_PATH, "Model file"), (SHAW_DICT_PATH, "Shavian dictionary"), (SHAVIAN_FONT_PATH, "Shavian font")):
        if not path.exists():
            raise FileNotFoundError(f"{what} not found at {path} (run init.py)")

    model = YOLO(MODEL_PATH)
    print(f"Loaded YOLO model from {MODEL_PATH}")

    image_files = sorted(p for ext in ("png", "jpg", "jpeg", "webp") for p in INPUT_DIR.glob(f"*.{ext}"))
    print(f"Found {len(image_files)} images in {INPUT_DIR}")

    for image_path in image_files:
        print(f"\n--- Processing: {image_path.name} ---")
        try:
            original_image = Image.open(image_path).convert("RGBA")
            image_rgb = np.array(original_image.convert("RGB"))

            bubbles = detect_speech_bubbles(image_path, original_image, model)
            if not bubbles:
                print("  No speech bubbles detected. Skipping.")
                continue

            output_image = original_image.copy()
            for i, bubble in enumerate(bubbles):
                print(f"\nProcessing Bubble {i+1} (bbox: {bubble['bbox']})")
                # One bad bubble shouldn't lose the whole image
                try:
                    process_bubble(original_image, image_rgb, output_image, bubble, f"{image_path.stem}_bubble_{i+1}")
                except Exception as e:
                    print(f"  Error processing bubble {i+1}: {e!r}; leaving it untouched.")

            output_filename = OUTPUT_DIR / f"{image_path.stem}_shavian.png"
            output_image.save(output_filename)
            print(f"\nSaved modified image to: {output_filename}")

        except Exception as e:
            print(f"An unexpected error occurred processing {image_path.name}: {e!r}")

    print("\n--- Pipeline Finished ---")


if __name__ == "__main__":
    main()
