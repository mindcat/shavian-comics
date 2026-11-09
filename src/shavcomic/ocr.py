"""OCR backends. Each takes a list of cropped images and returns their English text."""
import re

import numpy as np
import torch
from PIL import Image, ImageFilter
from scipy import ndimage

from . import config, models
from .regions import mask_bbox


def crop_for_ocr(image, region):
    """The region's text on a clean background, upscaled 2x (helps with small lettering)."""
    if region.kind == "bubble":
        mask = ndimage.binary_dilation(region.layout_mask, iterations=2)
        box = mask_bbox(mask)
        crop = image.convert("RGB").crop(box)
        canvas = Image.new("RGB", crop.size, (255, 255, 255))
        canvas.paste(crop, (0, 0), Image.fromarray(mask.astype(np.uint8) * 255).crop(box))
    else:
        pad = int(max(6, 0.5 * region.letter_height))
        x1, y1, x2, y2 = region.box
        w, h = image.size
        canvas = image.convert("RGB").crop((max(0, x1 - pad), max(0, y1 - pad), min(w, x2 + pad), min(h, y2 + pad)))
    up = canvas.resize((canvas.width * 2, canvas.height * 2), Image.Resampling.LANCZOS)
    return up.filter(ImageFilter.SHARPEN)


# Credits and signatures look like text but shouldn't be transliterated. The signature
# is often misread ("by HATTERSON"), so any lone "by <name> [year]" counts as a credit.
SKIP_TEXT = re.compile(
    r"watterson|syndicat|distributed by|universal press|andrews mc ?meel|copyright"
    r"|^by \w+( \d{4})?$"
    r"|^(no text|empty|none|n/a)\.?$",
    re.I,
)


def _clean(text):
    text = " ".join(text.strip().strip('"').split())
    if SKIP_TEXT.search(text):
        return ""
    return text if any(ch.isalpha() for ch in text) else ""


def moondream(crops):
    model = models.moondream()
    return [_clean(model.query(c, config.OCR_PROMPT)["answer"]) for c in crops]


def qwen(crops):
    model, processor = models.qwen()
    out = []
    for c in crops:
        messages = [{"role": "user", "content": [
            {"type": "image", "image": c},
            {"type": "text", "text": config.OCR_PROMPT + " Reply with the text only."},
        ]}]
        inputs = processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt"
        ).to(model.device)
        with torch.inference_mode():
            ids = model.generate(**inputs, max_new_tokens=160, do_sample=False)
        text = processor.batch_decode(ids[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]
        out.append(_clean(text))
    return out


BACKENDS = {"moondream": moondream, "qwen": qwen}
