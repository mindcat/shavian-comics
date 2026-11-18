"""OCR backends. Each takes a list of cropped images and returns their English text."""
import re

import numpy as np
import torch
from PIL import Image, ImageFilter
from scipy import ndimage

from . import config, models, translit
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


def florence(crops):
    """Florence-2 '<OCR>': reads lettering exactly as written (capitals, line breaks),
    so lines are joined and the text re-cased with the Shavian dictionary's help."""
    model, processor = models.florence()
    out = []
    for c in crops:
        inputs = processor(text="<OCR>", images=c, return_tensors="pt").to(model.device, model.dtype)
        with torch.inference_mode():
            ids = model.generate(**inputs, max_new_tokens=200, num_beams=3, do_sample=False)
        raw = processor.batch_decode(ids, skip_special_tokens=False)[0]
        text = processor.post_process_generation(raw, task="<OCR>", image_size=c.size)["<OCR>"]
        out.append(_clean(translit.recase(join_lines(text))))
    return out


SUFFIXES = {"s", "es", "ed", "d", "ing", "er", "ers", "est", "ly", "y", "tion", "tions", "ment", "ness", "able", "ful"}


def join_lines(text):
    """
    Join OCR'd lines. A word hyphenated across a line break is re-joined when the
    second half is a common suffix or the dictionary knows the whole word
    ("PAINT-/ED" -> "painted"); otherwise the hyphen stays ("BRAND-/NEW").
    """
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    joined = ""
    for line in lines:
        m = re.search(r"(\w+)-$", joined)
        first = re.match(r"\w+", line)
        if m and first:
            merge = first.group(0).lower() in SUFFIXES or translit.known_word(m.group(1) + first.group(0))
            joined = (joined[:-1] if merge else joined) + line
        else:
            joined = f"{joined} {line}" if joined else line
    # OCR often puts a space before closing punctuation ("HERE !")
    return re.sub(r"\s+([!?.,;:])", r"\1", joined)


GROQ_URL = "https://api.groq.com/openai/v1"
GROQ_PREFERRED = ("llama-4-scout", "llama-4-maverick", "vision", "-vl", "qwen")


def groq_model():
    if config.GROQ_MODEL:
        return config.GROQ_MODEL
    import requests

    r = requests.get(f"{GROQ_URL}/models", headers={"Authorization": f"Bearer {config.groq_key()}"}, timeout=30)
    r.raise_for_status()
    ids = [m["id"] for m in r.json()["data"]]
    for hint in GROQ_PREFERRED:
        for i in ids:
            if hint in i.lower():
                return i
    raise RuntimeError(f"No vision model recognised among Groq's models; set GROQ_MODEL. Available: {ids}")


def groq(crops):
    """Hosted OCR through Groq's API (needs a key; images leave this machine)."""
    import base64
    import io
    import time

    import requests

    key = config.groq_key()
    if not key:
        raise RuntimeError(f"No Groq key: set GROQ_API_KEY or put it in {config.GROQ_KEY_FILE}")
    model = groq_model()
    print(f"  [groq] model {model}")
    out = []
    for c in crops:
        buf = io.BytesIO()
        c.convert("RGB").save(buf, "JPEG", quality=92)
        payload = {
            "model": model, "temperature": 0, "max_completion_tokens": 200,
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": config.OCR_PROMPT + " Reply with the text only."},
                {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()}},
            ]}],
        }
        for attempt in range(6):
            r = requests.post(f"{GROQ_URL}/chat/completions", json=payload, timeout=60,
                              headers={"Authorization": f"Bearer {key}"})
            if r.status_code == 429:  # free-tier rate limit: wait as told, then retry
                time.sleep(float(r.headers.get("retry-after", 2 ** attempt)))
                continue
            r.raise_for_status()
            break
        else:
            r.raise_for_status()
        out.append(_clean(r.json()["choices"][0]["message"]["content"]))
    return out


BACKENDS = {"moondream": moondream, "qwen": qwen, "florence": florence, "groq": groq}
