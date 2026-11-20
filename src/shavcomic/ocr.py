"""OCR backends. Each takes a list of cropped images and returns their English text."""
import re

import numpy as np
import torch
from PIL import Image, ImageFilter
from scipy import ndimage

from . import config, models, textproc, translit
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


_clean = textproc.clean
SKIP_TEXT = textproc.SKIP_TEXT
SUFFIXES = textproc.SUFFIXES


def join_lines(text):
    return textproc.join_lines(text, translit.engine().dictionary)


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
    """
    Florence-2 '<OCR_WITH_REGION>': reads each line of lettering separately, exactly
    as written (capitals). Lines are then joined and the text re-cased with the
    Shavian dictionary's help. (Plain '<OCR>' sometimes drops the line breaks,
    gluing words together: "80%WATER".)
    """
    model, processor = models.florence()
    task = "<OCR_WITH_REGION>"
    out = []
    for c in crops:
        inputs = processor(text=task, images=c, return_tensors="pt").to(model.device, model.dtype)
        with torch.inference_mode():
            ids = model.generate(**inputs, max_new_tokens=200, num_beams=config.FLORENCE_BEAMS, do_sample=False)
        raw = processor.batch_decode(ids, skip_special_tokens=False)[0]
        lines = processor.post_process_generation(raw, task=task, image_size=c.size)[task]["labels"]
        lines = [l.replace("</s>", "").replace("<s>", "").strip() for l in lines]
        out.append(_clean(translit.recase(join_lines("\n".join(l for l in lines if l)))))
    return out


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


SHEET_MAX_HEIGHT = 1600  # px; taller sheets risk being downscaled by the API
SHEET_MAX_CROPS = 16


def _groq_request(content, max_tokens, json_mode=False):
    import time

    import requests

    payload = {"model": groq_model(), "temperature": 0, "max_completion_tokens": max_tokens,
               "messages": [{"role": "user", "content": content}]}
    if json_mode:
        payload["response_format"] = {"type": "json_object"}
    for attempt in range(8):
        r = requests.post(f"{GROQ_URL}/chat/completions", json=payload, timeout=120,
                          headers={"Authorization": f"Bearer {config.groq_key()}"})
        if r.status_code == 429:  # free-tier rate limit: wait as told, then retry
            wait = float(r.headers.get("retry-after", 2 ** attempt))
            if wait > 60:  # e.g. the daily quota: give up so the caller can fall back
                raise RuntimeError(f"Groq rate limit, retry in {wait:.0f}s")
            print(f"  [groq] rate limited, waiting {wait:.0f}s")
            time.sleep(wait)
            continue
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]
    r.raise_for_status()


def _data_url(image):
    import base64
    import io

    buf = io.BytesIO()
    image.convert("RGB").save(buf, "JPEG", quality=90)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def _sheets(crops):
    """Group crops (shrunk back to 1x) into numbered sheets: [(first_index, sheet_image, n)]."""
    from PIL import ImageDraw, ImageFont

    small = [c.resize((max(1, c.width // 2), max(1, c.height // 2)), Image.Resampling.LANCZOS) for c in crops]
    groups, cur, h = [], [], 0
    for i, c in enumerate(small):
        if cur and (h + c.height + 8 > SHEET_MAX_HEIGHT or len(cur) >= SHEET_MAX_CROPS):
            groups.append(cur)
            cur, h = [], 0
        cur.append(i)
        h += c.height + 8
    if cur:
        groups.append(cur)
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", 20)
    except OSError:
        font = ImageFont.load_default()
    out = []
    for g in groups:
        label, pad = 40, 8
        sheet = Image.new("RGB", (max(small[i].width for i in g) + label + pad, sum(small[i].height + pad for i in g)), "white")
        d = ImageDraw.Draw(sheet)
        y = 0
        for n, i in enumerate(g, 1):
            c = small[i]
            d.rectangle((0, y, sheet.width - 1, y + c.height + pad - 1), outline=(200, 0, 0), width=2)
            d.text((6, y + 6), str(n), fill=(200, 0, 0), font=font)
            sheet.paste(c.convert("RGB"), (label, y + pad // 2))
            y += c.height + pad
        out.append((g, sheet))
    return out


def groq(crops):
    """
    Hosted OCR through Groq's API (needs a key; images leave this machine). Every
    image costs a flat ~800 input tokens against a free-tier budget of a few
    thousand per minute, so crops are tiled into numbered sheets and read in one
    request per sheet; a sheet whose answer can't be parsed falls back to one
    request per crop.
    """
    import json

    if not config.groq_key():
        raise RuntimeError(f"No Groq key: set GROQ_API_KEY or put it in {config.GROQ_KEY_FILE}")
    print(f"  [groq] model {groq_model()}")
    out = [""] * len(crops)
    for idx, sheet in _sheets(crops):
        prompt = (f"This image is a stack of {len(idx)} boxes, numbered in red on the left, each cut from a comic. "
                  "For each box, transcribe its lettering all lowercase unless a word is a proper name or place; "
                  "use an empty string if a box has no real text. "
                  "Reply with only a JSON object mapping each box number to its text.")
        try:
            answer = json.loads(_groq_request(
                [{"type": "text", "text": prompt}, {"type": "image_url", "image_url": {"url": _data_url(sheet)}}],
                max_tokens=150 * len(idx) + 100, json_mode=True))
            for n, i in enumerate(idx, 1):
                out[i] = _clean(str(answer.get(str(n), "")))
        except (ValueError, AttributeError) as e:
            print(f"  [groq] couldn't use the sheet's answer ({e}); reading one by one")
            for i in idx:
                out[i] = _clean(_groq_request([
                    {"type": "text", "text": config.OCR_PROMPT + " Reply with the text only."},
                    {"type": "image_url", "image_url": {"url": _data_url(crops[i])}}], max_tokens=200))
    return out


BACKENDS = {"moondream": moondream, "qwen": qwen, "florence": florence, "groq": groq}
