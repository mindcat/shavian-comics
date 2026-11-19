#!/usr/bin/env python3
"""
Web tool: drop in a comic, inspect and correct each stage, download the Shavian version.

    uv run python src/server.py [--host 127.0.0.1] [--port 8765]

Serves the front end (src/web/) at /, the JSON API at /api/, and the latest
comparison report at /report/. The front end only talks to /api/, so it can also
be hosted separately (e.g. GitHub Pages) and pointed at this server with ?api=URL.
"""
import argparse
import io
import os
import sys
import threading
import time
import uuid
from collections import OrderedDict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import uvicorn  # noqa: E402
from fastapi import FastAPI, File, HTTPException, Request, UploadFile  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from fastapi.responses import FileResponse, RedirectResponse, Response  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402
from PIL import Image, ImageOps  # noqa: E402
from pydantic import BaseModel  # noqa: E402

from shavcomic import config, detect_ctd, layout, models, ocr, pipeline, translit  # noqa: E402

WEB_DIR = Path(__file__).resolve().parent / "web"
REPORT_DIR = config.OUTPUT_DIR / "compare"
OCR_BACKEND = os.environ.get("SHAVIAN_OCR", "qwen")  # qwen (GPU) | florence (CPU-friendly) | groq (hosted API)
# Limits, configurable for public hosting (see deploy/space/)
MAX_JOBS = int(os.environ.get("SHAVIAN_MAX_JOBS", 20))        # jobs kept in memory
MAX_SIDE = int(os.environ.get("SHAVIAN_MAX_SIDE", 2400))      # downscale bigger uploads
MAX_UPLOAD_MB = float(os.environ.get("SHAVIAN_MAX_UPLOAD_MB", 15))
UPLOADS_PER_HOUR = int(os.environ.get("SHAVIAN_UPLOADS_PER_HOUR", 0))  # per client IP; 0 = unlimited
ALLOWED_ORIGINS = [o.strip() for o in os.environ.get("SHAVIAN_ALLOWED_ORIGINS", "*").split(",") if o.strip()]
OCR_FALLBACK = os.environ.get("SHAVIAN_OCR_FALLBACK", "")  # used if the main backend fails (e.g. Groq quota)

app = FastAPI(title="Shavian comics")
app.add_middleware(CORSMiddleware, allow_origins=ALLOWED_ORIGINS, allow_methods=["*"], allow_headers=["*"],
                   expose_headers=["X-Font-Sizes"])
uploads = {}  # client IP -> recent upload times

gpu = threading.Lock()   # one model call at a time
jobs = OrderedDict()     # job id -> Job
status = {"ready": False, "loading": "starting"}


class Job:
    def __init__(self, name, image):
        self.id = uuid.uuid4().hex[:12]
        self.name = name
        self.image = image
        self.ctx = {}
        self.regions = {}  # id -> Region
        self.next_id = 1
        self.created = time.time()

    def add(self, region, enabled=True):
        region.id = self.next_id
        region.settings = {"enabled": enabled, "eraser": "auto", "size": "match"}
        self.regions[region.id] = region
        self.next_id += 1
        return region


def outline(mask):
    """Simplified polygon of a mask's largest outer contour, as [[x, y], ...]."""
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return []
    c = max(contours, key=cv2.contourArea)
    c = cv2.approxPolyDP(c, 1.5, True)
    return c[:, 0, :].tolist()


def region_json(r):
    return {
        "id": r.id, "kind": r.kind, "box": [int(v) for v in r.box], "outline": outline(r.layout_mask),
        "note": r.note, "letter_height": round(r.letter_height, 1),
        "english": r.ocr.get(OCR_BACKEND), "shavian": r.shavian.get(OCR_BACKEND, ""),
        "read": OCR_BACKEND in r.ocr, "font_size": getattr(r, "font_size", None),
        **r.settings,
    }


def job_json(job):
    return {"id": job.id, "name": job.name, "width": job.image.width, "height": job.image.height,
            "regions": [region_json(r) for r in job.regions.values()]}


def get_job(job_id):
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(404, "Unknown or expired job; upload the image again")
    return job


def warm_up():
    """Load everything up front so the first upload isn't slow."""
    try:
        ocr_loader = {"qwen": models.qwen, "florence": models.florence, "moondream": models.moondream}.get(
            OCR_BACKEND, lambda: ocr.groq_model() if config.groq_key() else None)
        steps = [("comic-text-detector", models.ctd), ("shaw.py + spaCy", translit.engine),
                 (f"{OCR_BACKEND} OCR", ocr_loader), ("LaMa", models.lama)]
        if OCR_FALLBACK in ("florence", "qwen", "moondream"):
            steps.append((f"{OCR_FALLBACK} OCR (fallback)", getattr(models, OCR_FALLBACK)))
        for name, fn in steps:
            status["loading"] = name
            with gpu:
                fn()
        status.update(ready=True, loading=None)
        print("[server] all models loaded")
    except Exception as e:  # keep serving; the error shows in /api/info
        status["loading"] = f"failed: {e!r}"
        raise


# ---------------------------------------------------------------- API

@app.get("/api/info")
def info():
    ocr_name = {"qwen": config.QWEN_ID, "florence": config.FLORENCE_ID, "moondream": config.MOONDREAM_ID}.get(
        OCR_BACKEND, f"Groq {config.GROQ_MODEL or '(auto model)'}")
    return {**status, "device": config.DEVICE, "ocr": ocr_name, "ocr_batch": OCR_BACKEND == "groq",
            "ocr_fallback": OCR_FALLBACK or None, "max_upload_mb": MAX_UPLOAD_MB, "dictionary": config.DICTIONARY,
            "font": config.FONT, "fonts": sorted(p.name for p in config.FONT_DIR.glob("*.otf"))}


def client_ip(request):
    # Behind a proxy (e.g. Hugging Face Spaces) the first X-Forwarded-For entry is the client
    fwd = request.headers.get("x-forwarded-for", "")
    return fwd.split(",")[0].strip() or (request.client.host if request.client else "?")


def check_rate(request):
    if not UPLOADS_PER_HOUR:
        return
    now, ip = time.time(), client_ip(request)
    recent = [t for t in uploads.get(ip, []) if now - t < 3600]
    if len(recent) >= UPLOADS_PER_HOUR:
        raise HTTPException(429, f"Limit of {UPLOADS_PER_HOUR} comics an hour reached; try again later")
    uploads[ip] = recent + [now]
    if len(uploads) > 10000:  # forget idle clients
        for k in [k for k, v in uploads.items() if now - v[-1] > 3600]:
            del uploads[k]


@app.post("/api/jobs")
async def create_job(request: Request, file: UploadFile = File(...)):
    """Upload an image and detect its text. OCR happens separately, region by region."""
    check_rate(request)
    data = await file.read(int(MAX_UPLOAD_MB * 1024 * 1024) + 1)
    if len(data) > MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(413, f"Images up to {MAX_UPLOAD_MB:g} MB, please")
    try:
        image = Image.open(io.BytesIO(data))
        image = ImageOps.exif_transpose(image).convert("RGBA")
    except Exception:
        raise HTTPException(400, "That doesn't look like an image")
    if max(image.size) > MAX_SIDE:
        image.thumbnail((MAX_SIDE, MAX_SIDE), Image.Resampling.LANCZOS)
    job = Job(Path(file.filename or "comic").stem, image)
    with gpu:
        regions, skipped = detect_ctd.detect(image, ctx=job.ctx)
        for r in regions:
            job.add(r)
        for box, reason in skipped:  # offered, but off by default
            job.add(detect_ctd.region_from_box(job.ctx, box, note=f"skipped: {reason}"), enabled=False)
    jobs[job.id] = job
    while len(jobs) > MAX_JOBS:
        jobs.popitem(last=False)
    return job_json(job)


@app.get("/api/jobs/{job_id}")
def read_job(job_id: str):
    return job_json(get_job(job_id))


@app.get("/api/jobs/{job_id}/original.png")
def original(job_id: str):
    buf = io.BytesIO()
    get_job(job_id).image.convert("RGB").save(buf, "PNG")
    return Response(buf.getvalue(), media_type="image/png")


class Ids(BaseModel):
    ids: list[int]


@app.post("/api/jobs/{job_id}/ocr")
def read_regions(job_id: str, body: Ids):
    """OCR + transliterate the given regions. Returns them."""
    job = get_job(job_id)
    regs = [job.regions[i] for i in body.ids if i in job.regions]
    crops = [ocr.crop_for_ocr(job.image, r) for r in regs]
    with gpu:
        try:
            texts = ocr.BACKENDS[OCR_BACKEND](crops)
        except Exception as e:
            if not OCR_FALLBACK:
                raise
            print(f"[server] {OCR_BACKEND} OCR failed ({e!r}); using {OCR_FALLBACK}")
            texts = ocr.BACKENDS[OCR_FALLBACK](crops)
    for r, t in zip(regs, texts):
        r.ocr[OCR_BACKEND] = t
    for r, s in zip(regs, translit.transliterate(texts)):
        r.shavian[OCR_BACKEND] = s
    return [region_json(r) for r in regs]


class Box(BaseModel):
    box: list[float]


@app.post("/api/jobs/{job_id}/regions")
def add_region(job_id: str, body: Box):
    """A region from a box drawn by hand (not read yet)."""
    job = get_job(job_id)
    x1, y1, x2, y2 = body.box
    if abs(x2 - x1) < 4 or abs(y2 - y1) < 4:
        raise HTTPException(400, "Box too small")
    box = (min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))
    return region_json(job.add(detect_ctd.region_from_box(job.ctx, box)))


class Edit(BaseModel):
    english: str | None = None
    shavian: str | None = None
    enabled: bool | None = None
    eraser: str | None = None
    size: str | None = None


@app.patch("/api/jobs/{job_id}/regions/{rid}")
def edit_region(job_id: str, rid: int, body: Edit):
    """Change a region. New English is re-transliterated unless Shavian is given too."""
    job = get_job(job_id)
    r = job.regions.get(rid)
    if r is None:
        raise HTTPException(404, "No such region")
    if body.english is not None:
        r.ocr[OCR_BACKEND] = body.english.strip()
        if body.shavian is None:
            r.shavian[OCR_BACKEND] = translit.transliterate([r.ocr[OCR_BACKEND]])[0]
    if body.shavian is not None:
        r.shavian[OCR_BACKEND] = body.shavian.strip()
    for key in ("enabled", "eraser", "size"):
        value = getattr(body, key)
        if value is not None:
            if key == "eraser" and value not in pipeline.erase.BACKENDS:
                raise HTTPException(400, f"Unknown eraser {value}")
            if key == "size" and value not in ("match", "fit") and not value.isdigit():
                raise HTTPException(400, "size must be match, fit or a number")
            r.settings[key] = value
    return region_json(r)


@app.delete("/api/jobs/{job_id}/regions/{rid}")
def delete_region(job_id: str, rid: int):
    get_job(job_id).regions.pop(rid, None)
    return {"ok": True}


def erased_image(job):
    """The image with every active region's text removed. Cached: erasing (LaMa) is the
    slow part, and it only depends on which regions are on and how they're erased."""
    regs = list(job.regions.values())
    key = tuple((r.id, bool(r.shavian.get(OCR_BACKEND)) and r.settings["enabled"], r.settings["eraser"]) for r in regs)
    with gpu:
        if getattr(job, "erase_key", None) != key:
            job.erased = pipeline.erase_only(job.image, regs, "auto", OCR_BACKEND)
            job.erase_key = key
        return job.erased


@app.get("/api/jobs/{job_id}/render.png")
def render(job_id: str, stage: str = "final"):
    """stage = "final" (erased + Shavian) or "erased" (text removed only)."""
    job = get_job(job_id)
    out = erased_image(job).copy()
    sizes = ""
    if stage != "erased":
        for r in job.regions.values():
            text = r.shavian.get(OCR_BACKEND)
            if text and r.settings["enabled"]:
                r.note_layout, r.font_size = layout.draw_text(out, r, text, r.settings["size"])
        sizes = ",".join(f"{r.id}:{r.font_size}" for r in job.regions.values()
                         if r.settings["enabled"] and r.shavian.get(OCR_BACKEND) and getattr(r, "font_size", None))
    buf = io.BytesIO()
    out.convert("RGB").save(buf, "PNG", compress_level=1)
    return Response(buf.getvalue(), media_type="image/png",
                    headers={"Cache-Control": "no-store", "X-Font-Sizes": sizes})


@app.get("/fonts/{name}")
def font_file(name: str):
    path = config.FONT_DIR / Path(name).name
    if not path.exists():
        raise HTTPException(404)
    return FileResponse(path)


@app.get("/")
def index():
    return FileResponse(WEB_DIR / "index.html")


@app.get("/report")
def report_redirect():
    return RedirectResponse("/report/")


app.mount("/web", StaticFiles(directory=WEB_DIR), name="web")
if REPORT_DIR.exists():
    app.mount("/report", StaticFiles(directory=REPORT_DIR, html=True), name="report")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default=os.environ.get("SHAVIAN_HOST", "127.0.0.1"))
    ap.add_argument("--port", type=int, default=int(os.environ.get("SHAVIAN_PORT", 8765)))
    args = ap.parse_args()
    threading.Thread(target=warm_up, daemon=True).start()
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
