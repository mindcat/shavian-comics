#!/usr/bin/env python3
"""
Compares OCR backends on the test strips and writes an HTML report to
src/img/out/compare/index.html (also served at /report/ by server.py).

Detection is comic-text-detector (pipeline B); each OCR backend's text is
transliterated and typeset (auto eraser, matched size) so they can be judged
side by side. Backends that aren't available (e.g. Groq without a key) are
listed in the report with the reason.

    uv run python src/compare.py [--ocr qwen,florence[,groq]] [--detector ctd] [--cpu-bench] [image ...]
"""
import argparse
import html
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from shavcomic import config, models, ocr, pipeline  # noqa: E402

OUT = config.OUTPUT_DIR / "compare"
LABELS = {
    "qwen": "Qwen3-VL-2B",
    "florence": "Florence-2 (large)",
    "florence-base": "Florence-2-base",
    "florence-large": "Florence-2-large",
    "trocr-small": "TrOCR-small (per line)",
    "trocr-base": "TrOCR-base (per line)",
    "smolvlm-256m": "SmolVLM-256M",
    "smolvlm-500m": "SmolVLM-500M",
    "groq": "Groq (hosted API)",
    "moondream": "moondream2",
}


def save(img, name, stem):
    d = OUT / stem
    d.mkdir(parents=True, exist_ok=True)
    img.convert("RGB").save(d / f"{name}.png")
    return f"{stem}/{name}.png"


def cpu_bench(images, det, n=6):
    """Seconds per region for Florence-2 base/large on CPU (what a free hosted Space would have)."""
    import torch

    results = {}
    crops = [ocr.crop_for_ocr(images[s], r) for s, (regs, _) in det.items() for r in regs][:n]
    saved = (config.DEVICE, config.FLORENCE_ID)
    threads = torch.get_num_threads()
    torch.set_num_threads(2)  # a free Hugging Face Space has 2 vCPUs
    try:
        for mid in ("florence-community/Florence-2-base", "florence-community/Florence-2-large"):
            config.DEVICE, config.FLORENCE_ID = "cpu", mid
            models.unload("florence")
            ocr.BACKENDS["florence"](crops[:1])  # warm up
            t = time.time()
            ocr.BACKENDS["florence"](crops)
            results[mid.split("/")[-1]] = (time.time() - t) / len(crops)
            models.unload("florence")
    finally:
        config.DEVICE, config.FLORENCE_ID = saved
        torch.set_num_threads(threads)
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("images", nargs="*", type=Path)
    ap.add_argument("--ocr", default="qwen,florence", help="comma-separated; add groq explicitly (uses the free API quota)")
    ap.add_argument("--detector", default="ctd", choices=list(pipeline.DETECTORS))
    ap.add_argument("--cpu-bench", action="store_true", help="also time Florence-2 on 2 CPU threads")
    args = ap.parse_args()

    paths = args.images or sorted(p for ext in ("png", "jpg", "jpeg", "webp") for p in config.INPUT_DIR.glob(f"*.{ext}"))
    images = pipeline.load_images(paths)
    backends = [b.strip() for b in args.ocr.split(",") if b.strip()]
    timings, unavailable = {}, {}

    t = time.time()
    det = pipeline.detect_all(images, args.detector)
    timings[f"detect ({args.detector})"] = time.time() - t
    models.unload("moondream")

    n_regions = sum(len(r) for r, _ in det.values())
    ran = []
    for b in backends:
        if b == "groq" and not config.groq_key():
            unavailable[b] = f"no API key (set GROQ_API_KEY or write it to {config.GROQ_KEY_FILE})"
            continue
        t = time.time()
        try:
            pipeline.ocr_all(images, det, b)
        except Exception as e:
            unavailable[b] = f"failed: {e}"
            traceback.print_exc()
            continue
        timings[f"ocr {b}"] = time.time() - t
        ran.append(b)
        models.unload("qwen", "florence", "moondream")
        pipeline.translit_all([det], b)

    bench = cpu_bench(images, det) if args.cpu_bench else {}

    # Every region's text per backend, for scoring against a reference (tools/score_ocr.py)
    import json
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "results.json").write_text(json.dumps({
        "backends": ran,
        "seconds_per_region": {b: timings[f"ocr {b}"] / max(n_regions, 1) for b in ran},
        "regions": [{"strip": stem, "n": i, "kind": r.kind, "box": [int(v) for v in r.box],
                     "text": {b: r.ocr.get(b, "") for b in ran}}
                    for stem, (regs, _) in det.items() for i, r in enumerate(regs, 1)],
    }, indent=1, ensure_ascii=False), encoding="utf-8")

    report = {}
    for stem, img in images.items():
        print(f"[render] {stem}")
        regions, skipped = det[stem]
        rep = report[stem] = {"original": save(img, "0-original", stem),
                              "detect": save(pipeline.overlay(img, regions, skipped), "1-detect", stem)}
        for b in ran:
            rep[b] = save(pipeline.render(img, regions, "auto", b, "match"), f"3-final-{b}", stem)
    models.unload()

    write_html(images, det, report, ran, unavailable, timings, n_regions, bench)
    print(f"\nReport: {OUT / 'index.html'}")
    for k, v in timings.items():
        print(f"  {k:24s} {v:7.1f}s")
    for k, v in bench.items():
        print(f"  CPU (2 threads) {k}: {v:.1f}s/region")


def fig(src, caption):
    return (f'<figure><a href="{src}" target="_blank"><img src="{src}" loading="lazy"></a>'
            f"<figcaption>{html.escape(caption)}</figcaption></figure>")


def region_table(regions, ran):
    head = "".join(f"<th>{html.escape(LABELS.get(b, b))}</th>" for b in ran)
    rows = []
    for i, r in enumerate(regions, 1):
        texts = [r.ocr.get(b, "") for b in ran]
        differ = len({t.lower().strip(" .!?") for t in texts}) > 1
        cells = "".join(
            f"<td{' class=diff' if differ else ''}>{html.escape(r.ocr.get(b, '')) or '<em>(none)</em>'}"
            f"<div class='shaw'>{html.escape(r.shavian.get(b, ''))}</div></td>" for b in ran)
        rows.append(f"<tr><td>{i}</td><td>{r.kind}<br><small>{html.escape(r.note)}</small></td>{cells}</tr>")
    return f"<table><tr><th>#</th><th>region</th>{head}</tr>{''.join(rows)}</table>"


def write_html(images, det, report, ran, unavailable, timings, n_regions, bench):
    nav = " · ".join(f'<a href="#{html.escape(s)}">{html.escape(s)}</a>' for s in images)
    agree = sum(1 for regs, _ in det.values() for r in regs
                if len({r.ocr.get(b, "").lower().strip(" .!?") for b in ran}) == 1)
    summary = [f"<li>{n_regions} text regions across {len(images)} strips; all OCR backends agree (ignoring case and end punctuation) on {agree}.</li>"]
    for b in ran:
        summary.append(f"<li>{html.escape(LABELS.get(b, b))}: {timings[f'ocr {b}']:.0f}s total, "
                       f"{timings[f'ocr {b}'] / max(n_regions, 1):.2f}s per region on {config.DEVICE if b != 'groq' else 'Groq'}</li>")
    for b, why in unavailable.items():
        summary.append(f"<li><b>{html.escape(LABELS.get(b, b))} not run:</b> {html.escape(why)}</li>")
    for k, v in bench.items():
        summary.append(f"<li>CPU, 2 threads (like a free Hugging Face Space): {html.escape(k)} {v:.1f}s per region</li>")

    parts = []
    for stem in images:
        rep = report[stem]
        regs, skip = det[stem]
        finals = "".join(fig(rep[b], LABELS.get(b, b)) for b in ran)
        parts.append(f"""
<section id="{html.escape(stem)}"><h2>{html.escape(stem)}</h2>
<div class="row">{fig(rep['original'], 'original')}{fig(rep['detect'], f'detection: {len(regs)} regions, {len(skip)} skipped')}</div>
<h3>OCR → Shavian</h3>
<p class="legend">Yellow = the backends disagree. Numbers match the detection image.</p>
{region_table(regs, ran)}
<h3>Final, per OCR backend</h3>
<div class="row">{finals}</div>
</section>""")
    page = f"""<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>OCR comparison</title>
<style>
body{{font:14px/1.45 system-ui,sans-serif;margin:16px;background:#f6f6f4;color:#222}}
section,.summary{{background:#fff;border-radius:8px;padding:12px 16px;margin:18px 0;box-shadow:0 1px 3px #0002}}
.row{{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:10px;margin:8px 0}}
figure{{margin:0}} img{{width:100%;border:1px solid #ccc}} figcaption{{font-size:12px;color:#555}}
table{{border-collapse:collapse;width:100%;font-size:13px}} td,th{{border:1px solid #ddd;padding:4px 6px;vertical-align:top;text-align:left}}
.shaw{{font-size:16px;margin-top:3px}} .diff{{background:#fff4c2}} .legend{{color:#666;font-size:12px}}
nav{{position:sticky;top:0;background:#f6f6f4;padding:6px 0;z-index:1;overflow-x:auto;white-space:nowrap}}
@media (prefers-color-scheme: dark){{body{{background:#161615;color:#e8e7e2}} section,.summary{{background:#222120}} nav{{background:#161615}}
td,th{{border-color:#3a3936}} .diff{{background:#4a4220}} figcaption,.legend{{color:#a3a29c}}}}
</style></head><body><h1>OCR comparison (comic-text-detector + auto eraser)</h1>
<nav>{nav}</nav><div class="summary"><ul>{''.join(summary)}</ul></div>{''.join(parts)}</body></html>"""
    (OUT / "index.html").write_text(page, encoding="utf-8")


if __name__ == "__main__":
    main()
