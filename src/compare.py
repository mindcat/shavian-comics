#!/usr/bin/env python3
"""
Runs both pipelines over the test strips and writes a side-by-side HTML report,
stage by stage, to src/img/out/compare/index.html.

  A = YOLO bubbles + moondream          B = comic-text-detector text lines

Stages compared: detection (A vs B), OCR (moondream vs Qwen, on both), POS
tagging (off vs on), erasing (fill / telea / LaMa) and final output.

    uv run python src/compare.py [image ...]
"""
import html
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from shavcomic import config, models, pipeline  # noqa: E402

OUT = config.OUTPUT_DIR / "compare"
ERASERS = ["fill", "telea", "lama"]


def save(img, name, stem):
    d = OUT / stem
    d.mkdir(parents=True, exist_ok=True)
    img.convert("RGB").save(d / f"{name}.png")
    return f"{stem}/{name}.png"


def main():
    args = [Path(a) for a in sys.argv[1:]]
    paths = args or sorted(p for ext in ("png", "jpg", "jpeg", "webp") for p in config.INPUT_DIR.glob(f"*.{ext}"))
    images = pipeline.load_images(paths)
    timings = {}

    def timed(label, fn, *a, **k):
        t = time.time()
        r = fn(*a, **k)
        timings[label] = timings.get(label, 0) + time.time() - t
        return r

    # Detection + moondream OCR (moondream is already loaded for pipeline A's detection)
    det = {"A": timed("detect A (yolo+moondream)", pipeline.detect_all, images, "yolo"),
           "B": timed("detect B (ctd)", pipeline.detect_all, images, "ctd")}
    for p in "AB":
        timed("ocr moondream", pipeline.ocr_all, images, det[p], "moondream")
    models.unload("moondream")
    for p in "AB":
        timed("ocr qwen", pipeline.ocr_all, images, det[p], "qwen")
    models.unload("qwen")

    both = [det["A"], det["B"]]
    timed("transliterate", pipeline.translit_all, both, "moondream")
    timed("transliterate", pipeline.translit_all, both, "qwen")
    timed("transliterate", pipeline.translit_all, both, "moondream", pos_tagging=False, key="moondream-nopos")

    report = {}
    for stem, img in images.items():
        print(f"[render] {stem}")
        rep = report[stem] = {"original": save(img, "0-original", stem)}
        for p in "AB":
            regions, skipped = det[p][stem]
            rep[f"{p}-detect"] = save(pipeline.overlay(img, regions, skipped), f"1-{p}-detect", stem)
            for e in ERASERS:
                rep[f"{p}-erase-{e}"] = save(timed(f"erase {e}", pipeline.erase_only, img, regions, e, "moondream"),
                                             f"2-{p}-erase-{e}", stem)
        a_regions = det["A"][stem][0]
        rep["A-baseline"] = save(pipeline.render(img, a_regions, "fill", "moondream-nopos", "fit"), "3-A-baseline", stem)
        for p in "AB":
            for e in ("fill", "lama"):
                rep[f"{p}-final-{e}"] = save(
                    pipeline.render(img, det[p][stem][0], e, "moondream", "match"), f"3-{p}-final-{e}", stem)
        rep["B-final-lama-qwen"] = save(
            pipeline.render(img, det["B"][stem][0], "lama", "qwen", "match"), "3-B-final-lama-qwen", stem)
    models.unload()

    write_html(images, det, report, timings)
    print(f"\nReport: {OUT / 'index.html'}")
    for k, v in timings.items():
        print(f"  {k:28s} {v:7.1f}s")


def fig(src, caption):
    return (f'<figure><a href="{src}" target="_blank"><img src="{src}" loading="lazy"></a>'
            f"<figcaption>{html.escape(caption)}</figcaption></figure>")


def region_table(regions, label):
    rows = []
    for i, r in enumerate(regions, 1):
        md, qw = r.ocr.get("moondream", ""), r.ocr.get("qwen", "")
        diff = ' class="diff"' if md.lower() != qw.lower() else ""
        nopos, pos = r.shavian.get("moondream-nopos", ""), r.shavian.get("moondream", "")
        rows.append(
            f"<tr><td>{i}</td><td>{r.kind}<br><small>{html.escape(r.note)}<br>"
            f"letters ~{r.letter_height:.0f}px</small></td>"
            f"<td{diff}>{html.escape(md) or '<em>(none)</em>'}</td><td{diff}>{html.escape(qw) or '<em>(none)</em>'}</td>"
            f"<td class='shaw'>{html.escape(pos)}</td>"
            f"<td class='shaw'>{html.escape(r.shavian.get('qwen', ''))}</td>"
            + (f"<td class='shaw diff'>{html.escape(nopos)}</td>" if nopos != pos else "<td><small>same</small></td>")
            + "</tr>")
    return (f"<h4>{label}</h4><table><tr><th>#</th><th>region</th><th>moondream OCR</th><th>Qwen OCR</th>"
            "<th>Shavian (moondream, POS)</th><th>Shavian (Qwen, POS)</th><th>without POS tagging</th></tr>"
            + "".join(rows) + "</table>")


def write_html(images, det, report, timings):
    parts = []
    nav = " · ".join(f'<a href="#{html.escape(s)}">{html.escape(s)}</a>' for s in images)
    for stem in images:
        rep = report[stem]
        a_regs, a_skip = det["A"][stem]
        b_regs, b_skip = det["B"][stem]
        parts.append(f"""
<section id="{html.escape(stem)}"><h2>{html.escape(stem)}</h2>
<h3>1 · Detection</h3>
<p class="legend">green = bubble, orange = free text (B only), red = text pixels, grey box = found but skipped. Numbers match the tables.</p>
<div class="row">{fig(rep['original'], 'original')}
{fig(rep['A-detect'], f'A: YOLO + moondream — {len(a_regs)} regions, {len(a_skip)} skipped')}
{fig(rep['B-detect'], f'B: comic-text-detector — {len(b_regs)} regions, {len(b_skip)} skipped')}</div>
<details><summary>skipped detections</summary><ul>
{''.join(f'<li>A {b}: {html.escape(why)}</li>' for b, why in a_skip)}
{''.join(f'<li>B {b}: {html.escape(why)}</li>' for b, why in b_skip)}</ul></details>
<h3>2 · OCR and transliteration</h3>
<p class="legend">Yellow = the two OCR models disagree / POS tagging changed the Shavian.</p>
{region_table(a_regs, 'A regions')}
{region_table(b_regs, 'B regions')}
<h3>3 · Erasing (moondream text, regions with text only)</h3>
<div class="row">{''.join(fig(rep[f'A-erase-{e}'], f'A + {e}') for e in ERASERS)}</div>
<div class="row">{''.join(fig(rep[f'B-erase-{e}'], f'B + {e}') for e in ERASERS)}</div>
<h3>4 · Final</h3>
<div class="row">{fig(rep['A-baseline'], 'A baseline: previous commit (fill, fit sizing, no POS)')}
{fig(rep['A-final-fill'], 'A: fill, matched size, POS')}{fig(rep['A-final-lama'], 'A: LaMa, matched size, POS')}</div>
<div class="row">{fig(rep['B-final-fill'], 'B: fill, matched size, POS')}{fig(rep['B-final-lama'], 'B: LaMa, matched size, POS')}
{fig(rep['B-final-lama-qwen'], 'B: LaMa, matched size, POS, Qwen OCR')}</div>
</section>""")
    t = "".join(f"<li>{html.escape(k)}: {v:.1f}s</li>" for k, v in timings.items())
    page = f"""<!doctype html><html><head><meta charset="utf-8"><title>Pipeline comparison</title>
<style>
body{{font:14px/1.45 system-ui,sans-serif;margin:16px;background:#f6f6f4;color:#222}}
section{{background:#fff;border-radius:8px;padding:12px 16px;margin:18px 0;box-shadow:0 1px 3px #0002}}
.row{{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:10px;margin:8px 0}}
figure{{margin:0}} img{{width:100%;border:1px solid #ccc}} figcaption{{font-size:12px;color:#555}}
table{{border-collapse:collapse;width:100%;font-size:13px;margin-bottom:10px}}
td,th{{border:1px solid #ddd;padding:4px 6px;vertical-align:top;text-align:left}}
.shaw{{font-size:16px}} .diff{{background:#fff4c2}} .legend{{color:#666;font-size:12px}}
nav{{position:sticky;top:0;background:#f6f6f4;padding:6px 0;z-index:1}}
</style></head><body><h1>Shavian comics: pipeline A vs B</h1>
<nav>{nav}</nav><details><summary>timings</summary><ul>{t}</ul></details>{''.join(parts)}</body></html>"""
    (OUT / "index.html").write_text(page, encoding="utf-8")


if __name__ == "__main__":
    main()
