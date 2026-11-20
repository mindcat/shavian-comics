#!/usr/bin/env python3
"""
Scores OCR backends against a reference transcription.

    uv run python tools/score_ocr.py [results.json] [reference.json]

results.json is written by src/compare.py (src/img/out/compare/results.json).
The reference (default src/img/in/ocr_reference.json) maps "<strip>@x1,y1,x2,y2"
(a region's box from comic-text-detector) to what the region really says; ""
means the region has no text to read (a signature, a sound-effect fragment).

Text is compared case-insensitively with punctuation removed (apostrophes inside
words are kept), since casing is redone downstream anyway. Writes a summary to
src/img/out/compare/scores.html and prints it.
"""
import html
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "src/img/out/compare/results.json"
REFERENCE = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / "src/img/in/ocr_reference.json"


def normalise(text):
    text = text.lower().replace("’", "'").replace("‘", "'")
    text = re.sub(r"(?<![a-z])'|'(?![a-z])", " ", text)   # quotes, not apostrophes
    text = re.sub(r"[^a-z0-9' ]+", " ", text)
    return " ".join(text.split())


def edits(a, b):
    """Levenshtein distance between two sequences."""
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def key(region):
    return f"{region['strip']}@{','.join(map(str, region['box']))}"


def main():
    results = json.loads(RESULTS.read_text(encoding="utf-8"))
    reference = {k: v for k, v in json.loads(REFERENCE.read_text(encoding="utf-8")).items() if not k.startswith("_")}
    backends = results["backends"]
    totals = {b: {"word_err": 0, "words": 0, "char_err": 0, "chars": 0, "exact": 0, "n": 0,
                  "empty_ok": 0, "empty_n": 0} for b in backends}
    per_strip = {}
    missing = 0
    for r in results["regions"]:
        k = key(r)
        if k not in reference:
            missing += 1
            continue
        ref = normalise(reference[k])
        for b in backends:
            got = normalise(r["text"].get(b, ""))
            t = totals[b]
            if not ref:  # nothing to read: credit for returning nothing
                t["empty_n"] += 1
                t["empty_ok"] += not got
                continue
            we, ce = edits(ref.split(), got.split()), edits(ref, got)
            t["word_err"] += we
            t["words"] += len(ref.split())
            t["char_err"] += ce
            t["chars"] += len(ref)
            t["exact"] += ref == got
            t["n"] += 1
            s = per_strip.setdefault(r["strip"], {}).setdefault(b, [0, 0])
            s[0] += we
            s[1] += len(ref.split())

    rows = []
    for b in sorted(backends, key=lambda b: totals[b]["word_err"] / max(totals[b]["words"], 1)):
        t = totals[b]
        rows.append({
            "backend": b,
            "wer": 100 * t["word_err"] / max(t["words"], 1),
            "cer": 100 * t["char_err"] / max(t["chars"], 1),
            "exact": 100 * t["exact"] / max(t["n"], 1),
            "empty": f"{t['empty_ok']}/{t['empty_n']}",
            "secs": results["seconds_per_region"].get(b, 0),
        })
    print(f"{len(results['regions']) - missing} regions scored ({missing} without a reference)\n")
    print(f"{'backend':16} {'WER':>6} {'CER':>6} {'exact':>6}  {'no-text ok':>10}  {'s/region (GPU)':>14}")
    for row in rows:
        print(f"{row['backend']:16} {row['wer']:5.1f}% {row['cer']:5.1f}% {row['exact']:5.0f}%  {row['empty']:>10}  {row['secs']:14.2f}")
    print("\nWER by strip:")
    strips = sorted(per_strip)
    print(f"{'':16} " + " ".join(f"{s[:10]:>10}" for s in strips))
    for row in rows:
        b = row["backend"]
        print(f"{b:16} " + " ".join(
            f"{100 * per_strip[s].get(b, [0, 1])[0] / max(per_strip[s].get(b, [0, 1])[1], 1):9.0f}%" for s in strips))

    head = "".join(f"<th>{h}</th>" for h in ("backend", "word error rate", "character error rate",
                                             "regions exactly right", "no-text regions left empty", "s/region (GPU)"))
    body = "".join(
        f"<tr><td>{html.escape(r['backend'])}</td><td>{r['wer']:.1f}%</td><td>{r['cer']:.1f}%</td>"
        f"<td>{r['exact']:.0f}%</td><td>{r['empty']}</td><td>{r['secs']:.2f}</td></tr>" for r in rows)
    strip_head = "".join(f"<th>{html.escape(s)}</th>" for s in strips)
    strip_body = "".join(
        f"<tr><td>{html.escape(r['backend'])}</td>" + "".join(
            f"<td>{100 * per_strip[s].get(r['backend'], [0, 1])[0] / max(per_strip[s].get(r['backend'], [0, 1])[1], 1):.0f}%</td>"
            for s in strips) + "</tr>" for r in rows)
    page = f"""<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>OCR scores</title><style>body{{font:14px system-ui;margin:16px}} table{{border-collapse:collapse;margin:10px 0}}
td,th{{border:1px solid #ccc;padding:4px 8px;text-align:right}} td:first-child,th:first-child{{text-align:left}}
@media (prefers-color-scheme: dark){{body{{background:#161615;color:#e8e7e2}} td,th{{border-color:#444}}}}</style>
<h1>OCR scores against the reference transcription</h1>
<p>Lower is better for error rates. Case and punctuation are ignored. <a href="index.html">Full side-by-side report</a></p>
<table><tr>{head}</tr>{body}</table><h2>Word error rate by strip</h2>
<table><tr><th>backend</th>{strip_head}</tr>{strip_body}</table>"""
    (RESULTS.parent / "scores.html").write_text(page, encoding="utf-8")


if __name__ == "__main__":
    main()
