#!/usr/bin/env python3
"""
Rendering regression check: did a pipeline change alter the output, and where?

    uv run python tools/regress.py [REF]     # default REF: HEAD

Renders every strip in src/img/in/ with comic-text-detector + the auto eraser, using
the same placeholder Shavian for every region (so no OCR or GPU is needed), once
with the code at REF (via a temporary git worktree) and once with the working tree.
Prints which regions changed and writes a before/after sheet of them to
src/img/out/regress/sheet.png.
"""
import pickle
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "src" / "img" / "out" / "regress"
PLACEHOLDER = "𐑞 𐑒𐑨𐑑 𐑕𐑨𐑑 𐑪𐑯 𐑞 𐑥𐑨𐑑 𐑯 𐑦𐑑 𐑢𐑪𐑟 𐑣𐑨𐑐𐑦"

RENDER = r'''
import os, pickle, sys
from pathlib import Path
os.environ["SHAVIAN_DEVICE"] = "cpu"
sys.path.insert(0, sys.argv[1])
from shavcomic import config, pipeline
inputs, out = Path(sys.argv[2]), Path(sys.argv[3])
out.mkdir(parents=True, exist_ok=True)
paths = sorted(p for e in ("png", "jpg", "jpeg", "webp") for p in inputs.glob(f"*.{e}"))
boxes = {}
for stem, img in pipeline.load_images(paths).items():
    regions, _ = pipeline.DETECTORS["ctd"](img)
    for r in regions:
        r.shavian["x"] = sys.argv[4]
    pipeline.render(img, regions, "auto", "x", "match").convert("RGB").save(out / f"{stem}.png")
    boxes[stem] = [(r.kind, tuple(int(v) for v in r.box)) for r in regions]
pickle.dump(boxes, open(out / "boxes.pkl", "wb"))
'''


def render(src_dir, out_dir):
    subprocess.run([sys.executable, "-c", RENDER, str(src_dir), str(ROOT / "src" / "img" / "in"), str(out_dir), PLACEHOLDER],
                   check=True, cwd=src_dir, stdout=subprocess.DEVNULL)


def main():
    import numpy as np
    from PIL import Image, ImageDraw

    ref = sys.argv[1] if len(sys.argv) > 1 else "HEAD"
    before, after = OUT / "before", OUT / "after"
    with tempfile.TemporaryDirectory() as tmp:
        wt = Path(tmp) / "wt"
        subprocess.run(["git", "worktree", "add", "-q", "--detach", str(wt), ref], check=True, cwd=ROOT)
        try:
            for d in ("models", "dict", "fonts"):  # downloaded assets aren't in git
                target = wt / "src" / d
                if target.exists():
                    subprocess.run(["rm", "-rf", str(target)], check=True)
                target.symlink_to(ROOT / "src" / d)
            print(f"rendering with {ref}…")
            render(wt / "src", before)
        finally:
            subprocess.run(["git", "worktree", "remove", "--force", str(wt)], cwd=ROOT)
    print("rendering with the working tree…")
    render(ROOT / "src", after)

    boxes = pickle.load(open(before / "boxes.pkl", "rb"))
    boxes_after = pickle.load(open(after / "boxes.pkl", "rb"))
    tiles = []
    for stem, regs in boxes.items():
        a, b = Image.open(before / f"{stem}.png"), Image.open(after / f"{stem}.png")
        diff = np.abs(np.array(a).astype(int) - np.array(b).astype(int)).max(axis=2) > 24
        changed = []
        for kind, (x1, y1, x2, y2) in regs:
            box = (max(0, x1 - 30), max(0, y1 - 30), min(a.width, x2 + 30), min(a.height, y2 + 30))
            frac = diff[box[1]:box[3], box[0]:box[2]].mean()
            if frac > 0.002:
                changed.append(f"{kind}@{x1},{y1} {frac * 100:.1f}%")
                ca, cb = a.crop(box), b.crop(box)
                h = 130
                w = max(1, int(ca.width * h / ca.height))
                t = Image.new("RGB", (2 * w + 6, h + 14), "white")
                t.paste(ca.resize((w, h)), (0, 14))
                t.paste(cb.resize((w, h)), (w + 6, 14))
                ImageDraw.Draw(t).text((2, 1), f"{stem} {kind}  before | after", fill=(0, 0, 200))
                tiles.append(t)
        note = "" if regs == boxes_after.get(stem) else "  [regions differ]"
        print(f"{stem:24} {diff.mean() * 100:5.2f}% of pixels changed{note} | {', '.join(changed) or 'no region changed'}")

    if tiles:
        width, rows, row, x = 1800, [], [], 0
        for t in tiles:
            if x + t.width > width and row:
                rows.append(row)
                row, x = [], 0
            row.append(t)
            x += t.width + 8
        rows.append(row)
        sheet = Image.new("RGB", (width, sum(max(t.height for t in r) + 8 for r in rows)), (235, 235, 235))
        y = 0
        for r in rows:
            x = 0
            for t in r:
                sheet.paste(t, (x, y))
                x += t.width + 8
            y += max(t.height for t in r) + 8
        sheet.save(OUT / "sheet.png")
        print(f"\n{len(tiles)} changed regions: {OUT / 'sheet.png'}")
    else:
        print("\nNo changes.")


if __name__ == "__main__":
    main()
