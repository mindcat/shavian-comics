#!/usr/bin/env python3
"""
Transliterate comic speech bubbles into Shavian.

    uv run python src/shavian-comics.py [images...] [--detector ctd|yolo] [--ocr qwen|moondream]
                                        [--eraser auto|fill|telea|lama] [--size match|fit] [--no-pos]

With no images, converts everything in src/img/in/ into src/img/out/.
To compare the backends side by side, use src/compare.py instead.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from shavcomic import config, models, pipeline  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("images", nargs="*", type=Path, help="images to convert (default: src/img/in/*)")
    ap.add_argument("-o", "--out", type=Path, default=config.OUTPUT_DIR, help="output directory")
    ap.add_argument("--detector", choices=list(pipeline.DETECTORS), default="ctd")
    ap.add_argument("--ocr", choices=["moondream", "qwen"], default="qwen")
    ap.add_argument("--eraser", choices=["auto", "fill", "telea", "lama"], default="auto",
                    help="auto = flat fill in bubbles, LaMa elsewhere")
    ap.add_argument("--size", choices=["match", "fit"], default="match",
                    help="match the original lettering size, or use the largest size that fits")
    ap.add_argument("--no-pos", action="store_true", help="turn off part-of-speech tagging in shaw.py")
    ap.add_argument("--debug", action="store_true", help="also save detection overlays")
    args = ap.parse_args()

    paths = args.images or sorted(p for ext in ("png", "jpg", "jpeg", "webp") for p in config.INPUT_DIR.glob(f"*.{ext}"))
    for p in (config.YOLO_MODEL, config.CTD_MODEL, config.DICT_DIR / f"{config.DICTIONARY}.dict", config.FONT_DIR / config.FONT):
        if not p.exists():
            sys.exit(f"Missing {p} (run src/init.py)")
    print(f"Device: {config.DEVICE}; {len(paths)} image(s)")

    images = pipeline.load_images(paths)
    detections = pipeline.detect_all(images, args.detector)
    if args.ocr != "moondream":
        models.unload("moondream")
    pipeline.ocr_all(images, detections, args.ocr)
    pipeline.unload_ocr()
    pipeline.translit_all([detections], args.ocr, pos_tagging=not args.no_pos)

    args.out.mkdir(parents=True, exist_ok=True)
    for name, img in images.items():
        regions, skipped = detections[name]
        for i, r in enumerate(regions, 1):
            print(f"  {name} #{i}: {r.ocr.get(args.ocr, '')!r} -> {r.shavian.get(args.ocr, '')}")
        out = pipeline.render(img, regions, args.eraser, args.ocr, args.size)
        out.save(args.out / f"{name}_shavian.png")
        if args.debug:
            pipeline.overlay(img, regions, skipped).convert("RGB").save(args.out / f"{name}_detect.png")
        print(f"Saved {args.out / f'{name}_shavian.png'}")


if __name__ == "__main__":
    main()
