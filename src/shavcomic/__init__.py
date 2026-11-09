"""Shavian comics pipeline: detect text -> OCR -> transliterate -> erase -> typeset.

Each stage has interchangeable backends so they can be compared side by side
(see compare.py):

    detect:  "yolo" (YOLO bubble masks matched to moondream boxes)
             "ctd"  (comic-text-detector text lines, grouped by bubble)
    ocr:     "moondream" | "qwen"
    erase:   "fill" | "telea" | "lama"
    size:    "fit"   (largest size that fits the bubble)
             "match" (match the original lettering height, shrink if needed)
"""
