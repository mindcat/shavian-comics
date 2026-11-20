#!/usr/bin/env python3
"""
Generates src/web/i18n.js: the web tool's interface text in English and Shavian.

    uv run python tools/make_i18n.py

The English strings below are the source of truth; the Shavian is produced by
dechifro's shaw.py (with part-of-speech tagging), so it matches the comics.
{placeholders}, names (Florence-2, Groq, ...) and units stay as they are.
Strings are lowercased before transliterating so mid-sentence capitals don't get
a namer dot; entries in FIXED are written by hand.
"""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

STRINGS = {
    "title": "Shavian Comics",
    "lang_switch_title": "Switch the interface language",
    "mode_group": "Where it runs",
    "mode_server": "Server",
    "mode_server_title": "Process on the server",
    "mode_local": "This device",
    "mode_local_title": "Process everything in this browser: no server needed",
    "connecting": "connecting…",
    "read_with": "Read text with",
    "ocr_florence": "Florence-2 on this device (offline, ~270 MB once)",
    "ocr_florence_small": "Florence-2, smaller: for phones (offline, ~210 MB once)",
    "ocr_groq": "Groq, with your own API key (better, online)",
    "groq_key": "Groq key",
    "groq_hint": "kept only in this browser; text crops are sent to Groq",
    "resume_last": "Last comic on this device:",
    "resume_btn": "Resume it",
    "resume_hint": "Its text regions are kept; type the text yourself, or try reading it again.",
    "drop_main": "Drop a comic here",
    "drop_or": ", paste one, or",
    "drop_choose": "choose a file",
    "drop_hint": "Text is found, read and transliterated automatically; you can then fix anything below.",
    "step_detect": "1 · Find text",
    "step_read": "2 · Read",
    "step_render": "3 · Typeset",
    "tab_original": "Original",
    "tab_regions": "Regions",
    "tab_erased": "Erased",
    "tab_final": "Shavian",
    "addbox": "+ Box",
    "addbox_title": "Drag a box over text that was missed",
    "download": "Download",
    "legend_bubble": "bubble",
    "legend_free": "text on artwork",
    "legend_off": "skipped (tick to include)",
    "legend_tap": "tap a region to edit it",
    "regions": "Regions",
    "count": "{on} on · {all} found",
    "another": "new comic",
    "read_rest": "read {n} unread regions",
    "read_rest_one": "read 1 unread region",
    "remove": "remove",
    "remove_title": "Remove this region",
    "english": "English",
    "shavian": "Shavian",
    "not_read": "(not read yet)",
    "no_text": "(no text)",
    "no_text_note": "no text found (tap to edit)",
    "erase": "Erase",
    "eraser_auto": "auto",
    "eraser_fill": "flat fill",
    "size": "Size",
    "size_match": "match original",
    "size_fit": "as big as fits",
    "size_custom": "custom…",
    "drawn_at": "drawn at {n}px",
    "kind_bubble": "bubble",
    "kind_free": "on art",
    "note_bubble": "{n} lines in a bubble",
    "note_bubble_one": "1 line in a bubble",
    "note_free": "{n} lines on artwork",
    "note_free_one": "1 line on artwork",
    "note_hand": "added by hand",
    "note_skipped": "skipped: {reason}",
    "reason_sfx": "sound effect or logo",
    "reason_small": "small print",
    "reason_vertical": "vertical text",
    "status_local": "on this device · OCR {ocr} · models download on first use and are cached",
    "status_server": "ready · {device} · OCR {ocr} · {dict} dictionary",
    "status_loading": "loading models: {what}…",
    "status_local_error": "couldn't start on-device mode: {err}",
    "status_no_server": "can't reach the server · try “This device”",
    "allowance_left": "{left} of {per} pages left today",
    "allowance_none": "You've used today's {per} pages; come back tomorrow",
    "alert_no_pages": "You've used today's pages; please come back tomorrow.",
    "busy_finding": "finding text…",
    "busy_reading_all": "reading {n} regions…",
    "busy_reading": "reading {i} / {n}…",
    "busy_reopening": "reopening the last comic…",
    "busy_typesetting": "typesetting…",
    "busy_reading_one": "reading…",
    "busy_reading_box": "reading the new box…",
    "busy_downloading": "downloading {what}: {pct}%",
    "busy_loading": "loading {what}…",
    "what_detector": "text detector (95 MB, first time only)",
    "what_ocr": "OCR model (~{n} MB, first time only)",
    "what_python": "Python (Pyodide, first time only)",
    "what_dict": "Shavian dictionary (2.4 MB)",
    "what_tagger": "part-of-speech tagger",
    "what_reading": "reading text {i} / {n}",
    "step_finding": "finding the text",
    "step_reading": "reading the text",
    "step_typesetting": "typesetting",
    "alert_upload": "Upload failed: {err}",
    "alert_reopen": "Couldn't reopen it: {err}",
    "alert_reading": "Reading failed: {err}",
    "alert_render": "Rendering failed: {err}",
    "crash_title": "The last comic didn't finish:",
    "crash_where": "the page reloaded while {step}.",
    "crash_memory": "That usually means the browser closed the page for using too much memory.",
    "crash_tips": "Try a smaller image, close other tabs, or switch “Read text with” to Groq (it needs far less memory).",
    "crash_smaller": "The next try will use the smaller reading model (switched under “Read text with”). If that fails too, Groq needs far less memory.",
}

# Written by hand (names, or where the automatic result isn't what's wanted)
FIXED = {
    "title": "·𐑖𐑱𐑝𐑾𐑯 𐑒𐑪𐑥𐑦𐑒𐑕",
    "lang_switch": "·𐑖𐑱𐑝𐑾𐑯",   # the button in the English interface switches *to* Shavian
}
ENGLISH_ONLY = {"lang_switch": "English"}  # ...and in the Shavian interface, back to English

# Corrections where the part-of-speech tagger guesses wrong on a short UI fragment
# (imperative "read", the verb "close"): key -> [(wrong, right)], first occurrence
POST_FIX = {
    "read_with": [("𐑮𐑧𐑛", "𐑮𐑰𐑛")],
    "read_rest": [("𐑮𐑧𐑛", "𐑮𐑰𐑛")],
    "read_rest_one": [("𐑮𐑧𐑛", "𐑮𐑰𐑛")],
    "crash_tips": [("𐑒𐑤𐑴𐑕", "𐑒𐑤𐑴𐑟")],
}

# Names get a namer dot (lowercasing the source removed it)
NAMES = ["𐑖𐑱𐑝𐑾𐑯", "𐑦𐑙𐑜𐑤𐑦𐑖", "𐑐𐑲𐑔𐑪𐑯"]

# Kept as written in the Shavian interface
KEEP = ["Florence-2", "Groq", "Pyodide", "API", "OCR", "MB", "px", "~"]


def protect(text):
    """Swap placeholders and names for ⟦n⟧ tokens (which shaw.py passes through)."""
    keep = []

    def tok(m):
        keep.append(m.group(0))
        return f"⟦{len(keep) - 1}⟧"

    pattern = r"\{[a-z]+\}|" + "|".join(re.escape(k) for k in KEEP)
    return re.sub(pattern, tok, text), keep


def restore(text, keep):
    return re.sub(r"⟦(\d+)⟧", lambda m: keep[int(m.group(1))], text)


def main():
    from shavcomic import translit

    keys = [k for k in STRINGS if k not in FIXED]
    protected = [protect(STRINGS[k]) for k in keys]
    shaw = translit.transliterate([p[0].lower() for p in protected])
    shavian = {k: restore(s, p[1]) for k, s, p in zip(keys, shaw, protected)}
    for k, fixes in POST_FIX.items():
        for wrong, right in fixes:
            if wrong not in shavian[k]:
                raise SystemExit(f"POST_FIX for {k} no longer applies: {wrong!r} not in {shavian[k]!r}")
            shavian[k] = shavian[k].replace(wrong, right, 1)
    for k in shavian:
        for name in NAMES:
            shavian[k] = re.sub(r"(?<!·)" + name, "·" + name, shavian[k])
    shavian.update(FIXED)
    english = {**STRINGS, **ENGLISH_ONLY}
    bad = [k for k in keys if "⟦" in shavian[k] or not shavian[k]]
    if bad:
        raise SystemExit(f"transliteration lost something in: {bad}")
    out = ROOT / "src" / "web" / "i18n.js"
    out.write_text(
        "// Generated by tools/make_i18n.py: edit the English there and re-run it.\n"
        "window.SHAVIAN_I18N = " + json.dumps({"en": english, "shaw": shavian}, ensure_ascii=False, indent=1) + ";\n",
        encoding="utf-8")
    print(f"wrote {out} ({len(english)} strings)")
    for k in ("drop_main", "drop_hint", "read_rest", "status_local", "crash_tips"):
        print(f"  {k}: {shavian[k]}")


if __name__ == "__main__":
    main()
