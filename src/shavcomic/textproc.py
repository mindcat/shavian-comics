"""
Text handling shared by the server and the in-browser version (which runs this
file under Pyodide), so both behave identically. Pure Python: no torch, numpy, etc.

- ShawEngine: dechifro's shaw.py kept loaded in-process
- clean / join_lines / recase: tidying OCR output
- transliterate: batch English -> Shavian
"""
import contextlib
import io
import os
import re
import sys

SEPARATOR = "⁂"  # passes through shaw.py untouched
SHAVIAN_WORD = r"[\U00010450-\U0001047F·]+"
_SPLIT = "text = sys.stdin.read()"


class ShawEngine:
    """
    shaw.py is a script: it loads its dictionaries (and a POS tagger) and then
    transliterates stdin once. Starting it is slow (spaCy alone takes ~20 s), so we
    run its setup half once and only re-run the part after `text = sys.stdin.read()`
    for each call, resetting the two globals that part accumulates into.

    pos: None (no tagging), 1 (NLTK, pure Python: works in the browser) or 2 (spaCy).
    """

    def __init__(self, script_path, dict_path, pos=None, cwd=None):
        with open(script_path, encoding="utf-8") as f:
            src = f.read()
        if src.count(_SPLIT) != 1:
            raise RuntimeError(f"shaw.py layout changed: expected one '{_SPLIT}' line")
        head, tail = src.split(_SPLIT)
        self.globals = {"__name__": "shaw", "__file__": str(script_path)}
        argv = ["shaw.py"] + (["-p", str(pos)] if pos else []) + [str(dict_path)]
        old_argv, old_cwd = sys.argv, os.getcwd()
        try:
            sys.argv = argv
            if cwd:
                os.chdir(cwd)  # shaw.py looks for an optional config.py in the working directory
            exec(compile(head, str(script_path), "exec"), self.globals)
        finally:
            sys.argv = old_argv
            os.chdir(old_cwd)
        self.tail = compile("text = _shaw_input\n" + tail, str(script_path), "exec")

    @property
    def dictionary(self):
        return self.globals["dict"]

    def __call__(self, text):
        g = self.globals
        g["tokens"] = ["."]  # filled by shaw.py's HTML parser
        g["htags"] = {}
        g["_shaw_input"] = text
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            exec(self.tail, g)
        return buf.getvalue()


def resolve_alternatives(text):
    """shaw.py writes 'a@b' when it can't pick a pronunciation; take its first choice."""
    return re.sub(r"@" + SHAVIAN_WORD, "", text)


def transliterate(engine, texts):
    """Transliterates a list of English strings in one shaw.py run; empty strings stay empty."""
    todo = [(i, t) for i, t in enumerate(texts) if t.strip()]
    result = [""] * len(texts)
    if not todo:
        return result
    joined = f"\n\n{SEPARATOR}\n\n".join(t for _, t in todo)
    parts = engine(joined).split(SEPARATOR)
    if len(parts) != len(todo):
        raise RuntimeError(f"shaw.py returned {len(parts)} parts for {len(todo)} texts")
    for (i, _), part in zip(todo, parts):
        # Safety net: drop stray dictionary "_" markers if shaw.py and the dict ever drift apart
        part = re.sub(r"(?<![^\W_])_", "", part)
        result[i] = resolve_alternatives(" ".join(part.split()))
    return result


# ------------------------------------------------------------------ OCR clean-up

# Credits and signatures look like text but shouldn't be transliterated. The signature
# is often misread ("by HATTERSON"), so any lone "by <name> [year]" counts as a credit.
SKIP_TEXT = re.compile(
    r"\b[wh]?at+[ae]?rs[oa]n\b|wattsin|syndicat|distributed by|universal press|andrews mc ?meel|copyright|©"
    r"|^by \S+( \S*\d{4})?$"
    r"|^(no text|empty|none|n/a)\.?$",
    re.I,
)


def clean(text):
    """Tidy OCR output; '' for credits, signatures and non-text."""
    text = " ".join(str(text).strip().strip('"').split())
    if SKIP_TEXT.search(text):
        return ""
    return text if any(ch.isalpha() for ch in text) else ""


def known_word(dictionary, word):
    return word.lower() in dictionary or word.capitalize() in dictionary


SUFFIXES = {"s", "es", "ed", "d", "ing", "er", "ers", "est", "ly", "y", "tion", "tions", "ment", "ness", "able", "ful"}


def join_lines(text, dictionary):
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
            merge = first.group(0).lower() in SUFFIXES or known_word(dictionary, m.group(1) + first.group(0))
            joined = (joined[:-1] if merge else joined) + line
        else:
            joined = f"{joined} {line}" if joined else line
    # OCR often puts a space before closing punctuation ("HERE !")
    return re.sub(r"\s+([!?.,;:])", r"\1", joined)


def recase(text, dictionary):
    """
    Sentence-style casing for text OCR'd in capitals: lowercase, then restore
    capitals for "I" and for words the dictionary only knows capitalised (names,
    places), since Shavian marks those with a namer dot.
    """
    def fix(m):
        low = m.group(0).lower()
        if low == "i" or low.startswith("i'"):
            return "I" + low[1:]
        cap = low.capitalize()
        if cap in dictionary and low not in dictionary:
            return cap
        return low

    return re.sub(r"[A-Za-z]+(?:'[A-Za-z]+)?", fix, text)


def tidy_capitals(text, dictionary):
    """For OCR that returns lettering as written (in capitals): join lines, re-case, clean."""
    return clean(recase(join_lines(text, dictionary), dictionary))
