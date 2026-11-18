"""
English -> Shavian with dechifro's shaw.py, kept loaded in-process.

shaw.py is a script: it loads its dictionaries (and, with -p 2, spaCy) and then
transliterates stdin once. Starting it costs ~20 s (mostly importing spaCy), so
we run its setup half once and only re-run the part after `text = sys.stdin.read()`
for each request, resetting the two globals that part accumulates into.
"""
import contextlib
import io
import os
import re
import sys
import threading

from . import config

SEPARATOR = "⁂"  # passes through shaw.py untouched
SHAVIAN_WORD = r"[\U00010450-\U0001047F·]+"
_SPLIT = "text = sys.stdin.read()"


class Shaw:
    def __init__(self, dictionary, pos_tagging=True):
        src = config.SHAW_SCRIPT.read_text(encoding="utf-8")
        if src.count(_SPLIT) != 1:
            raise RuntimeError(f"shaw.py layout changed: expected one '{_SPLIT}' line")
        head, tail = src.split(_SPLIT)
        self.globals = {"__name__": "shaw", "__file__": str(config.SHAW_SCRIPT)}
        argv = ["shaw.py"] + (["-p", "2"] if pos_tagging else []) + [str(dictionary)]
        old_argv, old_cwd = sys.argv, os.getcwd()
        try:
            sys.argv = argv
            os.chdir(config.SRC_DIR)  # shaw.py looks for an optional config.py here
            exec(compile(head, str(config.SHAW_SCRIPT), "exec"), self.globals)
        finally:
            sys.argv = old_argv
            os.chdir(old_cwd)
        self.tail = compile("text = _shaw_input\n" + tail, str(config.SHAW_SCRIPT), "exec")
        self.lock = threading.Lock()
        if pos_tagging:
            _cache_spacy_load()

    def __call__(self, text):
        with self.lock:
            g = self.globals
            g["tokens"] = ["."]   # filled by shaw.py's HTML parser
            g["htags"] = {}
            g["_shaw_input"] = text
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                exec(self.tail, g)
            return buf.getvalue()


def _cache_spacy_load():
    """shaw.py calls spacy.load() on every run; load the model once."""
    import spacy

    if getattr(spacy.load, "_cached", False):
        return
    original, cache = spacy.load, {}

    def load(name, *a, **k):
        if name not in cache:
            cache[name] = original(name, *a, **k)
        return cache[name]

    load._cached = True
    spacy.load = load


_engines = {}
_engines_lock = threading.Lock()


def engine(pos_tagging=True):
    key = (config.DICTIONARY, pos_tagging)
    with _engines_lock:
        if key not in _engines:
            print(f"[translit] loading shaw.py ({config.DICTIONARY}, POS tagging {'on' if pos_tagging else 'off'})")
            _engines[key] = Shaw(config.DICT_DIR / f"{config.DICTIONARY}.dict", pos_tagging)
        return _engines[key]


def _dictionary():
    return engine().globals["dict"]


def known_word(word):
    d = _dictionary()
    return word.lower() in d or word.capitalize() in d


def recase(text):
    """
    Sentence-style casing for text OCR'd in capitals: lowercase, then restore
    capitals for "I" and for words the dictionary only knows capitalised (names,
    places), since Shavian marks those with a namer dot.
    """
    d = _dictionary()

    def fix(m):
        w = m.group(0)
        low = w.lower()
        if low == "i" or low.startswith("i'"):
            return "I" + low[1:]
        cap = low.capitalize()
        if cap in d and low not in d:
            return cap
        return low

    return re.sub(r"[A-Za-z]+(?:'[A-Za-z]+)?", fix, text)


def resolve_alternatives(text):
    """shaw.py writes 'a@b' when it can't pick a pronunciation; take its first choice."""
    return re.sub(r"@" + SHAVIAN_WORD, "", text)


def transliterate(texts, pos_tagging=True):
    """Transliterates a list of English strings; empty strings stay empty."""
    todo = [(i, t) for i, t in enumerate(texts) if t.strip()]
    result = [""] * len(texts)
    if not todo:
        return result
    joined = f"\n\n{SEPARATOR}\n\n".join(t for _, t in todo)
    parts = engine(pos_tagging)(joined).split(SEPARATOR)
    if len(parts) != len(todo):
        raise RuntimeError(f"shaw.py returned {len(parts)} parts for {len(todo)} texts")
    for (i, _), part in zip(todo, parts):
        # Safety net: drop stray dictionary "_" markers if shaw.py and the dict ever drift apart
        part = re.sub(r"(?<![^\W_])_", "", part)
        result[i] = resolve_alternatives(" ".join(part.split()))
    return result
