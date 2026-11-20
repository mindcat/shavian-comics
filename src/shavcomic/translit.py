"""
English -> Shavian with dechifro's shaw.py, kept loaded in-process (see
textproc.ShawEngine), with spaCy part-of-speech tagging on the server.
"""
import threading

from . import config, textproc
from .textproc import SEPARATOR, SHAVIAN_WORD, resolve_alternatives  # noqa: F401 (re-exported)


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


class Shaw(textproc.ShawEngine):
    """Thread-safe ShawEngine using spaCy tagging (shaw.py -p 2) when pos_tagging is on."""

    def __init__(self, dictionary, pos_tagging=True):
        if pos_tagging:
            _cache_spacy_load()
        super().__init__(config.SHAW_SCRIPT, dictionary, pos=2 if pos_tagging else None, cwd=config.SRC_DIR)
        self.lock = threading.Lock()

    def __call__(self, text):
        with self.lock:
            return super().__call__(text)


_engines = {}
_engines_lock = threading.Lock()


def engine(pos_tagging=True):
    key = (config.DICTIONARY, pos_tagging)
    with _engines_lock:
        if key not in _engines:
            print(f"[translit] loading shaw.py ({config.DICTIONARY}, POS tagging {'on' if pos_tagging else 'off'})")
            _engines[key] = Shaw(config.DICT_DIR / f"{config.DICTIONARY}.dict", pos_tagging)
        return _engines[key]


def known_word(word):
    return textproc.known_word(engine().dictionary, word)


def recase(text):
    return textproc.recase(text, engine().dictionary)


def transliterate(texts, pos_tagging=True):
    """Transliterates a list of English strings; empty strings stay empty."""
    return textproc.transliterate(engine(pos_tagging), texts)
