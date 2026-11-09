"""English -> Shavian with dechifro's shaw.py, run once for a whole batch of texts."""
import re
import subprocess
import sys

from . import config

SEPARATOR = "⁂"  # passes through shaw.py untouched
SHAVIAN_WORD = r"[\U00010450-\U0001047F·]+"


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
    cmd = [sys.executable, str(config.SHAW_SCRIPT)]
    if pos_tagging:
        cmd += ["-p", "2"]  # spaCy part-of-speech tagging: read/read, record/record, ...
    cmd.append(str(config.DICT_DIR / f"{config.DICTIONARY}.dict"))
    proc = subprocess.run(cmd, input=joined, capture_output=True, text=True, cwd=config.SRC_DIR)
    if proc.returncode != 0:
        raise RuntimeError(f"shaw.py failed: {proc.stderr.strip()[-500:]}")
    parts = proc.stdout.split(SEPARATOR)
    if len(parts) != len(todo):
        raise RuntimeError(f"shaw.py returned {len(parts)} parts for {len(todo)} texts")
    for (i, _), part in zip(todo, parts):
        # Safety net: drop stray dictionary "_" markers if shaw.py and the dict ever drift apart
        part = re.sub(r"(?<![^\W_])_", "", part)
        result[i] = resolve_alternatives(" ".join(part.split()))
    return result
