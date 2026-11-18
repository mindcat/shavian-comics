import os
from pathlib import Path

import torch

SRC_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = SRC_DIR / "models"
DICT_DIR = SRC_DIR / "dict"
FONT_DIR = SRC_DIR / "fonts"
INPUT_DIR = SRC_DIR / "img" / "in"
OUTPUT_DIR = SRC_DIR / "img" / "out"
TEMP_DIR = SRC_DIR / "img" / "tmp"

YOLO_MODEL = MODEL_DIR / "model.pt"
CTD_MODEL = MODEL_DIR / "comictextdetector.pt.onnx"
LAMA_MODEL = MODEL_DIR / "big-lama.pt"
SHAW_SCRIPT = SRC_DIR / "shaw.py"

MOONDREAM_ID = "vikhyatk/moondream2"
MOONDREAM_REVISION = "2025-06-21"
QWEN_ID = os.environ.get("SHAVIAN_QWEN", "Qwen/Qwen3-VL-2B-Instruct")
FLORENCE_ID = os.environ.get("SHAVIAN_FLORENCE", "florence-community/Florence-2-large")

# Groq (optional hosted OCR; off unless a key is configured). The key is read from
# GROQ_API_KEY or ~/.config/shavian-comics/groq_api_key, never from the repo.
GROQ_MODEL = os.environ.get("GROQ_MODEL", "")
GROQ_KEY_FILE = Path.home() / ".config" / "shavian-comics" / "groq_api_key"


def groq_key():
    if os.environ.get("GROQ_API_KEY"):
        return os.environ["GROQ_API_KEY"].strip()
    if GROQ_KEY_FILE.exists():
        return GROQ_KEY_FILE.read_text().strip()
    return ""

DICTIONARY = os.environ.get("SHAVIAN_DICT", "amer")  # or "brit"
FONT = os.environ.get("SHAVIAN_FONT", "backlogshavian.otf")
TEXT_COLOR = (0, 0, 0, 255)

OCR_PROMPT = (
    "Transcribe the text, all lowercase unless it is a proper name or place. "
    "If it isn't actual text, return an empty string."
)


def pick_device():
    """Use an NVIDIA GPU if present, then Apple Silicon, then fall back to CPU."""
    if os.environ.get("SHAVIAN_DEVICE"):
        return os.environ["SHAVIAN_DEVICE"]
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


DEVICE = pick_device()
