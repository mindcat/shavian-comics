

# core libs
import subprocess
from pathlib import Path
import shutil # For clearing temp directory if needed
import math

# image processing
from PIL import Image, ImageDraw, ImageFont, ImageOps, ImageFilter
import numpy as np

# machine vision image segmentation (for selecting speech bubbles)
from ultralytics import YOLO

# machine vision OCR (from moondream2)
from transformers import AutoModelForCausalLM, AutoTokenizer

# --- Knobs --- (this is stuff you can fiddle with)
HARDWARE_ACCEL = "mps" # ...or 'cuda', for nvidia gpu's
TRANSCRIPTION_PROMPT = "Transcribe the text, all lowercase unless it is a proper name or place. If it doesn't seem like a sentence, return an empty string." #
STARTING_CONFIDENCE_SEGMENTATION = 0.9

# --- Directories ---
INPUT_DIR = Path("img/in")
TEMP_DIR = Path("img/tmp") # For intermediate bubble images (optional use)
OUTPUT_DIR = Path("img/out")
MODEL_DIR = Path("models")
FONT_DIR = Path("fonts") # Directory containing the .otf Shavian font files
DICT_DIR = Path("dict") # Directory containing the dictionary for shaw.py

# --- Models ---
SEG_MODEL = "models/model.pt" # image segmentation vision model, from ultralytics, trained by kitsumed
OCR_MODEL = AutoModelForCausalLM.from_pretrained(
    "vikhyatk/moondream2",
    revision="2025-06-21",
    trust_remote_code=True,
    device_map={"": HARDWARE_ACCEL}
)

# --- Transliteration Script ---
SHAW_SCRIPT_PATH = "shaw.py"
