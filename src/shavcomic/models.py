"""Lazily loaded models. Big ones can be unloaded so the next fits in GPU memory."""
import gc

import torch

from . import config

_cache = {}


def _get(name, loader):
    if name not in _cache:
        print(f"[models] loading {name} on {config.DEVICE}")
        _cache[name] = loader()
    return _cache[name]


def unload(*names):  # noqa: D401
    for name in names or list(_cache):
        if _cache.pop(name, None) is not None:
            print(f"[models] unloaded {name}")
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def moondream():
    def load():
        from transformers import AutoModelForCausalLM

        return AutoModelForCausalLM.from_pretrained(
            config.MOONDREAM_ID,
            revision=config.MOONDREAM_REVISION,
            trust_remote_code=True,
            device_map={"": config.DEVICE},
        )

    return _get("moondream", load)


def qwen():
    def load():
        from transformers import AutoModelForImageTextToText, AutoProcessor

        model = AutoModelForImageTextToText.from_pretrained(
            config.QWEN_ID, dtype=torch.bfloat16, device_map={"": config.DEVICE}
        ).eval()
        processor = AutoProcessor.from_pretrained(config.QWEN_ID)
        return model, processor

    return _get("qwen", load)


def florence():
    def load():
        from transformers import AutoProcessor, Florence2ForConditionalGeneration

        dtype = torch.float16 if config.DEVICE == "cuda" else torch.float32
        model = Florence2ForConditionalGeneration.from_pretrained(config.FLORENCE_ID, dtype=dtype)
        return model.to(config.DEVICE).eval(), AutoProcessor.from_pretrained(config.FLORENCE_ID)

    return _get("florence", load)


def yolo():
    def load():
        from ultralytics import YOLO

        return YOLO(config.YOLO_MODEL)

    return _get("yolo", load)


def ctd():
    def load():
        import onnxruntime as ort

        return ort.InferenceSession(str(config.CTD_MODEL), providers=["CPUExecutionProvider"])

    return _get("ctd", load)


def lama():
    def load():
        return torch.jit.load(str(config.LAMA_MODEL), map_location=config.DEVICE).eval()

    return _get("lama", load)
