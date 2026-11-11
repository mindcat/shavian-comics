"""Erasers: remove a region's original lettering from the image (in place)."""
import cv2
import numpy as np
import torch
from PIL import Image, ImageFilter
from scipy import ndimage

from . import config, models
from .regions import mask_bbox


def fill(image, region):
    """Paint the region's erase mask in its background colour (perfect for flat bubbles)."""
    m = Image.fromarray(region.erase_mask.astype(np.uint8) * 255).filter(ImageFilter.GaussianBlur(0.7))
    image.paste(Image.new(image.mode, image.size, (*region.bg, 255)[: len(image.mode)]), (0, 0), m)


def _inpaint_mask(region):
    return ndimage.binary_dilation(region.text_mask, iterations=3)


def telea(image, region):
    """OpenCV's classic inpainting: no model, fine on simple backgrounds."""
    mask = _inpaint_mask(region)
    box = _crop_box(mask, image.size, pad=16)
    rgb = np.array(image.convert("RGB").crop(box))
    out = cv2.inpaint(rgb, mask[box[1]:box[3], box[0]:box[2]].astype(np.uint8) * 255, 5, cv2.INPAINT_TELEA)
    _paste(image, out, mask, box)


def lama(image, region):
    """LaMa (big-lama) inpainting on a crop around the text: handles artwork and gradients."""
    mask = _inpaint_mask(region)
    box = _crop_box(mask, image.size, pad=64)
    rgb = np.array(image.convert("RGB").crop(box))
    m = mask[box[1]:box[3], box[0]:box[2]]
    h, w = m.shape
    ph, pw = (-h) % 8, (-w) % 8  # LaMa wants sizes divisible by 8
    rgb_p = np.pad(rgb, ((0, ph), (0, pw), (0, 0)), mode="reflect")
    m_p = np.pad(m, ((0, ph), (0, pw)))
    img_t = torch.from_numpy(rgb_p).permute(2, 0, 1)[None].float().div(255).to(config.DEVICE)
    mask_t = torch.from_numpy(m_p)[None, None].float().to(config.DEVICE)
    with torch.inference_mode():
        out = models.lama()(img_t, mask_t)
    out = (out[0].permute(1, 2, 0).clamp(0, 1).mul(255).byte().cpu().numpy())[:h, :w]
    _paste(image, out, mask, box)


def _crop_box(mask, size, pad):
    x1, y1, x2, y2 = mask_bbox(mask)
    w, h = size
    return max(0, x1 - pad), max(0, y1 - pad), min(w, x2 + pad), min(h, y2 + pad)


def _paste(image, rgb, mask, box):
    """Paste rgb (the crop at box) back into image, only where mask is set (feathered)."""
    m = Image.fromarray(mask[box[1]:box[3], box[0]:box[2]].astype(np.uint8) * 255).filter(ImageFilter.GaussianBlur(0.7))
    patch = Image.fromarray(rgb).convert(image.mode)
    image.paste(patch, box[:2], m)


def auto(image, region):
    """Flat fill inside bubbles (exact), LaMa for text over artwork or panel backgrounds."""
    (fill if region.kind == "bubble" else lama)(image, region)


BACKENDS = {"auto": auto, "fill": fill, "telea": telea, "lama": lama}
