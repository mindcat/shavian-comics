"""Erasers: remove a region's original lettering from the image (in place)."""
import cv2
import numpy as np
import torch
from PIL import Image, ImageFilter
from scipy import ndimage

from . import config, models
from .regions import mask_bbox


def _exact_bg(image, region):
    """
    The real background colour around the text. region.bg comes from coarse colour
    bins (good for finding the bubble, but a few levels off: 252 instead of 255),
    so average the actual nearby pixels that match it.
    """
    near = ndimage.binary_dilation(region.erase_mask, iterations=6) & ~region.erase_mask & ~region.text_mask
    px = np.array(image.convert("RGB"))[near].astype(np.int16)
    if not len(px):
        return region.bg
    px = px[np.abs(px - np.array(region.bg, dtype=np.int16)).max(axis=1) <= 12]
    return tuple(int(round(v)) for v in px.mean(axis=0)) if len(px) else region.bg


def fill(image, region):
    """Paint the region's erase mask in its background colour (perfect for flat bubbles)."""
    m = Image.fromarray(region.erase_mask.astype(np.uint8) * 255).filter(ImageFilter.GaussianBlur(0.7))
    image.paste(Image.new(image.mode, image.size, (*_exact_bg(image, region), 255)[: len(image.mode)]), (0, 0), m)


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


def _flat_background(image, region):
    """True if the pixels just around the text are (almost) all the background colour."""
    ring = ndimage.binary_dilation(region.erase_mask, iterations=4) & ~region.erase_mask
    if not ring.any():
        return False
    px = np.array(image.convert("RGB"))[ring].astype(np.int16)
    off = np.abs(px - np.array(_exact_bg(image, region), dtype=np.int16)).max(axis=1) > 24
    return off.mean() < 0.10  # a few nearby lines (axis ticks) are fine: fill only paints the text itself


def auto(image, region):
    """Flat fill inside bubbles and on flat backgrounds (exact); LaMa over artwork and gradients."""
    (fill if region.kind == "bubble" or _flat_background(image, region) else lama)(image, region)


BACKENDS = {"auto": auto, "fill": fill, "telea": telea, "lama": lama}
