// Erasing and typesetting on a canvas: ports of src/shavcomic/erase.py and layout.py.
// In the browser, LaMa is replaced by a diffusion fill for text over artwork.
import * as ip from "./imgproc.js";

const MIN_FONT_SIZE = 8, TEXT_PADDING = 3;
const TALL_LETTERS = [..."𐑐𐑑𐑒𐑓𐑔𐑕𐑖𐑗𐑘𐑙"];
export const FONT_FAMILY = "Backlog Shavian";

// ---------------------------------------------------------------- erasing

function exactBg(img, r) {
  const { w, h, px } = img;
  const near = ip.andNot(ip.andNot(ip.dilate(r.erase, w, h, 6), r.erase), r.text);
  let s0 = 0, s1 = 0, s2 = 0, n = 0;
  for (let i = 0; i < near.length; i++) {
    if (!near[i]) continue;
    const a = px[4 * i], b = px[4 * i + 1], c = px[4 * i + 2];
    if (Math.max(Math.abs(a - r.bg[0]), Math.abs(b - r.bg[1]), Math.abs(c - r.bg[2])) <= 12) { s0 += a; s1 += b; s2 += c; n++; }
  }
  return n ? [Math.round(s0 / n), Math.round(s1 / n), Math.round(s2 / n)] : r.bg;
}

function flatBackground(img, r) {
  const { w, h, px } = img;
  const ring = ip.andNot(ip.dilate(r.erase, w, h, 4), r.erase);
  const bg = exactBg(img, r);
  let off = 0, n = 0;
  for (let i = 0; i < ring.length; i++) {
    if (!ring[i]) continue;
    n++;
    if (Math.max(Math.abs(px[4 * i] - bg[0]), Math.abs(px[4 * i + 1] - bg[1]), Math.abs(px[4 * i + 2] - bg[2])) > 24) off++;
  }
  return n > 0 && off / n < 0.10;
}

function fill(img, r) {
  const { w, h, px } = img;
  const bg = exactBg(img, r);
  // Feather the edge a little (the Python version blurs the mask by 0.7 px)
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    const i = y * w + x;
    let a;
    if (r.erase[i]) {
      const edge = (x > 0 && !r.erase[i - 1]) || (x < w - 1 && !r.erase[i + 1]) || (y > 0 && !r.erase[i - w]) || (y < h - 1 && !r.erase[i + w]);
      a = edge ? 0.8 : 1;
    } else continue;
    for (let c = 0; c < 3; c++) px[4 * i + c] = Math.round(a * bg[c] + (1 - a) * px[4 * i + c]);
  }
}

// Diffusion inpainting: masked pixels repeatedly take the average of their
// neighbours, so the surrounding colours flow in (coarse to fine for speed)
function diffuse(img, r) {
  const { w, h, px } = img;
  const mask = ip.dilate(r.text, w, h, 3);
  const b = ip.bbox(mask, w, h);
  if (!b) return;
  const pad = 16;
  const x1 = Math.max(0, b[0] - pad), y1 = Math.max(0, b[1] - pad), x2 = Math.min(w, b[2] + pad), y2 = Math.min(h, b[3] + pad);
  const cw = x2 - x1, ch = y2 - y1;
  const m = new Uint8Array(cw * ch), v = new Float32Array(cw * ch * 3);
  const bg = exactBg(img, r);
  for (let y = 0; y < ch; y++) for (let x = 0; x < cw; x++) {
    const i = (y + y1) * w + x + x1, j = y * cw + x;
    m[j] = mask[i];
    for (let c = 0; c < 3; c++) v[3 * j + c] = m[j] ? bg[c] : px[4 * i + c];
  }
  const iters = Math.min(400, 40 + 4 * Math.max(cw, ch));
  for (let k = 0; k < iters; k++) {
    for (let y = 0; y < ch; y++) for (let x = 0; x < cw; x++) {
      const j = y * cw + x;
      if (!m[j]) continue;
      let n = 0, s0 = 0, s1 = 0, s2 = 0;
      const add = jj => { s0 += v[3 * jj]; s1 += v[3 * jj + 1]; s2 += v[3 * jj + 2]; n++; };
      if (x > 0) add(j - 1);
      if (x < cw - 1) add(j + 1);
      if (y > 0) add(j - cw);
      if (y < ch - 1) add(j + cw);
      v[3 * j] = s0 / n; v[3 * j + 1] = s1 / n; v[3 * j + 2] = s2 / n;
    }
  }
  for (let y = 0; y < ch; y++) for (let x = 0; x < cw; x++) {
    const j = y * cw + x;
    if (!m[j]) continue;
    const i = (y + y1) * w + x + x1;
    for (let c = 0; c < 3; c++) px[4 * i + c] = Math.round(v[3 * j + c]);
  }
}

export const ERASERS = {
  auto: (img, r) => (r.kind === "bubble" || flatBackground(img, r) ? fill : diffuse)(img, r),
  fill, diffuse,
};

// ---------------------------------------------------------------- typesetting

let measureCtx;
function ctx2d() {
  if (!measureCtx) measureCtx = document.createElement("canvas").getContext("2d");
  return measureCtx;
}
const fontSpec = size => `${size}px "${FONT_FAMILY}"`;

let tallRatio;
export function tallLetterRatio() {
  if (!tallRatio) {
    const c = ctx2d();
    c.font = fontSpec(100);
    const asc = TALL_LETTERS.map(ch => c.measureText(ch).actualBoundingBoxAscent).sort((a, b) => a - b);
    tallRatio = asc[Math.floor(asc.length / 2)] / 100 || 0.7;
  }
  return tallRatio;
}

export const sizeForLetterHeight = lh => Math.max(MIN_FONT_SIZE, Math.round(lh / tallLetterRatio()));

function metrics(size) {
  const c = ctx2d();
  c.font = fontSpec(size);
  const m = c.measureText("𐑐𐑚𐑑𐑛");
  const ascent = m.actualBoundingBoxAscent, descent = m.actualBoundingBoxDescent;
  return { ascent, descent, step: ascent + descent + Math.max(2, 0.12 * size) };
}

function wrapWords(words, size, maxWidth) {
  const c = ctx2d();
  c.font = fontSpec(size);
  const lines = [];
  let cur = "";
  for (const word of words) {
    const test = cur ? `${cur} ${word}` : word;
    if (c.measureText(test).width <= maxWidth) cur = test;
    else if (!cur || c.measureText(word).width > maxWidth) return null;
    else { lines.push(cur); cur = word; }
  }
  if (cur) lines.push(cur);
  return lines;
}

function blockFits(lines, size, center, sat, w, h) {
  const c = ctx2d();
  c.font = fontSpec(size);
  const { step } = metrics(size);
  const total = lines.length * step;
  const top = center[1] - total / 2;
  for (let i = 0; i < lines.length; i++) {
    const lw = c.measureText(lines[i]).width;
    const x1 = Math.floor(center[0] - lw / 2), x2 = Math.ceil(center[0] + lw / 2);
    const y1 = Math.floor(top + i * step), y2 = Math.ceil(top + (i + 1) * step);
    if (x1 < 0 || y1 < 0 || x2 > w || y2 > h) return false;
    if (ip.rectSum(sat, w, x1, y1, x2, y2) !== (x2 - x1) * (y2 - y1)) return false;
  }
  return true;
}

// Largest size <= maxSize whose wrapped text fits inside the area (fit_text)
function fitText(text, area, w, h, maxSize, preferWide) {
  const words = text.split(/\s+/).filter(Boolean);
  const fitMask = ip.erode(area, w, h, TEXT_PADDING);
  const b = ip.bbox(fitMask, w, h);
  if (!words.length || !b) return null;
  const sat = ip.integral(fitMask, w, h);
  const dist = ip.distance(fitMask, w, h);
  let deep = 0;
  for (let i = 1; i < dist.length; i++) if (dist[i] > dist[deep]) deep = i;
  let sx = 0, sy = 0, n = 0;
  for (let i = 0; i < fitMask.length; i++) if (fitMask[i]) { sx += i % w; sy += Math.floor(i / w); n++; }
  const centers = [[deep % w, Math.floor(deep / w)]];
  const cx = sx / n, cy = sy / n;
  if (fitMask[Math.floor(cy) * w + Math.floor(cx)]) centers.unshift([cx, cy]);
  const maxWidth = b[2] - 1 - b[0];
  let hi = Math.max(MIN_FONT_SIZE, Math.floor((b[3] - 1 - b[1]) * 0.5));
  if (maxSize) hi = Math.min(hi, maxSize);
  let fracs = Array.from({ length: 14 }, (_, k) => 0.35 + k * 0.65 / 13);
  if (preferWide) fracs = fracs.reverse();
  const trySize = size => {
    for (const f of fracs) {
      const lines = wrapWords(words, size, maxWidth * f);
      if (!lines) continue;
      for (const c of centers) if (blockFits(lines, size, c, sat, w, h)) return { size, lines, center: c };
    }
    return null;
  };
  let best = trySize(hi);
  if (best) return best;
  let lo = MIN_FONT_SIZE;
  hi -= 1;
  while (lo <= hi) {
    const mid = (lo + hi) >> 1;
    const found = trySize(mid);
    if (found) { best = found; lo = mid + 1; } else hi = mid - 1;
  }
  return best;
}

function forcedLayout(text, area, w, h, size) {
  const b = ip.bbox(area, w, h);
  const lines = wrapWords(text.split(/\s+/).filter(Boolean), size, (b[2] - b[0]) * 0.95) || [text];
  const dist = ip.distance(area, w, h);
  let deep = 0;
  for (let i = 1; i < dist.length; i++) if (dist[i] > dist[deep]) deep = i;
  return { size, lines, center: [deep % w, Math.floor(deep / w)] };
}

// Draws text into the region on ctx; sizeMode: "match", "fit" or a number. Returns the size used.
export function drawText(ctx, r, text, sizeMode, w, h) {
  let fitted;
  if (/^\d+$/.test(String(sizeMode))) fitted = forcedLayout(text, r.layout, w, h, Number(sizeMode));
  else {
    const maxSize = sizeMode === "match" && r.letterHeight ? sizeForLetterHeight(r.letterHeight) : null;
    fitted = fitText(text, r.layout, w, h, maxSize, r.kind === "free") || forcedLayout(text, r.layout, w, h, MIN_FONT_SIZE);
  }
  const { size, lines, center } = fitted;
  const { ascent, step } = metrics(size);
  ctx.font = fontSpec(size);
  ctx.fillStyle = "#000";
  ctx.textAlign = "center";
  ctx.textBaseline = "alphabetic";
  const top = center[1] - (lines.length * step) / 2;
  lines.forEach((line, i) => ctx.fillText(line, center[0], top + i * step + ascent + (step - ascent - metrics(size).descent) / 2));
  return size;
}
