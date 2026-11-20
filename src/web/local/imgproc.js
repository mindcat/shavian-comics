// Binary-mask and colour helpers: JavaScript versions of the scipy.ndimage / numpy
// operations the Python pipeline uses, with the same conventions (4-connectivity,
// cross-shaped structuring element, zero outside the image) so results line up.
//
// A mask is a Uint8Array of w*h (0/1); an image is {w, h, px: Uint8ClampedArray RGBA}.

export const newMask = (w, h) => new Uint8Array(w * h);

export function rectMask(w, h, box, pad = 0) {
  const m = newMask(w, h);
  const x1 = Math.max(0, Math.floor(box[0]) - pad), y1 = Math.max(0, Math.floor(box[1]) - pad);
  const x2 = Math.min(w, Math.floor(box[2]) + pad), y2 = Math.min(h, Math.floor(box[3]) + pad);
  for (let y = y1; y < y2; y++) m.fill(1, y * w + x1, y * w + x2);
  return m;
}

export function count(m) {
  let n = 0;
  for (let i = 0; i < m.length; i++) n += m[i];
  return n;
}

export function bbox(m, w, h) {
  let x1 = w, y1 = h, x2 = -1, y2 = -1;
  for (let y = 0; y < h; y++) {
    const row = y * w;
    for (let x = 0; x < w; x++) {
      if (m[row + x]) {
        if (x < x1) x1 = x;
        if (x > x2) x2 = x;
        if (y < y1) y1 = y;
        y2 = y;
      }
    }
  }
  return x2 < 0 ? null : [x1, y1, x2 + 1, y2 + 1];
}

export const and = (a, b) => a.map((v, i) => v & b[i]);
export const andNot = (a, b) => a.map((v, i) => v & (1 - b[i]));
export const or = (a, b) => a.map((v, i) => v | b[i]);

// Cross-shaped (4-neighbour) dilation/erosion, repeated `iters` times, like
// scipy.ndimage.binary_dilation/erosion(..., iterations=iters). Erosion treats
// outside the image as 0, so masks shrink away from the image border too.
export function dilate(m, w, h, iters = 1) {
  let cur = m;
  for (let k = 0; k < iters; k++) {
    const out = cur.slice();
    for (let y = 0; y < h; y++) {
      const row = y * w;
      for (let x = 0; x < w; x++) {
        const i = row + x;
        if (cur[i]) continue;
        if ((x > 0 && cur[i - 1]) || (x < w - 1 && cur[i + 1]) || (y > 0 && cur[i - w]) || (y < h - 1 && cur[i + w])) out[i] = 1;
      }
    }
    cur = out;
  }
  return cur;
}

export function erode(m, w, h, iters = 1) {
  let cur = m;
  for (let k = 0; k < iters; k++) {
    const out = cur.slice();
    for (let y = 0; y < h; y++) {
      const row = y * w;
      for (let x = 0; x < w; x++) {
        const i = row + x;
        if (!cur[i]) continue;
        if (x === 0 || y === 0 || x === w - 1 || y === h - 1 || !cur[i - 1] || !cur[i + 1] || !cur[i - w] || !cur[i + w]) out[i] = 0;
      }
    }
    cur = out;
  }
  return cur;
}

// Connected components (4-connectivity). Returns labels (0 = background, 1..n),
// n, and per-label bounding boxes [x1, y1, x2, y2] and areas (index = label - 1).
export function label(m, w, h) {
  const labels = new Int32Array(w * h);
  const boxes = [], areas = [];
  const stack = new Int32Array(w * h);
  let n = 0;
  for (let start = 0; start < m.length; start++) {
    if (!m[start] || labels[start]) continue;
    n++;
    let sp = 0, x1 = w, y1 = h, x2 = 0, y2 = 0, area = 0;
    stack[sp++] = start;
    labels[start] = n;
    while (sp) {
      const i = stack[--sp];
      const x = i % w, y = (i - x) / w;
      area++;
      if (x < x1) x1 = x;
      if (x + 1 > x2) x2 = x + 1;
      if (y < y1) y1 = y;
      if (y + 1 > y2) y2 = y + 1;
      if (x > 0 && m[i - 1] && !labels[i - 1]) { labels[i - 1] = n; stack[sp++] = i - 1; }
      if (x < w - 1 && m[i + 1] && !labels[i + 1]) { labels[i + 1] = n; stack[sp++] = i + 1; }
      if (y > 0 && m[i - w] && !labels[i - w]) { labels[i - w] = n; stack[sp++] = i - w; }
      if (y < h - 1 && m[i + w] && !labels[i + w]) { labels[i + w] = n; stack[sp++] = i + w; }
    }
    boxes.push([x1, y1, x2, y2]);
    areas.push(area);
  }
  return { labels, n, boxes, areas };
}

// scipy.ndimage.binary_fill_holes: background not reachable from the border becomes foreground
export function fillHoles(m, w, h) {
  const reach = new Uint8Array(w * h);
  const stack = new Int32Array(w * h);
  let sp = 0;
  const push = i => { if (!m[i] && !reach[i]) { reach[i] = 1; stack[sp++] = i; } };
  for (let x = 0; x < w; x++) { push(x); push((h - 1) * w + x); }
  for (let y = 0; y < h; y++) { push(y * w); push(y * w + w - 1); }
  while (sp) {
    const i = stack[--sp];
    const x = i % w, y = (i - x) / w;
    if (x > 0) push(i - 1);
    if (x < w - 1) push(i + 1);
    if (y > 0) push(i - w);
    if (y < h - 1) push(i + w);
  }
  const out = new Uint8Array(w * h);
  for (let i = 0; i < out.length; i++) out[i] = reach[i] ? 0 : 1;
  return out;
}

// Approximate Euclidean distance to the nearest 0 pixel (two-pass chamfer 3-4)
export function distance(m, w, h) {
  const INF = 1e9, d = new Float32Array(w * h);
  for (let i = 0; i < d.length; i++) d[i] = m[i] ? INF : 0;
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    const i = y * w + x;
    if (!d[i]) continue;
    let v = d[i];
    if (x > 0) v = Math.min(v, d[i - 1] + 3);
    if (y > 0) {
      v = Math.min(v, d[i - w] + 3);
      if (x > 0) v = Math.min(v, d[i - w - 1] + 4);
      if (x < w - 1) v = Math.min(v, d[i - w + 1] + 4);
    }
    d[i] = v;
  }
  for (let y = h - 1; y >= 0; y--) for (let x = w - 1; x >= 0; x--) {
    const i = y * w + x;
    if (!d[i]) continue;
    let v = d[i];
    if (x < w - 1) v = Math.min(v, d[i + 1] + 3);
    if (y < h - 1) {
      v = Math.min(v, d[i + w] + 3);
      if (x < w - 1) v = Math.min(v, d[i + w + 1] + 4);
      if (x > 0) v = Math.min(v, d[i + w - 1] + 4);
    }
    d[i] = v;
  }
  for (let i = 0; i < d.length; i++) d[i] /= 3;
  return d;
}

// Pixels lying in a vertical or horizontal run of ink at least `run` long: the same
// as binary_opening with a 1-pixel-wide line, i.e. rules, axes and borders
export function longLines(m, w, h, run) {
  const out = new Uint8Array(w * h);
  for (let y = 0; y < h; y++) {
    let start = -1;
    for (let x = 0; x <= w; x++) {
      const on = x < w && m[y * w + x];
      if (on && start < 0) start = x;
      if (!on && start >= 0) { if (x - start >= run) out.fill(1, y * w + start, y * w + x); start = -1; }
    }
  }
  for (let x = 0; x < w; x++) {
    let start = -1;
    for (let y = 0; y <= h; y++) {
      const on = y < h && m[y * w + x];
      if (on && start < 0) start = y;
      if (!on && start >= 0) { if (y - start >= run) for (let k = start; k < y; k++) out[k * w + x] = 1; start = -1; }
    }
  }
  return out;
}

// Convex hull (monotone chain) of a mask's pixel corners; used for solidity and outlines
export function hull(m, w, h) {
  const pts = [];
  for (let y = 0; y < h; y++) {
    let first = -1, last = -1;
    for (let x = 0; x < w; x++) if (m[y * w + x]) { if (first < 0) first = x; last = x; }
    if (first >= 0) pts.push([first, y], [last + 1, y], [first, y + 1], [last + 1, y + 1]);
  }
  if (pts.length < 3) return pts;
  pts.sort((a, b) => a[0] - b[0] || a[1] - b[1]);
  const cross = (o, a, b) => (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]);
  const lower = [], upper = [];
  for (const p of pts) { while (lower.length >= 2 && cross(lower.at(-2), lower.at(-1), p) <= 0) lower.pop(); lower.push(p); }
  for (const p of pts.slice().reverse()) { while (upper.length >= 2 && cross(upper.at(-2), upper.at(-1), p) <= 0) upper.pop(); upper.push(p); }
  return lower.slice(0, -1).concat(upper.slice(0, -1));
}

export function solidity(m, w, h) {
  const hl = hull(m, w, h);
  let area = 0;
  for (let i = 0; i < hl.length; i++) {
    const [x1, y1] = hl[i], [x2, y2] = hl[(i + 1) % hl.length];
    area += x1 * y2 - x2 * y1;
  }
  return count(m) / Math.max(Math.abs(area) / 2, 1);
}

// Summed-area table of a mask, for "is this whole rectangle inside the mask?" checks
export function integral(m, w, h) {
  const s = new Int32Array((w + 1) * (h + 1));
  for (let y = 0; y < h; y++) {
    let row = 0;
    for (let x = 0; x < w; x++) {
      row += m[y * w + x];
      s[(y + 1) * (w + 1) + x + 1] = s[y * (w + 1) + x + 1] + row;
    }
  }
  return s;
}

export function rectSum(s, w, x1, y1, x2, y2) {
  const W = w + 1;
  return s[y2 * W + x2] - s[y1 * W + x2] - s[y2 * W + x1] + s[y1 * W + x1];
}

// ---------------------------------------------------------------- colours

// Most common colour, binned by 8 (bin centre), like regions.dominant_colour
export function dominantColour(img, mask) {
  const counts = new Map();
  const { px } = img;
  for (let i = 0; i < mask.length; i++) {
    if (!mask[i]) continue;
    const key = ((px[4 * i] >> 3) << 10) | ((px[4 * i + 1] >> 3) << 5) | (px[4 * i + 2] >> 3);
    counts.set(key, (counts.get(key) || 0) + 1);
  }
  let best = 0, bestN = -1;
  for (const [k, n] of counts) if (n > bestN) { best = k; bestN = n; }
  return [((best >> 10) & 31) * 8 + 4, ((best >> 5) & 31) * 8 + 4, (best & 31) * 8 + 4];
}

// Pixels in `area` whose colour differs from bg by more than tol in some channel
export function inkMask(img, area, bg, tol = 60) {
  const { px } = img, out = new Uint8Array(area.length);
  for (let i = 0; i < area.length; i++) {
    if (!area[i]) continue;
    const d = Math.max(Math.abs(px[4 * i] - bg[0]), Math.abs(px[4 * i + 1] - bg[1]), Math.abs(px[4 * i + 2] - bg[2]));
    if (d > tol) out[i] = 1;
  }
  return out;
}

// Typical letter height of the lettering in a text mask (70th percentile of blob heights)
export function letterHeight(m, w, h) {
  const { boxes, areas } = label(m, w, h);
  const hs = [];
  boxes.forEach(([x1, y1, x2, y2], i) => { const hh = y2 - y1; if ((x2 - x1) * hh >= 12 && hh >= 4) hs.push(hh); });
  return hs.length ? percentile(hs, 70) : 0;
}

export function percentile(values, p) {  // numpy's default (linear) percentile
  const v = values.slice().sort((a, b) => a - b);
  const pos = (v.length - 1) * p / 100, lo = Math.floor(pos), hi = Math.ceil(pos);
  return v[lo] + (v[hi] - v[lo]) * (pos - lo);
}

export const median = values => percentile(values, 50);
