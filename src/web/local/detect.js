// Text detection post-processing: a JavaScript port of src/shavcomic/detect_ctd.py.
// Takes comic-text-detector's text-pixel map (seg) and line map (det), already
// resized to the image, and returns regions with the same fields as the Python ones.
import * as ip from "./imgproc.js";

const SEG_THRESHOLD = 0.4, DET_THRESHOLD = 0.3;
const SFX_RATIO = 2.2, SMALL_RATIO = 0.45, BUBBLE_SEARCH = 10, MAX_BUBBLE_AREA = 15;
const MAX_NONTEXT_HOLES = 0.04, MIN_SOLIDITY = 0.75, STRONG_INK = 110;
const MAX_HOLE_HEIGHT = 3.0, SNUG_MARGIN = 2.0, MAX_OUTSIDE = 0.4;

export const unionBox = boxes => [
  Math.min(...boxes.map(b => b[0])), Math.min(...boxes.map(b => b[1])),
  Math.max(...boxes.map(b => b[2])), Math.max(...boxes.map(b => b[3]))];

// The bg-coloured region overlapping the seed most, with holes filled (colour_region)
function colourRegion(img, seed, bg, tol, margin) {
  const { w, h, px } = img;
  const box = ip.bbox(seed, w, h);
  const x1 = Math.max(0, box[0] - margin), y1 = Math.max(0, box[1] - margin);
  const x2 = Math.min(w, box[2] + margin), y2 = Math.min(h, box[3] + margin);
  const cw = x2 - x1, ch = y2 - y1;
  const isBg = new Uint8Array(cw * ch), seedC = new Uint8Array(cw * ch);
  for (let y = 0; y < ch; y++) for (let x = 0; x < cw; x++) {
    const i = (y + y1) * w + x + x1, j = y * cw + x;
    seedC[j] = seed[i];
    isBg[j] = Math.max(Math.abs(px[4 * i] - bg[0]), Math.abs(px[4 * i + 1] - bg[1]), Math.abs(px[4 * i + 2] - bg[2])) <= tol ? 1 : 0;
  }
  const { labels, n } = ip.label(isBg, cw, ch);
  if (!n) return null;
  const overlap = new Int32Array(n + 1);
  for (let j = 0; j < labels.length; j++) if (seedC[j] && labels[j]) overlap[labels[j]]++;
  let best = 1;
  for (let k = 2; k <= n; k++) if (overlap[k] > overlap[best]) best = k;
  if (!overlap[best]) return null;
  const raw = new Uint8Array(cw * ch);
  for (let j = 0; j < raw.length; j++) raw[j] = labels[j] === best ? 1 : 0;
  const filled = ip.fillHoles(raw, cw, ch);
  let touches = false;
  for (let x = 0; x < cw && !touches; x++) touches = (y1 > 0 && raw[x]) || (y2 < h && raw[(ch - 1) * cw + x]);
  for (let y = 0; y < ch && !touches; y++) touches = (x1 > 0 && raw[y * cw]) || (x2 < w && raw[y * cw + cw - 1]);
  const fullRaw = ip.newMask(w, h), fullFilled = ip.newMask(w, h);
  for (let y = 0; y < ch; y++) for (let x = 0; x < cw; x++) {
    fullRaw[(y + y1) * w + x + x1] = raw[y * cw + x];
    fullFilled[(y + y1) * w + x + x1] = filled[y * cw + x];
  }
  return { filled: fullFilled, raw: fullRaw, bg, enclosed: !touches };
}

function findLines(seg, det, w, h) {
  const ink = new Uint8Array(w * h), detBin = new Uint8Array(w * h);
  for (let i = 0; i < ink.length; i++) { ink[i] = seg[i] > SEG_THRESHOLD; detBin[i] = det[i] > DET_THRESHOLD; }
  const { boxes } = ip.label(detBin, w, h);
  const rowHasInk = (y, x1, x2) => { for (let x = x1; x < x2; x++) if (ink[y * w + x]) return true; return false; };
  const lines = new Map();
  for (const [cx1, cy1, cx2, cy2] of boxes) {
    const chh = cy2 - cy1;
    if (chh < 2 || cx2 - cx1 < 3) continue;
    const grow = Math.max(2, Math.floor(0.9 * chh));
    const x1 = Math.max(0, cx1 - chh), x2 = Math.min(w, cx2 + chh);
    let y1 = cy1, y2 = cy2;
    for (let k = 0; k < grow && y1 > 0 && rowHasInk(y1 - 1, x1, x2); k++) y1--;
    for (let k = 0; k < grow && y2 < h && rowHasInk(y2, x1, x2); k++) y2++;
    const idx = [];
    for (let x = x1; x < x2; x++) { for (let y = y1; y < y2; y++) if (ink[y * w + x]) { idx.push(x - x1); break; } }
    if (!idx.length) continue;
    const lh = y2 - y1;
    let run = [idx[0]];
    const runs = [];
    for (let k = 1; k < idx.length; k++) {
      if (idx[k] - idx[k - 1] > 1.2 * lh) { runs.push(run); run = []; }
      run.push(idx[k]);
    }
    runs.push(run);
    for (const r of runs) {
      const rx1 = x1 + r[0], rx2 = x1 + r.at(-1) + 1;
      let top = -1, bottom = -1;
      for (let y = y1; y < y2; y++) if (rowHasInk(y, rx1, rx2)) { if (top < 0) top = y; bottom = y; }
      if (top >= 0 && rx2 - rx1 >= 3) {
        const b = [rx1, top, rx2, bottom + 1];
        lines.set(b.join(","), b);
      }
    }
  }
  return [...lines.values()].sort((a, b) => a[1] - b[1] || a[0] - b[0]);
}

function groupFreeLines(lines) {
  const parent = lines.map((_, i) => i);
  const find = i => { while (parent[i] !== i) { parent[i] = parent[parent[i]]; i = parent[i]; } return i; };
  for (let i = 0; i < lines.length; i++) for (let j = i + 1; j < lines.length; j++) {
    const a = lines[i], b = lines[j];
    const ha = a[3] - a[1], hb = b[3] - b[1];
    const vgap = Math.max(a[1], b[1]) - Math.min(a[3], b[3]);
    const hover = Math.min(a[2], b[2]) - Math.max(a[0], b[0]);
    if (vgap <= 0.8 * Math.max(ha, hb) && hover > 0 && Math.max(ha, hb) <= 1.6 * Math.min(ha, hb)) parent[find(i)] = find(j);
  }
  const groups = new Map();
  lines.forEach((l, i) => { const r = find(i); if (!groups.has(r)) groups.set(r, []); groups.get(r).push(l); });
  const out = [...groups.values()];
  let merged = true;
  while (merged) {
    merged = false;
    outer: for (let i = 0; i < out.length; i++) for (let j = i + 1; j < out.length; j++) {
      const a = unionBox(out[i]), b = unionBox(out[j]);
      if (Math.min(a[2], b[2]) > Math.max(a[0], b[0]) && Math.min(a[3], b[3]) > Math.max(a[1], b[1])) {
        out[i] = out[i].concat(out.splice(j, 1)[0]);
        merged = true;
        break outer;
      }
    }
  }
  return out;
}

function findBubble(img, box, inkWide, typical) {
  const { w, h } = img;
  const lh = box[3] - box[1];
  const seed = ip.rectMask(w, h, box);
  const between = ip.andNot(seed, inkWide);
  const bg = ip.dominantColour(img, ip.count(between) > 20 ? between : seed);
  const found = colourRegion(img, seed, bg, 40, Math.floor(BUBBLE_SEARCH * Math.min(lh, 2 * typical)));
  if (!found) return null;
  const { filled, raw, enclosed } = found;
  const holes = ip.andNot(ip.andNot(filled, raw), inkWide);
  const nFilled = ip.count(filled);
  if (!(enclosed && ip.count(ip.and(filled, seed)) >= 0.9 * ip.count(seed) && ip.count(holes) <= MAX_NONTEXT_HOLES * nFilled
        && ip.solidity(filled, w, h) >= MIN_SOLIDITY)) return null;
  const { boxes } = ip.label(holes, w, h);
  if (boxes.length && Math.max(...boxes.map(b => b[3] - b[1])) > MAX_HOLE_HEIGHT * lh) return null;
  return { interior: filled, bg: found.bg };
}

function hugsText(interior, lines, w, h) {
  const lh = ip.median(lines.map(l => l[3] - l[1]));
  const near = ip.rectMask(w, h, unionBox(lines), Math.floor(SNUG_MARGIN * lh));
  return ip.count(ip.andNot(interior, near)) <= MAX_OUTSIDE * ip.count(interior);
}

function textMask(ink, lines, w, h) {
  const area = ip.newMask(w, h);
  for (const l of lines) { const r = ip.rectMask(w, h, l, 2); for (let i = 0; i < r.length; i++) area[i] |= r[i]; }
  return ip.and(ink, area);
}

function bubbleRegion(ink, interior, bg, lines, w, h, note) {
  return {
    kind: "bubble", box: unionBox(lines), layout: interior, erase: ip.erode(interior, w, h, 1),
    text: textMask(ink, lines, w, h), bg, letterHeight: ip.median(lines.map(l => l[3] - l[1])),
    note: note || `${lines.length} line(s) in bubble`,
  };
}

function freeRegion(img, ink, lines, note) {
  const { w, h } = img;
  const box = unionBox(lines);
  const lh = ip.median(lines.map(l => l[3] - l[1]));
  let text = textMask(ink, lines, w, h);
  const around = ip.andNot(ip.rectMask(w, h, box, Math.floor(0.5 * lh)), ip.dilate(text, w, h, 2));
  const bg = ip.count(around) ? ip.dominantColour(img, around) : [255, 255, 255];
  // Long straight lines next to the text (axes, panel borders) aren't lettering
  const run = Math.max(5, Math.floor(1.6 * lh));
  const drawn = ip.inkMask(img, ip.rectMask(w, h, box, Math.floor(lh)), bg, STRONG_INK);
  const art = ip.dilate(ip.longLines(drawn, w, h, run), w, h, 1);
  text = ip.andNot(text, art);
  // Room for the new text: the padded box minus any other drawing in it
  const room = ip.rectMask(w, h, box, Math.floor(0.3 * lh));
  const other = ip.dilate(ip.andNot(ip.inkMask(img, room, bg, STRONG_INK), ip.dilate(text, w, h, 3)), w, h, 2);
  let layout = ip.andNot(room, other);
  if (ip.count(layout) < 0.6 * ip.count(room)) layout = room;
  return {
    kind: "free", box, layout, erase: ip.andNot(ip.dilate(text, w, h, 3), art), text, bg,
    letterHeight: lh, note: note || `${lines.length} free line(s)`,
  };
}

// A region for any box (a skipped detection, or one drawn by hand): region_from_box
export function regionFromBox(ctx, box, note = "added by hand") {
  const { img, ink, inkWide, typical } = ctx;
  const { w, h } = img;
  let [x1, y1, x2, y2] = box.map(Math.round);
  x1 = Math.max(0, x1); y1 = Math.max(0, y1); x2 = Math.min(w, x2); y2 = Math.min(h, y2);
  let tx1 = w, ty1 = h, tx2 = -1, ty2 = -1;
  for (let y = y1; y < y2; y++) for (let x = x1; x < x2; x++) if (ink[y * w + x]) {
    tx1 = Math.min(tx1, x); tx2 = Math.max(tx2, x); ty1 = Math.min(ty1, y); ty2 = Math.max(ty2, y);
  }
  const tight = tx2 >= 0 ? [tx1, ty1, tx2 + 1, ty2 + 1] : [x1, y1, x2, y2];
  const found = findBubble(img, tight, inkWide, typical);
  const r = found ? bubbleRegion(ink, found.interior, found.bg, [tight], w, h, note) : freeRegion(img, ink, [tight], note);
  const lh = ip.letterHeight(r.text, w, h);
  if (lh) r.letterHeight = lh;
  return r;
}

// seg, det: Float32Array(w*h) in 0..1. Returns {regions, skipped: [[box, reason]], ctx}
export function detect(img, seg, det) {
  const { w, h } = img;
  const ink = new Uint8Array(w * h);
  for (let i = 0; i < ink.length; i++) ink[i] = seg[i] > SEG_THRESHOLD ? 1 : 0;
  const inkWide = ip.dilate(ink, w, h, 2);
  const lines = findLines(seg, det, w, h);
  const ctx = { img, ink, inkWide, typical: 0 };
  if (!lines.length) return { regions: [], skipped: [], ctx };
  const typical = ip.percentile(lines.map(b => b[3] - b[1]), 35);
  ctx.typical = typical;

  const skipped = [], normal = [], containers = [];
  for (const b of lines) {
    const lh = b[3] - b[1], lw = b[2] - b[0];
    if (lh < SMALL_RATIO * typical) { skipped.push([b, "small print"]); continue; }
    if (lh > 2.5 * lw && lh > 1.5 * typical) { skipped.push([b, "vertical text"]); continue; }
    const c = findBubble(img, b, inkWide, typical);
    if (lh > SFX_RATIO * typical && !c) { skipped.push([b, "sound effect / logo"]); continue; }
    normal.push(b);
    containers.push(c);
  }

  const bubbleGroups = [], freeLines = [];
  normal.forEach((b, k) => {
    const c = containers[k];
    if (!c) { freeLines.push(b); return; }
    const nc = ip.count(c.interior);
    for (const g of bubbleGroups) {
      const inter = ip.count(ip.and(g.interior, c.interior));
      if (inter >= 0.7 * Math.min(g.n, nc)) { g.lines.push(b); return; }
    }
    bubbleGroups.push({ interior: c.interior, bg: c.bg, lines: [b], n: nc });
  });

  const regions = [];
  for (const g of bubbleGroups) {
    const box = unionBox(g.lines);
    if (g.n > MAX_BUBBLE_AREA * (box[2] - box[0]) * (box[3] - box[1]) || !hugsText(g.interior, g.lines, w, h)) {
      freeLines.push(...g.lines);
      continue;
    }
    regions.push(bubbleRegion(ink, g.interior, g.bg, g.lines, w, h));
  }
  for (const group of groupFreeLines(freeLines)) regions.push(freeRegion(img, ink, group));
  regions.sort((a, b) => a.box[1] - b.box[1] || a.box[0] - b.box[0]);
  return { regions, skipped, ctx };
}
