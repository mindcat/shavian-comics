// "This device" mode: answers the web tool's /api/ calls entirely in the browser, so
// the page needs no server (e.g. on GitHub Pages). Mirrors src/server.py's API.
import * as ip from "./imgproc.js";
import { detect, regionFromBox } from "./detect.js";
import { ERASERS, drawText, FONT_FAMILY } from "./render.js";

const CTD_SIZE = 1024;

// Phones and tablets get a lower-memory profile: a smaller working image, CPU only,
// and only one big model in memory at a time (mobile browsers silently reload a tab
// that uses too much memory)
export const LOW_MEMORY = Boolean(
  navigator.userAgentData?.mobile || /iPhone|iPad|iPod|Android/i.test(navigator.userAgent) ||
  (navigator.platform === "MacIntel" && navigator.maxTouchPoints > 1) ||  // iPadOS
  (navigator.deviceMemory && navigator.deviceMemory <= 4));
const MAX_SIDE = LOW_MEMORY ? 1400 : 2000;
const GROQ_URL = "https://api.groq.com/openai/v1";
const GROQ_PREFERRED = ["llama-4-scout", "llama-4-maverick", "vision", "-vl", "qwen"];
const OCR_PROMPT = "Transcribe the text, all lowercase unless it is a proper name or place. " +
  "If it isn't actual text, return an empty string.";

const jsonResponse = (obj, status = 200) =>
  new Response(JSON.stringify(obj), { status, headers: { "Content-Type": "application/json" } });
const errorResponse = (detail, status = 400) => jsonResponse({ detail }, status);

// Each kind of model gets its own worker. WebAssembly memory never shrinks, so the
// only way to really give memory back (which phones need) is to terminate a worker.
const WORKER_FOR = { ctd: "ctd", florence: "ocr", translit: "py", tidyCapitals: "py", clean: "py" };

export class LocalBackend {
  constructor(assets) {
    this.assets = assets;
    this.jobs = new Map();
    this.onProgress = null;
    this.workers = {};
    this.nextCall = 1;
    this.lowMemory = LOW_MEMORY;
    this.ready = Promise.resolve();
    this.fontReady = new FontFace(FONT_FAMILY, `url(${assets.font})`).load()
      .then(f => document.fonts.add(f)).catch(() => {});
  }

  worker(kind) {
    if (!this.workers[kind]) {
      const w = new Worker(new URL("./worker.js", import.meta.url), { type: "module" });
      const calls = new Map();
      w.onmessage = e => {
        const { id, result, error, progress } = e.data;
        if (progress) { if (this.onProgress) this.onProgress(progress); return; }
        const call = calls.get(id);
        if (!call) return;
        calls.delete(id);
        error ? call.reject(new Error(error)) : call.resolve(result);
      };
      // Messages are handled in order, so init always runs before the first real call
      w.postMessage({ id: 0, op: "init", args: { urls: this.assets, lowMem: this.lowMemory, smallOcr: this.ocrMode === "florence-small" } });
      this.workers[kind] = { w, calls };
    }
    return this.workers[kind];
  }

  call(op, args = {}, transfer = []) {
    const { w, calls } = this.worker(WORKER_FOR[op]);
    const id = this.nextCall++;
    return new Promise((resolve, reject) => {
      calls.set(id, { resolve, reject });
      w.postMessage({ id, op, args }, transfer);
    });
  }

  terminate(kind) {
    const entry = this.workers[kind];
    if (!entry) return;
    entry.w.terminate();
    for (const call of entry.calls.values()) call.reject(new Error("cancelled"));
    delete this.workers[kind];
  }

  // Called by the page when it's done reading a comic, or is about to pick a new one
  releaseModels() {
    if (this.lowMemory) { this.terminate("ctd"); this.terminate("ocr"); }
  }

  get ocrMode() { return localStorage.getItem("shavian-local-ocr") || "florence"; }
  get groqKey() { return localStorage.getItem("shavian-groq-key") || ""; }

  // ------------------------------------------------------------ fetch-like entry point
  async fetch(path, opts = {}) {
    const method = (opts.method || "GET").toUpperCase();
    const url = new URL(path, location.href);
    const parts = url.pathname.replace(/^\/api\//, "").split("/");
    const body = opts.body && typeof opts.body === "string" ? JSON.parse(opts.body) : opts.body;
    try {
      if (parts[0] === "info") return jsonResponse(await this.info());
      if (parts[0] !== "jobs") return errorResponse("Not found", 404);
      if (parts.length === 1 && method === "POST") return jsonResponse(await this.createJob(body.get("file")));
      const job = this.jobs.get(parts[1]);
      if (!job) return errorResponse("Unknown or expired job; upload the image again", 404);
      if (parts[2] === "ocr") return jsonResponse(await this.readRegions(job, body.ids));
      if (parts[2] === "regions" && parts.length === 3) return jsonResponse(await this.addRegion(job, body.box));
      if (parts[2] === "regions" && method === "PATCH") return jsonResponse(await this.editRegion(job, Number(parts[3]), body));
      if (parts[2] === "regions" && method === "DELETE") { job.regions.delete(Number(parts[3])); return jsonResponse({ ok: true }); }
      if (parts[2] === "render.png") return await this.render(job, url.searchParams.get("stage") || "final");
      return errorResponse("Not found", 404);
    } catch (e) {
      console.error(e);
      return errorResponse(e.message || String(e), 500);
    }
  }

  async info() {
    const gpu = !!navigator.gpu;
    const groq = this.ocrMode === "groq";
    return {
      ready: true, loading: null, local: true,
      device: `this device${gpu ? " (WebGPU)" : ""}`,
      ocr: groq ? "Groq (your key)" : `Florence-2-base${this.ocrMode === "florence-small" ? ", smaller" : ""} (on device)`,
      low_memory: this.lowMemory,
      // Phones read every region in one go, so the OCR model can be unloaded before Python runs
      ocr_batch: groq || this.lowMemory, ocr_fallback: null, dictionary: "amer",
      uploads_per_day: null, uploads_left: null,
    };
  }

  // ------------------------------------------------------------ jobs
  async createJob(file) {
    await this.ready;
    this.releaseModels();
    const bitmap = await createImageBitmap(file, { imageOrientation: "from-image" });
    let w = bitmap.width, h = bitmap.height;
    const scale = Math.min(1, MAX_SIDE / Math.max(w, h));
    w = Math.round(w * scale); h = Math.round(h * scale);
    const canvas = new OffscreenCanvas(w, h);
    const c = canvas.getContext("2d", { willReadFrequently: true });
    c.fillStyle = "#fff";
    c.fillRect(0, 0, w, h);
    c.imageSmoothingQuality = "high";
    c.drawImage(bitmap, 0, 0, w, h);
    const img = { w, h, px: c.getImageData(0, 0, w, h).data };

    // comic-text-detector input: scaled into the top-left of a 1024x1024 black square
    const r = CTD_SIZE / Math.max(w, h), nw = Math.round(w * r), nh = Math.round(h * r);
    const sq = new OffscreenCanvas(CTD_SIZE, CTD_SIZE).getContext("2d", { willReadFrequently: true });
    sq.fillStyle = "#000";
    sq.fillRect(0, 0, CTD_SIZE, CTD_SIZE);
    sq.imageSmoothingQuality = "high";
    sq.drawImage(bitmap, 0, 0, nw, nh);
    const sp = sq.getImageData(0, 0, CTD_SIZE, CTD_SIZE).data;
    const N = CTD_SIZE * CTD_SIZE, input = new Float32Array(3 * N);
    for (let i = 0; i < N; i++) { input[i] = sp[4 * i] / 255; input[N + i] = sp[4 * i + 1] / 255; input[2 * N + i] = sp[4 * i + 2] / 255; }
    const { seg, det } = await this.call("ctd", { input }, [input.buffer]);
    if (this.lowMemory) this.terminate("ctd");  // give its memory back before the OCR model loads
    const segFull = resizeMap(seg, nw, nh, w, h), detFull = resizeMap(det, nw, nh, w, h);

    const { regions, skipped, ctx } = detect(img, segFull, detFull);
    const job = { id: Math.random().toString(36).slice(2, 14), name: (file.name || "comic").replace(/\.[^.]+$/, ""),
                  img, ctx, regions: new Map(), nextId: 1, eraseKey: null, erased: null };
    for (const reg of regions) addTo(job, reg, true);
    for (const [box, reason] of skipped) addTo(job, regionFromBox(ctx, box, `skipped: ${reason}`), false);
    this.jobs.set(job.id, job);
    while (this.jobs.size > 8) this.jobs.delete(this.jobs.keys().next().value);
    return { ...jobJson(job), uploads_left: null };
  }

  async readRegions(job, ids) {
    const regs = ids.map(i => job.regions.get(i)).filter(Boolean);
    if (!regs.length) return [];
    const crops = regs.map(r => cropForOcr(job.img, r));
    let texts;
    if (this.ocrMode === "groq" && this.groqKey) {
      texts = await this.call("clean", { texts: await groqOcr(crops, this.groqKey) });
    } else {
      const raw = await this.call("florence", { crops: crops.map(c => c.up) });
      if (this.lowMemory) this.terminate("ocr");  // never hold the OCR model and Python at once on phones
      texts = await this.call("tidyCapitals", { texts: raw });
    }
    const shavian = await this.call("translit", { texts });
    regs.forEach((r, k) => { r.english = texts[k]; r.shavian = shavian[k]; r.read = true; });
    return regs.map(regionJson);
  }

  async addRegion(job, box) {
    const [x1, y1, x2, y2] = box;
    if (Math.abs(x2 - x1) < 4 || Math.abs(y2 - y1) < 4) throw new Error("Box too small");
    const r = regionFromBox(job.ctx, [Math.min(x1, x2), Math.min(y1, y2), Math.max(x1, x2), Math.max(y1, y2)]);
    return regionJson(addTo(job, r, true));
  }

  async editRegion(job, rid, body) {
    const r = job.regions.get(rid);
    if (!r) throw new Error("No such region");
    if (body.english != null) {
      r.english = body.english.trim();
      r.read = true;
      if (body.shavian == null) r.shavian = (await this.call("translit", { texts: [r.english] }))[0];
    }
    if (body.shavian != null) r.shavian = body.shavian.trim();
    for (const k of ["enabled", "eraser", "size"]) if (body[k] != null) r.settings[k] = body[k];
    return regionJson(r);
  }

  async render(job, stage) {
    await this.fontReady;
    const { w, h } = job.img;
    const regs = [...job.regions.values()];
    const key = JSON.stringify(regs.map(r => [r.id, !!r.shavian && r.settings.enabled, r.settings.eraser]));
    if (job.eraseKey !== key) {
      const img = { w, h, px: job.img.px.slice() };
      for (const r of regs) {
        if (!r.shavian || !r.settings.enabled) continue;
        const eraser = ERASERS[r.settings.eraser] || (r.settings.eraser === "fill" ? ERASERS.fill : ERASERS.auto);
        (r.settings.eraser === "lama" || r.settings.eraser === "telea" ? ERASERS.diffuse : eraser)(img, r);
      }
      job.erased = img;
      job.eraseKey = key;
    }
    // A regular canvas, so the Shavian web font loaded into the page is available
    const canvas = document.createElement("canvas");
    canvas.width = w; canvas.height = h;
    const c = canvas.getContext("2d");
    c.putImageData(new ImageData(job.erased.px.slice(), w, h), 0, 0);
    const sizes = [];
    if (stage !== "erased") {
      for (const r of regs) {
        if (!r.shavian || !r.settings.enabled) continue;
        r.fontSize = drawText(c, r, r.shavian, r.settings.size, w, h);
        sizes.push(`${r.id}:${r.fontSize}`);
      }
    }
    const blob = await new Promise(res => canvas.toBlob(res, "image/png"));
    return new Response(blob, { headers: { "Content-Type": "image/png", "X-Font-Sizes": sizes.join(",") } });
  }
}

// ---------------------------------------------------------------- helpers

function addTo(job, r, enabled) {
  r.id = job.nextId++;
  r.settings = { enabled, eraser: "auto", size: "match" };
  r.english = null; r.shavian = ""; r.read = false;
  job.regions.set(r.id, r);
  return r;
}

function regionJson(r) {
  return {
    id: r.id, kind: r.kind, box: r.box.map(Math.round), outline: r.outline || [], note: r.note,
    letter_height: Math.round(r.letterHeight * 10) / 10, english: r.english, shavian: r.shavian || "",
    read: r.read, font_size: r.fontSize || null, ...r.settings,
  };
}

function jobJson(job) {
  for (const r of job.regions.values()) if (!r.outline) r.outline = ip.hull(r.layout, job.img.w, job.img.h);
  return { id: job.id, name: job.name, width: job.img.w, height: job.img.h, regions: [...job.regions.values()].map(regionJson) };
}

// Bilinear resize of the top-left nw x nh part of a 1024x1024 map to w x h (cv2 INTER_LINEAR)
function resizeMap(src, nw, nh, w, h) {
  const out = new Float32Array(w * h);
  for (let y = 0; y < h; y++) {
    const fy = Math.min(Math.max((y + 0.5) * nh / h - 0.5, 0), nh - 1);
    const y0 = Math.floor(fy), y1 = Math.min(y0 + 1, nh - 1), dy = fy - y0;
    for (let x = 0; x < w; x++) {
      const fx = Math.min(Math.max((x + 0.5) * nw / w - 0.5, 0), nw - 1);
      const x0 = Math.floor(fx), x1 = Math.min(x0 + 1, nw - 1), dx = fx - x0;
      const a = src[y0 * CTD_SIZE + x0], b = src[y0 * CTD_SIZE + x1], c = src[y1 * CTD_SIZE + x0], d = src[y1 * CTD_SIZE + x1];
      out[y * w + x] = (a * (1 - dx) + b * dx) * (1 - dy) + (c * (1 - dx) + d * dx) * dy;
    }
  }
  return out;
}

// The region's text on a clean background (crop_for_ocr), at 1x and upscaled 2x (RGB)
function cropForOcr(img, r) {
  const { w, h, px } = img;
  let box, mask = null;
  if (r.kind === "bubble") {
    mask = ip.dilate(r.layout, w, h, 2);
    box = ip.bbox(mask, w, h);
  } else {
    const pad = Math.floor(Math.max(6, 0.5 * r.letterHeight));
    box = [Math.max(0, r.box[0] - pad), Math.max(0, r.box[1] - pad), Math.min(w, r.box[2] + pad), Math.min(h, r.box[3] + pad)];
  }
  const [x1, y1, x2, y2] = box, cw = x2 - x1, ch = y2 - y1;
  const one = new OffscreenCanvas(cw, ch);
  const oc = one.getContext("2d", { willReadFrequently: true });
  const data = new ImageData(cw, ch);
  for (let y = 0; y < ch; y++) for (let x = 0; x < cw; x++) {
    const i = (y + y1) * w + x + x1, j = 4 * (y * cw + x);
    const keep = !mask || mask[i];
    for (let c = 0; c < 3; c++) data.data[j + c] = keep ? px[4 * i + c] : 255;
    data.data[j + 3] = 255;
  }
  oc.putImageData(data, 0, 0);
  const upC = new OffscreenCanvas(cw * 2, ch * 2).getContext("2d", { willReadFrequently: true });
  upC.imageSmoothingQuality = "high";
  upC.drawImage(one, 0, 0, cw * 2, ch * 2);
  const upPx = upC.getImageData(0, 0, cw * 2, ch * 2).data;
  const rgb = new Uint8ClampedArray(cw * 2 * ch * 2 * 3);
  for (let i = 0, j = 0; i < upPx.length; i += 4, j += 3) { rgb[j] = upPx[i]; rgb[j + 1] = upPx[i + 1]; rgb[j + 2] = upPx[i + 2]; }
  return { one, up: { w: cw * 2, h: ch * 2, data: rgb } };
}

// ---------------------------------------------------------------- Groq (visitor's own key)

let groqModelId = null;
async function groqModel(key) {
  if (groqModelId) return groqModelId;
  const res = await fetch(`${GROQ_URL}/models`, { headers: { Authorization: `Bearer ${key}` } });
  if (!res.ok) throw new Error(`Groq: ${res.status} (is the key right?)`);
  const ids = (await res.json()).data.map(m => m.id);
  for (const hint of GROQ_PREFERRED) { const id = ids.find(i => i.toLowerCase().includes(hint)); if (id) return (groqModelId = id); }
  throw new Error("Groq: no vision model found");
}

// Crops tiled into numbered sheets and read in one request per sheet (like ocr.groq)
async function groqOcr(crops, key) {
  const model = await groqModel(key);
  const out = new Array(crops.length).fill("");
  const groups = [];
  let cur = [], height = 0;
  crops.forEach((c, i) => {
    if (cur.length && (height + c.one.height + 8 > 1600 || cur.length >= 16)) { groups.push(cur); cur = []; height = 0; }
    cur.push(i); height += c.one.height + 8;
  });
  if (cur.length) groups.push(cur);
  for (const g of groups) {
    const label = 40, pad = 8;
    const W = Math.max(...g.map(i => crops[i].one.width)) + label + pad;
    const H = g.reduce((s, i) => s + crops[i].one.height + pad, 0);
    const sheet = new OffscreenCanvas(W, H), c = sheet.getContext("2d");
    c.fillStyle = "#fff"; c.fillRect(0, 0, W, H);
    let y = 0;
    g.forEach((i, n) => {
      const one = crops[i].one;
      c.strokeStyle = "#c80000"; c.lineWidth = 2; c.strokeRect(1, y + 1, W - 2, one.height + pad - 2);
      c.fillStyle = "#c80000"; c.font = "bold 20px sans-serif"; c.fillText(String(n + 1), 6, y + 24);
      c.drawImage(one, label, y + pad / 2);
      y += one.height + pad;
    });
    const blob = await sheet.convertToBlob({ type: "image/jpeg", quality: 0.9 });
    const dataUrl = await new Promise(res => { const fr = new FileReader(); fr.onload = () => res(fr.result); fr.readAsDataURL(blob); });
    const prompt = `This image is a stack of ${g.length} boxes, numbered in red on the left, each cut from a comic. ` +
      "For each box, transcribe its lettering all lowercase unless a word is a proper name or place; " +
      "use an empty string if a box has no real text. Reply with only a JSON object mapping each box number to its text.";
    const res = await fetch(`${GROQ_URL}/chat/completions`, {
      method: "POST", headers: { Authorization: `Bearer ${key}`, "Content-Type": "application/json" },
      body: JSON.stringify({ model, temperature: 0, max_completion_tokens: 150 * g.length + 100,
        response_format: { type: "json_object" },
        messages: [{ role: "user", content: [{ type: "text", text: prompt }, { type: "image_url", image_url: { url: dataUrl } }] }] }),
    });
    if (res.status === 429) throw new Error("Groq rate limit reached; wait a minute or switch to on-device OCR");
    if (!res.ok) throw new Error(`Groq: ${res.status} ${(await res.text()).slice(0, 200)}`);
    const answer = JSON.parse((await res.json()).choices[0].message.content);
    g.forEach((i, n) => { out[i] = String(answer[String(n + 1)] ?? ""); });
  }
  return out;
}
