// Runs the heavy parts of the in-browser pipeline off the main thread:
//   ctd       comic-text-detector (ONNX Runtime Web: WebGPU, else WASM)
//   florence  Florence-2-base OCR (transformers.js)
//   shaw      dechifro's shaw.py + our textproc.py (Pyodide), with NLTK POS tagging
// Each loads on first use. Messages: {id, op, args} -> {id, result} | {id, error};
// progress is posted as {progress: {what, loaded, total}}.

const ORT_URL = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.22.0/dist/";
const TRANSFORMERS_URL = "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.7.1";
const PYODIDE_URL = "https://cdn.jsdelivr.net/pyodide/v0.28.0/full/";
const FLORENCE_ID = "onnx-community/Florence-2-base-ft";
const CTD_SIZE = 1024;

let assets = null;       // {ctd, dict, shaw, textproc, nltk: {weights, tagdict, classes}}
let lowMemory = false;   // phones/tablets: CPU only, and models are released between comics
let ctdSession = null, florence = null, py = null;
const loading = {};

const progress = (what, loaded, total) => postMessage({ progress: { what, loaded, total } });

async function fetchWithProgress(url, what) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${what}: HTTP ${res.status} for ${url}`);
  const total = Number(res.headers.get("content-length")) || 0;
  const reader = res.body.getReader();
  const chunks = [];
  let loaded = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    loaded += value.length;
    progress(what, loaded, total);
  }
  const out = new Uint8Array(loaded);
  let off = 0;
  for (const c of chunks) { out.set(c, off); off += c.length; }
  return out;
}

// Cache big downloads (the detector model) across visits
async function cachedBytes(url, what) {
  try {
    const cache = await caches.open("shavian-comics-v1");
    const hit = await cache.match(url);
    if (hit) return new Uint8Array(await hit.arrayBuffer());
    const bytes = await fetchWithProgress(url, what);
    await cache.put(url, new Response(bytes));
    return bytes;
  } catch (e) {
    if (String(e).includes("HTTP")) throw e;
    return fetchWithProgress(url, what);  // Cache API unavailable (e.g. private mode)
  }
}

async function hasWebGPU() {
  if (lowMemory) return false;  // keep phones on the CPU path: lower memory, fewer GPU-process crashes
  try { return !!(navigator.gpu && await navigator.gpu.requestAdapter()); } catch { return false; }
}

// ---------------------------------------------------------------- comic-text-detector
async function loadCtd() {
  const ort = await import(ORT_URL + "ort.webgpu.min.mjs");
  ort.env.wasm.wasmPaths = ORT_URL;
  ort.env.wasm.numThreads = self.crossOriginIsolated ? Math.min(4, navigator.hardwareConcurrency || 1) : 1;
  const bytes = await cachedBytes(assets.ctd, "text detector (95 MB, first time only)");
  const providers = (await hasWebGPU()) ? ["webgpu", "wasm"] : ["wasm"];
  try {
    ctdSession = await ort.InferenceSession.create(bytes, { executionProviders: providers });
  } catch (e) {
    ctdSession = await ort.InferenceSession.create(bytes, { executionProviders: ["wasm"] });
  }
  ctdSession.ort = ort;
  return ctdSession;
}

async function runCtd({ input }) {
  const s = ctdSession || await (loading.ctd ||= loadCtd());
  const tensor = new s.ort.Tensor("float32", input, [1, 3, CTD_SIZE, CTD_SIZE]);
  const out = await s.run({ images: tensor });
  const seg = out.seg.data.slice(0, CTD_SIZE * CTD_SIZE);
  const det = out.det.data.slice(0, CTD_SIZE * CTD_SIZE);  // channel 0 of [1, 2, H, W]
  return { seg, det };
}

// ---------------------------------------------------------------- Florence-2 OCR
async function loadFlorence() {
  const tf = await import(TRANSFORMERS_URL);
  tf.env.allowLocalModels = false;
  const files = {};
  const onProgress = p => {
    if (p.status === "progress" && p.file) {
      files[p.file] = [p.loaded || 0, p.total || 0];
      const loaded = Object.values(files).reduce((a, v) => a + v[0], 0);
      const total = Object.values(files).reduce((a, v) => a + v[1], 0);
      progress("OCR model (~270 MB, first time only)", loaded, total);
    }
  };
  const device = (await hasWebGPU()) ? "webgpu" : "wasm";
  const model = await tf.Florence2ForConditionalGeneration.from_pretrained(FLORENCE_ID, {
    dtype: "q8", device, progress_callback: onProgress });
  const processor = await tf.AutoProcessor.from_pretrained(FLORENCE_ID);
  const tokenizer = await tf.AutoTokenizer.from_pretrained(FLORENCE_ID);
  florence = { tf, model, processor, tokenizer, device };
  return florence;
}

async function runFlorence({ crops }) {
  const f = florence || await (loading.florence ||= loadFlorence());
  const task = "<OCR_WITH_REGION>";
  const out = [];
  for (const c of crops) {
    const image = new f.tf.RawImage(c.data, c.w, c.h, 3);
    const prompts = f.processor.construct_prompts(task);
    const textInputs = f.tokenizer(prompts);
    const visionInputs = await f.processor(image);
    const ids = await f.model.generate({ ...textInputs, ...visionInputs, max_new_tokens: 200 });
    const raw = f.tokenizer.batch_decode(ids, { skip_special_tokens: false })[0];
    const res = f.processor.post_process_generation(raw, task, image.size)[task];
    const lines = (res.labels || []).map(l => l.replace(/<\/?s>/g, "").trim()).filter(Boolean);
    out.push(lines.join("\n"));
  }
  return out;
}

// ---------------------------------------------------------------- shaw.py via Pyodide
async function loadPy() {
  const { loadPyodide } = await import(PYODIDE_URL + "pyodide.mjs");
  progress("Python (Pyodide, first time only)", 0, 0);
  const p = await loadPyodide({ indexURL: PYODIDE_URL });
  await p.loadPackage(["nltk"]);
  const text = async (url, what) => new TextDecoder().decode(await fetchWithProgress(url, what));
  p.FS.mkdirTree("/shaw");
  p.FS.writeFile("/shaw/shaw.py", await text(assets.shaw, "shaw.py"));
  p.FS.writeFile("/shaw/textproc.py", await text(assets.textproc, "textproc.py"));
  p.FS.writeFile("/shaw/dict.dict", await text(assets.dict, "Shavian dictionary (2.4 MB)"));
  const tdir = "/nltk_data/taggers/averaged_perceptron_tagger_eng";
  p.FS.mkdirTree(tdir);
  for (const [key, name] of [["weights", "weights"], ["tagdict", "tagdict"], ["classes", "classes"]]) {
    p.FS.writeFile(`${tdir}/averaged_perceptron_tagger_eng.${name}.json`, await text(assets.nltk[key], `POS tagger (${name})`));
  }
  await p.runPythonAsync(`
import sys
sys.path.insert(0, "/shaw")
import nltk
nltk.data.path.insert(0, "/nltk_data")
import textproc
ENGINE = textproc.ShawEngine("/shaw/shaw.py", "/shaw/dict.dict", pos=1, cwd="/shaw")
`);
  py = p;
  return py;
}

async function pyCall(fn, texts) {
  const p = py || await (loading.py ||= loadPy());
  p.globals.set("_texts", p.toPy(texts));
  const result = await p.runPythonAsync(`(${fn})(list(_texts))`);
  const js = result.toJs();
  result.destroy();
  return js;
}

// Free models so the browser doesn't kill the tab for using too much memory
async function release({ ctd = false, ocr = false, python = false }) {
  if (ctd && ctdSession) { try { await ctdSession.release(); } catch {} ctdSession = null; delete loading.ctd; }
  if (ocr && florence) { try { await florence.model.dispose(); } catch {} florence = null; delete loading.florence; }
  if (python && py) { py = null; delete loading.py; }
  return true;
}

const ops = {
  init: async ({ urls, lowMem }) => { assets = urls; lowMemory = !!lowMem; return true; },
  release,
  ctd: runCtd,
  florence: runFlorence,
  // shaw.py (with NLTK tagging) on a batch of English texts
  translit: ({ texts }) => pyCall("lambda t: textproc.transliterate(ENGINE, t)", texts),
  // OCR text as written in capitals -> joined lines, re-cased, cleaned
  tidyCapitals: ({ texts }) => pyCall("lambda t: [textproc.tidy_capitals(x, ENGINE.dictionary) for x in t]", texts),
  clean: ({ texts }) => pyCall("lambda t: [textproc.clean(x) for x in t]", texts),
  warm: async ({ what }) => {
    if (what === "ctd") await (loading.ctd ||= loadCtd());
    if (what === "florence") await (loading.florence ||= loadFlorence());
    if (what === "shaw") await (loading.py ||= loadPy());
    return true;
  },
};

self.onmessage = async e => {
  const { id, op, args } = e.data;
  try {
    postMessage({ id, result: await ops[op](args || {}) });
  } catch (err) {
    postMessage({ id, error: String(err && err.stack || err) });
  }
};
