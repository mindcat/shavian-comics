// Shavian Comics front end. Talks only to the JSON API, so it can be hosted on its
// own (e.g. GitHub Pages) and pointed at a server with ?api=https://host:port
"use strict";

const API = (new URLSearchParams(location.search).get("api") || window.SHAVIAN_API || "").replace(/\/$/, "");
const $ = (s, el = document) => el.querySelector(s);
const $$ = (s, el = document) => [...el.querySelectorAll(s)];

const state = { job: null, regions: new Map(), view: "regions", selected: null, drawing: false,
                renderToken: 0, urls: {}, dirty: true,
                mode: localStorage.getItem("shavian-mode") || (window.SHAVIAN_DEFAULT_MODE || "server") };

// ------------------------------------------------------------------ "this device" mode
// Where the in-browser pipeline downloads its files from: config.js can point these
// elsewhere (e.g. a Hugging Face repo when the page is on GitHub Pages)
const ASSET_BASE = (window.SHAVIAN_ASSETS || `${API || location.origin}/assets`).replace(/[/]$/, "");
const ASSETS = {
  ctd: `${ASSET_BASE}/comictextdetector.onnx`, shaw: `${ASSET_BASE}/shaw.py`, textproc: `${ASSET_BASE}/textproc.py`,
  dict: `${ASSET_BASE}/amer.dict`, font: window.SHAVIAN_FONT_URL || `${API}/fonts/backlogshavian.otf`,
  nltk: { weights: `${ASSET_BASE}/nltk-tagger-weights.json`, tagdict: `${ASSET_BASE}/nltk-tagger-tagdict.json`,
          classes: `${ASSET_BASE}/nltk-tagger-classes.json` },
};
let localBackend = null;

// Crash note: mobile browsers silently reload a tab that runs out of memory. Record the
// current step while working on this device, so after such a reload we can say where.
const CRUMB = "shavian-local-step";
function crumb(step) {
  try {
    if (step) localStorage.setItem(CRUMB, JSON.stringify({ step, at: Date.now() }));
    else localStorage.removeItem(CRUMB);
  } catch {}
}
function crashNote() {
  let c = null;
  try { c = JSON.parse(localStorage.getItem(CRUMB) || "null"); } catch {}
  crumb(null);
  if (!c || Date.now() - c.at > 15 * 60 * 1000) return;
  const el = $("#crash-note");
  el.hidden = false;
  let next = "Try a smaller image, close other tabs, or switch “Read text with” to Groq (it needs far less memory).";
  // Died loading or running the OCR model: switch to the smaller one for the next try
  const current = localStorage.getItem("shavian-local-ocr") || "florence";
  if (/OCR model|reading the text/.test(c.step) && current === "florence") {
    localStorage.setItem("shavian-local-ocr", "florence-small");
    next = "The next try will use the smaller reading model (switched under “Read text with”). " +
      "If that fails too, Groq needs far less memory.";
  }
  el.innerHTML = `<b>The last comic didn't finish:</b> the page reloaded while <i>${c.step}</i>. ` +
    `That usually means the browser closed the page for using too much memory. ${next}`;
}
async function local() {
  if (!localBackend) {
    const { LocalBackend } = await import(new URL("web/local/backend.js", document.baseURI).href);
    localBackend = new LocalBackend(ASSETS);
    localBackend.onProgress = ({ what, loaded, total }) => {
      crumb(`loading ${what}`);
      busy(total ? `downloading ${what}: ${Math.round(100 * loaded / total)}%` : `loading ${what}…`);
    };
  }
  return localBackend;
}

function setMode(mode) {
  if (mode === state.mode && state.modeShown) return;
  state.mode = mode;
  state.modeShown = true;
  localStorage.setItem("shavian-mode", mode);
  $$("#mode button").forEach(b => b.classList.toggle("active", b.dataset.mode === mode));
  $("#local-settings").hidden = mode !== "local";
  if (state.job) reset();
  pollStatus();
}

// ------------------------------------------------------------------ API helpers
async function api(path, opts = {}) {
  const res = state.mode === "local" ? await (await local()).fetch(path, opts) : await fetch(API + path, opts);
  if (!res.ok) {
    let msg = res.statusText;
    try { msg = (await res.json()).detail || msg; } catch {}
    throw new Error(msg);
  }
  return res;
}
const json = (path, method = "GET", body) =>
  api(path, body === undefined ? { method } : {
    method, headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) }).then(r => r.json());

// Shavian font comes from the server, so it works when the page is hosted elsewhere
new FontFace("Backlog Shavian", `url(${window.SHAVIAN_FONT_URL || `${API}/fonts/backlogshavian.otf`})`).load()
  .then(f => document.fonts.add(f)).catch(() => {});

// ------------------------------------------------------------------ status
async function pollStatus() {
  const el = $("#status");
  try {
    const info = await json("/api/info");
    state.info = info;
    if (info.ready) {
      el.textContent = info.local
        ? `on this device · OCR ${info.ocr} · models download on first use and are cached`
        : `ready · ${info.device} · OCR ${info.ocr.split("/").pop()} · ${info.dictionary} dictionary`;
      showAllowance(info.uploads_left, info.uploads_per_day);
      el.className = "status ready";
      return;
    }
    el.textContent = `loading models: ${info.loading}…`;
  } catch {
    el.textContent = "can't reach the server · try “This device”";
  }
  if (state.mode === "server") setTimeout(pollStatus, 2000);
}

function step(name, how) {
  const el = $(`#steps [data-step="${name}"]`);
  if (el) el.className = how || "";
}
function busy(text) {
  const el = $("#busy");
  el.hidden = !text;
  el.textContent = text || "";
}

// ------------------------------------------------------------------ daily allowance
function showAllowance(left, perDay) {
  const el = $("#allowance");
  if (left == null || !perDay) { el.hidden = true; return; }
  state.left = left;
  el.hidden = false;
  el.textContent = left > 0 ? `${left} of ${perDay} pages left today` : `You've used today's ${perDay} pages; come back tomorrow`;
  el.classList.toggle("none-left", left === 0);
}

// ------------------------------------------------------------------ upload
async function upload(file) {
  if (!file || !file.type.startsWith("image/")) return;
  if (state.left === 0) {
    alert("You've used today's pages; please come back tomorrow.");
    return;
  }
  $("#drop").hidden = true;
  $("#work").hidden = false;
  ["detect", "read", "render"].forEach(s => step(s));
  step("detect", "doing");
  state.regions.clear();
  state.selected = null;
  Object.values(state.urls).forEach(URL.revokeObjectURL);
  state.urls = { original: URL.createObjectURL(file) };
  showImage();
  $("#regions").innerHTML = "";
  $("#download").disabled = true;
  busy("finding text…");

  const form = new FormData();
  form.append("file", file);
  if (state.mode === "local") crumb("finding the text");
  try {
    state.job = await (await api("/api/jobs", { method: "POST", body: form })).json();
  } catch (e) {
    busy();
    crumb(null);
    alert(`Upload failed: ${e.message}`);
    return reset();
  }
  if (state.mode === "server") state.urls.original = `${API}/api/jobs/${state.job.id}/original.png`;
  if (state.info) showAllowance(state.job.uploads_left, state.info.uploads_per_day);
  state.job.regions.forEach(r => state.regions.set(r.id, r));
  step("detect", "done");
  drawAll();

  // Read the enabled regions one by one so the cards fill in as we go (or all at
  // once, for backends that are faster that way, like the Groq API)
  step("read", "doing");
  if (state.mode === "local") crumb("reading the text");
  const todo = [...state.regions.values()].filter(r => r.enabled);
  if (state.info && state.info.ocr_batch) {
    busy(`reading ${todo.length} regions…`);
    (await json(`/api/jobs/${state.job.id}/ocr`, "POST", { ids: todo.map(r => r.id) })).forEach(update);
  } else {
    for (let i = 0; i < todo.length; i++) {
      busy(`reading ${i + 1} / ${todo.length}…`);
      await readRegion(todo[i].id);
    }
  }
  step("read", "done");
  if (state.mode === "local") {
    crumb("typesetting");
    (await local()).releaseModels();
  }
  await rerender();
  crumb(null);
}

async function readRegion(id) {
  const [r] = await json(`/api/jobs/${state.job.id}/ocr`, "POST", { ids: [id] });
  update(r);
}

function reset() {
  state.job = null;
  $("#work").hidden = true;
  $("#drop").hidden = false;
}

// ------------------------------------------------------------------ rendering
let renderTimer;
function scheduleRender(delay = 500) {
  state.dirty = true;
  clearTimeout(renderTimer);
  renderTimer = setTimeout(rerender, delay);
}

async function rerender() {
  if (!state.job) return;
  const token = ++state.renderToken;
  step("render", "doing");
  busy("typesetting…");
  try {
    const [fin, erased] = await Promise.all(["final", "erased"].map(stage =>
      api(`/api/jobs/${state.job.id}/render.png?stage=${stage}&t=${Date.now()}`)));
    if (token !== state.renderToken) return; // a newer render is on its way
    const sizes = fin.headers.get("X-Font-Sizes") || "";
    for (const pair of sizes.split(",").filter(Boolean)) {
      const [id, px] = pair.split(":").map(Number);
      const r = state.regions.get(id);
      if (r) { r.font_size = px; refreshCard(r); }
    }
    for (const [stage, res] of [["final", fin], ["erased", erased]]) {
      if (state.urls[stage]) URL.revokeObjectURL(state.urls[stage]);
      state.urls[stage] = URL.createObjectURL(await res.blob());
    }
    state.dirty = false;
    step("render", "done");
    $("#download").disabled = false;
    if (state.view === "regions" && !state.seenFinal) { state.seenFinal = true; setView("final"); }
    else showImage();
  } catch (e) {
    if (token === state.renderToken) alert(`Rendering failed: ${e.message}`);
  } finally {
    if (token === state.renderToken) busy();
  }
}

// ------------------------------------------------------------------ viewer
function setView(v) {
  state.view = v;
  $$(".tabs button").forEach(b => b.classList.toggle("active", b.dataset.view === v));
  showImage();
}

function showImage() {
  const src = { original: state.urls.original, regions: state.urls.original,
                erased: state.urls.erased, final: state.urls.final }[state.view] || state.urls.original;
  if ($("#img").src !== src) $("#img").src = src;
  $("#overlay").style.display = state.view === "regions" || state.drawing ? "" : "none";
}

const SVG = "http://www.w3.org/2000/svg";
function drawOverlay() {
  const svg = $("#overlay");
  if (!state.job) return;
  svg.setAttribute("viewBox", `0 0 ${state.job.width} ${state.job.height}`);
  svg.innerHTML = "";
  for (const r of state.regions.values()) {
    const cls = `region ${r.enabled ? r.kind : "off"}${state.selected === r.id ? " sel" : ""}`;
    let shape;
    if (r.outline && r.outline.length > 2) {
      shape = document.createElementNS(SVG, "polygon");
      shape.setAttribute("points", r.outline.map(p => p.join(",")).join(" "));
    } else {
      shape = document.createElementNS(SVG, "rect");
      const [x1, y1, x2, y2] = r.box;
      Object.entries({ x: x1, y: y1, width: x2 - x1, height: y2 - y1 }).forEach(([k, v]) => shape.setAttribute(k, v));
    }
    shape.setAttribute("class", cls);
    shape.addEventListener("click", e => { e.stopPropagation(); select(r.id, true); });
    svg.appendChild(shape);
    const label = document.createElementNS(SVG, "text");
    label.setAttribute("x", r.box[0]);
    label.setAttribute("y", Math.max(14, r.box[1] - 3));
    label.textContent = r.id;
    svg.appendChild(label);
  }
}

function select(id, scroll) {
  state.selected = id;
  if (state.view !== "regions") setView("regions");
  drawOverlay();
  $$(".card").forEach(c => c.classList.toggle("sel", Number(c.dataset.id) === id));
  const card = $(`.card[data-id="${id}"]`);
  if (scroll && card) card.scrollIntoView({ behavior: "smooth", block: "nearest" });
}

// ------------------------------------------------------------------ region cards
function drawAll() {
  drawOverlay();
  const list = $("#regions");
  list.innerHTML = "";
  for (const r of state.regions.values()) list.appendChild(makeCard(r));
  updateCount();
}

function updateCount() {
  const all = [...state.regions.values()];
  $("#count").textContent = `${all.filter(r => r.enabled).length} on · ${all.length} found`;
}

function makeCard(r) {
  const card = $("#card").content.firstElementChild.cloneNode(true);
  card.dataset.id = r.id;
  card.addEventListener("click", e => {
    if (e.target === card || (e.target.closest(".card-head") && !e.target.closest("input, button"))) {
      card.classList.toggle("open");
      select(r.id);
    }
  });

  $(".enabled", card).addEventListener("change", async e => {
    const enabled = e.target.checked;
    update(await json(`/api/jobs/${state.job.id}/regions/${r.id}`, "PATCH", { enabled }));
    if (enabled && !state.regions.get(r.id).read) {
      busy("reading…");
      await readRegion(r.id);
      busy();
    }
    scheduleRender(0);
  });

  const english = $(".english", card), shavian = $(".shavian", card);
  english.addEventListener("input", () => english.classList.add("pending"));
  english.addEventListener("change", async () => {
    update(await json(`/api/jobs/${state.job.id}/regions/${r.id}`, "PATCH", { english: english.value }));
    english.classList.remove("pending");
    scheduleRender();
  });
  shavian.addEventListener("input", () => shavian.classList.add("pending"));
  shavian.addEventListener("change", async () => {
    update(await json(`/api/jobs/${state.job.id}/regions/${r.id}`, "PATCH", { shavian: shavian.value }));
    shavian.classList.remove("pending");
    scheduleRender();
  });

  $(".eraser", card).addEventListener("change", async e => {
    update(await json(`/api/jobs/${state.job.id}/regions/${r.id}`, "PATCH", { eraser: e.target.value }));
    scheduleRender(0);
  });
  const size = $(".size", card), px = $(".px", card);
  size.addEventListener("change", async () => {
    px.hidden = size.value !== "custom";
    if (size.value === "custom") {
      px.value = state.regions.get(r.id).font_size || 16;
      px.focus();
    }
    const value = size.value === "custom" ? String(px.value) : size.value;
    update(await json(`/api/jobs/${state.job.id}/regions/${r.id}`, "PATCH", { size: value }));
    scheduleRender(0);
  });
  px.addEventListener("change", async () => {
    update(await json(`/api/jobs/${state.job.id}/regions/${r.id}`, "PATCH", { size: String(Math.round(px.value)) }));
    scheduleRender(0);
  });

  $(".del", card).addEventListener("click", async () => {
    await json(`/api/jobs/${state.job.id}/regions/${r.id}`, "DELETE");
    state.regions.delete(r.id);
    card.remove();
    drawOverlay();
    updateCount();
    scheduleRender(0);
  });

  fillCard(card, r);
  return card;
}

function fillCard(card, r) {
  card.classList.toggle("off", !r.enabled);
  card.classList.toggle("empty", r.read && !r.english && !r.shavian);
  $(".enabled", card).checked = r.enabled;
  $(".num", card).textContent = `#${r.id}`;
  const kind = $(".kind", card);
  kind.textContent = r.kind === "bubble" ? "bubble" : "on art";
  kind.className = `kind ${r.kind}`;
  $(".note", card).textContent = r.note;
  const english = $(".english", card);
  if (!english.classList.contains("pending")) english.value = r.english ?? "";
  english.placeholder = r.read ? "(no text)" : "(not read yet)";
  const shavian = $(".shavian", card);
  if (!shavian.classList.contains("pending")) shavian.value = r.shavian || "";
  $(".eraser", card).value = r.eraser;
  const custom = !["match", "fit"].includes(r.size);
  $(".size", card).value = custom ? "custom" : r.size;
  $(".px", card).hidden = !custom;
  if (custom) $(".px", card).value = r.size;
  $(".fs", card).textContent = r.font_size ? `drawn at ${r.font_size}px` : "";
}

function refreshCard(r) {
  const card = $(`.card[data-id="${r.id}"]`);
  if (card) fillCard(card, r);
}

function update(r) {
  const known = state.regions.has(r.id);
  state.regions.set(r.id, { ...state.regions.get(r.id), ...r });
  if (known) refreshCard(state.regions.get(r.id));
  else $("#regions").appendChild(makeCard(r));
  drawOverlay();
  updateCount();
}

// ------------------------------------------------------------------ drawing a new box
function toggleDrawing(on = !state.drawing) {
  state.drawing = on;
  $("#viewer").classList.toggle("drawing", on);
  $("#addbox").classList.toggle("active", on);
  if (on && state.view !== "regions") setView("regions");
  showImage();
}

function svgPoint(e) {
  const svg = $("#overlay");
  const rect = svg.getBoundingClientRect();
  return [(e.clientX - rect.left) * state.job.width / rect.width, (e.clientY - rect.top) * state.job.height / rect.height];
}

function setupDrawing() {
  const svg = $("#overlay");
  let start = null, draft = null;
  svg.addEventListener("pointerdown", e => {
    if (!state.drawing) return;
    e.preventDefault();
    svg.setPointerCapture(e.pointerId);
    start = svgPoint(e);
    draft = document.createElementNS(SVG, "rect");
    draft.setAttribute("class", "draft");
    svg.appendChild(draft);
  });
  svg.addEventListener("pointermove", e => {
    if (!start) return;
    const [x, y] = svgPoint(e);
    draft.setAttribute("x", Math.min(x, start[0])); draft.setAttribute("y", Math.min(y, start[1]));
    draft.setAttribute("width", Math.abs(x - start[0])); draft.setAttribute("height", Math.abs(y - start[1]));
  });
  svg.addEventListener("pointerup", async e => {
    if (!start) return;
    const end = svgPoint(e), s = start;
    start = null;
    draft.remove();
    toggleDrawing(false);
    if (Math.abs(end[0] - s[0]) < 6 || Math.abs(end[1] - s[1]) < 6) return;
    try {
      busy("reading new box…");
      const r = await json(`/api/jobs/${state.job.id}/regions`, "POST", { box: [s[0], s[1], end[0], end[1]] });
      update(r);
      await readRegion(r.id);
      select(r.id, true);
      scheduleRender(0);
    } catch (err) {
      busy();
      alert(err.message);
    }
  });
}

// ------------------------------------------------------------------ download
async function download() {
  if (state.dirty) await rerender();
  const a = document.createElement("a");
  a.href = state.urls.final;
  a.download = `${state.job.name}_shavian.png`;
  a.click();
}

// ------------------------------------------------------------------ wiring
function setupMode() {
  if (window.SHAVIAN_NO_SERVER) {  // e.g. GitHub Pages without a server: always in the browser
    state.mode = "local";
    $("#mode").hidden = true;
  }
  $$("#mode button").forEach(b => b.addEventListener("click", () => setMode(b.dataset.mode)));
  const ocrSel = $("#local-ocr"), key = $("#groq-key");
  ocrSel.value = localStorage.getItem("shavian-local-ocr") || "florence";
  key.value = localStorage.getItem("shavian-groq-key") || "";
  $("#groq-row").hidden = ocrSel.value !== "groq";
  ocrSel.addEventListener("change", () => {
    localStorage.setItem("shavian-local-ocr", ocrSel.value);
    if (localBackend) localBackend.terminate("ocr");  // reload with the chosen model
    $("#groq-row").hidden = ocrSel.value !== "groq";
    pollStatus();
  });
  key.addEventListener("change", () => localStorage.setItem("shavian-groq-key", key.value.trim()));
  setMode(state.mode);
}

function setup() {
  // Free on-device models before the photo picker opens (phones may unload the page meanwhile)
  $("#pick").addEventListener("click", () => {
    if (localBackend) localBackend.releaseModels();
    $("#file").click();
  });
  $("#file").addEventListener("change", e => upload(e.target.files[0]));
  $("#another").addEventListener("click", () => { reset(); $("#file").value = ""; });
  ["dragenter", "dragover"].forEach(t => document.addEventListener(t, e => {
    e.preventDefault(); document.body.classList.add("dragging"); }));
  ["dragleave", "drop"].forEach(t => document.addEventListener(t, e => {
    e.preventDefault(); document.body.classList.remove("dragging"); }));
  document.addEventListener("drop", e => upload(e.dataTransfer.files[0]));
  document.addEventListener("paste", e => {
    const item = [...e.clipboardData.items].find(i => i.type.startsWith("image/"));
    if (item) upload(item.getAsFile());
  });
  $$(".tabs button").forEach(b => b.addEventListener("click", () => setView(b.dataset.view)));
  $("#addbox").addEventListener("click", () => toggleDrawing());
  $("#download").addEventListener("click", download);
  $("#overlay").addEventListener("click", () => { if (!state.drawing) select(null); });
  setupDrawing();
  crashNote();
  setupMode();
}
setup();
