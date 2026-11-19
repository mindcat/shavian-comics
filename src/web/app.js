// Shavian Comics front end. Talks only to the JSON API, so it can be hosted on its
// own (e.g. GitHub Pages) and pointed at a server with ?api=https://host:port
"use strict";

const API = (new URLSearchParams(location.search).get("api") || window.SHAVIAN_API || "").replace(/\/$/, "");
const $ = (s, el = document) => el.querySelector(s);
const $$ = (s, el = document) => [...el.querySelectorAll(s)];

const state = { job: null, regions: new Map(), view: "regions", selected: null, drawing: false,
                renderToken: 0, urls: {}, dirty: true };

// ------------------------------------------------------------------ API helpers
async function api(path, opts = {}) {
  const res = await fetch(API + path, opts);
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
new FontFace("Backlog Shavian", `url(${API}/fonts/backlogshavian.otf)`).load()
  .then(f => document.fonts.add(f)).catch(() => {});

// ------------------------------------------------------------------ status
async function pollStatus() {
  const el = $("#status");
  try {
    const info = await json("/api/info");
    state.info = info;
    if (info.ready) {
      el.textContent = `ready · ${info.device} · OCR ${info.ocr.split("/").pop()} · ${info.dictionary} dictionary`;
      el.className = "status ready";
      return;
    }
    el.textContent = `loading models: ${info.loading}…`;
  } catch {
    el.textContent = "can't reach the server";
  }
  setTimeout(pollStatus, 2000);
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

// ------------------------------------------------------------------ upload
async function upload(file) {
  if (!file || !file.type.startsWith("image/")) return;
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
  try {
    state.job = await (await api("/api/jobs", { method: "POST", body: form })).json();
  } catch (e) {
    busy();
    alert(`Upload failed: ${e.message}`);
    return reset();
  }
  state.urls.original = `${API}/api/jobs/${state.job.id}/original.png`;
  state.job.regions.forEach(r => state.regions.set(r.id, r));
  step("detect", "done");
  drawAll();

  // Read the enabled regions one by one so the cards fill in as we go (or all at
  // once, for backends that are faster that way, like the Groq API)
  step("read", "doing");
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
  await rerender();
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
function setup() {
  $("#pick").addEventListener("click", () => $("#file").click());
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
  pollStatus();
}
setup();
