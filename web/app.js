/**
 * FaceSwapper main thread: UI, drag&drop, progress, preview, download.
 * No inference here — everything heavy runs in worker.js.
 *
 * Privacy: this file never uploads anything. Decode via createImageBitmap
 * (EXIF-orientation aware), ship pixels to the worker, get pixels back.
 */

const MAX_LONG_SIDE = 4096; // output cap; detection letterboxes internally
// Absolute-from-page URLs: workers resolve relative fetch() against the
// *worker script* location (dist/assets/), NOT the page — so never send the
// worker a relative path. document.baseURI keeps this correct at any subpath.
const APP_BASE = new URL('.', document.baseURI).href;
const MODEL_BASE = `${APP_BASE}models`;
const SOURCE_CANDIDATES = ['example.jpg', 'example.jpeg', 'example.png', 'example.webp'].map(
  (n) => `${APP_BASE}source/${n}`,
);

const $ = (id) => document.getElementById(id);
const els = {
  backend: $('backend-badge'),
  source: $('source-badge'),
  sourceWarning: $('source-warning'),
  dropzone: $('dropzone'),
  fileInput: $('file-input'),
  threshold: $('threshold'),
  thresholdVal: $('threshold-val'),
  debugBoxes: $('debug-boxes'),
  processBtn: $('process-btn'),
  resetBtn: $('reset-btn'),
  resultPanel: $('result-panel'),
  progress: $('progress'),
  compare: $('compare'),
  canvasBefore: $('canvas-before'),
  canvasAfter: $('canvas-after'),
  afterWrap: $('after-wrap'),
  slider: $('slider'),
  resultMeta: $('result-meta'),
  dlPng: $('download-png'),
  dlJpg: $('download-jpg'),
  dlbar: $('download-bar'),
  dlfill: $('download-fill'),
  dllabel: $('download-label'),
  sourceCanvas: $('canvas-source'),
  sourceName: $('source-name'),
  sourceInput: $('source-input'),
  sourceReset: $('source-reset'),
};

const worker = new Worker(new URL('./worker.js', import.meta.url), { type: 'module' });

/**
 * EXIF-aware decode. NOTE: the enum value is "from-image" (hyphenated);
 * "fromImage" is invalid and throws (seen in the wild — hence this helper).
 * Falls back to plain decode where the option is unsupported (EXIF rotation
 * is then not applied — documented limitation, faces still detect).
 */
async function decodeImageBitmap(blob) {
  try {
    return await createImageBitmap(blob, { imageOrientation: 'from-image' });
  } catch {
    return await createImageBitmap(blob);
  }
}
let seq = 0;
const pending = new Map();
worker.onmessage = (ev) => {
  const msg = ev.data;
  if (msg.type === 'progress') {
    pushStage(msg.stage);
    return;
  }
  if (msg.type === 'download') {
    showDownload(msg.label, msg.received, msg.total, msg.tag);
    return;
  }
  const slot = pending.get(msg.id);
  if (slot) {
    pending.delete(msg.id);
    slot(msg);
  }
};
worker.onerror = (e) => {
  pushStage(`Worker error: ${e.message || e}`, 'error');
};

function callWorker(msg, transfer) {
  const id = ++seq;
  return new Promise((resolve) => {
    pending.set(id, resolve);
    worker.postMessage({ ...msg, id }, transfer || []);
  });
}

// --- theme ---------------------------------------------------------------
const root = document.documentElement;
try {
  const saved = localStorage.getItem('faceswapper-theme');
  if (saved) root.dataset.theme = saved;
} catch { /* private mode */ }
$('theme-toggle').addEventListener('click', (e) => {
  const next = root.dataset.theme === 'dark' ? 'light' : 'dark';
  root.dataset.theme = next;
  e.currentTarget.setAttribute('aria-pressed', String(next === 'dark'));
  try { localStorage.setItem('faceswapper-theme', next); } catch { /* ignore */ }
});

// --- state ---------------------------------------------------------------
let original = null; // { rgba: Uint8Array, w, h, name }
let result = null;   // { rgba: Uint8Array, w, h, faces, timings }
let sourceReady = false;
let sourceName = 'Charlie Kirk';
let sourceBusy = false; // worker handles one prepare-source at a time

// --- worker init + fixed source identity ----------------------------------
async function boot() {
  if (location.protocol === 'file:') {
    pushStage('This app cannot run from file:// — browsers block workers and modules there. Serve it: `npm run dev` (or serve dist/ over http).', 'error');
    els.backend.textContent = 'unavailable';
    els.source.textContent = 'Source: —';
    return;
  }
  // Ask the browser to keep our origin storage (incl. the cached model
  // weights) instead of evicting it under pressure. Best-effort: browsers
  // may grant, prompt, or ignore it.
  try {
    if (navigator.storage?.persist) {
      const persisted = await navigator.storage.persist();
      const est = navigator.storage.estimate ? await navigator.storage.estimate() : null;
      const quota = est?.quota ? ` (quota ~${(est.quota / 1073741824).toFixed(1)} GB)` : '';
      console.info(`[faceswapper] persistent storage: ${persisted}${quota}`);
    }
  } catch {
    /* private mode / unsupported — the app still works, cache just won't stick */
  }
  const ready = await callWorker({ type: 'init', modelBase: MODEL_BASE });
  if (ready.type === 'ready') {
    els.backend.textContent = ready.backend === 'webgpu' ? 'WebGPU' : 'WASM';
    els.backend.title =
      ready.backend === 'webgpu'
        ? 'ONNX Runtime Web · WebGPU execution provider'
        : 'ONNX Runtime Web · WASM fallback (WebGPU unavailable)';
  } else {
    els.backend.textContent = 'unavailable';
    pushStage(`${ready.code || 'Error'}: ${ready.message || 'worker init failed'}`, 'error');
  }
  await loadBundledSource();
}

/** Load the fixed built-in portrait. Used at boot and by the reset button. */
async function loadBundledSource() {
  // The portrait file ships separately for rights reasons (see SOURCE-LICENSING.md).
  // NOTE: a missing portrait and missing MODELS are different failures and
  // are reported differently (they used to share one message).
  let portrait = null;
  for (const url of SOURCE_CANDIDATES) {
    try {
      const res = await fetch(url);
      if (res.ok) {
        portrait = await res.blob();
        break;
      }
    } catch {
      /* try next candidate/extension */
    }
  }
  if (!portrait) {
    els.source.textContent = 'Source: missing';
    els.sourceWarning.hidden = false;
    pushStage(
      'Built-in portrait not found. Tried: ' +
        SOURCE_CANDIDATES.map((u) => new URL(u).pathname).join(', ') +
        '. Check the exact filename directly under public/source/ (reload on dev, rebuild for dist/). ' +
        'You can still upload a custom source face in step 1.',
      'error',
    );
    console.error('[faceswapper] source portrait 404 for all candidates:', SOURCE_CANDIDATES);
    return false;
  }
  els.sourceWarning.hidden = true;
  return submitSourceBlob(portrait, 'Charlie Kirk');
}

/**
 * Prepare any source image (built-in portrait or user upload) in the worker.
 * Returns true on success; the previous source is kept on failure.
 */
async function submitSourceBlob(blob, name) {
  if (sourceBusy) return false;
  sourceBusy = true;
  try {
    const bmp = await decodeImageBitmap(blob);
    const { rgba, w, h } = bitmapToCappedRgba(bmp, 1024);
    bmp.close();
    const r = await callWorker({ type: 'prepare-source', buf: rgba.buffer, w, h, modelBase: MODEL_BASE }, [rgba.buffer]);
    if (r.type === 'source-ready') {
      sourceReady = true;
      sourceName = name;
      els.source.textContent = `Source: ${name}`;
      els.source.title = `prepared in ${r.ms}ms via ${r.backend || '?'}`;
      els.sourceName.textContent = name;
      drawSourcePreview(new Uint8Array(r.previewBuf), r.previewW, r.previewH);
      return true;
    }
    els.source.textContent = 'Source: error';
    pushStage(`${r.code || 'Error'}: ${r.message || ''}`, 'error');
    if ((r.code || '') === 'Model loading failed') {
      pushStage('Setup needed: copy the model files into public/models/ (see public/models/MODELS.md, or run: node tools/fetch-models.mjs) and reload/rebuild.', 'error');
    }
    return false;
  } catch (e) {
    els.source.textContent = 'Source: error';
    pushStage(`Source image could not be decoded: ${e.message || e}`, 'error');
    return false;
  } finally {
    sourceBusy = false;
  }
}

function drawSourcePreview(rgba, w, h) {
  fitCanvas(els.sourceCanvas, w, h);
  els.sourceCanvas.getContext('2d').putImageData(new ImageData(new Uint8ClampedArray(rgba), w, h), 0, 0);
}

async function handleSourceFile(file) {
  if (!file || !file.type.startsWith('image/')) {
    pushStage('Unsupported image: please choose a JPEG, PNG or WebP file.', 'error');
    return;
  }
  const shortName = (file.name.replace(/\.[^.]+$/, '') || 'Custom face').slice(0, 40);
  if (await submitSourceBlob(file, shortName)) {
    pushStage(`Source face set to “${shortName}”. Now upload a photo in step 2.`);
  }
}

// --- image intake ----------------------------------------------------------
function bitmapToCappedRgba(bmp, cap) {
  let { width: w, height: h } = bmp;
  const long = Math.max(w, h);
  if (long > cap) {
    const s = cap / long;
    w = Math.max(1, Math.round(w * s));
    h = Math.max(1, Math.round(h * s));
  }
  const canvas = document.createElement('canvas');
  canvas.width = w;
  canvas.height = h;
  const ctx = canvas.getContext('2d', { willReadFrequently: true });
  ctx.drawImage(bmp, 0, 0, w, h);
  const data = ctx.getImageData(0, 0, w, h).data;
  return { rgba: new Uint8Array(data.buffer.slice(0)), w, h };
}

async function handleFile(file) {
  clearStages();
  hideDownload();
  result = null;
  els.resultPanel.hidden = true;
  if (!file || !file.type.startsWith('image/')) {
    pushStage('Unsupported image: please choose a JPEG, PNG or WebP file.', 'error');
    return;
  }
  let bmp;
  try {
    bmp = await decodeImageBitmap(file);
  } catch (e) {
    pushStage(`Unsupported image: this file could not be decoded (${e.message || e}).`, 'error');
    return;
  }
  try {
    const { rgba, w, h } = bitmapToCappedRgba(bmp, MAX_LONG_SIDE);
    original = { rgba, w, h, name: file.name.replace(/\.[^.]+$/, '') || 'image' };
    drawPreview();
    els.processBtn.disabled = false;
    els.resetBtn.disabled = false;
    pushStage(`Loaded ${w}×${h} — press “Detect & Swap”.`);
  } finally {
    bmp.close();
  }
}

els.dropzone.addEventListener('click', (e) => {
  if (e.target.tagName !== 'INPUT' && e.target.tagName !== 'LABEL') els.fileInput.click();
});
els.dropzone.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' || e.key === ' ') {
    e.preventDefault();
    els.fileInput.click();
  }
});
els.fileInput.addEventListener('change', (e) => handleFile(e.target.files[0]));
els.sourceInput.addEventListener('change', (e) => {
  handleSourceFile(e.target.files[0]);
  e.target.value = ''; // allow re-picking the same file
});
els.sourceReset.addEventListener('click', () => loadBundledSource());
for (const evt of ['dragenter', 'dragover']) {
  els.dropzone.addEventListener(evt, (e) => { e.preventDefault(); els.dropzone.classList.add('over'); });
}
for (const evt of ['dragleave', 'drop']) {
  els.dropzone.addEventListener(evt, (e) => { e.preventDefault(); els.dropzone.classList.remove('over'); });
}
els.dropzone.addEventListener('drop', (e) => handleFile(e.dataTransfer.files[0]));
// Pasting an image also works (keyboard-first users).
window.addEventListener('paste', (e) => {
  const f = [...(e.clipboardData?.files || [])].find((x) => x.type.startsWith('image/'));
  if (f) handleFile(f);
});

els.threshold.addEventListener('input', () => {
  els.thresholdVal.textContent = Number(els.threshold.value).toFixed(2);
});

// --- processing --------------------------------------------------------------
els.processBtn.addEventListener('click', async () => {
  if (!original) return;
  if (!sourceReady) {
    pushStage('Source identity is not installed — processing is disabled until a licensed portrait is provided.', 'error');
    return;
  }
  els.processBtn.disabled = true;
  clearStages();
  hideDownload();
  pushStage('Starting…');
  // Fresh copy: the worker composites onto its own buffer; keep `original`
  // intact for the before/after view. Transfer the copy.
  const buf = original.rgba.slice().buffer;
  const r = await callWorker(
    { type: 'process', buf, w: original.w, h: original.h, threshold: Number(els.threshold.value), maxFaces: 50, modelBase: MODEL_BASE },
    [buf],
  );
  els.processBtn.disabled = false;
  if (r.type === 'error') {
    hideDownload();
    pushStage(`${r.code}: ${r.message}`, 'error');
    return;
  }
  hideDownload();
  result = { rgba: new Uint8Array(r.buf), w: r.w, h: r.h, faces: r.faces, timings: r.timings };
  drawPreview();
  if (els.debugBoxes.checked) drawDebugBoxes();
  els.resultPanel.hidden = false;
  const t = r.timings || {};
  els.resultMeta.textContent =
    `${r.faces.length} face${r.faces.length === 1 ? '' : 's'} swapped · ` +
    `detect ${t.detectMs ?? '?'}ms · swap [${(t.swapMsPerFace || []).join(', ')}]ms · total ${t.totalMs ?? '?'}ms · backend ${r.backend} · source: ${sourceName}`;
  els.resultPanel.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
});

els.resetBtn.addEventListener('click', () => {
  original = null;
  result = null;
  els.fileInput.value = '';
  els.processBtn.disabled = true;
  els.resetBtn.disabled = true;
  els.resultPanel.hidden = true;
  clearStages();
  hideDownload();
});

// --- preview + before/after slider ---------------------------------------------
function fitCanvas(canvas, w, h) {
  canvas.width = w;
  canvas.height = h;
}

function drawPreview() {
  if (!original) return;
  fitCanvas(els.canvasBefore, original.w, original.h);
  els.canvasBefore.getContext('2d').putImageData(new ImageData(new Uint8ClampedArray(original.rgba), original.w, original.h), 0, 0);
  const after = result ? result.rgba : original.rgba;
  const w = result ? result.w : original.w;
  const h = result ? result.h : original.h;
  fitCanvas(els.canvasAfter, w, h);
  els.canvasAfter.getContext('2d').putImageData(new ImageData(new Uint8ClampedArray(after), w, h), 0, 0);
  els.resultPanel.hidden = !result;
  applySlider();
}

function applySlider() {
  els.afterWrap.style.width = `${els.slider.value}%`;
}
els.slider.addEventListener('input', applySlider);

function drawDebugBoxes() {
  if (!result) return;
  const ctx = els.canvasAfter.getContext('2d');
  ctx.save();
  ctx.strokeStyle = '#a78bfa';
  ctx.lineWidth = Math.max(2, result.w / 400);
  for (const f of result.faces) {
    const [x1, y1, x2, y2] = f.bbox;
    ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
  }
  ctx.restore();
}

// --- download progress bar (single live element, not a card per chunk) -------
function showDownload(label, received, total, tag) {
  els.dlbar.hidden = false;
  const pct = total ? Math.min(100, (received / total) * 100) : null;
  els.dlfill.style.width = pct === null ? '100%' : `${pct}%`;
  els.dlfill.classList.toggle('indeterminate', pct === null);
  els.dlbar.setAttribute('aria-valuenow', pct === null ? '0' : String(Math.round(pct)));
  const have = (received / 1048576).toFixed(0);
  const want = total ? ` / ${(total / 1048576).toFixed(0)}` : '';
  const where = tag === 'remote' ? ' · mirror' : '';
  els.dllabel.textContent = `${label || 'Downloading'} · ${have}${want} MB${where}${pct === null ? '' : ` · ${pct.toFixed(0)}%`}`;
}

function hideDownload() {
  els.dlbar.hidden = true;
  els.dlfill.style.width = '0';
  els.dlfill.classList.remove('indeterminate');
}

// --- progress ------------------------------------------------------------------
function clearStages() {
  els.progress.innerHTML = '';
}
function pushStage(text, kind) {
  for (const li of els.progress.children) {
    li.classList.remove('active');
    li.classList.add('done');
  }
  const li = document.createElement('li');
  li.textContent = text;
  if (kind === 'error') li.style.borderColor = 'red';
  else li.classList.add('active');
  els.progress.appendChild(li);
}

// --- export (visible mark + PNG tEXt metadata) -----------------------------------
function exportCanvas(mime, quality) {
  if (!result) return;
  const c = document.createElement('canvas');
  c.width = result.w;
  c.height = result.h;
  const ctx = c.getContext('2d');
  ctx.putImageData(new ImageData(new Uint8ClampedArray(result.rgba), result.w, result.h), 0, 0);
  burnDisclosureMark(ctx, result.w, result.h);
  c.toBlob(async (blob) => {
    if (!blob) {
      pushStage('Export failed: the browser could not encode the image.', 'error');
      return;
    }
    let out = blob;
    if (mime === 'image/png') out = new Blob([embedPngText(await blob.arrayBuffer())], { type: 'image/png' });
    const a = document.createElement('a');
    a.href = URL.createObjectURL(out);
    a.download = `${original?.name || 'image'}-faceswapped-ai.${mime === 'image/png' ? 'png' : 'jpg'}`;
    a.click();
    setTimeout(() => URL.revokeObjectURL(a.href), 60_000);
  }, mime, quality);
}

/** Visible disclosure burned into the downloaded file (not just the preview). */
function burnDisclosureMark(ctx, w, h) {
  const label = 'AI-MANIPULATED · FaceSwapper';
  const px = Math.max(14, Math.round(Math.min(w, h) / 45));
  ctx.save();
  ctx.font = `700 ${px}px system-ui, sans-serif`;
  const tw = ctx.measureText(label).width;
  const pad = px * 0.7;
  const bw = tw + pad * 2;
  const bh = px + pad * 1.4;
  const x = px * 0.8;
  const y = h - bh - px * 0.8;
  ctx.fillStyle = 'rgba(0,0,0,0.62)';
  roundRect(ctx, x, y, bw, bh, bh / 3);
  ctx.fill();
  ctx.fillStyle = '#fff';
  ctx.textBaseline = 'middle';
  ctx.fillText(label, x + pad, y + bh / 2 + 1);
  ctx.restore();
}

function roundRect(ctx, x, y, w, h, r) {
  ctx.beginPath();
  ctx.moveTo(x + r, y);
  ctx.arcTo(x + w, y, x + w, y + h, r);
  ctx.arcTo(x + w, y + h, x, y + h, r);
  ctx.arcTo(x, y + h, x, y, r);
  ctx.arcTo(x, y, x + w, y, r);
  ctx.closePath();
}

const CRC_TABLE = (() => {
  const t = new Uint32Array(256);
  for (let n = 0; n < 256; n++) {
    let c = n;
    for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
    t[n] = c;
  }
  return t;
})();

function crc32(bytes) {
  let c = 0xffffffff;
  for (let i = 0; i < bytes.length; i++) c = CRC_TABLE[(c ^ bytes[i]) & 0xff] ^ (c >>> 8);
  return (c ^ 0xffffffff) >>> 0;
}

/** Insert a tEXt chunk (AI disclosure) before IEND. Returns a new ArrayBuffer. */
export function embedPngText(pngBuffer) {
  const src = new Uint8Array(pngBuffer);
  const sig = [137, 80, 78, 71, 13, 10, 26, 10];
  for (let i = 0; i < 8; i++) if (src[i] !== sig[i]) throw new Error('Not a PNG file');
  const keyword = 'AI-Disclosure';
  const text = 'AI-manipulated image. Face identity synthetically replaced with FaceSwapper. Do not present as an authentic photograph.';
  const enc = new TextEncoder();
  const kb = enc.encode(keyword);
  const tb = enc.encode(text);
  const data = new Uint8Array(kb.length + 1 + tb.length);
  data.set(kb, 0);
  data[kb.length] = 0;
  data.set(tb, kb.length + 1);
  const type = enc.encode('tEXt');
  const chunk = new Uint8Array(4 + 4 + data.length + 4);
  const dv = new DataView(chunk.buffer);
  dv.setUint32(0, data.length);
  chunk.set(type, 4);
  chunk.set(data, 8);
  dv.setUint32(8 + data.length, crc32(new Uint8Array([...type, ...data])));
  // Walk chunks to find IEND.
  let pos = 8;
  while (pos + 8 <= src.length) {
    const len = new DataView(src.buffer, src.byteOffset + pos, 4).getUint32(0);
    const name = String.fromCharCode(...src.subarray(pos + 4, pos + 8));
    if (name === 'IEND') break;
    pos += 12 + len;
  }
  const out = new Uint8Array(src.length + chunk.length);
  out.set(src.subarray(0, pos), 0);
  out.set(chunk, pos);
  out.set(src.subarray(pos), pos + chunk.length);
  return out.buffer;
}

els.dlPng.addEventListener('click', () => exportCanvas('image/png'));
els.dlJpg.addEventListener('click', () => exportCanvas('image/jpeg', 0.92));

boot();
