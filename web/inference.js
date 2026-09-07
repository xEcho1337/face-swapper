/**
 * FaceSwapper inference layer (runs inside the Web Worker).
 *
 * Real ONNX pipeline — no mocks:
 *   SCRFD detection -> ArcFace embedding -> inswapper_128 swap -> paste-back
 * Geometry/masking/blending kernels live in Rust/WASM (`src/`); this module
 * only does what the browser cannot do from Rust conveniently: fetch models,
 * drive ONNX Runtime Web (WebGPU with WASM fallback), and orchestrate crops.
 *
 * ---------------------------------------------------------------------------
 * MODEL CONTRACTS (verified against upstream source, NOT assumed)
 * ---------------------------------------------------------------------------
 * [1] Detector: SCRFD 2.5G with 5 keypoints (`scrfd_2.5g_bnkps.onnx`).
 *     Source: deepinsight/insightface `detection/scrfd/tools/scrfd.py`
 *             + `tools/scrfd2onnx.py`.
 *     VERIFIED 2026-09-07 against the cysin/scrfd_onnx export with the
 *     `onnx` Python package:
 *     - Input  "input.1": FLOAT [1,3,640,640] (this export is FIXED-shape;
 *       dynamic InsightFace exports accept any H/W — both work because the
 *       app always feeds a 640x640 letterbox).
 *     - Outputs (9, order as exported): score_8 [1,12800,1], score_16
 *       [1,3200,1], score_32 [1,800,1], bbox_8 [1,12800,4], bbox_16
 *       [1,3200,4], bbox_32 [1,800,4], kps_8 [1,12800,10], kps_16
 *       [1,3200,10], kps_32 [1,800,10]. 12800 = 80*80 grid * 2 anchors,
 *       etc. — confirms strides [8,16,32] with 2 anchors/stride.
 *       Fixed-shape exports are UNBATCHED-2D? No — this one is BATCHED
 *       rank-3; the code accepts both rank-2 and rank-3.
 *     - Outputs carry post-sigmoid scores (reference thresholds raw values
 *       at 0.5 with no extra sigmoid — `scrfd.py::forward`).
 *     Letterbox/pad/normalize/NMS details as below.
 *     - Input  "input.1": float32 NCHW, dynamic H/W (we feed 640x640
 *       letterboxed, aspect-preserving, top-left, zero-padded — exactly like
 *       `SCRFD.detect`). Preprocessing: BGR->RGB, `(x - 127.5) / 128.0`
 *       (i.e. `cv2.dnn.blobFromImage(img, 1/128, size, (127.5,...),
 *       swapRB=True)`).
 *     - Outputs (9, `_bnkps` variant, fmc=3, strides [8,16,32],
 *       2 anchors/stride):
 *         score_8/16/32, bbox_8/16/32, kps_8/16/32.
 *       Each may be UNBATCHED `[N,1] / [N,4] / [N,10]` (fixed-shape exports)
 *       or BATCHED `[1,N,1] / [1,N,4] / [1,N,10]` (dynamic exports) — both
 *       are handled. bbox/kps predictions are distances scaled by stride.
 *     - Post: anchor-center grid per stride (centers at (jx*s, iy*s),
 *       duplicated for 2 anchors), `distance2bbox` / `distance2kps`,
 *       score threshold, greedy NMS @ IoU 0.4 (with the +1px convention),
 *       divide by `det_scale = new_h / img_h`.
 *     - License: InsightFace code is MIT, but the pretrained weights follow
 *       the InsightFace model license (non-commercial research unless you
 *       arrange otherwise). See public/models/MODELS.md. Weights are NOT
 *       bundled; install them separately.
 *
 * [2] Recognizer: ArcFace ResNet50 from the `buffalo_l` pack (`w600k_r50.onnx`).
 *     Source: `python-package/insightface/model_zoo/arcface_onnx.py`.
 *     - Input: float32 [1,3,112,112]; crop via `norm_crop` (112 template);
 *       BGR->RGB, `(x - 127.5) / 127.5` (buffalo models have no Sub/Mul
 *       prefix nodes -> mean=127.5, std=127.5).
 *     - Output: single float32 [1,512] embedding; L2-normalize for
 *       `normed_embedding`.
 *
 * [3] Swapper: `reswapper-1019500.onnx` (ReSwapper, somanchiu) — the default,
 *     openly-licensed swap model. An independent reimplementation of the
 *     inswapper architecture (AGPL-3.0, code + weights) with an explicitly
 *     inswapper-compatible I/O contract ("usable with the original INSwapper
 *     class"). Preprocessing, latent math and paste-back are therefore
 *     identical to InsightFace `model_zoo/inswapper.py::get`.
 *     - Inputs (matched BY SHAPE, not by name — exports rename them):
 *         image-like 4D float32 [1,3,128,128]: target crop from
 *           `norm_crop2(img, kps, 128)`, BGR->RGB, `x / 255.0`
 *           (mean=0, std=255).
 *         latent-like 2D float32 [1,512]: `normalize(normed_embedding @ emap)`
 *           where `emap` is the LAST initializer of the ONNX graph
 *           (float [512,512], row-major). ORT Web cannot read initializers,
 *           so `emap` is extracted once offline into
 *           `models/reswapper.emap.json` (see tools/extract_emap.py —
 *           generic for any file whose last initializer is float32 [512,512]).
 *     - Output: single float32 [1,3,128,128] in [0,1] (RGB); `*255` -> BGR.
 *       VERIFIED 2026-09-07 with the `onnx` package on reswapper-1019500.onnx
 *       (554 MB): inputs `target` FLOAT [1,3,128,128] + `source` FLOAT
 *       [1,512]; output `output` FLOAT [1,3,128,128] (symbolic batch/spatial
 *       dims, concrete [1,3,128,128] at runtime); 65 initializers, last one
 *       float32 [512,512] (= emap); opset 11. Inputs are matched by shape
 *       (robust to renamed exports); the output is validated at warmup.
 *       E2E quality on a 6-face photo (tools/verify-swap.mjs, ArcFace cosine
 *       to source identity): 0.58–0.87, every face closer to the source than
 *       to its original identity. (inswapper_128 measured 0.84–0.96 on the
 *       same protocol — stronger, but license-gated, kept as documented
 *       alternative in MODELS.md.)
 *     - License: AGPL-3.0 (GitHub somanchiu/ReSwapper and HF
 *       somanchiu/reswapper agree). Copyleft, NOT permissive: the weights
 *       stay OUT of this repository and are installed separately by the
 *       operator (see MODELS.md); license notices must be preserved.
 *       Provenance note (author's own disclosure): training used inswapper
 *       outputs among its targets — documented, no InsightFace rights are
 *       claimed or conveyed by this project.
 */

/**
 * ORT is loaded lazily from CDN (or a self-hosted copy) via dynamic import
 * so `dist/` stays small and no import-map is needed inside workers.
 * Library code only — never user pixels.
 */
const ORT_CDN_MODULE = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.22.0/dist/ort.webgpu.bundle.min.mjs';
let ort = null;

async function ensureOrt(ortModuleUrl) {
  if (ort) return ort;
  const url = ortModuleUrl || ORT_CDN_MODULE;
  try {
    ort = await import(/* @vite-ignore */ url);
  } catch (e) {
    fail(
      'Model loading failed',
      `Could not load the ONNX Runtime Web library from ${url}. Check connectivity (or self-host via \`npm run vendor-ort\`). No image data was sent anywhere — this is library code only.`,
      e,
    );
  }
  return ort;
}

export const CONTRACTS = {
  detector: {
    model: 'SCRFD-2.5G-BNKPS',
    input: { dtype: 'float32', layout: 'NCHW', size: [640, 640], mean: 127.5, std: 128.0, rgb: true },
    strides: [8, 16, 32],
    numAnchors: 2,
    nmsThreshold: 0.4,
  },
  recognizer: {
    model: 'ArcFace-w600k-r50 (buffalo_l)',
    input: { dtype: 'float32', shape: [1, 3, 112, 112], mean: 127.5, std: 127.5, rgb: true },
  },
  swapper: {
    model: 'ReSwapper-1019500 (inswapper-compatible, AGPL-3.0)',
    inputImage: { dtype: 'float32', shape: [1, 3, 128, 128], mean: 0.0, std: 255.0, rgb: true },
    inputLatent: { dtype: 'float32', shape: [1, 512] },
    output: { dtype: 'float32', shape: [1, 3, 128, 128], range: '[0,1] RGB' },
  },
};

const state = {
  config: null,
  wasm: null, // Rust/WASM kernel module
  backend: 'unknown', // 'webgpu' | 'wasm'
  sessions: { detector: null, recognizer: null, swapper: null },
  manifest: null, // public/models/manifest.json (URL map: local path + remote fallback)
  emap: null, // Float32Array(512*512)
  source: null, // { latent: Float32Array(512), width, height, preview: Uint8Array(RGBA 256) }
};

function fail(code, message, cause) {
  const err = new Error(message);
  err.code = code;
  if (cause) err.cause = cause;
  throw err;
}

/** One-line summary of an EP failure for UI display (full object -> console). */
function shortErr(e) {
  const m = (e && e.message ? String(e.message) : String(e)).replace(/\s+/g, ' ').trim();
  return m.length > 240 ? m.slice(0, 240) + '…' : m;
}

/** Validate a live ORT output tensor against the contract (rank/dtype/dims). */
export function validateTensor(name, tensor, shapeSpec, dtype = 'float32') {
  if (tensor.type !== dtype) {
    fail('Invalid model', `Invalid model: tensor '${name}' dtype is '${tensor.type}', expected '${dtype}'`);
  }
  if (tensor.dims.length !== shapeSpec.length) {
    fail('Invalid model', `Invalid model: tensor '${name}' rank ${tensor.dims.length} != expected ${shapeSpec.length}`);
  }
  shapeSpec.forEach((want, i) => {
    if (want != null && tensor.dims[i] !== want) {
      fail('Invalid model', `Invalid model: tensor '${name}' dim ${i} is ${tensor.dims[i]}, expected ${want}`);
    }
  });
}

async function createSessionWithFallback(modelUrl, { warmup, label }) {
  const failures = [];
  for (const ep of ['webgpu', 'wasm']) {
    try {
      const session = await ort.InferenceSession.create(modelUrl, {
        executionProviders: [ep],
      });
      if (warmup) await warmup(session, ep);
      if (state.backend === 'unknown') state.backend = ep;
      return { session, backend: ep };
    } catch (e) {
      // Full error -> DevTools console; one-line summary -> UI status list.
      console.error(`[faceswapper] ${label}: '${ep}' execution provider failed`, e);
      failures.push(`${ep}: ${shortErr(e)}`);
    }
  }
  fail(
    'Inference failed',
    `${label}: no execution provider worked. ` +
      failures.map((f) => `[${f}]`).join(' ') +
      ` Open DevTools console for the full errors.`,
    new Error(failures.join(' | ')),
  );
}

import { cachedFetch } from './model-cache.js';

/** Test seam: set the model base without loading ORT (used by tools/). */
export function configureModelBase(modelBase) {
  state.config = { modelBase, numThreads: 1 };
}

// Throttle for byte-progress updates: first chunk fires immediately (so the
// bar appears at once), then 5 MB / 2 s granularity.
const byteProgress = { bytes: 0, time: 0 };

function emitBytes(onBytes, label, tag, received, total) {
  if (!onBytes) return;
  const now = (typeof performance !== 'undefined' && performance.now()) || 0;
  const first = byteProgress.bytes === 0 && byteProgress.time === 0;
  if (!first && received - byteProgress.bytes < 5 * 1048576 && now - byteProgress.time < 2000) return;
  byteProgress.bytes = received;
  byteProgress.time = now;
  onBytes(received, total, label, tag);
}

/**
 * Local-first model fetch with remote fallback (GitHub Pages pattern) and
 * persistent on-device cache for remote downloads.
 * Returns `{ buf, source }` with source ∈ 'local' | 'remote' | 'cache'.
 */
export async function fetchModelFile(key, { label, minBytes }, { onProgress, onBytes } = {}) {
  const manifest = await loadManifest();
  const entry = manifest.files?.[key];
  if (!entry) fail('Model loading failed', `manifest.json has no '${key}' entry.`);
  const attempts = [{ url: `${state.config.modelBase}/${entry.path}`, cache: false, tag: 'local' }];
  if (entry.remote) attempts.push({ url: entry.remote, cache: true, tag: 'remote' });
  const problems = [];
  for (const { url, cache, tag } of attempts) {
    try {
      const { buf, source } = await cachedFetch(url, {
        // Only remote downloads are cached: same-host dev files are mutable.
        key: cache ? entry.path : null,
        expectedBytes: minBytes,
        sha256: entry.sha256 || null,
        onProgress: (received, total) => emitBytes(onBytes, label, tag, received, total),
      });
      const origin = source === 'cache' ? 'cache' : tag;
      if (origin === 'cache') onProgress?.(`Loading models... (${label} from on-device cache ✓)`);
      return { buf, source: origin };
    } catch (e) {
      problems.push(`${tag}: ${shortErr(e)}`);
    }
  }
  fail(
    'Model loading failed',
    `Could not fetch ${entry.path} (${problems.join(' / ')}). Install it per public/models/MODELS.md.`,
  );
}

async function loadManifest() {
  if (state.manifest) return state.manifest;
  let res;
  try {
    res = await fetch(`${state.config.modelBase}/manifest.json`);
  } catch (e) {
    fail('Model loading failed', `Could not fetch manifest.json from ${state.config.modelBase}: ${shortErr(e)}`, e);
  }
  if (!res.ok) fail('Model loading failed', `Could not fetch manifest.json (HTTP ${res.status}).`);
  state.manifest = await res.json();
  return state.manifest;
}

export async function initInference({ wasmModule, modelBase, ortWasmBase, ortModuleUrl, numThreads } = {}) {
  state.wasm = wasmModule;
  await ensureOrt(ortModuleUrl);
  state.config = { modelBase: modelBase || './models', numThreads: numThreads || Math.min(4, (navigator.hardwareConcurrency || 4)) };
  ort.env.wasm.wasmPaths = ortWasmBase || 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.22.0/dist/';
  // WASM threads need SharedArrayBuffer, i.e. COOP/COEP headers
  // (crossOriginIsolated). Without them, force 1 thread instead of failing.
  const isolated = typeof crossOriginIsolated !== 'undefined' && crossOriginIsolated === true;
  ort.env.wasm.numThreads = isolated ? state.config.numThreads : 1;
  if (!isolated) {
    console.warn(
      '[faceswapper] page is not cross-origin isolated — WASM backend limited to 1 thread. ' +
        'Serve with Cross-Origin-Opener-Policy: same-origin + Cross-Origin-Embedder-Policy: require-corp for full speed (npm run dev already does).',
    );
  }
  ort.env.wasm.simd = true;
  try {
    if (navigator.gpu) {
      await navigator.gpu.requestAdapter().then((a) => a?.requestDevice().then((d) => d?.destroy()));
    }
  } catch {
    // WebGPU probe failed — session creation will fall back to WASM.
  }
  return { backend: state.backend };
}

export function currentBackend() {
  return state.backend;
}

async function ensureDetector(onProgress, onBytes) {
  if (state.sessions.detector) return state.sessions.detector;
  const { buf } = await fetchModelFile(
    'detector',
    { label: 'Face detector', minBytes: 1_000_000 },
    { onProgress, onBytes },
  );
  const { session, backend } = await createSessionWithFallback(buf, {
    label: 'Face detector',
    warmup: async (s) => {
      // Warmup also validates the 9-tensor bnkps contract. NOTE: common
      // fixed-shape exports (e.g. cysin/scrfd_onnx, [1,3,640,640]) reject any
      // other size, so warm up at the real 640x640 letterbox size.
      const t = new ort.Tensor('float32', new Float32Array(1 * 3 * 640 * 640), [1, 3, 640, 640]);
      const out = await s.run({ [s.inputNames[0]]: t });
      const names = Object.keys(out);
      if (names.length !== 9 && names.length !== 6) {
        fail('Invalid model', `Face detector returned ${names.length} outputs, expected 9 (bnkps) or 6 (no-kps). Wrong model file?`);
      }
    },
  });
  state.sessions.detector = { session, backend, hasKps: true };
  return state.sessions.detector;
}

async function ensureRecognizer(onProgress, onBytes) {
  if (state.sessions.recognizer) return state.sessions.recognizer;
  const { buf } = await fetchModelFile(
    'recognizer',
    { label: 'Face recognizer', minBytes: 50_000_000 },
    { onProgress, onBytes },
  );
  const { session, backend } = await createSessionWithFallback(buf, {
    label: 'Face recognizer',
    warmup: async (s) => {
      const t = new ort.Tensor('float32', new Float32Array(1 * 3 * 112 * 112), [1, 3, 112, 112]);
      const out = await s.run({ [s.inputNames[0]]: t });
      const names = Object.keys(out);
      if (names.length !== 1) fail('Invalid model', `Face recognizer returned ${names.length} outputs, expected 1.`);
      validateTensor(names[0], out[names[0]], [1, 512]);
    },
  });
  state.sessions.recognizer = { session, backend };
  return state.sessions.recognizer;
}

async function ensureEmap() {
  if (state.emap) return state.emap;
  const url = `${state.config.modelBase}/reswapper.emap.json`;
  let res;
  try {
    res = await fetch(url);
  } catch (e) {
    fail('Model loading failed', 'Could not fetch the swapper emap matrix. Extract it once with tools/extract_emap.py (see MODELS.md).', e);
  }
  if (!res.ok) {
    fail(
      'Model loading failed',
      'reswapper.emap.json is missing. The emap matrix must be extracted once from YOUR copy of ' +
        'reswapper-1019500.onnx: python3 tools/extract_emap.py public/models/reswapper-1019500.onnx public/models/reswapper.emap.json',
    );
  }
  const contentType = res.headers.get('content-type') || '';
  const bodyText = await res.text();
  if (contentType.includes('text/html')) {
    fail(
      'Model loading failed',
      'reswapper.emap.json is missing (the server returned an HTML page — SPA fallback for a missing file). ' +
        'Extract it from YOUR copy of reswapper-1019500.onnx: ' +
        'python3 tools/extract_emap.py public/models/reswapper-1019500.onnx public/models/reswapper.emap.json',
    );
  }
  let j;
  try {
    j = JSON.parse(bodyText);
  } catch {
    fail(
      'Invalid model',
      'reswapper.emap.json is not valid JSON. Re-generate it with tools/extract_emap.py.',
    );
  }
  if (!j || j.shape?.join(',') !== '512,512' || !Array.isArray(j.data) || j.data.length !== 512 * 512) {
    fail('Invalid model', 'reswapper.emap.json must be {shape:[512,512], data:[262144 floats]}.');
  }
  state.emap = Float32Array.from(j.data);
  return state.emap;
}

async function ensureSwapper(onProgress, onBytes) {
  if (state.sessions.swapper) return state.sessions.swapper;
  const { buf } = await fetchModelFile(
    'swapper',
    { label: 'Face swap model', minBytes: 400_000_000 },
    { onProgress, onBytes },
  );
  const { session, backend } = await createSessionWithFallback(buf, {
    label: 'Face swap model',
    warmup: async (s) => {
      if (s.inputNames.length !== 2) {
        fail('Invalid model', `Face swap model has ${s.inputNames.length} inputs, expected 2 (target image + source latent). Wrong model file?`);
      }
      // Role detection by probe: feed the two candidate tensors and see
      // which assignment the graph accepts (ORT validates shapes).
      // NOTE: both probe errors are reported — an earlier version only
      // surfaced the second probe's error, masking the real failure.
      const img = new ort.Tensor('float32', new Float32Array(1 * 3 * 128 * 128), [1, 3, 128, 128]);
      const lat = new ort.Tensor('float32', new Float32Array(1 * 512), [1, 512]);
      const [n0, n1] = s.inputNames;
      let out = null;
      let roles = null;
      let firstErr = null;
      try {
        out = await s.run({ [n0]: img, [n1]: lat });
        roles = { image: n0, latent: n1 };
      } catch (e) {
        firstErr = e;
      }
      if (!out) {
        try {
          out = await s.run({ [n0]: lat, [n1]: img });
          roles = { image: n1, latent: n0 };
        } catch (e2) {
          throw new Error(
            `swap input-role probe failed for both assignments: ` +
              `[${n0}=image] ${shortErr(firstErr)} // [${n0}=latent] ${shortErr(e2)}`,
          );
        }
      }
      const onames = Object.keys(out);
      if (onames.length !== 1) fail('Invalid model', `Face swap model returned ${onames.length} outputs, expected 1.`);
      validateTensor(onames[0], out[onames[0]], [1, 3, 128, 128]);
      state.sessions.swapperRoles = { ...roles, output: onames[0] };
    },
  });
  state.sessions.swapper = { session, backend };
  return state.sessions.swapper;
}

// ---------------------------------------------------------------------------
// Tensor helpers (NCHW float32, RGB)
// ---------------------------------------------------------------------------

export function rgbaToNchw(rgba, w, h, mean, std) {
  const chw = new Float32Array(3 * w * h);
  for (let i = 0; i < w * h; i++) {
    const r = rgba[i * 4];
    const g = rgba[i * 4 + 1];
    const b = rgba[i * 4 + 2];
    chw[i] = (r - mean) / std;
    chw[w * h + i] = (g - mean) / std;
    chw[2 * w * h + i] = (b - mean) / std;
  }
  return chw;
}

function l2normalize(v) {
  let s = 0;
  for (let i = 0; i < v.length; i++) s += v[i] * v[i];
  const n = Math.sqrt(s) || 1;
  const o = new Float32Array(v.length);
  for (let i = 0; i < v.length; i++) o[i] = v[i] / n;
  return o;
}

// ---------------------------------------------------------------------------
// SCRFD decode (port of detection/scrfd/tools/scrfd.py)
// ---------------------------------------------------------------------------

function anchorCenters(h, w, stride, numAnchors, cache) {
  const key = `${h}x${w}x${stride}x${numAnchors}`;
  let c = cache.get(key);
  if (!c) {
    c = new Float32Array(h * w * numAnchors * 2);
    let p = 0;
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        for (let a = 0; a < numAnchors; a++) {
          c[p++] = x * stride;
          c[p++] = y * stride;
        }
      }
    }
    if (cache.size < 100) cache.set(key, c);
  }
  return c;
}

function flat2D(t) {
  // Accept [N,C] or [1,N,C].
  const d = t.dims;
  const v = t.data;
  if (d.length === 3) return { n: d[1], c: d[2], stride: d[2], base: 0, data: v };
  if (d.length === 2) return { n: d[0], c: d[1], stride: d[1], base: 0, data: v };
  fail('Invalid model', `Unexpected detector output rank ${d.length} (dims ${d}).`);
}

export function decodeScrfd(outputs, inW, inH, threshold, cache) {
  // Output order for bnkps exports: score_x3, bbox_x3, kps_x3.
  const names = Object.keys(outputs);
  const hasKps = names.length === 9;
  if (names.length !== 9 && names.length !== 6) {
    fail('Invalid model', `Detector outputs [${names.join(', ')}]: expected 9 (bnkps) or 6 tensors.`);
  }
  const strides = CONTRACTS.detector.strides;
  const numAnchors = CONTRACTS.detector.numAnchors;
  const boxes = [];
  const scores = [];
  const kps = [];
  for (let sIdx = 0; sIdx < 3; sIdx++) {
    const stride = strides[sIdx];
    const scoreT = flat2D(outputs[names[sIdx]]);
    const bboxT = flat2D(outputs[names[sIdx + 3]]);
    const kpsT = hasKps ? flat2D(outputs[names[sIdx + 6]]) : null;
    const featH = Math.floor(inH / stride);
    const featW = Math.floor(inW / stride);
    const centers = anchorCenters(featH, featW, stride, numAnchors, cache);
    const rows = Math.min(scoreT.n, bboxT.n, centers.length / 2);
    for (let i = 0; i < rows; i++) {
      const sc = scoreT.data[scoreT.base + i * scoreT.stride];
      if (sc < threshold) continue;
      const cx = centers[i * 2];
      const cy = centers[i * 2 + 1];
      const bo = bboxT.base + i * bboxT.stride;
      const l = bboxT.data[bo] * stride;
      const t = bboxT.data[bo + 1] * stride;
      const r = bboxT.data[bo + 2] * stride;
      const b = bboxT.data[bo + 3] * stride;
      boxes.push([cx - l, cy - t, cx + r, cy + b]);
      scores.push(sc);
      if (kpsT) {
        const ko = kpsT.base + i * kpsT.stride;
        const pts = [];
        for (let k = 0; k < 5; k++) {
          pts.push([cx + kpsT.data[ko + k * 2] * stride, cy + kpsT.data[ko + k * 2 + 1] * stride]);
        }
        kps.push(pts);
      }
    }
  }
  return { boxes, scores, kps, hasKps };
}

export function nmsJs(boxes, scores, thresh) {
  const order = scores.map((_, i) => i).sort((a, b) => scores[b] - scores[a]);
  const areas = boxes.map((b) => (b[2] - b[0] + 1) * (b[3] - b[1] + 1));
  const keep = [];
  const dead = new Uint8Array(boxes.length);
  for (let oi = 0; oi < order.length; oi++) {
    const i = order[oi];
    if (dead[i]) continue;
    keep.push(i);
    for (let oj = oi + 1; oj < order.length; oj++) {
      const j = order[oj];
      if (dead[j]) continue;
      const xx1 = Math.max(boxes[i][0], boxes[j][0]);
      const yy1 = Math.max(boxes[i][1], boxes[j][1]);
      const xx2 = Math.min(boxes[i][2], boxes[j][2]);
      const yy2 = Math.min(boxes[i][3], boxes[j][3]);
      const w = Math.max(0, xx2 - xx1 + 1);
      const h = Math.max(0, yy2 - yy1 + 1);
      const ovr = (w * h) / (areas[i] + areas[j] - w * h);
      if (ovr > thresh) dead[j] = 1;
    }
  }
  return keep;
}

/**
 * Detect faces in an RGBA image. Returns faces in ORIGINAL pixel coords:
 * [{ bbox:[x1,y1,x2,y2], landmarks:[[x,y]x5], score }].
 */
export async function detectFaces(rgba, w, h, { threshold = 0.5, onProgress, onBytes } = {}) {
  const det = await ensureDetector(onProgress, onBytes);
  onProgress?.('Detecting faces...');
  // Letterbox into 640x640 exactly like SCRFD.detect (top-left, zero-pad).
  const S = 640;
  const imRatio = h / w;
  const modelRatio = 1;
  let newW, newH;
  if (imRatio > modelRatio) {
    newH = S;
    newW = Math.max(1, Math.round(newH / imRatio));
  } else {
    newW = S;
    newH = Math.max(1, Math.round(newW * imRatio));
  }
  const detScale = newH / h;
  // Resize with the WASM bilinear kernel (or canvas fallback in tests).
  const resized = state.wasm
    ? state.wasm.resizeRgba(rgba, w, h, newW, newH)
    : neonatalResize(rgba, w, h, newW, newH);
  const padded = new Uint8Array(S * S * 4); // zeros = padding
  for (let y = 0; y < newH; y++) {
    padded.set(resized.subarray(y * newW * 4, (y + 1) * newW * 4), y * S * 4);
  }
  const chw = rgbaToNchw(padded, S, S, 127.5, 128.0);
  const input = new ort.Tensor('float32', chw, [1, 3, S, S]);
  let out;
  try {
    out = await det.session.run({ [det.session.inputNames[0]]: input });
  } catch (e) {
    fail('Inference failed', `Face detection inference failed: ${e.message || e}`, e);
  }
  const t0 = performance.now();
  const { boxes, scores, kps, hasKps } = decodeScrfd(out, S, S, threshold, (detectFaces._cache ||= new Map()));
  if (!hasKps) {
    fail(
      'Invalid model',
      'The detector has no landmark outputs (expected the `_bnkps` variant with 9 outputs). ' +
        'Landmarks are required for alignment — use scrfd_2.5g_bnkps.onnx.',
    );
  }
  let keep;
  try {
    const flat = new Float32Array(boxes.length * 4);
    boxes.forEach((b, i) => flat.set(b, i * 4));
    keep = state.wasm
      ? Array.from(state.wasm.nmsBoxes(flat, Float32Array.from(scores), 0.4))
      : nmsJs(boxes, scores, 0.4);
  } catch {
    keep = nmsJs(boxes, scores, 0.4);
  }
  const faces = keep.map((i) => ({
    bbox: boxes[i].map((v) => v / detScale),
    landmarks: kps[i].map((p) => [p[0] / detScale, p[1] / detScale]),
    score: scores[i],
  }));
  // Clamp to image.
  for (const f of faces) {
    f.bbox = [Math.max(0, f.bbox[0]), Math.max(0, f.bbox[1]), Math.min(w, f.bbox[2]), Math.min(h, f.bbox[3])];
  }
  void t0;
  return faces;
}

function neonatalResize(rgba, w, h, nw, nh) {
  // Minimal fallback when WASM is unavailable (tests). Real path uses WASM.
  const out = new Uint8Array(nw * nh * 4);
  for (let y = 0; y < nh; y++) {
    for (let x = 0; x < nw; x++) {
      const sx = Math.min(w - 1, Math.floor(((x + 0.5) * w) / nw));
      const sy = Math.min(h - 1, Math.floor(((y + 0.5) * h) / nh));
      out.set(rgba.subarray((sy * w + sx) * 4, (sy * w + sx) * 4 + 4), (y * nw + x) * 4);
    }
  }
  return out;
}

// ---------------------------------------------------------------------------
// Alignment (via WASM) + embedding + latent
// ---------------------------------------------------------------------------

function landmarksFlat(lm) {
  return Float32Array.from([lm[0][0], lm[0][1], lm[1][0], lm[1][1], lm[2][0], lm[2][1], lm[3][0], lm[3][1], lm[4][0], lm[4][1]]);
}

/** Aligned crop (RGBA bytes) + matrix M, like `norm_crop2`. */
export function alignCrop(rgba, w, h, landmarks, size) {
  const M = state.wasm.estimateNorm(landmarksFlat(landmarks), size);
  const crop = state.wasm.warpRgba(rgba, w, h, M, size, size);
  return { crop, M: Array.from(M) };
}

export async function embeddingForFace(rgba, w, h, landmarks, onProgress, onBytes) {
  const rec = await ensureRecognizer(onProgress, onBytes);
  const { crop } = alignCrop(rgba, w, h, landmarks, 112);
  const chw = rgbaToNchw(crop, 112, 112, 127.5, 127.5);
  const input = new ort.Tensor('float32', chw, [1, 3, 112, 112]);
  let out;
  try {
    out = await rec.session.run({ [rec.session.inputNames[0]]: input });
  } catch (e) {
    fail('Inference failed', `Face embedding inference failed: ${e.message || e}`, e);
  }
  const name = Object.keys(out)[0];
  validateTensor(name, out[name], [1, 512]);
  return l2normalize(Array.from(out[name].data));
}

/** latent = normalize(normed_embedding @ emap) — exactly like inswapper.py. */
export async function latentForEmbedding(normedEmbedding) {
  const emap = await ensureEmap();
  if (normedEmbedding.length !== 512) fail('Invalid model', 'Embedding must be 512-d (buffalo_l ArcFace).');
  const lat = new Float32Array(512);
  for (let j = 0; j < 512; j++) {
    let s = 0;
    for (let i = 0; i < 512; i++) s += normedEmbedding[i] * emap[i * 512 + j];
    lat[j] = s;
  }
  return l2normalize(lat);
}

// ---------------------------------------------------------------------------
// Source identity (prepared ONCE, cached)
// ---------------------------------------------------------------------------

export async function prepareSourceIdentity(rgba, w, h, { onProgress, onBytes } = {}) {
  onProgress?.('Preparing source identity...');
  const faces = await detectFaces(rgba, w, h, { threshold: 0.4, onProgress, onBytes });
  if (faces.length === 0) {
    fail('No face detected', 'No face found in the source identity image. Replace public/source/example.jpg with a clear, front-facing portrait.');
  }
  // Largest face = the subject.
  faces.sort((a, b) => area(b.bbox) - area(a.bbox));
  const face = faces[0];
  const emb = await embeddingForFace(rgba, w, h, face.landmarks, onProgress, onBytes);
  const latent = await latentForEmbedding(emb);
  const { crop } = alignCrop(rgba, w, h, face.landmarks, 256);
  state.source = { latent, width: w, height: h, preview: crop.slice(0), previewSize: 256, landmarks: face.landmarks, bbox: face.bbox };
  return state.source;
}

function area(b) {
  return Math.max(0, b[2] - b[0]) * Math.max(0, b[3] - b[1]);
}

// ---------------------------------------------------------------------------
// Swap + paste-back for ONE target face
// ---------------------------------------------------------------------------

export async function swapOneFace(targetRgba, tw, th, face, sourceLatent, { colorMatch = true, onBytes = null } = {}) {
  const sw = await ensureSwapper(undefined, onBytes);
  const S = 128;
  // 1. align target face (norm_crop2 @128)
  const { crop: aimg, M } = alignCrop(targetRgba, tw, th, face.landmarks, S);
  // 2. swap inference: blob = RGB/255
  const chw = rgbaToNchw(aimg, S, S, 0.0, 255.0);
  const roles = state.sessions.swapperRoles;
  if (!roles) fail('Invalid model', 'Swap session roles were not established during warmup.');
  const feeds = {
    [roles.image]: new ort.Tensor('float32', chw, [1, 3, S, S]),
    [roles.latent]: new ort.Tensor('float32', sourceLatent, [1, 512]),
  };
  let out;
  try {
    out = await sw.session.run(feeds);
  } catch (e) {
    fail('Inference failed', `Face swap inference failed: ${e.message || e}`, e);
  }
  const t = out[roles.output];
  validateTensor(roles.output, t, [1, 3, S, S]);
  // 3. to RGB bytes (output is [0,1] RGB)
  const d = t.data;
  const fakeRgb = new Uint8Array(S * S * 3);
  for (let i = 0; i < S * S; i++) {
    fakeRgb[i * 3] = Math.max(0, Math.min(255, Math.round(d[i] * 255)));
    fakeRgb[i * 3 + 1] = Math.max(0, Math.min(255, Math.round(d[S * S + i] * 255)));
    fakeRgb[i * 3 + 2] = Math.max(0, Math.min(255, Math.round(d[2 * S * S + i] * 255)));
  }
  const aimgRgb = new Uint8Array(S * S * 3);
  for (let i = 0; i < S * S; i++) {
    aimgRgb[i * 3] = aimg[i * 4];
    aimgRgb[i * 3 + 1] = aimg[i * 4 + 1];
    aimgRgb[i * 3 + 2] = aimg[i * 4 + 2];
  }
  // 4. crop-space feathered mask (white, 2px zero border, erode+blur)
  const white = new Float32Array(S * S).fill(255);
  for (let y = 0; y < S; y++) {
    for (let x = 0; x < S; x++) {
      if (x < 2 || y < 2 || x >= S - 2 || y >= S - 2) white[y * S + x] = 0;
    }
  }
  const maskCrop = state.wasm.buildCropMask(white, S);
  // 5. illumination adaptation (masked moment match in crop space)
  const matched = colorMatch ? state.wasm.colorMatchCrop(aimgRgb, fakeRgb, maskCrop, S * S) : fakeRgb;
  // 6. warp swapped face + mask back with IM
  const IM = state.wasm.invertAffine(Float32Array.from(M));
  const matchedRgba = new Uint8Array(S * S * 4);
  for (let i = 0; i < S * S; i++) {
    matchedRgba[i * 4] = matched[i * 3];
    matchedRgba[i * 4 + 1] = matched[i * 3 + 1];
    matchedRgba[i * 4 + 2] = matched[i * 3 + 2];
    matchedRgba[i * 4 + 3] = 255;
  }
  const backRgb = state.wasm.warpRgba(matchedRgba, S, S, IM, tw, th);
  // Warp mask as RGBA (pack weight into all channels, bilinear = smooth).
  const maskRgba = new Uint8Array(S * S * 4);
  for (let i = 0; i < S * S; i++) {
    const v = Math.round(maskCrop[i] * 255);
    maskRgba[i * 4] = maskRgba[i * 4 + 1] = maskRgba[i * 4 + 2] = v;
    maskRgba[i * 4 + 3] = 255;
  }
  const backMaskRgba = state.wasm.warpRgba(maskRgba, S, S, IM, tw, th);
  const backMask = new Float32Array(tw * th);
  for (let i = 0; i < tw * th; i++) backMask[i] = backMaskRgba[i * 4] / 255;
  const backRgb3 = new Uint8Array(tw * th * 3);
  for (let i = 0; i < tw * th; i++) {
    backRgb3[i * 3] = backRgb[i * 4];
    backRgb3[i * 3 + 1] = backRgb[i * 4 + 1];
    backRgb3[i * 3 + 2] = backRgb[i * 4 + 2];
  }
  // 7. composite
  return { backRgb3, backMask, M };
}

export function getSource() {
  return state.source;
}
