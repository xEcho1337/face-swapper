/**
 * FaceSwapper Web Worker — all heavy work off the main thread.
 *
 * Main thread owns: UI, drag&drop, progress display, preview, download.
 * This worker owns: WASM kernels, ONNX sessions, detection, swap, blending.
 * Large buffers move via Transferables (zero-copy where possible).
 *
 * Protocol (main -> worker):
 *   { id, type: 'init', modelBase, ortWasmBase }
 *   { id, type: 'prepare-source', buf: ArrayBuffer, w, h }   (RGBA bytes)
 *   { id, type: 'process', buf: ArrayBuffer, w, h, threshold, maxFaces }
 *
 * Protocol (worker -> main):
 *   { id, type: 'ready', backend }
 *   { id, type: 'source-ready' }
 *   { id, type: 'progress', stage }
 *   { id, type: 'done', buf, w, h, faces, timings, backend }
 *   { id, type: 'error', code, message }
 */

import { initInference, currentBackend, sessionBackends, epReport, webgpuExposed, detectFaces, prepareSourceIdentity, swapOneFace, getSource } from './inference.js';
import initWasm, * as wasm from './pkg/faceswapper.js';

let wasmReady = false;

async function ensureWasm() {
  if (!wasmReady) {
    await initWasm();
    wasmReady = true;
  }
  return wasm;
}

function post(msg, transfer) {
  postMessage(msg, transfer || []);
}

function errToCode(e) {
  if (e && e.code) return e.code;
  const m = String((e && e.message) || e);
  if (/memory|allocation|OOM/i.test(m)) return 'Out of memory';
  return 'Inference failed';
}

onmessage = async (ev) => {
  const msg = ev.data;
  const { id, type } = msg;
  const reply = (m, t) => post({ ...m, id }, t);
  // Byte progress goes to the progress BAR (one live element); discrete
  // stages go to the status list. Never one card per chunk.
  const forwardBytes = (received, total, label, tag) =>
    reply({ type: 'download', received, total, label, tag });
  try {
    if (type === 'init') {
      const wk = await ensureWasm();
      await initInference({ wasmModule: wk, modelBase: msg.modelBase, ortWasmBase: msg.ortWasmBase, ortModuleUrl: msg.ortModuleUrl, swapperModel: msg.swapperModel, backendPreference: msg.backendPreference });
      reply({ type: 'ready', backend: currentBackend(), gpu: webgpuExposed(), epLog: epReport() });
      return;
    }
    if (type === 'prepare-source') {
      const wk = await ensureWasm();
      await initInference({ wasmModule: wk, modelBase: msg.modelBase, ortWasmBase: msg.ortWasmBase, ortModuleUrl: msg.ortModuleUrl, swapperModel: msg.swapperModel, backendPreference: msg.backendPreference });
      const rgba = new Uint8Array(msg.buf);
      const t0 = performance.now();
      await prepareSourceIdentity(rgba, msg.w, msg.h, {
        onProgress: (stage) => reply({ type: 'progress', stage }),
        onBytes: forwardBytes,
      });
      // Send the aligned source crop back for the UI preview (transferred).
      const src = getSource();
      const preview = src.preview;
      reply(
        {
          type: 'source-ready',
          ms: Math.round(performance.now() - t0),
          backend: currentBackend(),
          backends: sessionBackends(),
          epLog: epReport(),
          previewBuf: preview.buffer,
          previewW: src.previewSize,
          previewH: src.previewSize,
        },
        [preview.buffer],
      );
      return;
    }
    if (type === 'process') {
      const wk = await ensureWasm();
      await initInference({ wasmModule: wk, modelBase: msg.modelBase, ortWasmBase: msg.ortWasmBase, ortModuleUrl: msg.ortModuleUrl, swapperModel: msg.swapperModel, backendPreference: msg.backendPreference });
      const tAll = performance.now();
      const threshold = msg.threshold ?? 0.5;
      const maxFaces = msg.maxFaces ?? 50;
      const rgba = new Uint8Array(msg.buf);
      const { w, h } = msg;

      const timings = {};
      let t = performance.now();
      const progress = (stage) => reply({ type: 'progress', stage });

      const faces = await detectFaces(rgba, w, h, { threshold, onProgress: progress, onBytes: forwardBytes });
      timings.detectMs = Math.round(performance.now() - t);

      if (faces.length === 0) {
        const e = new Error('No face detected in this image. Try a photo with a clear, front-facing face.');
        e.code = 'No face detected';
        throw e;
      }
      if (faces.length > maxFaces) {
        const e = new Error(`Too many faces: found ${faces.length}, limit is ${maxFaces}.`);
        e.code = 'Too many faces';
        throw e;
      }

      const source = getSource();
      if (!source) {
        const e = new Error('Source identity is not ready. The fixed source portrait failed to load — see status.');
        e.code = 'Model loading failed';
        throw e;
      }

      // Composite incrementally on a working RGBA copy (one full-res buffer,
      // reused per face) to bound memory on very large images.
      let working = rgba.slice();
      const perFaceMs = [];
      for (let i = 0; i < faces.length; i++) {
        progress(`Swapping ${i + 1} / ${faces.length} faces...`);
        const tF = performance.now();
        const { backRgb3, backMask } = await swapOneFace(working, w, h, faces[i], source.latent, { colorMatch: msg.colorMatch ?? false, onBytes: forwardBytes });
        progress('Blending result...');
        working = wk.blendFullres(working, backRgb3, backMask, w, h);
        perFaceMs.push(Math.round(performance.now() - tF));
      }
      timings.swapMsPerFace = perFaceMs;
      timings.totalMs = Math.round(performance.now() - tAll);

      progress('Done.');
      const out = working;
      reply(
        {
          type: 'done',
          buf: out.buffer,
          w,
          h,
          faces: faces.map((f) => ({ bbox: f.bbox, score: f.score })),
          timings,
          backend: currentBackend(),
          backends: sessionBackends(),
          epLog: epReport(),
        },
        [out.buffer],
      );
      return;
    }
    throw new Error(`Unknown worker message: ${type}`);
  } catch (e) {
    reply({ type: 'error', code: errToCode(e), message: (e && e.message) || String(e) });
  }
};
