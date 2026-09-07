#!/usr/bin/env node
/**
 * REAL end-to-end face-swap verification (not a mock):
 * detection (SCRFD weights) -> ArcFace embedding (w600k_r50 weights) ->
 * contract-compatible swap model (ReSwapper / inswapper_128) ->
 * WASM paste-back -> identity assertions.
 *
 * Uses the EXACT shipped code: web/inference.js decode + web/pkg/*.wasm
 * kernels. Source identity = largest face of the input photo (self-swap
 * protocol: every face must move TOWARD the source identity).
 *
 * Setup: npm i --no-save onnxruntime-node@1.22.0 sharp
 * Usage: node tools/verify-swap.mjs <scrfd.onnx> <w600k_r50.onnx> <swap-model.onnx> <emap.json> <photo> [out.jpg]
 */
import { readFileSync, writeFileSync } from 'node:fs';
import * as ort from 'onnxruntime-node';
import sharp from 'sharp';
import initWasm, * as wasm from '../web/pkg/faceswapper.js';
import { decodeScrfd, nmsJs, rgbaToNchw } from '../web/inference.js';

const [scrfdPath, recPath, swapPath, emapPath, photoPath, outPath] = process.argv.slice(2);
if (!scrfdPath || !recPath || !swapPath || !emapPath || !photoPath) {
  console.error('usage: verify-swap.mjs <scrfd> <w600k> <swap-model> <emap.json> <photo> [out.jpg]');
  process.exit(2);
}

await initWasm(readFileSync(new URL('../web/pkg/faceswapper_bg.wasm', import.meta.url)));
const emapJson = JSON.parse(readFileSync(emapPath, 'utf8'));
const emap = Float32Array.from(emapJson.data);

const S = 640;
const raw = await sharp(photoPath).removeAlpha().raw().toBuffer({ resolveWithObject: true });
const W = raw.info.width, H = raw.info.height;
const rgba = new Uint8Array(W * H * 4);
for (let i = 0; i < W * H; i++) {
  rgba[i * 4] = raw.data[i * 3]; rgba[i * 4 + 1] = raw.data[i * 3 + 1];
  rgba[i * 4 + 2] = raw.data[i * 3 + 2]; rgba[i * 4 + 3] = 255;
}
console.log(`photo: ${W}x${H}`);

// --- sessions (reused, like the app) ---
const det = await ort.InferenceSession.create(readFileSync(scrfdPath), { executionProviders: ['cpu'] });
const rec = await ort.InferenceSession.create(readFileSync(recPath), { executionProviders: ['cpu'] });
const swp = await ort.InferenceSession.create(readFileSync(swapPath), { executionProviders: ['cpu'] });
console.log('swap inputs:', swp.inputNames, 'outputs:', swp.outputNames);

// --- detect (app-identical letterbox via WASM resize) ---
const imRatio = H / W;
let newW, newH;
if (imRatio > 1) { newH = S; newW = Math.max(1, Math.round(newH / imRatio)); }
else { newW = S; newH = Math.max(1, Math.round(newW * imRatio)); }
const detScale = newH / H;
const resized = wasm.resizeRgba(rgba, W, H, newW, newH);
const padded = new Uint8Array(S * S * 4);
for (let y = 0; y < newH; y++) padded.set(resized.subarray(y * newW * 4, (y + 1) * newW * 4), y * S * 4);
const detOut = await det.run({ [det.inputNames[0]]: new ort.Tensor('float32', rgbaToNchw(padded, S, S, 127.5, 128.0), [1, 3, S, S]) });
const adapted = {};
for (const [k, v] of Object.entries(detOut)) adapted[k] = { dims: Array.from(v.dims), data: v.data, type: v.type };
const { boxes, scores, kps } = decodeScrfd(adapted, S, S, 0.5, new Map());
const keep = nmsJs(boxes, scores, 0.4);
const faces = keep.map((i) => ({
  bbox: boxes[i].map((v) => v / detScale),
  landmarks: kps[i].map((p) => [p[0] / detScale, p[1] / detScale]),
  score: scores[i],
}));
console.log(`faces: ${faces.length}`);
if (faces.length === 0) { console.error('FAIL: no faces'); process.exit(1); }

const flat = (lm) => Float32Array.from(lm.flat());
const l2n = (v) => { const n = Math.hypot(...v) || 1; return v.map((x) => x / n); };
const cos = (a, b) => a.reduce((s, x, i) => s + x * b[i], 0);

async function embed(rgbaImg, w, h, landmarks) {
  const M = wasm.estimateNorm(flat(landmarks), 112);
  const crop = wasm.warpRgba(rgbaImg, w, h, M, 112, 112);
  const out = await rec.run({ [rec.inputNames[0]]: new ort.Tensor('float32', rgbaToNchw(crop, 112, 112, 127.5, 127.5), [1, 3, 112, 112]) });
  return l2n(Array.from(Object.values(out)[0].data));
}

// --- source identity: largest face, prepared ONCE ---
const byArea = [...faces].sort((a, b) =>
  ((b.bbox[2] - b.bbox[0]) * (b.bbox[3] - b.bbox[1])) - ((a.bbox[2] - a.bbox[0]) * (a.bbox[3] - a.bbox[1])));
const sourceEmb = await embed(rgba, W, H, byArea[0].landmarks);
const latent = (() => {
  const l = new Array(512).fill(0);
  for (let j = 0; j < 512; j++) { let s = 0; for (let i = 0; i < 512; i++) s += sourceEmb[i] * emap[i * 512 + j]; l[j] = s; }
  return Float32Array.from(l2n(l));
})();
console.log('source identity ready (largest face)');

// --- swap every face ---
let working = rgba.slice();
const SS = 128;
for (let f = 0; f < faces.length; f++) {
  const t0 = performance.now();
  const M = wasm.estimateNorm(flat(faces[f].landmarks), SS);
  const aimg = wasm.warpRgba(working, W, H, M, SS, SS);
  const chw = rgbaToNchw(aimg, SS, SS, 0.0, 255.0);
  // role detection by shape (app-identical)
  const in0 = swp.inputNames[0], in1 = swp.inputNames[1];
  const tImg = new ort.Tensor('float32', chw, [1, 3, SS, SS]);
  const tLat = new ort.Tensor('float32', latent, [1, 512]);
  let out;
  try { out = await swp.run({ [in0]: tImg, [in1]: tLat }); }
  catch { out = await swp.run({ [in0]: tLat, [in1]: tImg }); }
  const d = Object.values(out)[0].data;
  const fakeRgb = new Uint8Array(SS * SS * 3);
  for (let i = 0; i < SS * SS; i++) {
    fakeRgb[i * 3] = Math.max(0, Math.min(255, Math.round(d[i] * 255)));
    fakeRgb[i * 3 + 1] = Math.max(0, Math.min(255, Math.round(d[SS * SS + i] * 255)));
    fakeRgb[i * 3 + 2] = Math.max(0, Math.min(255, Math.round(d[2 * SS * SS + i] * 255)));
  }
  const aimgRgb = new Uint8Array(SS * SS * 3);
  for (let i = 0; i < SS * SS; i++) { aimgRgb[i*3] = aimg[i*4]; aimgRgb[i*3+1] = aimg[i*4+1]; aimgRgb[i*3+2] = aimg[i*4+2]; }
  const white = new Float32Array(SS * SS).fill(255);
  for (let y = 0; y < SS; y++) for (let x = 0; x < SS; x++)
    if (x < 2 || y < 2 || x >= SS - 2 || y >= SS - 2) white[y * SS + x] = 0;
  const maskCrop = wasm.buildCropMask(white, SS);
  const matched = wasm.colorMatchCrop(aimgRgb, fakeRgb, maskCrop, SS * SS);
  const matchedRgba = new Uint8Array(SS * SS * 4);
  for (let i = 0; i < SS * SS; i++) { matchedRgba[i*4] = matched[i*3]; matchedRgba[i*4+1] = matched[i*3+1]; matchedRgba[i*4+2] = matched[i*3+2]; matchedRgba[i*4+3] = 255; }
  const IM = wasm.invertAffine(M);
  const backRgb = wasm.warpRgba(matchedRgba, SS, SS, IM, W, H);
  const maskRgba = new Uint8Array(SS * SS * 4);
  maskCrop.forEach((v, i) => { const b = Math.round(v * 255); maskRgba[i*4]=maskRgba[i*4+1]=maskRgba[i*4+2]=b; maskRgba[i*4+3]=255; });
  const backMaskRgba = wasm.warpRgba(maskRgba, SS, SS, IM, W, H);
  const backMask = new Float32Array(W * H);
  for (let i = 0; i < W * H; i++) backMask[i] = backMaskRgba[i * 4] / 255;
  const back3 = new Uint8Array(W * H * 3);
  for (let i = 0; i < W * H; i++) { back3[i*3] = backRgb[i*4]; back3[i*3+1] = backRgb[i*4+1]; back3[i*3+2] = backRgb[i*4+2]; }
  working = wasm.blendFullres(working, back3, backMask, W, H);
  console.log(`swapped face ${f + 1}/${faces.length} in ${Math.round(performance.now() - t0)}ms`);
}

// --- identity assertions ---
let ok = true;
// (a) output differs from input inside face regions (not a copy)
let faceDiff = 0, faceN = 0;
for (const f of faces) {
  const [x1, y1, x2, y2] = f.bbox.map(Math.round);
  for (let y = Math.max(0, y1); y < Math.min(H, y2); y++)
    for (let x = Math.max(0, x1); x < Math.min(W, x2); x++) {
      const i = (y * W + x) * 4;
      faceDiff += Math.abs(working[i] - rgba[i]) + Math.abs(working[i+1] - rgba[i+1]) + Math.abs(working[i+2] - rgba[i+2]);
      faceN++;
    }
}
const meanFaceDiff = faceDiff / Math.max(1, faceN * 3);
console.log(`mean abs diff in face regions: ${meanFaceDiff.toFixed(2)}/255`);
if (!(meanFaceDiff > 3)) { console.error('FAIL: output identical to input (no swap happened)'); ok = false; }

// (b) every swapped face moved TOWARD the source identity
for (let f = 0; f < faces.length; f++) {
  const origEmb = await embed(rgba, W, H, faces[f].landmarks);
  const swapEmb = await embed(working, W, H, faces[f].landmarks);
  const toSource = cos(swapEmb, sourceEmb);
  const toOrig = cos(swapEmb, origEmb);
  const srcVsOrig = cos(sourceEmb, origEmb);
  console.log(`face ${f + 1}: cos(swapped,source)=${toSource.toFixed(3)} cos(swapped,orig-target)=${toOrig.toFixed(3)} cos(source,orig-target)=${srcVsOrig.toFixed(3)}`);
  if (!(toSource > 0.55)) { console.error(`FAIL: face ${f + 1} did not take source identity (sim ${toSource.toFixed(3)} <= 0.55)`); ok = false; }
  const isSelfSwap = srcVsOrig > 0.99; // target IS the source face: identity must be preserved, not moved
  if (!isSelfSwap && !(toSource > toOrig)) { console.error(`FAIL: face ${f + 1} is still closer to its original identity`); ok = false; }
}

if (outPath) {
  const rgb = Buffer.alloc(W * H * 3);
  for (let i = 0; i < W * H; i++) { rgb[i*3] = working[i*4]; rgb[i*3+1] = working[i*4+1]; rgb[i*3+2] = working[i*4+2]; }
  await sharp(rgb, { raw: { width: W, height: H, channels: 3 } }).jpeg({ quality: 92 }).toFile(outPath);
  console.log(`wrote ${outPath}`);
}
console.log(ok ? 'E2E FACE-SWAP VERIFICATION: ALL OK' : 'E2E FACE-SWAP VERIFICATION: FAILED');
// NOTE: onnxruntime-node may print a native teardown warning AFTER this line;
// the verdict above is authoritative. Browser runs use ORT Web, not this.
process.exitCode = ok ? 0 : 1;
