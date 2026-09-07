#!/usr/bin/env node
/**
 * REAL end-to-end verification of the SCRFD decode port (not a mock):
 * runs the actual scrfd_2.5g_bnkps.onnx weights with onnxruntime-node on a
 * real photo, decoding with the EXACT functions shipped in web/inference.js
 * (decodeScrfd, nmsJs, rgbaToNchw).
 *
 * One-time setup (verification tooling only — not part of the app):
 *   npm i --no-save onnxruntime-node@1.22.0 sharp
 *   node tools/fetch-models.mjs   # or place scrfd_2.5g_bnkps.onnx manually
 *   node tools/verify-detection.mjs public/models/scrfd_2.5g_bnkps.onnx <photo> [threshold]
 *
 * Exit code 0 + printed faces = the port works against real weights.
 */
import { readFileSync } from 'node:fs';
import * as ort from 'onnxruntime-node';
import sharp from 'sharp';
import { decodeScrfd, nmsJs, rgbaToNchw } from '../web/inference.js';

const [modelPath, imagePath, thrArg] = process.argv.slice(2);
if (!modelPath || !imagePath) {
  console.error('usage: verify-detection.mjs <model.onnx> <photo> [threshold]');
  process.exit(2);
}
const threshold = Number(thrArg || 0.5);

// Letterbox EXACTLY like the app + SCRFD.detect: aspect fit, top-left, zero pad.
const S = 640;
const meta = await sharp(imagePath).metadata();
const { width: W, height: H } = meta;
const imRatio = H / W;
let newW, newH;
if (imRatio > 1) { newH = S; newW = Math.max(1, Math.round(newH / imRatio)); }
else { newW = S; newH = Math.max(1, Math.round(newW * imRatio)); }
const detScale = newH / H;
const { data, info } = await sharp(imagePath)
  .removeAlpha()
  .resize(newW, newH, { fit: 'fill', kernel: 'lanczos3' })
  .extend({ top: 0, left: 0, bottom: S - newH, right: S - newW, background: { r: 0, g: 0, b: 0 } })
  .raw()
  .toBuffer({ resolveWithObject: true });
if (info.width !== S || info.height !== S) throw new Error('letterbox failed');
// RGB -> RGBA for rgbaToNchw parity with the browser path.
const rgba = new Uint8Array(S * S * 4);
for (let i = 0; i < S * S; i++) {
  rgba[i * 4] = data[i * 3];
  rgba[i * 4 + 1] = data[i * 3 + 1];
  rgba[i * 4 + 2] = data[i * 3 + 2];
  rgba[i * 4 + 3] = 255;
}

const session = await ort.InferenceSession.create(readFileSync(modelPath), { executionProviders: ['cpu'] });
console.log('inputs:', session.inputNames, 'outputs:', session.outputNames);
const chw = rgbaToNchw(rgba, S, S, 127.5, 128.0);
const feeds = { [session.inputNames[0]]: new ort.Tensor('float32', chw, [1, 3, S, S]) };
const t0 = performance.now();
const out = await session.run(feeds);
console.log(`inference: ${Math.round(performance.now() - t0)}ms (onnxruntime-node CPU)`);
// Adapt ORT-node tensors to the {dims, data, type} shape decodeScrfd expects.
const adapted = {};
for (const [k, v] of Object.entries(out)) adapted[k] = { dims: Array.from(v.dims), data: v.data, type: v.type };
const { boxes, scores, kps } = decodeScrfd(adapted, S, S, threshold, new Map());
const keep = nmsJs(boxes, scores, 0.4);
console.log(`raw candidates >= ${threshold}: ${boxes.length}, after NMS: ${keep.length}`);
for (const i of keep) {
  const b = boxes[i].map((v) => Math.round((v / detScale) * 10) / 10);
  const l = kps[i].map((p) => p.map((v) => Math.round((v / detScale) * 10) / 10));
  console.log(`face score=${scores[i].toFixed(3)} bbox=[${b}] kps=${JSON.stringify(l)}`);
}
