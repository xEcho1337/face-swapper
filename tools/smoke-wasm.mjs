/** Node smoke test for the REAL built WASM artifact (web/pkg/). */
import { readFileSync } from 'node:fs';
import { strict as assert } from 'node:assert';
import init, * as wasm from '../web/pkg/faceswapper.js';

const bytes = readFileSync(new URL('../web/pkg/faceswapper_bg.wasm', import.meta.url));
await init(bytes);

// 1. estimateNorm on the InsightFace 112 template mapped through a known
//    similarity: recovery must be exact.
const ang = (30 * Math.PI) / 180;
const sc = 2, c = Math.cos(ang), s = Math.sin(ang);
const M = [sc * c, -sc * s, 7, sc * s, sc * c, -4];
const tmpl = [38.2946, 51.6963, 73.5318, 51.5014, 56.0252, 71.7366, 41.5493, 92.3655, 70.7299, 92.2041];
const warped = [];
for (let i = 0; i < 5; i++) {
  const x = tmpl[i * 2], y = tmpl[i * 2 + 1];
  warped.push(M[0] * x + M[1] * y + M[2], M[3] * x + M[4] * y + M[5]);
}
// estimateNorm maps landmarks -> template; feed it warped points with a
// template scaled to match: use image_size trick — instead directly verify
// invertAffine round-trip + nms + warp + mask + blend on the artifact.
const inv = wasm.invertAffine(Float32Array.from(M));
assert.equal(inv.length, 6);
for (const [x, y] of [[0, 0], [128, 5], [40, 111]]) {
  const a = M[0] * x + M[1] * y + M[2], b = M[3] * x + M[4] * y + M[5];
  const cx = inv[0] * a + inv[1] * b + inv[2], cy = inv[3] * a + inv[4] * b + inv[5];
  assert.ok(Math.abs(cx - x) < 1e-3 && Math.abs(cy - y) < 1e-3, `roundtrip ${x},${y}`);
}

// 2. estimateNorm smoke: template landmarks -> ~identity-ish for size 112.
const Mest = wasm.estimateNorm(Float32Array.from(tmpl), 112);
assert.equal(Mest.length, 6);
assert.ok(Math.abs(Mest[0] - 1) < 1e-3 && Math.abs(Mest[4] - 1) < 1e-3);

// 3. warp identity round-trips bytes.
const src = new Uint8Array([10, 0, 0, 255, 20, 0, 0, 255, 30, 0, 0, 255, 40, 0, 0, 255]);
const id = wasm.warpRgba(src, 4, 1, Float32Array.from([1, 0, 0, 0, 1, 0]), 4, 1);
assert.deepEqual(Array.from(id), Array.from(src));

// 4. NMS: overlapping lower-score box suppressed.
const keep = wasm.nmsBoxes(
  Float32Array.from([0, 0, 10, 10, 1, 1, 11, 11, 50, 50, 60, 60]),
  Float32Array.from([0.9, 0.8, 0.7]),
  0.4,
);
assert.deepEqual(Array.from(keep), [0, 2]);

// 5. Crop mask: weights in [0,1], opaque center, feathered border.
const S = 128;
const white = new Float32Array(S * S).fill(255);
for (let y = 0; y < S; y++)
  for (let x = 0; x < S; x++) if (x < 2 || y < 2 || x >= S - 2 || y >= S - 2) white[y * S + x] = 0;
const mask = wasm.buildCropMask(white, S);
assert.equal(mask.length, S * S);
assert.ok(mask.every((v) => v >= 0 && v <= 1));
assert.ok(mask[64 * S + 64] > 0.99 && mask[0] < 0.01);

// 6. Full paste-back loop at full res: warp out, warp back, blend.
const W = 64, H = 64;
const face = new Uint8Array(W * H * 4).fill(200);
const Mw = wasm.estimateNorm(Float32Array.from([20, 20, 44, 20, 32, 34, 24, 48, 40, 48]), 128);
const crop = wasm.warpRgba(face, W, H, Mw, 128, 128);
const IM = wasm.invertAffine(Mw);
const back = wasm.warpRgba(crop, 128, 128, IM, W, H);
const wm = new Float32Array(128 * 128).fill(255);
const mCrop = wasm.buildCropMask(wm, 128);
const mRgba = new Uint8Array(128 * 128 * 4);
mCrop.forEach((v, i) => { const b = Math.round(v * 255); mRgba[i*4]=mRgba[i*4+1]=mRgba[i*4+2]=b; mRgba[i*4+3]=255; });
const mBack = wasm.warpRgba(mRgba, 128, 128, IM, W, H);
const mBackF = new Float32Array(W * H);
for (let i = 0; i < W * H; i++) mBackF[i] = mBack[i * 4] / 255;
const back3 = new Uint8Array(W * H * 3);
for (let i = 0; i < W * H; i++) { back3[i*3]=back[i*4]; back3[i*3+1]=back[i*4+1]; back3[i*3+2]=back[i*4+2]; }
const comp = wasm.blendFullres(face, back3, mBackF, W, H);
assert.equal(comp.length, W * H * 4);

console.log('WASM artifact smoke test: ALL OK');
