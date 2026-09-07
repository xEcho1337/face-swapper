#!/usr/bin/env node
/**
 * Self-host the ONNX Runtime Web WASM assets instead of using the CDN.
 * Copies node_modules/onnxruntime-web/dist/*.wasm (+ .mjs) to public/ort/
 * and prints the worker init snippet. No pixels involved — library code only.
 *
 *   npm run vendor-ort   # then set ortWasmBase: './ort/' in app boot
 */
import { copyFileSync, existsSync, mkdirSync, readdirSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const here = dirname(fileURLToPath(import.meta.url));
const src = resolve(here, '..', 'node_modules/onnxruntime-web/dist');
const dst = resolve(here, '..', 'public/ort');
if (!existsSync(src)) {
  console.error('node_modules/onnxruntime-web/dist not found — run `npm install` first.');
  process.exit(1);
}
mkdirSync(dst, { recursive: true });
for (const f of readdirSync(src)) {
  if (/\.(wasm|mjs|js)$/.test(f)) {
    copyFileSync(resolve(src, f), resolve(dst, f));
    console.log(`vendored ${f}`);
  }
}
console.log('\nSelf-hosted. In web/app.js boot, pass ortWasmBase: "./ort/" AND ortModuleUrl: "./ort/ort.webgpu.bundle.min.mjs" to the worker init message.');
