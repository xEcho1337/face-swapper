#!/usr/bin/env node
/**
 * Download the REDISTRIBUTABLE model files into public/models/.
 *
 * Included: SCRFD + buffalo_l recognizer (InsightFace non-commercial
 * research terms) and ReSwapper-1019500 (AGPL-3.0 — open source; keep its
 * license notices, see MODELS.md).
 *
 * Deliberately DOES NOT fetch inswapper_128.onnx: that model requires a
 * separate license (contact@insightface.ai). Pass an explicit
 * `--with-inswapper <url>` only if YOU hold the rights — the script prints
 * the licensing reminder and requires `--i-hold-inswapper-rights`.
 *
 * After fetching the swapper, extract its emap sidecar:
 *   python3 tools/extract_emap.py public/models/reswapper-1019500.onnx public/models/reswapper.emap.json
 *
 * Usage:
 *   node tools/fetch-models.mjs [--models-dir public/models]
 *   node tools/fetch-models.mjs --with-inswapper https://…/inswapper_128.onnx --i-hold-inswapper-rights
 */
import { createWriteStream, existsSync, mkdirSync, statSync } from 'node:fs';
import { get } from 'node:https';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const here = dirname(fileURLToPath(import.meta.url));
const args = process.argv.slice(2);
const opt = (name, dflt) => {
  const i = args.indexOf(name);
  return i >= 0 && args[i + 1] ? args[i + 1] : dflt;
};
const flag = (name) => args.includes(name);

const modelsDir = resolve(here, '..', opt('--models-dir', 'public/models'));
mkdirSync(modelsDir, { recursive: true });

// Sources are mirrors of the exact upstream InsightFace exports. SCRFD +
// YuNet-class weights are small; the buffalo_l recognizer comes from the
// official InsightFace release bundle.
const FILES = [
  {
    name: 'scrfd_2.5g_bnkps.onnx',
    // NOTE: use the direct media URL, not the github.com/.../raw/... page URL.
    // The page URL answers with a 302 that carries no CORS headers, so
    // browsers on GitHub Pages refuse to follow it ("Failed to fetch") and
    // raw.githubusercontent/jsDelivr only serve the 132-byte Git-LFS pointer.
    // This media URL serves the real 3.3 MB bytes with ACAO: * (verified).
    url: 'https://media.githubusercontent.com/media/cysin/scrfd_onnx/refs/heads/main/scrfd_2.5g_bnkps.onnx',
    minBytes: 1_000_000,
    license: 'InsightFace model license (non-commercial research unless arranged)',
  },
  {
    name: 'w600k_r50.onnx',
    url: 'https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip',
    minBytes: 10_000_000,
    license: 'InsightFace model license (contact recognition-oss-pack@insightface.ai)',
    note: 'This is a .zip containing the buffalo_l pack — unzip and keep w600k_r50.onnx as public/models/w600k_r50.onnx',
    zip: true,
  },
  {
    name: 'reswapper-1019500.onnx',
    url: 'https://huggingface.co/somanchiu/reswapper/resolve/main/reswapper-1019500.onnx',
    minBytes: 400_000_000,
    license: 'AGPL-3.0 (somanchiu/ReSwapper, code + weights). Keep license notices; see MODELS.md.',
    note: 'Default open swap model. Then run: python3 tools/extract_emap.py public/models/reswapper-1019500.onnx public/models/reswapper.emap.json',
  },
];

function download(url, dest) {
  return new Promise((resolveP, reject) => {
    get(url, { headers: { 'User-Agent': 'faceswapper-fetch-models' } }, (res) => {
      if (res.statusCode >= 300 && res.statusCode < 400 && res.headers.location) {
        res.resume();
        return download(res.headers.location, dest).then(resolveP, reject);
      }
      if (res.statusCode !== 200) {
        res.resume();
        return reject(new Error(`HTTP ${res.statusCode} for ${url}`));
      }
      const out = createWriteStream(dest);
      res.pipe(out);
      out.on('finish', () => resolveP());
      out.on('error', reject);
    }).on('error', reject);
  });
}

for (const f of FILES) {
  const dest = resolve(modelsDir, f.zip ? f.name + '.zip' : f.name);
  if (existsSync(dest) && statSync(dest).size >= f.minBytes) {
    console.log(`ok (cached): ${dest}`);
    continue;
  }
  console.log(`downloading ${f.url}\n  license: ${f.license}`);
  if (f.note) console.log(`  note: ${f.note}`);
  try {
    await download(f.url, dest);
    const size = statSync(dest).size;
    if (size < f.minBytes) throw new Error(`suspiciously small (${size} bytes) — mirror may have changed`);
    console.log(`ok: ${dest} (${(size / 1048576).toFixed(1)} MB)`);
  } catch (e) {
    console.error(`FAILED: ${f.name}: ${e.message}`);
    console.error('Download it manually from the upstream InsightFace release and verify the tensor contract in MODELS.md.');
  }
}

const inswapperUrl = opt('--with-inswapper', null);
if (inswapperUrl) {
  if (!flag('--i-hold-inswapper-rights')) {
    console.error('\nREFUSED: inswapper_128.onnx requires a separate license (contact@insightface.ai).');
    console.error('Re-run with --i-hold-inswapper-rights only if you actually hold redistribution/usage rights.');
    process.exit(1);
  }
  console.log('\nYou asserted inswapper rights. Downloading (554 MB, takes a while)…');
  const dest = resolve(modelsDir, 'inswapper_128.onnx');
  await download(inswapperUrl, dest);
  console.log(`ok: ${dest}`);
  console.log('Next: python3 tools/extract_emap.py public/models/inswapper_128.onnx public/models/inswapper_128.emap.json');
} else {
  console.log('\nSkipped inswapper_128.onnx (license-gated, never auto-fetched). Provide your licensed copy as public/models/inswapper_128.onnx, then extract the emap matrix (see MODELS.md).');
}
