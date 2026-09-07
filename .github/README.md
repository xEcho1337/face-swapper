# FaceSwapper

**Swap any face, right in your browser.** A face-swap web app that runs **100%
in your browser**: open the page, pick a source face (Charlie Kirk by
default, or any face you upload), drop in a photo, every detected face is
replaced with it, and you download the result.
No backend. No uploads.

## Privacy

- All inference runs in a Web Worker in your tab (ONNX Runtime Web).
- The Rust/WASM module touches raw pixels locally; nothing is logged.
- Network activity is limited to: the page itself, the JS/WASM/ORT library
  code, and model weights — served from the same host when present,
  otherwise fetched once from hash-verified public mirrors and cached
  on-device. Pixels are never sent anywhere.
- There is no upload endpoint, no analytics carrying pixels, no telemetry
  with embeddings/landmarks. The project has no server component at all.

## AI models

See [`public/models/MODELS.md`](public/models/MODELS.md) for the full
per-model sheets (format, license, input/output shapes, dtypes,
preprocessing, postprocessing, backend, memory). Summary:

| Detect | SCRFD-2.5G-BNKPS | research-only weights |
| Embed | ArcFace R50 (`buffalo_l`) | research-only weights |
| Swap | ReSwapper-1019500 | AGPL-3.0 |

Weights are never bundled: local-first, remote-mirror fallback, cached
on-device. InsightFace weights = non-commercial research unless arranged
(`recognition-oss-pack@insightface.ai`); inswapper needs a separate license
(`contact@insightface.ai`). See `MODELS.md` + `EMAP-LICENSE.md`.

## Building

```bash
npm run build       # wasm-pack --release + vite build → dist/
npm run preview     # serve dist/ locally
```

`dist/` is a plain static site: deploy to any static host. No Node/Python/
Rust server, no database, no API at runtime. For COOP/COEP (ORT threads when
self-hosting) set `Cross-Origin-Opener-Policy: same-origin` and
`Cross-Origin-Embedder-Policy: require-corp`.

## Running locally

Any static server over `dist/`, e.g. `npx serve dist`. WebGPU needs a secure
context (`https` or `localhost`). Without WebGPU the app falls back to WASM
automatically and says so in the header badge.

## Deployment (GitHub Pages)

Push the repo (weights stay out — gitignored). Commit the portrait with
`git add -f public/source/example.jpg`, enable Pages → GitHub Actions
(workflow in `.github/workflows/`). Visitors download weights once from the
mirrors, then run from on-device cache. No COOP/COEP on Pages → the WASM
fallback runs single-threaded there (WebGPU unaffected).

## Responsible use

Full policy: `web/responsible-use.html` (also linked in the UI). No
server-side moderation exists by design (client-side app) — the policy binds
you as the publisher of anything you share.
