# AI models

FaceSwapper runs three ONNX models, all **locally in your browser** via ONNX
Runtime Web (WebGPU preferred, WASM fallback). **No weights are bundled in
this repository** — for licensing reasons (below) and size (up to ~554 MB).
Copy them next to `manifest.json` (see `tools/fetch-models.mjs` for the
redistributable ones).

## 1. Face detector — SCRFD-2.5G-BNKPS

- Model: `scrfd_2.5g_bnkps.onnx` (~3.3 MB)
- Format: ONNX (opset 11, dynamic H/W axes)
- License: **InsightFace model license — non-commercial research only** unless
  you arrange otherwise. The InsightFace *code* is MIT; the *pretrained
  weights* are not. Do not redistribute the weights commercially.
- Input: `input.1`, float32 NCHW. Verified 2026-09-07 with the `onnx`
  package: the widely mirrored export is **fixed-shape `[1,3,640,640]`**
  (outputs `score_8 [1,12800,1]`, `score_16 [1,3200,1]`, `score_32
  [1,800,1]`, `bbox_* [1,N,4]`, `kps_* [1,N,10]` — 12800 = 80×80×2
  anchors, confirming strides `[8,16,32]` × 2 anchors). Dynamic InsightFace
  exports accept any H/W; the app always feeds a 640×640 letterbox so both
  work. Scores are post-sigmoid (threshold raw values, no extra sigmoid —
  exactly like `scrfd.py`).
- Input dtype/layout: float32, RGB, `(x − 127.5) / 128.0`.
- Outputs (9 tensors, batched `[1,N,…]` or unbatched `[N,…]` — both handled):
  `score_8/16/32`, `bbox_8/16/32`, `kps_8/16/32` for strides `[8,16,32]`,
  2 anchors per stride.
- Preprocessing: letterbox + RGB + mean/std above (exact port of
  `detection/scrfd/tools/scrfd.py::forward`).
- Postprocessing: anchor-center grid, `distance2bbox`/`distance2kps`,
  confidence threshold (UI, default 0.5), greedy NMS @ IoU 0.4 with the `+1`
  pixel convention, coordinates divided by `det_scale`.
- Browser backend: WebGPU, WASM fallback. Expected memory: ~150 MB.

## 2. Face recognizer — ArcFace R50 (`w600k_r50.onnx`, buffalo_l pack)

- Model: `w600k_r50.onnx` (~250 MB)
- Format: ONNX
- License: **InsightFace model license** — for the open-sourced recognition
  models contact `recognition-oss-pack@insightface.ai` for licensing terms.
  Not bundled; install separately.
- Input: float32 `[1,3,112,112]`; crop via `norm_crop` (112 template);
  RGB, `(x − 127.5) / 127.5`. VERIFIED 2026-09-07 on the official
  `buffalo_l.zip` (release v0.7): input `input.1` FLOAT `[None,3,112,112]`,
  output `683` FLOAT `[1,512]`; first-8 graph nodes are
  `Conv/PRelu/BatchNorm…` (no Sub/Mul prefix), which per
  `arcface_onnx.py` means exactly mean=127.5, std=127.5.
- Output: single float32 `[1,512]` embedding, L2-normalized.
- Preprocessing/Postprocessing: per `model_zoo/arcface_onnx.py`.
- Browser backend: WebGPU, WASM fallback.

## 3. Face swap — `reswapper-1019500.onnx` (ReSwapper, default)

- Model: `reswapper-1019500.onnx` (~554 MB, `somanchiu/reswapper` on
  HuggingFace, code at `somanchiu/ReSwapper`)
- Format: ONNX, opset 11
- License: **AGPL-3.0** (repo and model card agree) — genuinely open-source,
  but **copyleft, not permissive**. Consequences for this project: the weights
  are NEVER bundled and never auto-downloaded into the repo; the operator
  installs them separately next to `manifest.json`; license notices must be
  preserved. Serving the weights yourself from your own host counts as
  conveying them — read the AGPL (notably the network-use source-offer
  section) before doing that; the "bring your own file" setup avoids the
  issue by default.
- Provenance: the author's own disclosure says training used inswapper
  outputs among its targets. Documented here as-is; this project claims and
  conveys no InsightFace rights.
- Input (matched **by shape**, robust to renamed exports) — VERIFIED
  2026-09-07 with the `onnx` package: `target` FLOAT `[1,3,128,128]`
  (128px `norm_crop2` crop, RGB, `x/255`) + `source` FLOAT `[1,512]`
  (`normalize(normed_embedding @ emap)`); output `output` FLOAT
  `[1,3,128,128]` (`[0,1]` RGB); 65 initializers, last float32 `[512,512]`
  (= emap); contract-identical to `inswapper_128` by design ("usable with
  the original INSwapper class").
- Quality (measured, `tools/verify-swap.mjs`, 6-face photo, ArcFace cosine
  to source identity): **0.58–0.87**, every face closer to the source than
  to its original identity. The earlier `reswapper-429500` checkpoint scored
  0.48–0.84 on the same protocol — hence `-1019500` is the default.
- The `emap` (last graph initializer, float `[512,512]`) cannot be read via
  ORT Web, so it is extracted **once, offline, from your copy**:
  `python3 tools/extract_emap.py public/models/reswapper-1019500.onnx public/models/reswapper.emap.json`
- Browser backend: WebGPU strongly recommended; WASM fallback works but is
  slow on this 554 MB graph. Expected memory: ~1.2 GB. Known quirk: unlike
  inswapper (static output shape), reswapper exports declare *symbolic*
  output dims — some WebGPU builds reject dynamic shapes, in which case the
  app automatically falls back to WASM (still 100% local, ~1–2 s/face on a
  modern laptop CPU).

### Higher-resolution option: ReSwapper 256 (`reswapper_256-1567500.onnx`)

Same author, same AGPL-3.0 terms, same inswapper-compatible contract at
256px (verified 2026-09-08: inputs `target [1,3,256,256]` + `source [1,512]`,
output symbolic `[1,3,256,256]`, 65 initializers, last float32 `[512,512]`).
Install as `public/models/reswapper_256-1567500.onnx` plus
`public/models/reswapper_256.emap.json` (same `extract_emap.py`), then pick
"ReSwapper 256" in the UI. 4x the pixels of the 128 models for the face
region; slower on WASM, WebGPU recommended.

### Gated alternative: InsightFace `inswapper_128.onnx`

Same tensor contract (verified: inputs `target`/`source`, output `output`,
76 initializers, last float32 `[512,512]`), stronger measured identity
transfer (**0.84–0.96** on the same protocol) — but **REQUIRES A SEPARATE
LICENSE: contact `contact@insightface.ai`**. Do not assume MIT. Never
auto-downloaded, never bundled; `tools/fetch-models.mjs` only fetches it
with an explicit `--with-inswapper <url> --i-hold-inswapper-rights`.
To use it, point the `swapper.path`/`emap` entries at your files (same
`extract_emap.py` works — it accepts any file whose last initializer is
float32 `[512,512]`).

In the app UI (card 2, “Face swap model”) you can switch between ReSwapper
(default) and `inswapper_128` at runtime. The inswapper option is
license-gated behind an acknowledgment checkbox and needs YOUR licensed copy
as `public/models/inswapper_128.onnx` plus
`public/models/inswapper_128.emap.json` (extract once with the same tool):
its official GitHub release URL answers with a CORS-less redirect that
browsers refuse, so the remote fallback alone is not reliable.

## License-compatibile alternatives

If you cannot obtain inswapper rights, do **not** silently swap in an
“equivalent” file: tensor contracts differ and results would be wrong. The
honest paths are: (a) obtain the license, (b) train/port a swap model whose
license you hold and adapt `web/inference.js` + `manifest.json` to its
*verified* contract. The pipeline validates shapes at warmup and refuses to
run mismatched models.
