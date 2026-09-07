# EMAP sidecar license

`reswapper.emap.json` in this directory is **not** original project code. It
is the last-initializer projection matrix (512×512 floats) extracted
verbatim from the ReSwapper weights (`reswapper-1019500.onnx`) with
`tools/extract_emap.py`, and is therefore covered by the same license as
those weights:

- **GNU Affero General Public License v3.0 (AGPL-3.0)**
- Copyright holder: the ReSwapper author(s) — see
  https://github.com/somanchiu/ReSwapper (repo LICENSE) and
  https://huggingface.co/somanchiu/reswapper (model card)
- Keep this notice next to the file wherever it is distributed.

Everything else in this repository remains under its own MIT license.
