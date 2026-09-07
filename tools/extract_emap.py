#!/usr/bin/env python3
"""Extract the `emap` projection matrix from a swap-model ONNX file.

InsightFace-style swappers (inswapper_128 and the contract-compatible
ReSwapper) compute the swap latent as::

    latent = normalize(normed_embedding @ emap)

where `emap` is the LAST initializer of the ONNX graph (float [512,512]).
ONNX Runtime Web cannot read graph initializers, so FaceSwapper loads this
matrix from a sidecar JSON file instead. Run this script ONCE, offline, on a
model file you are entitled to use::

    pip install onnx numpy
    python3 tools/extract_emap.py public/models/reswapper-1019500.onnx public/models/reswapper.emap.json

The script verifies the initializer is float32 [512,512] and refuses to write
anything else — this guards against silently using a wrong/foreign file whose
tensor contract would produce garbage swaps.
"""
import json
import sys

try:
    import numpy as np
    import onnx
    from onnx import numpy_helper
except ImportError:
    sys.exit("need `pip install onnx numpy` (one-time offline tooling; not part of the browser app)")


def main() -> None:
    if len(sys.argv) != 3 or sys.argv[1] in ("-h", "--help"):
        sys.exit("usage: extract_emap.py <swap-model.onnx> <out.emap.json>")
    src, dst = sys.argv[1], sys.argv[2]

    model = onnx.load(src)
    if not model.graph.initializer:
        sys.exit("refusing: model has no initializers — not a compatible swap model?")
    emap = numpy_helper.to_array(model.graph.initializer[-1])
    print(f"last initializer: dtype={emap.dtype} shape={emap.shape}")
    if emap.dtype != np.float32 or tuple(emap.shape) != (512, 512):
        sys.exit(
            "refusing: expected float32 [512,512]. "
            "This file does not match the swap-model contract — do not use it."
        )
    payload = {"shape": [512, 512], "data": emap.astype(np.float64).ravel().tolist()}
    with open(dst, "w") as f:
        json.dump(payload, f)
    print(f"wrote {dst} ({len(payload['data'])} floats)")


if __name__ == "__main__":
    main()
