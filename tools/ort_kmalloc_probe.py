"""Minimal ONNXRuntime CUDA slab-leak probe for Jetson.

Run inside a maintenance window (stop your ALPR API first), then monitor
kmalloc-128 growth with tools/gpu_slab_watch.sh in another shell.
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import onnxruntime as ort


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", required=True, help="Path to ONNX model")
    parser.add_argument(
        "--provider", default="cuda", choices=["cuda", "cpu"], help="Execution provider"
    )
    parser.add_argument("--iters", type=int, default=20000, help="Inference iterations")
    args = parser.parse_args()

    providers = ["CPUExecutionProvider"]
    if args.provider == "cuda":
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

    print(f"[probe] loading {args.onnx} with providers={providers}")
    sess = ort.InferenceSession(args.onnx, providers=providers)

    inp = sess.get_inputs()[0]
    shape = [d if isinstance(d, int) else 1 for d in inp.shape]  # replace dynamic dims
    dummy = np.random.randint(0, 255, size=shape, dtype=np.uint8)

    print(f"[probe] input name={inp.name}, shape={shape}, starting loop...")
    for i in range(args.iters):
        _ = sess.run(None, {inp.name: dummy})
        if i % 500 == 0:
            print(f"[probe] iter={i}")
            time.sleep(0.1)


if __name__ == "__main__":
    main()
