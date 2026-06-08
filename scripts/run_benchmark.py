#!/usr/bin/env python3
"""Benchmark backends for a model (§7; roadmap step 4).

Times PyTorch / ONNX Runtime / TensorRT on the SAME input and prints the
latency / FPS / P95 / P99 / speedup table, saving it to results/benchmark/.
Run on the VM for the server-GPU numbers and on the Jetson for the edge
numbers; the two JSONs together form the server-vs-edge comparison (§14).

    # on the VM:
    python scripts/run_benchmark.py --model rgb_seg --backends pytorch onnx
    # on the Jetson:
    python scripts/run_benchmark.py --model rgb_seg --backends trt --tag jetson
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import load_config


def make_runner(model_key: str, backend: str, cfg: dict):
    """Build a ``(runner, sample, sync)`` triple for one backend.

    The benchmark only needs a no-arg-ish callable plus a fixed input. We feed
    a representative input at the model's configured ``imgsz``.

    TODO(step 4): finalize per-backend preprocessing so the timed call matches
    the real inference path:
      • pytorch : the wrapped model's forward on a CUDA tensor (+ cuda sync)
      • onnx    : ``ort_session.run`` on a numpy batch (synchronous; sync=None)
      • trt     : ``TRTInference.infer`` (+ its ``.sync`` as the barrier)
    """
    import numpy as np

    m = cfg["models"][model_key]
    imgsz = cfg["project"]["rgb_imgsz"] if model_key == "rgb_seg" else cfg["project"]["sonar_imgsz"]
    sample = np.zeros((imgsz, imgsz, 3), dtype=np.uint8)  # placeholder input

    if backend in ("pytorch", "onnx", "engine"):
        from src.inference.rgb_segmenter import RGBSegmenter
        from src.inference.sonar_detector import SonarDetector

        weights = m["engine"] if backend == "engine" else m["weights"]
        cls = RGBSegmenter if m["task"] == "segment" else SonarDetector
        model = cls(weights, backend=backend, conf=m["conf"], iou=m["iou"], imgsz=imgsz)
        sync = None
        if backend == "pytorch":
            try:
                import torch
                sync = torch.cuda.synchronize if torch.cuda.is_available() else None
            except Exception:
                sync = None
        return (lambda x: model.infer(x)), sample, sync

    if backend == "trt":
        from src.optimization.tensorrt_inference import TRTInference

        runner = TRTInference(m["engine"], precision=cfg["optimization"]["precision"])
        return (lambda x: runner.infer(x)), sample, runner.sync

    raise ValueError(f"unknown backend {backend!r}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Benchmark inference backends.")
    ap.add_argument("--config", default="configs/model_config.yaml")
    ap.add_argument("--model", default="rgb_seg", choices=["rgb_seg", "sonar_det"])
    ap.add_argument("--backends", nargs="+", default=["pytorch", "onnx", "trt"])
    ap.add_argument("--baseline", default="pytorch", help="backend to compute speedup against")
    ap.add_argument("--tag", default="server", help="label for the output file (e.g. server/jetson)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    bcfg = cfg["benchmark"]

    from src.optimization.benchmark import (add_speedups, benchmark, save_json, to_dataframe)

    results = []
    for backend in args.backends:
        print(f"[bench] {args.model} / {backend} …")
        runner, sample, sync = make_runner(args.model, backend, cfg)
        results.append(benchmark(
            runner, sample, name=backend,
            warmup_iters=bcfg["warmup_iters"], measure_iters=bcfg["measure_iters"],
            percentiles=bcfg["percentiles"], sync=sync,
            meta={"model": args.model, "tag": args.tag},
        ))

    if args.baseline in args.backends:
        add_speedups(results, args.baseline)

    print("\n" + to_dataframe(results).to_string(index=False))
    out = save_json(results, f"results/benchmark/{args.model}_{args.tag}.json")
    print(f"\n[bench] saved → {out}")


if __name__ == "__main__":
    main()
