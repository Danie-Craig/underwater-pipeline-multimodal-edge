"""Backend-agnostic inference benchmark (§7).

Times any callable "runner" — PyTorch, ONNX Runtime, or TensorRT — under one
harness so the PyTorch / ONNX / TensorRT-FP16 rows and the server-GPU vs
edge-device comparison are measured identically. Reports average + std
latency, FPS, P95, P99, and speedup over a chosen baseline.

This module is complete; the runners that feed it come from the inference
wrappers and ``tensorrt_inference``.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np


@dataclass
class BenchmarkResult:
    name: str
    iters: int
    latency_ms_avg: float
    latency_ms_std: float
    latency_ms_min: float
    fps: float
    percentiles_ms: dict[int, float] = field(default_factory=dict)
    speedup_vs_baseline: float | None = None
    meta: dict[str, Any] = field(default_factory=dict)


def _percentile(samples_ms: np.ndarray, p: int) -> float:
    return float(np.percentile(samples_ms, p))


def benchmark(
    runner: Callable[[Any], Any],
    sample: Any,
    name: str,
    warmup_iters: int = 50,
    measure_iters: int = 500,
    percentiles: Sequence[int] = (95, 99),
    sync: Callable[[], None] | None = None,
    meta: dict[str, Any] | None = None,
) -> BenchmarkResult:
    """Benchmark ``runner(sample)`` over many iterations.

    Parameters
    ----------
    runner : one inference call; its return value is ignored.
    sample : the (already-preprocessed) input passed to ``runner`` each iter.
    sync   : optional barrier called after each inference so async backends are
             timed correctly — e.g. ``torch.cuda.synchronize`` for PyTorch, or
             a CUDA stream sync for the TensorRT runner. Leave ``None`` for
             synchronous backends (ONNX Runtime).
    """
    # Warmup: trigger lazy init, autotuning, and clock ramp; not timed.
    for _ in range(warmup_iters):
        runner(sample)
    if sync:
        sync()

    samples_ms = np.empty(measure_iters, dtype=np.float64)
    for i in range(measure_iters):
        t0 = time.perf_counter()
        runner(sample)
        if sync:
            sync()
        samples_ms[i] = (time.perf_counter() - t0) * 1e3

    avg = float(samples_ms.mean())
    return BenchmarkResult(
        name=name,
        iters=measure_iters,
        latency_ms_avg=avg,
        latency_ms_std=float(samples_ms.std()),
        latency_ms_min=float(samples_ms.min()),
        fps=(1000.0 / avg if avg > 0 else float("inf")),
        percentiles_ms={int(p): _percentile(samples_ms, int(p)) for p in percentiles},
        meta=meta or {},
    )


def add_speedups(results: list[BenchmarkResult], baseline_name: str) -> list[BenchmarkResult]:
    """Fill ``speedup_vs_baseline`` (baseline_latency / this_latency) in place."""
    base = next((r for r in results if r.name == baseline_name), None)
    if base is None:
        raise ValueError(f"baseline {baseline_name!r} not among results")
    for r in results:
        r.speedup_vs_baseline = base.latency_ms_avg / r.latency_ms_avg
    return results


def to_dataframe(results: list[BenchmarkResult]):
    """Render results as a pandas DataFrame (one row per backend)."""
    import pandas as pd

    rows = []
    for r in results:
        row = {
            "backend": r.name,
            "avg_ms": round(r.latency_ms_avg, 3),
            "std_ms": round(r.latency_ms_std, 3),
            "fps": round(r.fps, 1),
        }
        for p, v in sorted(r.percentiles_ms.items()):
            row[f"p{p}_ms"] = round(v, 3)
        if r.speedup_vs_baseline is not None:
            row["speedup"] = round(r.speedup_vs_baseline, 2)
        rows.append(row)
    return pd.DataFrame(rows)


def save_json(results: list[BenchmarkResult], path: str | Path) -> Path:
    """Persist results to ``results/benchmark/*.json`` for the writeup."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump([asdict(r) for r in results], fh, indent=2)
    return path
