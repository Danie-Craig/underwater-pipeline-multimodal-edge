"""TensorRT engine build + inference — JETSON-SIDE (§7, §11C).

A TensorRT engine is specific to the GPU and the TensorRT version, so it
**must** be built on the Orin itself — never checked in, never built on the
server. This module: (1) builds an FP16 (optionally INT8) engine from an ONNX
file on-device, caching it under ``models/trt_cache/``, and (2) runs inference
through it, exposing a plain ``infer(input)`` so it can be wrapped by the
inference classes and timed by the benchmark harness.

The build flow is outlined with the real TensorRT 10.x calls; host/device
buffer management and output decoding are completed on-device in roadmap
step 4 (they depend on the exact engine I/O shapes).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


class TRTInference:
    """Build (if needed) and run a TensorRT engine for one model."""

    def __init__(
        self,
        onnx_or_engine: str | Path,
        precision: str = "fp16",
        workspace_gib: int = 4,
        int8_calibrator: Any | None = None,
    ) -> None:
        self.path = Path(onnx_or_engine)
        self.precision = precision
        self.workspace_gib = workspace_gib
        self.int8_calibrator = int8_calibrator
        self.engine = None
        self.context = None

        if self.path.suffix == ".engine":
            self.engine = self._load_engine(self.path)
        else:
            engine_path = self._engine_cache_path(self.path)
            if engine_path.exists():
                self.engine = self._load_engine(engine_path)
            else:
                self.engine = self.build_engine(self.path, engine_path)
        self._allocate_buffers()

    # ------------------------------------------------------------------ #
    @staticmethod
    def _engine_cache_path(onnx_path: Path) -> Path:
        root = Path(__file__).resolve().parents[2]
        return root / "models" / "trt_cache" / f"{onnx_path.stem}_fp16.engine"

    # ------------------------------------------------------------------ #
    def build_engine(self, onnx_path: Path, engine_path: Path) -> Any:
        """Build a TensorRT engine from ONNX and cache it on the Jetson.

        Outline (TensorRT 10.x):
          logger  = trt.Logger(trt.Logger.WARNING)
          builder = trt.Builder(logger)
          network = builder.create_network(EXPLICIT_BATCH)
          parser  = trt.OnnxParser(network, logger)         # parse onnx_path
          config  = builder.create_builder_config()
          config.set_memory_pool_limit(WORKSPACE, workspace_gib << 30)
          if precision == "fp16": config.set_flag(trt.BuilderFlag.FP16)
          if precision == "int8":
              config.set_flag(trt.BuilderFlag.INT8)
              config.int8_calibrator = self.int8_calibrator   # §7 stretch goal
          serialized = builder.build_serialized_network(network, config)
          engine_path.write_bytes(serialized); return deserialize(serialized)

        TODO(step 4, on-device): implement the above and persist to
        ``engine_path``. Confirm TensorRT 10.x / CUDA 12.6 first.
        """
        raise NotImplementedError(
            "TensorRT engine build runs on the Jetson — roadmap step 4."
        )

    # ------------------------------------------------------------------ #
    def _load_engine(self, engine_path: Path) -> Any:
        """Deserialize a cached ``.engine`` from disk.

        TODO(step 4): trt.Runtime(logger).deserialize_cuda_engine(bytes).
        """
        raise NotImplementedError("Engine load: roadmap step 4 (on-device).")

    # ------------------------------------------------------------------ #
    def _allocate_buffers(self) -> None:
        """Allocate paired host/device buffers for each binding.

        TODO(step 4): walk engine I/O tensors, allocate pagelocked host arrays
        + device memory (pycuda), and create a CUDA stream. Keep a ``sync()``
        that calls ``stream.synchronize`` so the benchmark can time correctly.
        """
        # Left as a no-op scaffold so the object can be constructed in tooling
        # that doesn't reach inference; real allocation happens on-device.
        self.stream = None

    # ------------------------------------------------------------------ #
    def sync(self) -> None:
        """CUDA stream barrier — passed to benchmark(..., sync=runner.sync)."""
        if self.stream is not None:
            self.stream.synchronize()

    # ------------------------------------------------------------------ #
    def infer(self, input_array: np.ndarray) -> Any:
        """Run one forward pass through the engine and return raw outputs.

        TODO(step 4): H2D copy → context.execute_async_v3(stream) → D2H copy →
        return the output tensors (decoded into masks/boxes by the inference
        wrappers).
        """
        raise NotImplementedError("TensorRT inference: roadmap step 4 (on-device).")
