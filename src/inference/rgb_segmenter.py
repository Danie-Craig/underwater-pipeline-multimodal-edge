"""RGB stream — pipe segmentation (§5).

A thin, backend-agnostic wrapper around a YOLO11-seg model. The same class
serves the PyTorch baseline, the ONNX Runtime path, and the TensorRT engine,
so the rest of the pipeline (fusion, benchmark, robustness) never has to know
which backend is live.

Backends
--------
- ``"pytorch"``  : the trained ``.pt`` checkpoint (server baseline)
- ``"onnx"``     : the exported ``.onnx`` via ONNX Runtime
- ``"engine"``   : a TensorRT engine, loaded through Ultralytics
- ``"trt"``      : the raw TensorRT runner in ``src/optimization/tensorrt_inference``
                   (lowest-overhead edge path; used for the headline FPS number)

Filled in during roadmap step 3 (training) and step 4 (export/benchmark).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np

Backend = Literal["pytorch", "onnx", "engine", "trt"]


@dataclass
class SegResult:
    """Result of segmenting one RGB frame.

    Attributes
    ----------
    present:    whether the pipe was detected above ``conf``.
    score:      confidence of the best pipe instance (0.0 if absent).
    mask:       HxW uint8 mask (1 = pipe) or ``None`` if absent.
    centroid:   (x, y) pixel centroid of the mask, or ``None``. The fusion
                stage projects this into the along-track coordinate (§6).
    latency_ms: wall-clock inference time for this frame.
    """

    present: bool = False
    score: float = 0.0
    mask: np.ndarray | None = None
    centroid: tuple[float, float] | None = None
    latency_ms: float = 0.0
    extra: dict = field(default_factory=dict)


class RGBSegmenter:
    """Segment the submarine pipe in an RGB frame."""

    def __init__(
        self,
        weights: str | Path,
        backend: Backend = "pytorch",
        conf: float = 0.25,
        iou: float = 0.50,
        imgsz: int = 640,
        device: str = "cuda:0",
    ) -> None:
        self.weights = str(weights)
        self.backend = backend
        self.conf = conf
        self.iou = iou
        self.imgsz = imgsz
        self.device = device
        self._model = None  # lazily loaded on first use
        self._load()

    # ------------------------------------------------------------------ #
    def _load(self) -> None:
        """Load the model for the configured backend (lazy, heavy imports)."""
        if self.backend in ("pytorch", "onnx", "engine"):
            # Ultralytics transparently loads .pt / .onnx / .engine by suffix.
            from ultralytics import YOLO

            self._model = YOLO(self.weights, task="segment")
        elif self.backend == "trt":
            # Lowest-overhead path: our own TensorRT runner.
            from src.optimization.tensorrt_inference import TRTInference

            self._model = TRTInference(self.weights)
        else:  # pragma: no cover - guarded by the Literal type
            raise ValueError(f"Unknown backend: {self.backend!r}")

    # ------------------------------------------------------------------ #
    def infer(self, frame: np.ndarray) -> SegResult:
        """Run segmentation on a single BGR frame and return a :class:`SegResult`.

        TODO(step 3/4): finalize post-processing for each backend —
          • Ultralytics backends: pull ``results[0].masks`` + the top score,
            threshold by ``self.conf``, union instance masks into one pipe
            mask, and compute its centroid.
          • ``trt`` backend: decode the raw engine output tensors (proto +
            mask coeffs for YOLO-seg) into a binary mask here.
        Keep timing around the forward pass only (exclude pre/post) so the
        benchmark in §7 measures the model, not the Python overhead.
        """
        raise NotImplementedError(
            "RGBSegmenter.infer is a scaffold — implemented in roadmap step 3/4."
        )
