"""Sonar stream — side-scan object detection (§5).

Backend-agnostic wrapper around a YOLO11 detector, mirroring
:class:`~src.inference.rgb_segmenter.RGBSegmenter` so the two streams share an
interface. Returns axis-aligned boxes for the pipe in the sonar image; the
fusion stage turns a confident box into an along-track measurement (§6).

Filled in during roadmap step 3 (training) and step 4 (export/benchmark).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np

Backend = Literal["pytorch", "onnx", "engine", "trt"]


@dataclass
class Detection:
    """A single detected box in the sonar image."""

    xyxy: tuple[float, float, float, float]  # (x1, y1, x2, y2) in pixels
    score: float
    cls: int = 0

    @property
    def center(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.xyxy
        return (0.5 * (x1 + x2), 0.5 * (y1 + y2))


@dataclass
class DetResult:
    """Result of detecting in one sonar frame."""

    detections: list[Detection] = field(default_factory=list)
    latency_ms: float = 0.0
    extra: dict = field(default_factory=dict)

    @property
    def present(self) -> bool:
        return len(self.detections) > 0

    @property
    def best(self) -> Detection | None:
        return max(self.detections, key=lambda d: d.score) if self.detections else None


class SonarDetector:
    """Detect the pipe in a side-scan sonar frame."""

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
        self._model = None
        self._load()

    # ------------------------------------------------------------------ #
    def _load(self) -> None:
        if self.backend in ("pytorch", "onnx", "engine"):
            from ultralytics import YOLO

            self._model = YOLO(self.weights, task="detect")
        elif self.backend == "trt":
            from src.optimization.tensorrt_inference import TRTInference

            self._model = TRTInference(self.weights)
        else:  # pragma: no cover
            raise ValueError(f"Unknown backend: {self.backend!r}")

    # ------------------------------------------------------------------ #
    def infer(self, frame: np.ndarray) -> DetResult:
        """Run detection on a single sonar frame and return a :class:`DetResult`.

        TODO(step 3/4): finalize post-processing for each backend —
          • Ultralytics backends: read ``results[0].boxes`` (xyxy + conf),
            filter by ``self.conf``, and wrap each as a :class:`Detection`.
          • ``trt`` backend: decode raw engine output (boxes + scores) and run
            NMS at ``self.iou`` here.
        Time only the forward pass (exclude pre/post) for a clean §7 benchmark.
        """
        raise NotImplementedError(
            "SonarDetector.infer is a scaffold — implemented in roadmap step 3/4."
        )
