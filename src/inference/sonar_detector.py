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

        Ultralytics backends (pytorch / onnx / engine) run through ``predict``
        and each kept box becomes a :class:`Detection` (xyxy already in
        original-image pixels). The raw ``trt`` backend is timed for the
        benchmark; its box decode + NMS are finalized on-device.
        """
        import time

        if self.backend == "trt":
            t0 = time.perf_counter()
            raw = self._model.infer(frame)
            return DetResult(latency_ms=(time.perf_counter() - t0) * 1000.0,
                             extra={"raw_output": raw})

        t0 = time.perf_counter()
        results = self._model.predict(
            frame, imgsz=self.imgsz, conf=self.conf, iou=self.iou,
            device=self.device, verbose=False,
        )
        dt = (time.perf_counter() - t0) * 1000.0

        r = results[0]
        boxes = getattr(r, "boxes", None)
        dets: list[Detection] = []
        if boxes is not None and len(boxes):
            xyxy = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy()
            clss = boxes.cls.cpu().numpy().astype(int)
            for (x1, y1, x2, y2), c, k in zip(xyxy, confs, clss):
                dets.append(Detection(
                    xyxy=(float(x1), float(y1), float(x2), float(y2)),
                    score=float(c), cls=int(k),
                ))
        return DetResult(detections=dets, latency_ms=dt)
