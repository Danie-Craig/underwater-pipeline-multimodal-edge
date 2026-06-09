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

        Ultralytics backends (pytorch / onnx / engine) run through ``predict``;
        we union the instance masks into one pipe mask, take the top score, and
        read the best box centre (already in original-image pixels) as the
        centroid the fusion stage projects along-track. The raw ``trt`` backend
        is timed for the benchmark; its mask decode is finalized on-device.
        """
        import time

        import cv2

        if self.backend == "trt":
            t0 = time.perf_counter()
            raw = self._model.infer(frame)
            return SegResult(latency_ms=(time.perf_counter() - t0) * 1000.0,
                             extra={"raw_output": raw})

        t0 = time.perf_counter()
        results = self._model.predict(
            frame, imgsz=self.imgsz, conf=self.conf, iou=self.iou,
            device=self.device, verbose=False,
        )
        dt = (time.perf_counter() - t0) * 1000.0

        r = results[0]
        boxes = getattr(r, "boxes", None)
        masks = getattr(r, "masks", None)
        if boxes is None or len(boxes) == 0:
            return SegResult(present=False, score=0.0, mask=None, centroid=None, latency_ms=dt)

        conf = boxes.conf.cpu().numpy()
        best = int(conf.argmax())
        score = float(conf[best])
        x1, y1, x2, y2 = (float(v) for v in boxes.xyxy[best].cpu().numpy())
        centroid = (0.5 * (x1 + x2), 0.5 * (y1 + y2))

        mask = None
        H, W = frame.shape[:2]
        if masks is not None:
            polys = getattr(masks, "xy", None)
            if polys:
                # Ultralytics gives polygons in original-image pixels; rasterize
                # them into one frame-resolution union mask so the mask and the
                # centroid share the same (frame) coordinate system.
                mask = np.zeros((H, W), dtype=np.uint8)
                for poly in polys:
                    if poly is not None and len(poly) >= 3:
                        cv2.fillPoly(mask, [np.asarray(poly, dtype=np.int32)], 1)
            elif masks.data is not None and len(masks.data):
                # Fallback: upsample the model-resolution union to the frame.
                m = (masks.data.cpu().numpy().sum(axis=0) > 0).astype(np.uint8)
                mask = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)

        return SegResult(present=True, score=score, mask=mask,
                         centroid=centroid, latency_ms=dt)
