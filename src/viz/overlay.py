"""Drawing helpers for the annotated demo (§14).

Render the RGB segmentation mask, the sonar detection boxes, and a small
heads-up display of the fused pipe track onto frames. Functional as-is; the
demo script composes these into the final video.
"""

from __future__ import annotations

import cv2
import numpy as np

from src.fusion.late_fusion import FusedFrameResult
from src.inference.rgb_segmenter import SegResult
from src.inference.sonar_detector import DetResult

# BGR colors
PIPE_MASK = (0, 200, 255)     # amber
SONAR_BOX = (0, 255, 120)     # green
HUD_BG = (30, 30, 30)
HUD_FG = (240, 240, 240)


def draw_segmentation(frame: np.ndarray, seg: SegResult, alpha: float = 0.45) -> np.ndarray:
    """Blend the pipe mask over the RGB frame and mark its centroid."""
    out = frame.copy()
    if seg.present and seg.mask is not None:
        color = np.zeros_like(out)
        color[seg.mask.astype(bool)] = PIPE_MASK
        out = cv2.addWeighted(color, alpha, out, 1.0, 0.0)
        if seg.centroid is not None:
            cx, cy = map(int, seg.centroid)
            cv2.circle(out, (cx, cy), 5, PIPE_MASK, -1)
        cv2.putText(out, f"pipe {seg.score:.2f}", (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, PIPE_MASK, 2, cv2.LINE_AA)
    return out


def draw_detections(frame: np.ndarray, det: DetResult) -> np.ndarray:
    """Draw sonar detection boxes with confidence labels."""
    out = frame.copy()
    if out.ndim == 2:  # promote grayscale sonar to BGR for colored overlays
        out = cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)
    for d in det.detections:
        x1, y1, x2, y2 = map(int, d.xyxy)
        cv2.rectangle(out, (x1, y1), (x2, y2), SONAR_BOX, 2)
        cv2.putText(out, f"{d.score:.2f}", (x1, max(y1 - 6, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, SONAR_BOX, 2, cv2.LINE_AA)
    return out


def draw_track_hud(frame: np.ndarray, fused: FusedFrameResult) -> np.ndarray:
    """Overlay the fused track state + which modality is carrying it."""
    out = frame.copy()
    h, w = out.shape[:2]
    panel_h = 86
    cv2.rectangle(out, (0, h - panel_h), (w, h), HUD_BG, -1)

    if fused.track is not None:
        t = fused.track
        lines = [
            f"track  pos={t.position:7.2f} m   hdg={np.degrees(t.heading):6.1f} deg",
            f"source RGB:{'Y' if fused.rgb_contributed else '-'}  "
            f"SONAR:{'Y' if fused.sonar_contributed else '-'}  "
            f"{'COASTING ' + str(t.coast_frames) if fused.coasting else 'LOCKED'}",
        ]
    else:
        lines = ["track  (uninitialized)"]

    y = h - panel_h + 30
    for line in lines:
        cv2.putText(out, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62,
                    HUD_FG, 2, cv2.LINE_AA)
        y += 30
    return out


def hstack_panels(*panels: np.ndarray, height: int = 480) -> np.ndarray:
    """Resize each panel to a common height and stack them side by side."""
    resized = []
    for p in panels:
        if p.ndim == 2:
            p = cv2.cvtColor(p, cv2.COLOR_GRAY2BGR)
        scale = height / p.shape[0]
        resized.append(cv2.resize(p, (int(p.shape[1] * scale), height)))
    return cv2.hconcat(resized)
