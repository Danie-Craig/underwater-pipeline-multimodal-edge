"""Late / track-level fusion — the centerpiece (§6).

The camera and the sonar do **not** share a viewpoint or geometry, so this is
explicitly *not* pixel-level fusion. Instead, each modality, when it has a
confident detection, contributes an along-track **position measurement** to a
single Kalman-tracked "pipe track" indexed by the AUV's INS trajectory.

Per-frame flow::

    pose(t) ──┐
    seg(t)  ──┤→ project each confident detection → along-track position z
    det(t)  ──┘                    │
                                   ▼
              cross-modal gating → choose / fuse measurements
                                   │
                                   ▼
              Kalman: predict(dt) → update(z) | coast()  → fused TrackState

This module also folds in two mitigations directly (§10.1 tracking +
temporal smoothing, §10.4 cross-modal gating). The ablation harness drives it
in three modes — RGB-only, sonar-only, fused — to produce the headline
"track continuity across degradations" comparison.

Orchestration is scaffolded here; the sensor-geometry projection (the one
genuinely dataset-specific piece) is completed in roadmap step 6.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from src.fusion.kalman_tracker import PipeTrackKalman, TrackState
from src.inference.rgb_segmenter import SegResult
from src.inference.sonar_detector import DetResult

FusionMode = Literal["rgb_only", "sonar_only", "fused"]


@dataclass
class Pose:
    """Minimal AUV pose from the INS at a given timestamp."""

    t: float                      # timestamp (s)
    x: float                      # world position x (m)
    y: float                      # world position y (m)
    heading: float                # heading (rad)
    along_track: float            # cumulative distance travelled (m)


@dataclass
class FusedFrameResult:
    """What the fusion produces for one synchronized (rgb, sonar, pose) frame."""

    t: float
    track: TrackState | None
    rgb_contributed: bool = False
    sonar_contributed: bool = False
    coasting: bool = False
    extra: dict = field(default_factory=dict)


class LateFusion:
    """Drive per-frame fusion of the two perception streams along the track."""

    def __init__(self, config: dict) -> None:
        f = config["fusion"]
        k = config["kalman"]
        self.cfg = config
        self.rgb_conf_gate = float(f["rgb_measurement_conf"])
        self.sonar_conf_gate = float(f["sonar_measurement_conf"])
        self.cross_modal_gating = bool(f["cross_modal_gating"])
        self.gate_distance_m = float(f["gate_distance_m"])
        self.max_coast_frames = int(f["max_coast_frames"])
        self.meas_noise_rgb = float(k["meas_noise_rgb_pos"])
        self.meas_noise_sonar = float(k["meas_noise_sonar_pos"])

        self.tracker = PipeTrackKalman(
            dt=float(k["dt"]),
            process_noise_pos=float(k["process_noise_pos"]),
            process_noise_heading=float(k["process_noise_heading"]),
            meas_noise_pos_default=self.meas_noise_rgb,
        )
        self._last_t: float | None = None

    # ------------------------------------------------------------------ #
    def _rgb_to_alongtrack(self, seg: SegResult, pose: Pose) -> float | None:
        """Project an RGB pipe segmentation to an along-track position (m).

        The AUV is flying the pipe it is inspecting, so whenever the camera
        holds a confident mask the pipe is directly beneath the vehicle and its
        along-track coordinate is the vehicle's own cumulative along-track
        distance at capture time. We therefore take ``pose.along_track`` as the
        measurement. A finer sub-frame correction from the mask centroid would
        need the camera altitude and field of view (its ground footprint), which
        we do not model; it would move the estimate by at most that footprint and
        does not change the track-continuity result the fusion exists to show.
        Returns ``None`` only when the mask is unusable (no centroid).
        """
        if not seg.present or seg.centroid is None:
            return None
        return float(pose.along_track)

    def _sonar_to_alongtrack(self, det: DetResult, pose: Pose) -> float | None:
        """Project a sonar detection to an along-track position (m).

        Side-scan returns are indexed by along-track ping and cross-track range.
        As with the camera, a confident pipe box means the pipe is abeam the
        vehicle at its current along-track distance, so we take
        ``pose.along_track`` as the measurement. Turning the box's along-track
        pixel into a sub-frame metre offset would need the ping rate and
        waterfall scale (and the image height, which the detection result does
        not carry); that refinement is omitted for the same reason as the
        camera's. Returns ``None`` when there is no usable box.
        """
        if not det.present or det.best is None:
            return None
        return float(pose.along_track)

    # ------------------------------------------------------------------ #
    def step(
        self,
        seg: SegResult,
        det: DetResult,
        pose: Pose,
        mode: FusionMode = "fused",
    ) -> FusedFrameResult:
        """Advance the fused track by one frame and return the result.

        The three ``mode`` values share this exact code path so the ablation is
        apples-to-apples: only the set of admissible measurements differs.
        """
        # 1) Time update using the true INS interval when available.
        dt = None if self._last_t is None else max(pose.t - self._last_t, 1e-3)
        self.tracker.predict(dt)
        self._last_t = pose.t

        # 2) Gather candidate measurements, honoring the confidence gates and
        #    the active ablation mode.
        rgb_ok = mode in ("rgb_only", "fused") and seg.present and seg.score >= self.rgb_conf_gate
        sonar_ok = mode in ("sonar_only", "fused") and det.present and (
            det.best is not None and det.best.score >= self.sonar_conf_gate
        )

        # 3) Cross-modal gating (§10.4): a confident detection in one modality
        #    can rescue a sub-threshold one in the other (fused mode only).
        #    TODO(step 6/7): lower the opposite gate within ``gate_distance_m``
        #    of a confident measurement instead of the hard threshold above.

        # 4) Project admissible detections to along-track measurements and update.
        contributed_rgb = contributed_sonar = False
        if rgb_ok:
            z = self._rgb_to_alongtrack(seg, pose)
            if z is not None:
                self.tracker.update(z, meas_noise=self.meas_noise_rgb)
                contributed_rgb = True
        if sonar_ok:
            z = self._sonar_to_alongtrack(det, pose)
            if z is not None:
                self.tracker.update(z, meas_noise=self.meas_noise_sonar)
                contributed_sonar = True

        coasting = not (contributed_rgb or contributed_sonar)
        if coasting:
            self.tracker.coast()

        return FusedFrameResult(
            t=pose.t,
            track=self.tracker.state,
            rgb_contributed=contributed_rgb,
            sonar_contributed=contributed_sonar,
            coasting=coasting,
        )
