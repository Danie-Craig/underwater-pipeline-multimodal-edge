"""Kalman tracker for the fused pipe track (§6).

Maintains a single estimate of the pipe along the AUV's INS-derived trajectory.
Both modalities feed *position* measurements into this one filter; between
measurements the track ``coast``s (predict-only), which is exactly what lets
the fused output survive either sensor dropping out.

State vector (4D)::

    x = [ p,  v,  h,  hr ]
          │   │   │   └─ heading rate         (rad / s)
          │   │   └───── heading              (rad)
          │   └───────── along-track velocity (m / s)
          └───────────── along-track position (m)

Constant-velocity model for both the position/velocity and heading/rate pairs.
This is generic, working filtering; tuning Q/R against the data is the
roadmap-step-6 work.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class TrackState:
    position: float          # along-track position (m)
    velocity: float          # along-track velocity (m/s)
    heading: float           # heading (rad)
    heading_rate: float      # heading rate (rad/s)
    covariance: np.ndarray   # 4x4 state covariance
    age: int                 # frames since the track was created
    coast_frames: int        # consecutive frames with no measurement


class PipeTrackKalman:
    """A constant-velocity Kalman filter over [position, velocity, heading, rate]."""

    def __init__(
        self,
        dt: float = 0.1,
        process_noise_pos: float = 0.05,
        process_noise_heading: float = 0.02,
        meas_noise_pos_default: float = 0.10,
    ) -> None:
        from filterpy.common import Q_discrete_white_noise
        from filterpy.kalman import KalmanFilter

        self.dt = dt
        self._default_R = meas_noise_pos_default

        kf = KalmanFilter(dim_x=4, dim_z=1)
        kf.x = np.zeros(4)
        kf.F = self._transition_matrix(dt)
        # We measure along-track position; H picks p out of the state.
        kf.H = np.array([[1.0, 0.0, 0.0, 0.0]])
        kf.P *= 10.0  # diffuse prior until the first measurement lands
        kf.R = np.array([[meas_noise_pos_default]])
        # Block-diagonal process noise: one block for (p, v), one for (h, hr).
        q_pos = Q_discrete_white_noise(dim=2, dt=dt, var=process_noise_pos)
        q_head = Q_discrete_white_noise(dim=2, dt=dt, var=process_noise_heading)
        kf.Q = np.block([[q_pos, np.zeros((2, 2))],
                         [np.zeros((2, 2)), q_head]])

        self.kf = kf
        self._initialized = False
        self.age = 0
        self.coast_frames = 0

    # ------------------------------------------------------------------ #
    @staticmethod
    def _transition_matrix(dt: float) -> np.ndarray:
        return np.array(
            [
                [1.0, dt, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, dt],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )

    # ------------------------------------------------------------------ #
    def predict(self, dt: float | None = None) -> None:
        """Advance the state by ``dt`` seconds (defaults to the nominal dt).

        Pass the true inter-frame interval from the INS timestamps when you
        have it, rather than relying on the nominal ``dt``.
        """
        if dt is not None and dt != self.dt:
            self.kf.F = self._transition_matrix(dt)
        self.kf.predict()
        self.age += 1

    # ------------------------------------------------------------------ #
    def update(self, position: float, meas_noise: float | None = None) -> None:
        """Fold in an along-track position measurement from either modality."""
        R = self._default_R if meas_noise is None else meas_noise
        self.kf.R = np.array([[R]])
        if not self._initialized:
            # Seed position from the first measurement; keep velocity at 0.
            self.kf.x[0] = position
            self._initialized = True
        self.kf.update(np.array([position]))
        self.coast_frames = 0

    # ------------------------------------------------------------------ #
    def coast(self) -> None:
        """No measurement this frame — count the gap (call after ``predict``)."""
        self.coast_frames += 1

    # ------------------------------------------------------------------ #
    @property
    def state(self) -> TrackState:
        p, v, h, hr = self.kf.x
        return TrackState(
            position=float(p),
            velocity=float(v),
            heading=float(h),
            heading_rate=float(hr),
            covariance=self.kf.P.copy(),
            age=self.age,
            coast_frames=self.coast_frames,
        )
