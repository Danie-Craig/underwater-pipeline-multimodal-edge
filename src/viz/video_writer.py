"""Tiny video-writer wrapper for the annotated demo (§14).

Auto-sizes from the first frame and works as a context manager::

    with VideoWriter("results/inference/demo.mp4", fps=10) as vw:
        for frame in annotated_frames:
            vw.add(frame)

Note: ``results/**/*.mp4`` is gitignored — host the final demo externally or
via git-lfs (see README / .gitignore).
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


class VideoWriter:
    def __init__(self, path: str | Path, fps: float = 10.0, fourcc: str = "mp4v") -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fps = fps
        self._fourcc = cv2.VideoWriter_fourcc(*fourcc)
        self._writer: cv2.VideoWriter | None = None
        self._size: tuple[int, int] | None = None  # (w, h)

    def add(self, frame: np.ndarray) -> None:
        if frame.ndim == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        if self._writer is None:
            h, w = frame.shape[:2]
            self._size = (w, h)
            self._writer = cv2.VideoWriter(str(self.path), self._fourcc, self.fps, self._size)
        # Keep frame size consistent with the first frame.
        if (frame.shape[1], frame.shape[0]) != self._size:
            frame = cv2.resize(frame, self._size)
        self._writer.write(frame)

    def close(self) -> None:
        if self._writer is not None:
            self._writer.release()
            self._writer = None

    def __enter__(self) -> "VideoWriter":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
