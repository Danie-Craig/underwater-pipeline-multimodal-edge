"""Per-modality underwater degradation library (§8, §10.3).

A configurable bank of degradations, one registry per modality, each function
with the signature ``fn(img: np.ndarray, severity: float) -> np.ndarray`` where
``severity`` is in ``[0, 1]``. The same bank serves two jobs:

  • the robustness sweep (§8) — apply one condition at several severities and
    measure detection/segmentation rate, confidence, and FPS; and
  • augmentation-based fine-tuning (§10.3) — sample conditions during training.

These are first-pass, physically-motivated models — good enough to drive the
sweep immediately. Refining and validating them (e.g. a proper dark-channel
turbidity model, Rayleigh speckle statistics) is roadmap step 5.

Condition names match ``robustness.rgb_conditions`` / ``sonar_conditions`` in
``configs/model_config.yaml``.
"""

from __future__ import annotations

import inspect
from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np

Augmentation = Callable[..., np.ndarray]


def _f32(img: np.ndarray) -> np.ndarray:
    return img.astype(np.float32)


def _u8(img: np.ndarray) -> np.ndarray:
    return np.clip(img, 0, 255).astype(np.uint8)


# ============================================================================
#  RGB conditions
# ============================================================================
def turbidity_haze(img: np.ndarray, severity: float) -> np.ndarray:
    """Scattering haze via an airlight model: I = J·t + A·(1−t)."""
    t = float(np.exp(-1.5 * severity))            # transmission falls with severity
    airlight = np.array([170, 178, 150], np.float32)  # blue-green-ish veiling light
    out = _f32(img) * t + airlight * (1.0 - t)
    return _u8(out)


def color_attenuation(img: np.ndarray, severity: float) -> np.ndarray:
    """Depth-dependent Beer-Lambert attenuation (red lost first). BGR order."""
    # Per-channel attenuation coefficients (B, G, R): red attenuates fastest.
    beta = np.array([0.4, 0.7, 1.8], np.float32) * severity
    factors = np.exp(-beta)                       # (3,)
    return _u8(_f32(img) * factors)


def low_light(img: np.ndarray, severity: float) -> np.ndarray:
    """Global dimming plus a slight gamma darkening."""
    gain = 1.0 - 0.85 * severity
    out = _f32(img) * gain
    return _u8(out)


def motion_blur(img: np.ndarray, severity: float, angle_deg: float = 0.0) -> np.ndarray:
    """Linear motion blur; kernel length scales with image size and severity."""
    reach = 0.025 * max(img.shape[0], img.shape[1])   # blur length ∝ resolution
    k = int(round(severity * reach)) | 1              # odd kernel length
    k = max(3, min(81, k))
    kernel = np.zeros((k, k), np.float32)
    kernel[k // 2, :] = 1.0
    M = cv2.getRotationMatrix2D((k / 2 - 0.5, k / 2 - 0.5), angle_deg, 1.0)
    kernel = cv2.warpAffine(kernel, M, (k, k))
    kernel /= max(kernel.sum(), 1e-6)
    return cv2.filter2D(img, -1, kernel)


def gaussian_noise(img: np.ndarray, severity: float) -> np.ndarray:
    sigma = severity * 50.0
    noise = np.random.normal(0.0, sigma, img.shape).astype(np.float32)
    return _u8(_f32(img) + noise)


def overexposure(img: np.ndarray, severity: float) -> np.ndarray:
    """Artificial-light hotspot: a bright radial blob blown toward white."""
    h, w = img.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]
    cx, cy = w * 0.5, h * 0.45
    r2 = ((xx - cx) ** 2 + (yy - cy) ** 2) / float((0.4 * max(h, w)) ** 2)
    hotspot = np.exp(-r2)[..., None] * (255.0 * severity)
    return _u8(_f32(img) + hotspot)


def backscatter(img: np.ndarray, severity: float) -> np.ndarray:
    """Illumination backscatter: a contrast-killing veiling glow (brightest where
    the light points) plus sparse bright particulates ("marine snow"). This is
    physically distinct from sensor ``gaussian_noise`` — the veil is the signature.
    """
    out = _f32(img).copy()
    h, w = img.shape[:2]

    # Veiling glow: blend toward a blue-green-grey light, strongest at the centre.
    yy, xx = np.mgrid[0:h, 0:w]
    r = np.sqrt(((xx - w * 0.5) / w) ** 2 + ((yy - h * 0.5) / h) ** 2)
    veil = (0.65 * severity) * (1.0 - np.clip(r / 0.75, 0.0, 1.0))[..., None]
    glow = np.array([155, 170, 168], np.float32)
    out = out * (1.0 - veil) + glow[None, None, :] * veil

    # Sparse bright particles (far fewer than a noise field).
    n = int(severity * 0.0012 * h * w)
    if n:
        ys = np.random.randint(0, h, n)
        xs = np.random.randint(0, w, n)
        for y, x in zip(ys, xs):
            cv2.circle(out, (int(x), int(y)), int(np.random.randint(1, 3)), (245, 245, 245), -1)
    return _u8(out)


def mask_from_yolo_label(label_path, w: int, h: int) -> Optional[np.ndarray]:
    """Rasterize a YOLO label into a binary pipe mask of shape (h, w).

    Handles both segmentation polygons (class + normalized x1 y1 x2 y2 ...) and
    detection boxes (class + cx cy bw bh). Returns uint8 {0, 255}, or None when
    the label file is missing or has no pipe.
    """
    p = Path(label_path)
    if not p.exists():
        return None
    mask = np.zeros((h, w), np.uint8)
    found = False
    for line in p.read_text(encoding="utf-8").splitlines():
        vals = line.split()
        if len(vals) < 5:
            continue
        coords = [float(v) for v in vals[1:]]
        if len(coords) == 4:                              # bbox: cx cy bw bh
            cx, cy, bw, bh = coords
            x0, y0 = int((cx - bw / 2) * w), int((cy - bh / 2) * h)
            x1, y1 = int((cx + bw / 2) * w), int((cy + bh / 2) * h)
            cv2.rectangle(mask, (x0, y0), (x1, y1), 255, -1)
            found = True
        elif len(coords) >= 6 and len(coords) % 2 == 0:   # polygon
            pts = np.array(coords, np.float32).reshape(-1, 2)
            pts[:, 0] *= w
            pts[:, 1] *= h
            cv2.fillPoly(mask, [pts.astype(np.int32)], 255)
            found = True
    return mask if found else None


def _blobby_alpha(h: int, w: int, coverage: float,
                  region: Optional[np.ndarray]) -> np.ndarray:
    """Soft-edged irregular coverage map in [0, 1].

    A low-frequency random field thresholded so that ~`coverage` of the pixels
    (within `region` if given, else the whole frame) are covered, with soft
    edges so the result looks organic rather than stamped.
    """
    coverage = float(np.clip(coverage, 0.0, 1.0))
    if coverage <= 0.0:
        return np.zeros((h, w), np.float32)
    small = np.random.random((max(2, h // 22), max(2, w // 22))).astype(np.float32)
    field = cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)
    field = cv2.GaussianBlur(field, (0, 0), sigmaX=max(h, w) / 70.0)
    sel = field[region > 0] if region is not None else field.ravel()
    if sel.size == 0:
        return np.zeros((h, w), np.float32)
    thr = float(np.quantile(sel, 1.0 - coverage))
    span = (field.max() - thr) or 1.0
    edge = 0.18 * span                                # thin soft edge -> opaque interior
    alpha = np.clip((field - thr) / edge, 0.0, 1.0)
    if region is not None:
        soft = cv2.GaussianBlur((region > 0).astype(np.float32), (0, 0),
                                sigmaX=max(h, w) / 120.0)
        alpha = alpha * soft
    return alpha


def sand_occlusion(img: np.ndarray, severity: float,
                   mask: Optional[np.ndarray] = None) -> np.ndarray:
    """Bury part of the pipe under irregular, textured sand.

    With a pipe `mask`, sand creeps over ~`severity` of the pipe (this is the
    intended use: it tests detection of a partially buried pipe). Without a
    mask the location of the pipe is unknown, so it falls back to scattering
    irregular sandy patches over a fraction of the frame. Either way the patch
    is textured with soft edges, not a flat block.
    """
    h, w = img.shape[:2]
    out = _f32(img).copy()
    sev = float(np.clip(severity, 0.0, 1.0))

    if mask is not None and int(np.count_nonzero(mask)) > 0:
        region = cv2.dilate((mask > 0).astype(np.uint8), np.ones((11, 11), np.uint8))
        coverage = sev                                   # fraction of the pipe to bury
    else:
        region = None
        coverage = 0.08 + 0.30 * sev                     # frame fraction (pipe unknown)

    alpha = _blobby_alpha(h, w, coverage, region)
    if float(alpha.max()) <= 0.0:
        return _u8(out)

    sand = np.array([108, 152, 178], np.float32)         # muted BGR sand tone
    grain = np.random.normal(0.0, 16.0, (h, w, 1)).astype(np.float32)
    low2 = np.random.normal(0.0, 20.0, (max(2, h // 30), max(2, w // 30))).astype(np.float32)
    low = cv2.resize(low2, (w, h), interpolation=cv2.INTER_CUBIC)[..., None]
    sand_img = np.clip(sand[None, None, :] + grain + low, 0.0, 255.0)

    a = alpha[..., None]
    out = out * (1.0 - a) + sand_img * a
    return _u8(out)


# ============================================================================
#  Sonar conditions  (operate on single- or multi-channel intensity images)
# ============================================================================
def speckle_noise(img: np.ndarray, severity: float) -> np.ndarray:
    """Multiplicative speckle: I' = I · (1 + n), n ~ N(0, severity)."""
    noise = np.random.normal(0.0, severity, img.shape).astype(np.float32)
    return _u8(_f32(img) * (1.0 + noise))


def beam_dropout(img: np.ndarray, severity: float) -> np.ndarray:
    """Zero out a random fraction of beams (rows) — low-SNR return loss."""
    out = _f32(img).copy()
    h = img.shape[0]
    n = int(severity * 0.4 * h)
    rows = np.random.choice(h, size=n, replace=False) if n else []
    out[rows] = 0.0
    return _u8(out)


def range_falloff(img: np.ndarray, severity: float) -> np.ndarray:
    """Range-dependent intensity falloff along the range axis (assumed rows)."""
    h = img.shape[0]
    ramp = np.linspace(1.0, 1.0 - 0.9 * severity, h).astype(np.float32)
    ramp = ramp.reshape(h, *([1] * (img.ndim - 1)))
    return _u8(_f32(img) * ramp)


def motion_smear(img: np.ndarray, severity: float) -> np.ndarray:
    """Along-track motion smear (vertical blur)."""
    return motion_blur(img, severity, angle_deg=90.0)


def reverberation_clutter(img: np.ndarray, severity: float) -> np.ndarray:
    """Additive structured horizontal-streak clutter."""
    h, w = img.shape[:2]
    streaks = np.zeros((h, w), np.float32)
    n = int(severity * 0.15 * h)
    for _ in range(n):
        y = np.random.randint(0, h)
        streaks[y, :] = np.random.uniform(40, 120)
    streaks = cv2.GaussianBlur(streaks, (1, 9), 0)
    if img.ndim == 3:
        streaks = streaks[..., None]
    return _u8(_f32(img) + streaks)


# ============================================================================
#  Registries + driver
# ============================================================================
RGB_AUGMENTATIONS: dict[str, Augmentation] = {
    "turbidity_haze": turbidity_haze,
    "color_attenuation": color_attenuation,
    "low_light": low_light,
    "motion_blur": motion_blur,
    "gaussian_noise": gaussian_noise,
    "overexposure": overexposure,
    "backscatter": backscatter,
    "sand_occlusion": sand_occlusion,
}

SONAR_AUGMENTATIONS: dict[str, Augmentation] = {
    "speckle_noise": speckle_noise,
    "beam_dropout": beam_dropout,
    "range_falloff": range_falloff,
    "motion_smear": motion_smear,
    "reverberation_clutter": reverberation_clutter,
}


class DegradationPipeline:
    """Apply a named degradation at a given severity to a given modality."""

    def __init__(self, modality: str) -> None:
        if modality == "rgb":
            self.bank = RGB_AUGMENTATIONS
        elif modality == "sonar":
            self.bank = SONAR_AUGMENTATIONS
        else:
            raise ValueError(f"modality must be 'rgb' or 'sonar', got {modality!r}")
        self.modality = modality

    def conditions(self) -> list[str]:
        return list(self.bank)

    def apply(self, img: np.ndarray, condition: str, severity: float,
              mask: Optional[np.ndarray] = None) -> np.ndarray:
        if condition not in self.bank:
            raise KeyError(f"unknown {self.modality} condition: {condition!r}")
        fn = self.bank[condition]
        if "mask" in inspect.signature(fn).parameters:
            return fn(img, float(severity), mask=mask)
        return fn(img, float(severity))
