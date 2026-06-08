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

from typing import Callable

import cv2
import numpy as np

Augmentation = Callable[[np.ndarray, float], np.ndarray]


def _f32(img: np.ndarray) -> np.ndarray:
    return img.astype(np.float32)


def _u8(img: np.ndarray) -> np.ndarray:
    return np.clip(img, 0, 255).astype(np.uint8)


# ============================================================================
#  RGB conditions
# ============================================================================
def turbidity_haze(img: np.ndarray, severity: float) -> np.ndarray:
    """Scattering haze via an airlight model: I = J·t + A·(1−t)."""
    t = float(np.exp(-3.0 * severity))            # transmission falls with severity
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
    """Linear motion blur; kernel length scales with severity (see §10.5)."""
    k = max(3, int(round(severity * 30)) | 1)     # odd kernel length
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
    """"Marine snow": sparse bright particulates."""
    out = _f32(img).copy()
    h, w = img.shape[:2]
    n = int(severity * 0.004 * h * w)             # particle count
    ys = np.random.randint(0, h, n)
    xs = np.random.randint(0, w, n)
    for y, x in zip(ys, xs):
        cv2.circle(out, (int(x), int(y)), np.random.randint(1, 3), (255, 255, 255), -1)
    return _u8(out)


def sand_occlusion(img: np.ndarray, severity: float) -> np.ndarray:
    """Occlude part of the pipe with a sandy band (covering fraction ∝ severity)."""
    out = _f32(img).copy()
    h, w = img.shape[:2]
    band = int(severity * 0.5 * h)
    if band <= 0:
        return _u8(out)
    sand = np.array([120, 170, 200], np.float32)  # BGR sandy tone
    texture = np.random.normal(0, 15, (band, w, 1)).astype(np.float32)
    y0 = h - band
    out[y0:h] = np.clip(sand + texture, 0, 255)
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

    def apply(self, img: np.ndarray, condition: str, severity: float) -> np.ndarray:
        if condition not in self.bank:
            raise KeyError(f"unknown {self.modality} condition: {condition!r}")
        return self.bank[condition](img, float(severity))
