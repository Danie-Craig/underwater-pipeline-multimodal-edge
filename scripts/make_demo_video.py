#!/usr/bin/env python3
"""Per-sensor annotated demo video.

Runs one trained model over its continuous SubPipe footage and renders an MP4
with the model output overlaid (pipe mask for RGB, detection boxes for sonar)
and a HUD banner showing frame number, detection, live inference time and FPS,
and a footer citing the Jetson edge numbers.

    python scripts/make_demo_video.py --modality rgb   --max-frames 600
    python scripts/make_demo_video.py --modality sonar --max-frames 400 --stride 1
    python scripts/make_demo_video.py --modality rgb   --images path/to/frames

The fused RGB+sonar video comes later, after the fusion stage. This script is
self-contained and only needs one sensor's frames. Writes
results/inference/<modality>_demo.mp4 (gitignored; host externally or via LFS).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import REPO_ROOT, load_config

MODELS = {
    "rgb":   {"cfg": "rgb_seg",   "root": "rgb_root",   "imgsz_key": "rgb_imgsz",
              "title": "RGB segmentation"},
    "sonar": {"cfg": "sonar_det", "root": "sonar_root", "imgsz_key": "sonar_imgsz",
              "title": "Sonar detection"},
}
HUD_FG = (80, 240, 80)       # green, BGR
HUD_DIM = (170, 170, 170)


def _labeled_ts_range(modality: str, cfg: dict):
    """(min, max) timestamp of frames annotated with a pipe, or None."""
    from src.data.loaders import _parse_timestamp
    out_base = REPO_ROOT / cfg["data"].get("prep", {}).get("out_dir", "data/subpipe/yolo")
    subdir = MODELS[modality]["cfg"]
    ts = []
    for split in ("train", "val"):
        d = out_base / subdir / "labels" / split
        if not d.is_dir():
            continue
        for f in d.glob("*.txt"):
            if f.read_text(encoding="utf-8").strip():     # non-empty => pipe present
                try:
                    ts.append(_parse_timestamp(f.stem))
                except Exception:
                    pass
    return (min(ts), max(ts)) if ts else None


def _find_frames(modality: str, cfg: dict, images_override: str | None,
                 stride: int, max_frames: int, segment: str = "labeled") -> list[Path]:
    from src.data.loaders import (RGB_DIR, SONAR_DIRS, SONAR_IMG_SUBDIR,
                                  ULTRALYTICS_IMG_EXTS, _parse_timestamp,
                                  discover_chunks, _sorted_by_timestamp)

    if images_override:
        d = Path(images_override)
        frames = [p for p in d.glob("*") if p.suffix.lower() in (".pbm", *ULTRALYTICS_IMG_EXTS)]
        return _sorted_by_timestamp(frames)[::max(1, stride)][:max_frames]

    root = REPO_ROOT / cfg["data"][MODELS[modality]["root"]]
    frames = []
    for chunk in discover_chunks(root):
        if modality == "rgb":
            d, exts = chunk / RGB_DIR, ULTRALYTICS_IMG_EXTS
        else:
            d = chunk / SONAR_DIRS.get("HF", "SSS_HF_images") / SONAR_IMG_SUBDIR
            exts = (".pbm", *ULTRALYTICS_IMG_EXTS)
        if d.is_dir():
            frames += [p for p in d.glob("*") if p.suffix.lower() in exts]
    frames = _sorted_by_timestamp(frames)

    # Restrict to the inspection window (where the pipe is annotated) so the
    # demo opens on the pipe instead of the approach/transit footage.
    if segment == "labeled":
        rng = _labeled_ts_range(modality, cfg)
        if rng is not None:
            lo, hi = rng
            kept = []
            for p in frames:
                try:
                    t = _parse_timestamp(p.stem)
                except Exception:
                    continue
                if lo <= t <= hi:
                    kept.append(p)
            if kept:
                frames = kept

    if stride > 1:
        frames = frames[::stride]
    return frames[:max_frames]


def _edge_note(modality: str) -> str | None:
    """FP16 edge line from the committed Jetson benchmark, if present."""
    p = REPO_ROOT / "results" / "benchmark" / f"{MODELS[modality]['cfg']}_jetson.json"
    if not p.exists():
        return None
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        fp16 = next(r for r in data["runs"] if r["precision"] == "fp16")
        return (f"edge ref: Jetson Orin TRT FP16  "
                f"{fp16['gpu_compute_ms']['mean']:.1f} ms / {fp16['throughput_qps']:.0f} FPS")
    except Exception:
        return None


def _measure_fps(model, frames, cv2, k: int = 40) -> float | None:
    """Average inference FPS over the first k frames (cold starts skipped)."""
    lat = []
    for i, fp in enumerate(frames[:min(k, len(frames))]):
        img = cv2.imread(str(fp), cv2.IMREAD_COLOR)
        if img is None:
            continue
        r = model.infer(img)
        if i >= 2:
            lat.append(r.latency_ms)
    if not lat:
        return None
    return 1000.0 / (sum(lat) / len(lat))


def _banner(width: int, lines: list[tuple[str, tuple]], line_h: int = 30, pad: int = 12):
    import numpy as np
    import cv2
    h = pad * 2 + line_h * len(lines)
    img = np.zeros((h, width, 3), dtype=np.uint8)
    y = pad + line_h - 8
    for text, color in lines:
        cv2.putText(img, text, (14, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, color, 2, cv2.LINE_AA)
        y += line_h
    return img


def _compose(frame_bgr, banner, width: int):
    import cv2
    h, w = frame_bgr.shape[:2]
    resized = cv2.resize(frame_bgr, (width, max(1, round(h * width / w))))
    return cv2.vconcat([banner, resized])


def render_demo(modality: str, cfg: dict, *, max_frames: int, stride: int, fps: float,
                width: int, images_override: str | None = None, backend: str = "pytorch",
                segment: str = "labeled", fps_source: str = "measured",
                model=None, ov=None) -> Path:
    import cv2
    from src.viz import overlay as ov_mod
    from src.viz.video_writer import VideoWriter
    if ov is None:
        ov = ov_mod

    spec = MODELS[modality]
    mcfg = cfg["models"][spec["cfg"]]
    imgsz = int(cfg["project"][spec["imgsz_key"]])
    if model is None:
        if modality == "rgb":
            from src.inference.rgb_segmenter import RGBSegmenter
            model = RGBSegmenter(str(REPO_ROOT / mcfg["weights"]), backend=backend,
                                 conf=mcfg.get("conf", 0.25), iou=mcfg.get("iou", 0.5), imgsz=imgsz)
        else:
            from src.inference.sonar_detector import SonarDetector
            model = SonarDetector(str(REPO_ROOT / mcfg["weights"]), backend=backend,
                                  conf=mcfg.get("conf", 0.25), iou=mcfg.get("iou", 0.5), imgsz=imgsz)

    frames = _find_frames(modality, cfg, images_override, stride, max_frames, segment)
    if not frames:
        raise FileNotFoundError(
            f"no {modality} frames found (looked under data.{spec['root']}); "
            f"pass --images <dir> to point at a folder of frames")

    edge = _edge_note(modality)
    out_path = REPO_ROOT / "results" / "inference" / f"{modality}_demo.mp4"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Warm up once (model load + CUDA init).
    _warm = cv2.imread(str(frames[0]), cv2.IMREAD_COLOR)
    if _warm is not None:
        model.infer(_warm)

    # Pick playback fps: the real inference rate, not an arbitrary value.
    if fps_source == "measured":
        m = _measure_fps(model, frames, cv2)
        if m:
            fps = m
    # else "fixed": use the --fps value as given.

    ema_ms = None
    n_total = len(frames)
    print(f"[demo] {modality}: {n_total} frames -> {out_path.name} "
          f"(backend={backend}, playback {fps:.1f} fps [{fps_source}])")
    writer = VideoWriter(str(out_path), fps=fps)
    for i, fp in enumerate(frames):
        frame = cv2.imread(str(fp), cv2.IMREAD_COLOR)
        if frame is None:
            continue
        if modality == "rgb":
            r = model.infer(frame)
            painted = ov.draw_segmentation(frame, r)
            det_line = (f"pipe  conf {r.score:.2f}" if r.present else "pipe: not detected")
            present = r.present
            ms = r.latency_ms
        else:
            r = model.infer(frame)
            painted = ov.draw_detections(frame, r)
            best = r.best
            det_line = (f"boxes {len(r.detections)}   best {best.score:.2f}"
                        if best else "no detection")
            present = r.present
            ms = r.latency_ms

        ema_ms = ms if ema_ms is None else 0.85 * ema_ms + 0.15 * ms
        live_fps = 1000.0 / ema_ms if ema_ms and ema_ms > 0 else 0.0
        lines = [
            (f"{spec['title']}    frame {i + 1}/{n_total}", HUD_FG),
            (det_line, HUD_FG if present else (80, 80, 235)),
            (f"infer {ema_ms:4.0f} ms   ~{live_fps:4.0f} FPS   (PyTorch, A6000)", HUD_DIM),
        ]
        if edge:
            lines.append((edge, HUD_DIM))
        writer.add(_compose(painted, _banner(width, lines), width))
    writer.close()
    print(f"[demo] {modality}: wrote {out_path}")
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser(description="Per-sensor annotated demo video.")
    ap.add_argument("--config", default="configs/model_config.yaml")
    ap.add_argument("--modality", required=True, choices=["rgb", "sonar"])
    ap.add_argument("--max-frames", type=int, default=600)
    ap.add_argument("--stride", type=int, default=1, help="use every Nth frame")
    ap.add_argument("--fps", type=float, default=12.0, help="playback fps when --fps-source fixed")
    ap.add_argument("--fps-source", default="measured", choices=["measured", "fixed"],
                    help="'measured' = real A6000 inference rate (default); 'fixed' = the --fps value")
    ap.add_argument("--width", type=int, default=1100)
    ap.add_argument("--images", default=None, help="override: a folder of frames")
    ap.add_argument("--backend", default="pytorch", choices=["pytorch", "onnx", "engine", "trt"])
    ap.add_argument("--segment", default="labeled", choices=["labeled", "full"],
                    help="'labeled' opens on the inspection window; 'full' uses the whole run")
    args = ap.parse_args()

    cfg = load_config(args.config)
    render_demo(args.modality, cfg, max_frames=args.max_frames, stride=args.stride,
                fps=args.fps, width=args.width, images_override=args.images,
                backend=args.backend, segment=args.segment, fps_source=args.fps_source)


if __name__ == "__main__":
    main()
