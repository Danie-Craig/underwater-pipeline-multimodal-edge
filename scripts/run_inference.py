#!/usr/bin/env python3
"""Run the full pipeline over one sequence and render an annotated fused demo.

RGB segmentation + sonar detection + along-track fusion, composed into a demo
video (camera panel with the pipe mask on top, the sonar waterfall with detection
boxes below, and a heads-up display of the fused track and which sensor is
carrying it) plus a per-frame track JSON. The processed window can be cropped and
subsampled, and an optional staggered stress (sonar motion smear early, camera
turbidity late) makes the fusion handoff visible: one sensor visibly drops while
the fused track stays locked on the other.

    # clean fused run over a cropped window
    python scripts/run_inference.py --sequence data/subpipe/SubPipe/DATA/Chunk3 \
        --time-window 1693574476 1693574504 --fps 6

    # the headline demo: same window with the staggered failures injected
    python scripts/run_inference.py --sequence data/subpipe/SubPipe/DATA/Chunk3 \
        --time-window 1693574476 1693574504 --fps 6 --stress \
        --video results/inference/fusion_demo.mp4

The TensorRT path on the Orin uses the same script with --backend trt.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import load_config

# Staggered demo stress: sonar fails early (camera carries the track), camera
# fails late (sonar carries), so the fused estimate hands off without losing lock.
SONAR_WINDOW = (0.05, 0.50)
RGB_WINDOW = (0.50, 0.95)
RGB_CONDITION = "turbidity_haze"
SONAR_CONDITION = "motion_smear"
PANEL_WIDTH = 900
HUD_STRIP = 90      # blank space appended so draw_track_hud does not cover imagery


def _parse_pair(s: str) -> tuple[float, float]:
    lo, hi = (float(v) for v in s.split(","))
    return lo, hi


def _compose_frame(rgb_panel, son_panel, banner_text: str, width: int = PANEL_WIDTH):
    """Banner strip, camera panel, sonar strip, and HUD space stacked vertically.

    Each panel is resized to a common width (keeping its own aspect, so the wide
    sonar waterfall stays a thin strip rather than dominating). Returns None if
    neither panel is present at this tick.
    """
    import cv2
    import numpy as np

    strip = np.full((30, width, 3), (20, 20, 80) if banner_text else (0, 0, 0), np.uint8)
    if banner_text:
        cv2.putText(strip, banner_text, (12, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (255, 255, 255), 2, cv2.LINE_AA)
    rows = [strip]
    for p in (rgb_panel, son_panel):
        if p is None:
            continue
        if p.ndim == 2:
            p = cv2.cvtColor(p, cv2.COLOR_GRAY2BGR)
        h = max(1, int(round(width * p.shape[0] / p.shape[1])))
        rows.append(cv2.resize(p, (width, h)))
    if len(rows) == 1:          # no imagery this tick
        return None
    rows.append(np.zeros((HUD_STRIP, width, 3), np.uint8))
    return cv2.vconcat(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description="Run RGB seg + sonar detect + fused track, render a demo.")
    ap.add_argument("--config", default="configs/model_config.yaml")
    ap.add_argument("--sequence", required=True, help="path to a SubPipe sequence dir")
    ap.add_argument("--backend", default="pytorch", choices=["pytorch", "onnx", "engine", "trt"])
    ap.add_argument("--mode", default="fused", choices=["rgb_only", "sonar_only", "fused"])
    ap.add_argument("--video", default="results/inference/fusion_demo.mp4")
    ap.add_argument("--track-json", default="results/inference/track.json")
    ap.add_argument("--no-video", action="store_true", help="skip writing the demo video")
    ap.add_argument("--fps", type=float, default=6.0, help="processing and playback rate")
    ap.add_argument("--max-seconds", type=float, default=None)
    ap.add_argument("--time-window", type=float, nargs=2, default=None, metavar=("T0", "T1"))
    ap.add_argument("--stress", action="store_true",
                    help="inject the staggered sonar/camera failures to show the handoff")
    ap.add_argument("--severity", type=float, default=1.0)
    ap.add_argument("--rgb-condition", default=RGB_CONDITION)
    ap.add_argument("--sonar-condition", default=SONAR_CONDITION)
    ap.add_argument("--rgb-window", type=_parse_pair, default=RGB_WINDOW)
    ap.add_argument("--sonar-window", type=_parse_pair, default=SONAR_WINDOW)
    args = ap.parse_args()

    cfg = load_config(args.config)

    from src.data.augmentations import DegradationPipeline
    from src.data.loaders import SequenceLoader
    from src.fusion.late_fusion import LateFusion
    from src.inference.rgb_segmenter import RGBSegmenter, SegResult
    from src.inference.sonar_detector import DetResult, SonarDetector
    from src.viz import overlay
    from src.viz.video_writer import VideoWriter

    rgb_cfg, son_cfg = cfg["models"]["rgb_seg"], cfg["models"]["sonar_det"]
    rgb_weights = rgb_cfg["engine"] if args.backend in ("engine", "trt") else rgb_cfg["weights"]
    son_weights = son_cfg["engine"] if args.backend in ("engine", "trt") else son_cfg["weights"]

    segmenter = RGBSegmenter(rgb_weights, backend=args.backend,
                             conf=rgb_cfg["conf"], iou=rgb_cfg["iou"],
                             imgsz=cfg["project"]["rgb_imgsz"])
    detector = SonarDetector(son_weights, backend=args.backend,
                             conf=son_cfg["conf"], iou=son_cfg["iou"],
                             imgsz=cfg["project"]["sonar_imgsz"])
    fusion = LateFusion(cfg)
    loader = SequenceLoader(args.sequence, cfg)

    # Processing window: nav-clip, then optional crop / cap (mirrors the ablation).
    timeline = loader._timeline
    if len(timeline) == 0:
        raise SystemExit("empty timeline: no RGB or sonar frames under the sequence dir.")
    pose_ts = loader._pose_ts
    if len(pose_ts):
        t_start = max(float(timeline[0]), float(pose_ts[0]))
        t_end = min(float(timeline[-1]), float(pose_ts[-1]))
    else:
        t_start, t_end = float(timeline[0]), float(timeline[-1])
    if args.max_seconds:
        t_end = min(t_end, t_start + args.max_seconds)
    if args.time_window:
        tw0, tw1 = (float(v) for v in args.time_window)
        t_start, t_end = max(t_start, tw0), min(t_end, tw1)
        if t_end <= t_start:
            raise SystemExit("--time-window does not overlap the nav-clipped span")
    dur = max(t_end - t_start, 1e-3)

    schedule = None
    if args.stress:
        schedule = {
            "rgb": (t_start + args.rgb_window[0] * dur, t_start + args.rgb_window[1] * dur,
                    args.rgb_condition, args.severity),
            "sonar": (t_start + args.sonar_window[0] * dur, t_start + args.sonar_window[1] * dur,
                      args.sonar_condition, args.severity),
        }
    rgb_deg = DegradationPipeline("rgb")
    son_deg = DegradationPipeline("sonar")

    print(f"[seq] {args.sequence}")
    print(f"[seq] window [{t_start:.1f}, {t_end:.1f}] ({dur:.0f}s) at {args.fps} fps | "
          f"mode={args.mode} | stress={'on' if args.stress else 'off'}")

    writer = None if args.no_video else VideoWriter(args.video, fps=args.fps)
    track_log: list[dict] = []
    tick_dt = 1.0 / float(args.fps)
    last_t = None
    n_frames = 0

    for frame in loader:
        t = float(frame.t)
        if t < t_start:
            continue
        if t > t_end:
            break
        if last_t is not None and (t - last_t) < tick_dt:
            continue
        last_t = t

        rgb_img = frame.rgb
        son_img = frame.sonar
        banner = ""
        if schedule is not None:
            rw, sw = schedule["rgb"], schedule["sonar"]
            if son_img is not None and sw[0] <= t <= sw[1]:
                son_img = son_deg.apply(son_img, sw[2], sw[3])
                banner = f"STRESS   sonar: {sw[2].replace('_', ' ')}"
            if rgb_img is not None and rw[0] <= t <= rw[1]:
                rgb_img = rgb_deg.apply(rgb_img, rw[2], rw[3])
                banner = f"STRESS   camera: {rw[2].replace('_', ' ')}"

        seg = segmenter.infer(rgb_img) if rgb_img is not None else SegResult()
        det = detector.infer(son_img) if son_img is not None else DetResult()
        fused = fusion.step(seg, det, frame.pose, mode=args.mode)

        if fused.track is not None:
            track_log.append({
                "t": fused.t,
                "position": float(fused.track.position),
                "heading": float(fused.track.heading),
                "rgb": bool(fused.rgb_contributed),
                "sonar": bool(fused.sonar_contributed),
                "coasting": bool(fused.coasting),
            })

        if writer is not None:
            rgb_panel = overlay.draw_segmentation(rgb_img, seg) if rgb_img is not None else None
            son_panel = overlay.draw_detections(son_img, det) if son_img is not None else None
            composed = _compose_frame(rgb_panel, son_panel, banner)
            if composed is None:
                continue
            writer.add(overlay.draw_track_hud(composed, fused))
        n_frames += 1

    if writer is not None:
        writer.close()
        print(f"[inference] wrote {n_frames} frames -> {args.video}")

    Path(args.track_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.track_json, "w", encoding="utf-8") as fh:
        json.dump(track_log, fh, indent=2)
    print(f"[inference] wrote {len(track_log)} track samples -> {args.track_json}")


if __name__ == "__main__":
    main()
