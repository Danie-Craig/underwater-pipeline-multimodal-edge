#!/usr/bin/env python3
"""Run the full pipeline over one sequence (§6 demo; roadmap steps 2/6).

RGB segmentation + sonar detection + along-track fusion, with an annotated
demo video and a per-frame track JSON. Backend is selectable so the same
script drives the PyTorch baseline on the VM and the TensorRT path on the Orin.

    python scripts/run_inference.py --sequence data/subpipe/<seq> --backend pytorch
    python scripts/run_inference.py --sequence data/subpipe/<seq> --backend trt \
        --mode fused --video results/inference/demo.mp4
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import load_config


def main() -> None:
    ap = argparse.ArgumentParser(description="Run RGB seg + sonar detect + fused track.")
    ap.add_argument("--config", default="configs/model_config.yaml")
    ap.add_argument("--sequence", required=True, help="path to a SubPipe sequence dir")
    ap.add_argument("--backend", default="pytorch", choices=["pytorch", "onnx", "engine", "trt"])
    ap.add_argument("--mode", default="fused", choices=["rgb_only", "sonar_only", "fused"])
    ap.add_argument("--video", default="results/inference/demo.mp4")
    ap.add_argument("--track-json", default="results/inference/track.json")
    ap.add_argument("--no-video", action="store_true", help="skip writing the demo video")
    args = ap.parse_args()

    cfg = load_config(args.config)

    # Heavy imports are local so --help works without the full stack installed.
    from src.data.loaders import SequenceLoader
    from src.fusion.late_fusion import LateFusion
    from src.inference.rgb_segmenter import RGBSegmenter
    from src.inference.sonar_detector import SonarDetector
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

    writer = None if args.no_video else VideoWriter(args.video, fps=10.0)
    track_log: list[dict] = []

    for frame in loader:
        seg = segmenter.infer(frame.rgb) if frame.rgb is not None else _empty_seg()
        det = detector.infer(frame.sonar) if frame.sonar is not None else _empty_det()
        fused = fusion.step(seg, det, frame.pose, mode=args.mode)

        if fused.track is not None:
            track_log.append({
                "t": fused.t,
                "position": fused.track.position,
                "heading": fused.track.heading,
                "rgb": fused.rgb_contributed,
                "sonar": fused.sonar_contributed,
                "coasting": fused.coasting,
            })

        if writer is not None:
            rgb_panel = overlay.draw_segmentation(frame.rgb, seg) if frame.rgb is not None else None
            son_panel = overlay.draw_detections(frame.sonar, det) if frame.sonar is not None else None
            panels = [p for p in (rgb_panel, son_panel) if p is not None]
            composed = overlay.hstack_panels(*panels)
            writer.add(overlay.draw_track_hud(composed, fused))

    if writer is not None:
        writer.close()
        print(f"[inference] wrote demo video → {args.video}")

    Path(args.track_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.track_json, "w", encoding="utf-8") as fh:
        json.dump(track_log, fh, indent=2)
    print(f"[inference] wrote {len(track_log)} track samples → {args.track_json}")


def _empty_seg():
    from src.inference.rgb_segmenter import SegResult
    return SegResult()


def _empty_det():
    from src.inference.sonar_detector import DetResult
    return DetResult()


if __name__ == "__main__":
    main()
