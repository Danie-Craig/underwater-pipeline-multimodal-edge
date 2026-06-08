#!/usr/bin/env python3
"""Per-modality robustness sweep (§8; roadmap step 5).

Apply each degradation condition at several severities to one modality's
frames, run that modality's model, and record detection/segmentation rate,
average confidence, and FPS per (condition, severity). Results land in
results/robustness/ and feed the failure analysis (§9).

    python scripts/evaluate_robustness.py --modality rgb   --sequence data/subpipe/<seq>
    python scripts/evaluate_robustness.py --modality sonar --sequence data/subpipe/<seq>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import load_config


def main() -> None:
    ap = argparse.ArgumentParser(description="Run the per-modality degradation sweep.")
    ap.add_argument("--config", default="configs/model_config.yaml")
    ap.add_argument("--modality", required=True, choices=["rgb", "sonar"])
    ap.add_argument("--sequence", required=True, help="path to a SubPipe sequence dir")
    ap.add_argument("--backend", default="pytorch", choices=["pytorch", "onnx", "engine", "trt"])
    ap.add_argument("--out", default=None, help="output JSON (default: results/robustness/<modality>.json)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    rob = cfg["robustness"]
    severities = rob["severities"]
    conditions = rob["rgb_conditions"] if args.modality == "rgb" else rob["sonar_conditions"]

    from src.data.augmentations import DegradationPipeline
    from src.data.loaders import SequenceLoader

    degrader = DegradationPipeline(args.modality)

    # Pick the model for this modality.
    if args.modality == "rgb":
        from src.inference.rgb_segmenter import RGBSegmenter
        m = cfg["models"]["rgb_seg"]
        weights = m["engine"] if args.backend in ("engine", "trt") else m["weights"]
        model = RGBSegmenter(weights, backend=args.backend, conf=m["conf"],
                             iou=m["iou"], imgsz=cfg["project"]["rgb_imgsz"])
        def detected(res) -> tuple[bool, float, float]:
            return res.present, res.score, res.latency_ms
        pick_image = lambda fr: fr.rgb
    else:
        from src.inference.sonar_detector import SonarDetector
        m = cfg["models"]["sonar_det"]
        weights = m["engine"] if args.backend in ("engine", "trt") else m["weights"]
        model = SonarDetector(weights, backend=args.backend, conf=m["conf"],
                              iou=m["iou"], imgsz=cfg["project"]["sonar_imgsz"])
        def detected(res) -> tuple[bool, float, float]:
            best = res.best
            return res.present, (best.score if best else 0.0), res.latency_ms
        pick_image = lambda fr: fr.sonar

    # Sweep: for each condition × severity, accumulate metrics over the sequence.
    table: list[dict] = []
    for condition in conditions:
        for sev in severities:
            n = hits = 0
            conf_sum = lat_sum = 0.0
            for frame in SequenceLoader(args.sequence, cfg):
                img = pick_image(frame)
                if img is None:
                    continue
                degraded = degrader.apply(img, condition, sev)
                present, score, lat_ms = detected(model.infer(degraded))
                n += 1
                hits += int(present)
                conf_sum += score
                lat_sum += lat_ms
            if n:
                table.append({
                    "condition": condition,
                    "severity": sev,
                    "detection_rate": round(hits / n, 4),
                    "avg_confidence": round(conf_sum / n, 4),
                    "fps": round(1000.0 / (lat_sum / n), 1) if lat_sum else None,
                    "frames": n,
                })
                print(f"  {condition:<20} sev={sev:<4} rate={hits/n:5.2f} "
                      f"conf={conf_sum/n:5.2f} n={n}")

    out = Path(args.out or f"results/robustness/{args.modality}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        json.dump(table, fh, indent=2)
    print(f"\n[robustness] wrote {len(table)} rows → {out}")


if __name__ == "__main__":
    main()
