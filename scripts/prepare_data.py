#!/usr/bin/env python3
"""Build the Ultralytics training datasets from raw SubPipe (roadmap step 2).

Converts the segmentation masks into YOLO-seg polygons and the side-scan sonar
frames + YOLO labels into a detection dataset, writes train/val splits and the
two dataset YAMLs, then verifies the result.

    python scripts/prepare_data.py
    python scripts/prepare_data.py --no-verify --val-fraction 0.15
    python scripts/prepare_data.py --copy-images        # copy instead of symlink
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import load_config


def main() -> None:
    ap = argparse.ArgumentParser(description="SubPipe → Ultralytics dataset prep.")
    ap.add_argument("--config", default="configs/model_config.yaml")
    ap.add_argument("--val-fraction", type=float, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--sonar-freq", choices=["HF", "LF"], default=None)
    ap.add_argument("--copy-images", action="store_true",
                    help="copy images instead of symlinking (slower, more disk)")
    ap.add_argument("--no-verify", action="store_true")
    ap.add_argument("--verify-samples", type=int, default=8)
    args = ap.parse_args()

    cfg = load_config(args.config)
    prep = cfg["data"].setdefault("prep", {})
    if args.val_fraction is not None:
        prep["val_fraction"] = args.val_fraction
    if args.seed is not None:
        prep["seed"] = args.seed
    if args.sonar_freq is not None:
        prep["sonar_freq"] = args.sonar_freq
    if args.copy_images:
        prep["copy_images"] = True

    from src.data.loaders import prepare_ultralytics_datasets, verify_annotations

    rgb_yaml, sonar_yaml = prepare_ultralytics_datasets(cfg, verbose=True)
    print(f"\nRGB-seg dataset : {rgb_yaml}")
    print(f"Sonar-det dataset: {sonar_yaml}")

    if not args.no_verify:
        print()
        verify_annotations(cfg, n=args.verify_samples)


if __name__ == "__main__":
    main()
