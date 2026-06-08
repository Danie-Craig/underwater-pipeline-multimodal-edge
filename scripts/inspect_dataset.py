#!/usr/bin/env python3
"""Inspect a downloaded SubPipe tree and report its real layout (roadmap step 2).

Run this FIRST, right after unzipping, to confirm the on-disk structure matches
what the loaders expect before converting anything. It discovers chunk dirs and,
for each modality, reports counts, a sample image size, the unique mask values
(so we know if segmentation is binary vs. indexed), sample sonar label lines,
and the EstimatedState.csv columns + row count.

    python scripts/inspect_dataset.py
    python scripts/inspect_dataset.py --rgb-root data/subpipe/SubPipeMini \
        --sonar-root data/subpipe/SubPipeMini2
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import REPO_ROOT, load_config


def _resolve(p: str) -> Path:
    path = Path(p)
    return path if path.is_absolute() else REPO_ROOT / path


def main() -> None:
    ap = argparse.ArgumentParser(description="Report the real SubPipe on-disk layout.")
    ap.add_argument("--config", default="configs/model_config.yaml")
    ap.add_argument("--rgb-root", default=None, help="override data.rgb_root")
    ap.add_argument("--sonar-root", default=None, help="override data.sonar_root")
    ap.add_argument("--samples", type=int, default=3)
    args = ap.parse_args()

    cfg = load_config(args.config)
    data = cfg["data"]

    # Local imports so --help works without cv2/numpy installed.
    import cv2
    import numpy as np
    from src.data.loaders import (
        RGB_DIR, SEG_DIR, SEG_LABEL_SUFFIX, SONAR_DIRS, SONAR_IMG_SUBDIR,
        SONAR_YOLO_SUBDIR, NAV_FILE, ULTRALYTICS_IMG_EXTS, discover_chunks,
    )

    rgb_root = _resolve(args.rgb_root or data.get("rgb_root", data["root"]))
    sonar_root = _resolve(args.sonar_root or data.get("sonar_root", data["root"]))

    print("=" * 70)
    print(" SubPipe layout inspection")
    print("=" * 70)
    print(f"rgb_root  : {rgb_root}   (exists={rgb_root.exists()})")
    print(f"sonar_root: {sonar_root}   (exists={sonar_root.exists()})")

    # ---- RGB / segmentation ------------------------------------------------
    print("\n--- RGB segmentation (Cam0_images + Segmentation) ---")
    rgb_chunks = discover_chunks(rgb_root)
    print(f"discovered {len(rgb_chunks)} chunk dir(s)")
    for chunk in rgb_chunks:
        cam = chunk / RGB_DIR
        seg = chunk / SEG_DIR
        n_jpg = len(list(cam.glob("*.jpg"))) if cam.is_dir() else 0
        masks = sorted(seg.glob(f"*{SEG_LABEL_SUFFIX}.png")) if seg.is_dir() else []
        line = f"  {chunk.name}: Cam0={n_jpg} jpg | masks={len(masks)}"
        if masks:
            m = cv2.imread(str(masks[0]), cv2.IMREAD_GRAYSCALE)
            if m is not None:
                uniq = np.unique(m)
                line += f" | mask {m.shape[1]}x{m.shape[0]} values={uniq[:8].tolist()}"
        if n_jpg:
            sample = next(iter(cam.glob('*.jpg')))
            im = cv2.imread(str(sample))
            if im is not None:
                line += f" | rgb {im.shape[1]}x{im.shape[0]}"
        print(line)

    # ---- Sonar / detection -------------------------------------------------
    for freq, dirname in SONAR_DIRS.items():
        print(f"\n--- Sonar detection ({dirname}) ---")
        sonar_chunks = [c for c in discover_chunks(sonar_root)
                        if (c / dirname / SONAR_IMG_SUBDIR).is_dir()]
        print(f"discovered {len(sonar_chunks)} chunk dir(s) with {dirname}")
        for chunk in sonar_chunks:
            img_dir = chunk / dirname / SONAR_IMG_SUBDIR
            lbl_dir = chunk / dirname / SONAR_YOLO_SUBDIR
            imgs = sorted(img_dir.glob("*"))
            lbls = sorted(lbl_dir.glob("*.txt")) if lbl_dir.is_dir() else []
            line = f"  {chunk.name}: images={len(imgs)} | yolo_txt={len(lbls)}"
            if imgs:
                im = cv2.imread(str(imgs[0]), cv2.IMREAD_GRAYSCALE)
                if im is not None:
                    line += f" | sonar {im.shape[1]}x{im.shape[0]} ({imgs[0].suffix})"
            print(line)
            for lbl in lbls[: args.samples]:
                first = next((ln for ln in lbl.read_text().splitlines() if ln.strip()), "<empty>")
                print(f"      {lbl.name}: {first}")

    # ---- Navigation --------------------------------------------------------
    print("\n--- Navigation (EstimatedState.csv) ---")
    import pandas as pd
    seen = 0
    for chunk in sorted(set(rgb_chunks) | set(discover_chunks(sonar_root))):
        nav = chunk / NAV_FILE
        if not nav.exists():
            continue
        df = pd.read_csv(nav, nrows=5)
        print(f"  {chunk.name}/{NAV_FILE}: {len(pd.read_csv(nav))} rows")
        print(f"      columns: {list(df.columns)}")
        seen += 1
        if seen >= args.samples:
            break
    if seen == 0:
        print(f"  no {NAV_FILE} found — trajectory will be zero (fusion still runs)")

    print("\nIf the counts and shapes look right, run: python scripts/prepare_data.py")


if __name__ == "__main__":
    main()
