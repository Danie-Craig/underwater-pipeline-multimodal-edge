#!/usr/bin/env python3
"""Render a degradation gallery: a clean frame beside each condition's severities.

Makes the degradation suite tangible and pairs directly with the mAP-vs-severity
plots. For each modality this picks a representative validation frame (one with a
visible target), applies every condition at the configured severities, and tiles
the result into a labeled contact sheet at
results/robustness/<modality>_degradation_gallery.png.

    python scripts/make_degradation_gallery.py --modality both
    python scripts/make_degradation_gallery.py --modality rgb --image path/to/frame.png

Runs on the VM (needs the prepared val frames and the augmentation library).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import REPO_ROOT, load_config

MODELS = {
    "rgb":   {"subdir": "rgb_seg",   "conditions_key": "rgb_conditions"},
    "sonar": {"subdir": "sonar_det", "conditions_key": "sonar_conditions"},
}
IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp")


def _pick_val_image(cfg: dict, subdir: str, override: str | None) -> Path:
    if override:
        return Path(override)
    out_base = REPO_ROOT / cfg["data"].get("prep", {}).get("out_dir", "data/subpipe/yolo")
    root = out_base / subdir
    val_img, val_lbl = root / "images" / "val", root / "labels" / "val"
    imgs = sorted(p for p in val_img.glob("*") if p.suffix.lower() in IMG_EXTS)
    if not imgs:
        raise FileNotFoundError(f"no val images under {val_img} — run prepare_data.py first")
    labeled = [p for p in imgs
               if (val_lbl / f"{p.stem}.txt").exists()
               and (val_lbl / f"{p.stem}.txt").read_text(encoding="utf-8").strip()]
    pool = labeled or imgs
    return pool[len(pool) // 2]


def build_gallery(modality: str, cfg: dict, image_override: str | None = None,
                  out: str | None = None) -> Path:
    import cv2
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from src.data.augmentations import DegradationPipeline

    spec = MODELS[modality]
    conditions = cfg["robustness"][spec["conditions_key"]]
    severities = cfg["robustness"]["severities"]

    img_path = _pick_val_image(cfg, spec["subdir"], image_override)
    clean_bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    if clean_bgr is None:
        raise FileNotFoundError(f"could not read image {img_path}")
    clean_rgb = cv2.cvtColor(clean_bgr, cv2.COLOR_BGR2RGB)
    degrader = DegradationPipeline(modality)

    col_titles = ["clean"] + [f"severity {s}" for s in severities]
    n_rows, n_cols = len(conditions), len(col_titles)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2.6, n_rows * 2.1))
    axes = axes.reshape(n_rows, n_cols)

    for r, cond in enumerate(conditions):
        for c in range(n_cols):
            ax = axes[r][c]
            if c == 0:
                ax.imshow(clean_rgb)
            else:
                degraded = degrader.apply(clean_bgr, cond, severities[c - 1])
                ax.imshow(cv2.cvtColor(degraded, cv2.COLOR_BGR2RGB))
            ax.set_xticks([])
            ax.set_yticks([])
            if r == 0:
                ax.set_title(col_titles[c], fontsize=10)
            if c == 0:
                ax.set_ylabel(cond, fontsize=9, rotation=0, ha="right", va="center", labelpad=8)

    fig.suptitle(f"{modality.upper()} degradation suite   (sample frame: {img_path.name})",
                 fontsize=13)
    fig.tight_layout(rect=[0.10, 0.0, 1.0, 0.98])
    out_path = Path(out or REPO_ROOT / "results" / "robustness" / f"{modality}_degradation_gallery.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"[gallery] {modality}: wrote {out_path}  (sample frame {img_path.name})")
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser(description="Render a per-condition degradation gallery.")
    ap.add_argument("--config", default="configs/model_config.yaml")
    ap.add_argument("--modality", required=True, choices=["rgb", "sonar", "both"])
    ap.add_argument("--image", default=None, help="override the sample frame (path)")
    ap.add_argument("--out", default=None, help="output PNG (single modality only)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    selection = ["rgb", "sonar"] if args.modality == "both" else [args.modality]
    for name in selection:
        build_gallery(name, cfg, image_override=args.image,
                      out=args.out if args.modality != "both" else None)


if __name__ == "__main__":
    main()
