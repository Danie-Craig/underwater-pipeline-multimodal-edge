#!/usr/bin/env python3
"""Degradation gallery showing the MODEL'S OUTPUT under each condition.

For each modality this picks a representative validation frame, then for the
clean frame and every degradation condition it runs the trained model, overlays
the actual prediction (pipe mask for RGB, detection boxes for sonar), and stamps
the confidence and inference latency. Panels where the model fails to detect the
pipe are outlined in red. Sonar frames are cropped into the target region so the
pipe is visible rather than a thin black strip.

    python scripts/make_degradation_gallery.py --modality both
    python scripts/make_degradation_gallery.py --modality rgb --severity 0.5
    python scripts/make_degradation_gallery.py --modality sonar --image path/to/frame.png

Runs on the VM (needs the trained weights, the prepared val frames, and the
augmentation library). Writes results/robustness/<modality>_degradation_gallery.png.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import REPO_ROOT, load_config

MODELS = {
    "rgb":   {"cfg": "rgb_seg",   "subdir": "rgb_seg",   "imgsz_key": "rgb_imgsz",
              "conditions_key": "rgb_conditions",   "crop": False},
    "sonar": {"cfg": "sonar_det", "subdir": "sonar_det", "imgsz_key": "sonar_imgsz",
              "conditions_key": "sonar_conditions", "crop": True},
}
IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp")


def _val_paths(cfg: dict, subdir: str) -> tuple[Path, Path]:
    out_base = REPO_ROOT / cfg["data"].get("prep", {}).get("out_dir", "data/subpipe/yolo")
    root = out_base / subdir
    return root / "images" / "val", root / "labels" / "val"


def _pick_val_image(cfg: dict, subdir: str, override: str | None) -> Path:
    if override:
        return Path(override)
    val_img, val_lbl = _val_paths(cfg, subdir)
    imgs = sorted(p for p in val_img.glob("*") if p.suffix.lower() in IMG_EXTS)
    if not imgs:
        raise FileNotFoundError(f"no val images under {val_img} — run prepare_data.py first")
    labeled = [p for p in imgs
               if (val_lbl / f"{p.stem}.txt").exists()
               and (val_lbl / f"{p.stem}.txt").read_text(encoding="utf-8").strip()]
    pool = labeled or imgs
    return pool[len(pool) // 2]


def _sonar_crop_window(label_path: Path, W: int, H: int,
                       zoom: float = 3.0, min_side: int = 240) -> tuple[int, int, int, int] | None:
    """A rectangle around the first GT box (YOLO 'cls cx cy w h', normalized)."""
    if not label_path or not label_path.exists():
        return None
    lines = [l for l in label_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    if not lines:
        return None
    parts = lines[0].split()
    if len(parts) < 5:
        return None
    cx, cy, w, h = (float(parts[1]) * W, float(parts[2]) * H,
                    float(parts[3]) * W, float(parts[4]) * H)
    win_w = min(max(w * zoom, min_side), W)
    win_h = min(max(h * zoom, min_side), H)
    x1 = int(max(0, min(cx - win_w / 2, W - win_w)))
    y1 = int(max(0, min(cy - win_h / 2, H - win_h)))
    return x1, y1, int(x1 + win_w), int(y1 + win_h)


def _compose(modality: str, img_name: str, severity: float,
             panels: list[tuple[str, "object", str, bool]], out: Path) -> Path:
    """panels: list of (title, BGR image, stat string, present flag)."""
    import cv2
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(panels)
    cols = 3
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4.3, rows * 3.3))
    axes = np.array(axes).reshape(-1)

    for i, ax in enumerate(axes):
        if i >= n:
            ax.axis("off")
            continue
        title, img, stat, present = panels[i]
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax.set_title(title, fontsize=10, color="black")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.text(0.025, 0.965, stat, transform=ax.transAxes, fontsize=8.5,
                va="top", ha="left", color="white", family="monospace",
                bbox=dict(facecolor="black", alpha=0.6, pad=2.5, edgecolor="none"))
        edge = "#d62728" if not present else "#444444"
        for sp in ax.spines.values():
            sp.set_visible(True)
            sp.set_color(edge)
            sp.set_linewidth(2.2 if not present else 0.8)

    fig.suptitle(f"{modality.upper()} \u2014 model output under degradation "
                 f"(severity {severity}, sample frame: {img_name})", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


def build_gallery(modality: str, cfg: dict, severity: float = 0.5,
                  image_override: str | None = None, out: str | None = None) -> Path:
    import cv2
    from src.data.augmentations import DegradationPipeline
    from src.viz import overlay as ov

    spec = MODELS[modality]
    conditions = cfg["robustness"][spec["conditions_key"]]
    imgsz = int(cfg["project"][spec["imgsz_key"]])
    mcfg = cfg["models"][spec["cfg"]]
    weights = REPO_ROOT / mcfg["weights"]
    if not weights.exists():
        raise FileNotFoundError(f"weights not found: {weights}")

    img_path = _pick_val_image(cfg, spec["subdir"], image_override)
    clean_bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    if clean_bgr is None:
        raise FileNotFoundError(f"could not read image {img_path}")
    H, W = clean_bgr.shape[:2]
    degrader = DegradationPipeline(modality)

    if modality == "rgb":
        from src.inference.rgb_segmenter import RGBSegmenter
        model = RGBSegmenter(str(weights), backend="pytorch", imgsz=imgsz,
                             conf=mcfg.get("conf", 0.25), iou=mcfg.get("iou", 0.5))

        def annotate(frame):
            r = model.infer(frame)
            img = ov.draw_segmentation(frame, r)
            stat = f"conf {r.score:.2f} | {r.latency_ms:4.0f} ms" if r.present else "NO DETECTION"
            return img, stat, r.present
        crop = None
    else:
        from src.inference.sonar_detector import SonarDetector
        model = SonarDetector(str(weights), backend="pytorch", imgsz=imgsz,
                              conf=mcfg.get("conf", 0.25), iou=mcfg.get("iou", 0.5))
        _, val_lbl = _val_paths(cfg, spec["subdir"])
        crop = _sonar_crop_window(val_lbl / f"{img_path.stem}.txt", W, H)

        def annotate(frame):
            r = model.infer(frame)
            img = ov.draw_detections(frame, r)
            best = r.best
            stat = f"conf {best.score:.2f} | {r.latency_ms:4.0f} ms" if best else "NO DETECTION"
            if crop is not None:
                x1, y1, x2, y2 = crop
                img = img[y1:y2, x1:x2]
            return img, stat, r.present

    panels = []
    for label, frame in [("clean", clean_bgr)] + \
            [(c.replace("_", " "), degrader.apply(clean_bgr, c, severity)) for c in conditions]:
        img, stat, present = annotate(frame)
        panels.append((label, img, stat, present))

    out_path = Path(out or REPO_ROOT / "results" / "robustness" / f"{modality}_degradation_gallery.png")
    _compose(modality, img_path.name, severity, panels, out_path)
    print(f"[gallery] {modality}: wrote {out_path}  (sample frame {img_path.name})")
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser(description="Degradation gallery with model output overlaid.")
    ap.add_argument("--config", default="configs/model_config.yaml")
    ap.add_argument("--modality", required=True, choices=["rgb", "sonar", "both"])
    ap.add_argument("--severity", type=float, default=0.5, help="degradation severity for the grid")
    ap.add_argument("--image", default=None, help="override the sample frame (path)")
    ap.add_argument("--out", default=None, help="output PNG (single modality only)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    selection = ["rgb", "sonar"] if args.modality == "both" else [args.modality]
    for name in selection:
        build_gallery(name, cfg, severity=args.severity, image_override=args.image,
                      out=args.out if args.modality != "both" else None)


if __name__ == "__main__":
    main()
