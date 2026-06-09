#!/usr/bin/env python3
"""Degradation galleries showing the MODEL'S OUTPUT under each condition.

Two figures are produced per modality:

1. <modality>_degradation_gallery.png
   A single sample frame: the clean frame plus every condition at one severity,
   with the model's prediction overlaid (pipe mask for RGB, boxes for sonar),
   confidence + latency stamped, and a red border on panels where the model
   fails to detect the pipe.

2. <modality>_degradation_grid.png   (when --samples > 1)
   Several different sample frames: conditions down the rows, sample frames
   across the columns, all at one severity, with the model output overlaid.

Sonar frames are cropped into the target region so the pipe is visible.

    python scripts/make_degradation_gallery.py --modality both
    python scripts/make_degradation_gallery.py --modality both --samples 6 --severity 0.5
    python scripts/make_degradation_gallery.py --modality rgb --image path/to/frame.png

Runs on the VM (needs the trained weights, the prepared val frames, and the
augmentation library).
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


def _labeled_val_images(cfg: dict, subdir: str) -> list[Path]:
    val_img, val_lbl = _val_paths(cfg, subdir)
    imgs = sorted(p for p in val_img.glob("*") if p.suffix.lower() in IMG_EXTS)
    if not imgs:
        raise FileNotFoundError(f"no val images under {val_img} — run prepare_data.py first")
    labeled = [p for p in imgs
               if (val_lbl / f"{p.stem}.txt").exists()
               and (val_lbl / f"{p.stem}.txt").read_text(encoding="utf-8").strip()]
    return labeled or imgs


def _pick_spread(items: list[Path], n: int) -> list[Path]:
    """Evenly spaced selection of n items for visual variety."""
    if n >= len(items):
        return items
    idx = [round(i * (len(items) - 1) / (n - 1)) for i in range(n)] if n > 1 else [len(items) // 2]
    seen, out = set(), []
    for i in idx:
        if i not in seen:
            seen.add(i)
            out.append(items[i])
    return out


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


def _load_model(modality: str, cfg: dict):
    spec = MODELS[modality]
    mcfg = cfg["models"][spec["cfg"]]
    imgsz = int(cfg["project"][spec["imgsz_key"]])
    weights = REPO_ROOT / mcfg["weights"]
    if not weights.exists():
        raise FileNotFoundError(f"weights not found: {weights}")
    if modality == "rgb":
        from src.inference.rgb_segmenter import RGBSegmenter
        return RGBSegmenter(str(weights), backend="pytorch", imgsz=imgsz,
                            conf=mcfg.get("conf", 0.25), iou=mcfg.get("iou", 0.5))
    from src.inference.sonar_detector import SonarDetector
    return SonarDetector(str(weights), backend="pytorch", imgsz=imgsz,
                         conf=mcfg.get("conf", 0.25), iou=mcfg.get("iou", 0.5))


def _annotate(model, modality: str, frame, crop, ov):
    """Run the model on one frame; return (BGR overlay, stat string, present)."""
    if modality == "rgb":
        r = model.infer(frame)
        img = ov.draw_segmentation(frame, r)
        stat = f"conf {r.score:.2f} | {r.latency_ms:4.0f} ms" if r.present else "NO DETECTION"
        return img, stat, r.present
    r = model.infer(frame)
    img = ov.draw_detections(frame, r)
    best = r.best
    stat = f"conf {best.score:.2f} | {r.latency_ms:4.0f} ms" if best else "NO DETECTION"
    if crop is not None:
        x1, y1, x2, y2 = crop
        img = img[y1:y2, x1:x2]
    return img, stat, r.present


def _style_axis(ax, img_rgb, *, present, title=None, ylabel=None, stat=None):
    ax.imshow(img_rgb)
    ax.set_xticks([])
    ax.set_yticks([])
    if title is not None:
        ax.set_title(title, fontsize=10, color="black")
    if ylabel is not None:
        ax.set_ylabel(ylabel, fontsize=9, rotation=0, ha="right", va="center", labelpad=8)
    if stat is not None:
        ax.text(0.025, 0.965, stat, transform=ax.transAxes, fontsize=8.5, va="top",
                ha="left", color="white", family="monospace",
                bbox=dict(facecolor="black", alpha=0.6, pad=2.5, edgecolor="none"))
    edge = "#d62728" if not present else "#444444"
    for sp in ax.spines.values():
        sp.set_visible(True)
        sp.set_color(edge)
        sp.set_linewidth(2.2 if not present else 0.8)


def build_gallery(modality: str, cfg: dict, severity: float, image_override: str | None,
                  out: str | None, model=None, ov=None) -> Path:
    import cv2
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from src.data.augmentations import DegradationPipeline, mask_from_yolo_label
    if ov is None:
        from src.viz import overlay as ov
    if model is None:
        model = _load_model(modality, cfg)

    spec = MODELS[modality]
    conditions = cfg["robustness"][spec["conditions_key"]]
    degrader = DegradationPipeline(modality)

    img_path = (Path(image_override) if image_override
                else _pick_spread(_labeled_val_images(cfg, spec["subdir"]), 1)[0])
    clean = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    if clean is None:
        raise FileNotFoundError(f"could not read image {img_path}")
    H, W = clean.shape[:2]
    _, val_lbl = _val_paths(cfg, spec["subdir"])
    crop = _sonar_crop_window(val_lbl / f"{img_path.stem}.txt", W, H) if spec["crop"] else None
    pipe_mask = mask_from_yolo_label(val_lbl / f"{img_path.stem}.txt", W, H)

    frames = [("clean", clean)] + \
             [(c.replace("_", " "),
               degrader.apply(clean, c, severity,
                              mask=pipe_mask if c == "sand_occlusion" else None))
              for c in conditions]
    n, cols = len(frames), 3
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4.3, rows * 3.3))
    axes = np.array(axes).reshape(-1)
    for i, ax in enumerate(axes):
        if i >= n:
            ax.axis("off")
            continue
        title, frame = frames[i]
        img, stat, present = _annotate(model, modality, frame, crop, ov)
        _style_axis(ax, cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
                    present=present, title=title, stat=stat)
    fig.suptitle(f"{modality.upper()} \u2014 model output under degradation "
                 f"(severity {severity}, sample frame: {img_path.name})", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out_path = Path(out or REPO_ROOT / "results" / "robustness" / f"{modality}_degradation_gallery.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"[gallery] {modality}: wrote {out_path}  (sample frame {img_path.name})")
    return out_path


def build_sample_grid(modality: str, cfg: dict, n_samples: int, severity: float,
                      model=None, ov=None) -> Path:
    import cv2
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from src.data.augmentations import DegradationPipeline, mask_from_yolo_label
    if ov is None:
        from src.viz import overlay as ov
    if model is None:
        model = _load_model(modality, cfg)

    spec = MODELS[modality]
    conditions = cfg["robustness"][spec["conditions_key"]]
    degrader = DegradationPipeline(modality)
    _, val_lbl = _val_paths(cfg, spec["subdir"])

    samples = _pick_spread(_labeled_val_images(cfg, spec["subdir"]), n_samples)
    loaded = []
    for p in samples:
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if img is None:
            continue
        H, W = img.shape[:2]
        crop = _sonar_crop_window(val_lbl / f"{p.stem}.txt", W, H) if spec["crop"] else None
        pipe_mask = mask_from_yolo_label(val_lbl / f"{p.stem}.txt", W, H)
        loaded.append((p, img, crop, pipe_mask))

    row_labels = ["clean"] + [c.replace("_", " ") for c in conditions]
    n_rows, n_cols = len(row_labels), len(loaded)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3.4, n_rows * 2.4))
    axes = np.array(axes).reshape(n_rows, n_cols)

    for c, (p, clean, crop, pipe_mask) in enumerate(loaded):
        for r, label in enumerate(row_labels):
            ax = axes[r][c]
            cond = conditions[r - 1] if r > 0 else None
            frame = clean if r == 0 else degrader.apply(
                clean, cond, severity,
                mask=pipe_mask if cond == "sand_occlusion" else None)
            img, _, present = _annotate(model, modality, frame, crop, ov)
            _style_axis(ax, cv2.cvtColor(img, cv2.COLOR_BGR2RGB), present=present,
                        title=(f"sample {c + 1}" if r == 0 else None),
                        ylabel=(label if c == 0 else None))

    fig.suptitle(f"{modality.upper()} \u2014 model output across {n_cols} frames "
                 f"(rows = condition at severity {severity})", fontsize=13)
    fig.tight_layout(rect=[0.12, 0, 1, 0.97])
    out_path = REPO_ROOT / "results" / "robustness" / f"{modality}_degradation_grid.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"[grid] {modality}: wrote {out_path}  ({n_cols} frames x {n_rows} rows)")
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser(description="Degradation galleries with model output overlaid.")
    ap.add_argument("--config", default="configs/model_config.yaml")
    ap.add_argument("--modality", required=True, choices=["rgb", "sonar", "both"])
    ap.add_argument("--severity", type=float, default=0.5, help="degradation severity")
    ap.add_argument("--samples", type=int, default=4,
                    help="frames in the multi-sample grid (>1 enables it)")
    ap.add_argument("--image", default=None, help="override the single-gallery sample frame")
    ap.add_argument("--out", default=None, help="single-gallery output PNG (one modality only)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    selection = ["rgb", "sonar"] if args.modality == "both" else [args.modality]
    from src.viz import overlay as ov
    for name in selection:
        model = _load_model(name, cfg)
        build_gallery(name, cfg, severity=args.severity, image_override=args.image,
                      out=args.out if args.modality != "both" else None, model=model, ov=ov)
        if args.samples and args.samples > 1:
            build_sample_grid(name, cfg, n_samples=args.samples, severity=args.severity,
                              model=model, ov=ov)


if __name__ == "__main__":
    main()
