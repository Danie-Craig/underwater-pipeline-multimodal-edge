#!/usr/bin/env python3
"""Fine-tune the two baseline YOLO11-nano models on SubPipe (roadmap step 3).

Trains the RGB segmentation model (yolo11n-seg) and the sonar detection model
(yolo11n) from COCO-pretrained weights on the datasets built by
``scripts/prepare_data.py``, copies each best checkpoint to its configured
``models/*.pt`` path, records clean (undegraded) validation metrics to
``results/baselines/``, and copies the key training plots there too.

    python scripts/train.py                 # train both
    python scripts/train.py --model rgb     # just RGB segmentation
    python scripts/train.py --model sonar --epochs 60 --batch 8
    python scripts/train.py --model both --device 0 --no-pretrained

Run this on the Thunder VM (needs the GPU + the prepared datasets). The trained
weights are gitignored — copy them onward (ONNX export in step 4) before
deleting the instance, or snapshot.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import REPO_ROOT, load_config

# cli name → (config models key, dataset subdir, training imgsz key, data-yaml key)
MODELS = {
    "rgb":   ("rgb_seg",   "rgb_seg",   "rgb_imgsz",   "rgb_dataset_yaml"),
    "sonar": ("sonar_det", "sonar_det", "sonar_imgsz", "sonar_dataset_yaml"),
}
PLOTS = ("results.png", "confusion_matrix.png", "confusion_matrix_normalized.png",
         "BoxPR_curve.png", "MaskPR_curve.png", "PR_curve.png")


def resolve_dataset_yaml(cfg: dict, subdir: str, data_yaml_key: str) -> Path:
    """Locate the dataset YAML for a modality (built dir first, configured next)."""
    data = cfg["data"]
    out_base = REPO_ROOT / data.get("prep", {}).get("out_dir", "data/subpipe/yolo")
    built = out_base / subdir / f"{subdir}.yaml"
    if built.exists():
        return built
    raw = data.get(data_yaml_key, "")
    if raw:
        configured = REPO_ROOT / raw
        if configured.is_file():
            return configured
    raise FileNotFoundError(
        f"No dataset YAML for '{subdir}'. Expected {built} — run "
        f"scripts/prepare_data.py first."
    )


def save_metrics(results_dict: dict, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    # numpy scalars → plain floats so json is clean
    clean = {k: (float(v) if hasattr(v, "__float__") else v) for k, v in results_dict.items()}
    dest.write_text(json.dumps(clean, indent=2), encoding="utf-8")


def copy_plots(run_dir: Path, results_dir: Path, subdir: str) -> list[str]:
    results_dir.mkdir(parents=True, exist_ok=True)
    copied = []
    for name in PLOTS:
        src = run_dir / name
        if src.exists():
            shutil.copy2(src, results_dir / f"{subdir}_{name}")
            copied.append(name)
    return copied


def _headline(results_dict: dict) -> str:
    """Pull the most informative mAP numbers out of an Ultralytics results dict."""
    def g(*keys):
        for k in keys:
            if k in results_dict:
                return float(results_dict[k])
        return None
    box50 = g("metrics/mAP50(B)")
    box = g("metrics/mAP50-95(B)")
    seg50 = g("metrics/mAP50(M)")
    seg = g("metrics/mAP50-95(M)")
    parts = []
    if box50 is not None:
        parts.append(f"box mAP50={box50:.3f} mAP50-95={box:.3f}")
    if seg50 is not None:
        parts.append(f"mask mAP50={seg50:.3f} mAP50-95={seg:.3f}")
    return " | ".join(parts) if parts else "(no mAP keys found)"


def train_one(cli_name: str, cfg: dict, args: argparse.Namespace) -> dict:
    cfg_key, subdir, imgsz_key, data_yaml_key = MODELS[cli_name]
    mcfg = cfg["models"][cfg_key]
    tcfg = cfg["training"]

    dataset_yaml = resolve_dataset_yaml(cfg, subdir, data_yaml_key)
    imgsz = args.imgsz or int(tcfg[imgsz_key])
    epochs = args.epochs or int(tcfg["epochs"])
    batch = args.batch or int(tcfg["batch"])
    device = tcfg["device"] if args.device is None else args.device
    pretrained = tcfg.get("pretrained", True) and not args.no_pretrained
    runs_dir = REPO_ROOT / tcfg.get("runs_dir", "runs")
    results_dir = REPO_ROOT / tcfg.get("results_dir", "results/baselines")

    arch = mcfg["arch"]
    init = f"{arch}.pt" if pretrained else f"{arch}.yaml"

    print("=" * 70)
    print(f" Training {cli_name}  ({cfg_key}, task={mcfg['task']})")
    print("=" * 70)
    print(f"  init weights : {init}")
    print(f"  dataset      : {dataset_yaml}")
    print(f"  imgsz={imgsz}  epochs={epochs}  batch={batch}  device={device}")

    from ultralytics import YOLO

    model = YOLO(init)
    model.train(
        data=str(dataset_yaml),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        workers=int(tcfg.get("workers", 8)),
        seed=int(tcfg.get("seed", 0)),
        patience=int(tcfg.get("patience", 20)),
        project=str(runs_dir),
        name=subdir,
        exist_ok=True,
        plots=True,
    )

    run_dir = Path(model.trainer.save_dir)
    best = Path(model.trainer.best)                       # runs/<task>/<subdir>/weights/best.pt
    dst_weights = REPO_ROOT / mcfg["weights"]
    dst_weights.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best, dst_weights)

    # Clean validation metrics on the held-out split.
    metrics = model.val()
    results_dict = dict(getattr(metrics, "results_dict", {}) or {})
    metrics_path = results_dir / f"{subdir}_clean_metrics.json"
    save_metrics(results_dict, metrics_path)
    plots = copy_plots(run_dir, results_dir, subdir)

    print(f"\n  ✓ best weights → {dst_weights.relative_to(REPO_ROOT)}")
    print(f"  ✓ metrics      → {metrics_path.relative_to(REPO_ROOT)}")
    print(f"  ✓ plots copied : {', '.join(plots) if plots else '(none found)'}")
    print(f"  ▶ {_headline(results_dict)}")
    return {"model": cli_name, "weights": str(dst_weights), "metrics": results_dict,
            "headline": _headline(results_dict)}


def main() -> None:
    ap = argparse.ArgumentParser(description="Fine-tune the SubPipe baseline models.")
    ap.add_argument("--config", default="configs/model_config.yaml")
    ap.add_argument("--model", choices=["rgb", "sonar", "both"], default="both")
    ap.add_argument("--epochs", type=int, default=None, help="override training.epochs")
    ap.add_argument("--imgsz", type=int, default=None, help="override per-model imgsz")
    ap.add_argument("--batch", type=int, default=None, help="override training.batch")
    ap.add_argument("--device", default=None, help="override GPU index (e.g. 0 or cpu)")
    ap.add_argument("--no-pretrained", action="store_true", help="train from scratch")
    args = ap.parse_args()

    cfg = load_config(args.config)
    selection = ["rgb", "sonar"] if args.model == "both" else [args.model]

    summary = [train_one(name, cfg, args) for name in selection]

    print("\n" + "=" * 70)
    print(" baseline training summary")
    print("=" * 70)
    for s in summary:
        print(f"  {s['model']:6s}: {s['headline']}")
    print("\nClean metrics + plots are in results/baselines/ (commit those).")
    print("Weights are in models/ (gitignored) — keep them for step 4 (ONNX export).")


if __name__ == "__main__":
    main()
