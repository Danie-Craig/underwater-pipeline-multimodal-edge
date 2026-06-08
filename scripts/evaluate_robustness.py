#!/usr/bin/env python3
"""Per-modality robustness sweep.

Degrade the held-out validation set with each underwater condition across a
range of severities, re-run validation, and record how mAP falls off. This is
the evidence that motivates sensor fusion: each modality has conditions under
which it collapses, and they are rarely the same conditions.

For every (condition, severity) we write a degraded copy of the val images
(labels unchanged), run Ultralytics validation to get mAP, and tabulate it. A
clean baseline (no degradation) anchors every curve. Outputs, per modality, a
JSON grid and a mAP-vs-severity plot under results/robustness/.

    python scripts/evaluate_robustness.py --modality rgb
    python scripts/evaluate_robustness.py --modality sonar
    python scripts/evaluate_robustness.py --modality both

Runs on the VM (needs the trained weights and the prepared val splits).
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import REPO_ROOT, load_config

MODELS = {
    "rgb":   {"cfg": "rgb_seg",   "subdir": "rgb_seg",   "imgsz_key": "rgb_imgsz",
              "conditions_key": "rgb_conditions",   "task": "segment"},
    "sonar": {"cfg": "sonar_det", "subdir": "sonar_det", "imgsz_key": "sonar_imgsz",
              "conditions_key": "sonar_conditions", "task": "detect"},
}
IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp")


def _val_dirs(cfg: dict, subdir: str) -> tuple[Path, Path]:
    out_base = REPO_ROOT / cfg["data"].get("prep", {}).get("out_dir", "data/subpipe/yolo")
    root = out_base / subdir
    return root / "images" / "val", root / "labels" / "val"


def _write_dataset_yaml(out_dir: Path, names: dict) -> Path:
    import yaml
    yaml_path = out_dir / "data.yaml"
    with yaml_path.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(
            {"path": str(out_dir.resolve()), "train": "images/val",
             "val": "images/val", "names": names},
            fh, sort_keys=False,
        )
    return yaml_path


def degrade_val_set(images_dir: Path, labels_dir: Path, modality: str,
                    condition: str | None, severity: float,
                    out_dir: Path, names: dict) -> Path:
    """Build a degraded copy of the val set under out_dir; return its data.yaml.

    ``condition=None`` or ``severity == 0`` copies the images unchanged (the
    clean baseline). Labels are always copied verbatim so the geometry matches.
    """
    import cv2
    from src.data.augmentations import DegradationPipeline

    img_out = out_dir / "images" / "val"
    lbl_out = out_dir / "labels" / "val"
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)
    degrader = DegradationPipeline(modality) if condition else None

    for img_path in sorted(images_dir.glob("*")):
        if img_path.suffix.lower() not in IMG_EXTS:
            continue
        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img is None:
            continue
        if degrader is not None and severity > 0:
            img = degrader.apply(img, condition, severity)
        cv2.imwrite(str(img_out / f"{img_path.stem}.png"), img)
        lbl = labels_dir / f"{img_path.stem}.txt"
        (lbl_out / f"{img_path.stem}.txt").write_text(
            lbl.read_text(encoding="utf-8") if lbl.exists() else "", encoding="utf-8"
        )
    return _write_dataset_yaml(out_dir, names)


def _plot(out: dict, png_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    severities = sorted({r["severity"] for r in out["grid"]})
    conditions = list(dict.fromkeys(r["condition"] for r in out["grid"]))
    clean = out["clean"]["map50"]

    plt.figure(figsize=(9, 6))
    for cond in conditions:
        ys = [clean]
        for s in severities:
            match = next((r["map50"] for r in out["grid"]
                          if r["condition"] == cond and r["severity"] == s), None)
            ys.append(match)
        plt.plot([0.0] + severities, ys, marker="o", linewidth=1.8, label=cond)

    plt.axhline(clean, ls="--", color="grey", alpha=0.6, label="clean baseline")
    plt.xlabel("degradation severity")
    plt.ylabel(out["metric"])
    plt.title(f"{out['modality'].upper()} robustness: {out['metric']} vs severity")
    plt.ylim(0, 1.0)
    plt.grid(alpha=0.3)
    plt.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(png_path, dpi=130)
    plt.close()


def _extract(results_dict: dict, seg: bool) -> dict:
    def f(key):
        v = results_dict.get(key)
        return float(v) if v is not None else None
    if seg:
        return {"map50": f("metrics/mAP50(M)"), "map50_95": f("metrics/mAP50-95(M)"),
                "box_map50": f("metrics/mAP50(B)")}
    return {"map50": f("metrics/mAP50(B)"), "map50_95": f("metrics/mAP50-95(B)")}


def run_modality(cli_name: str, cfg: dict) -> dict:
    spec = MODELS[cli_name]
    mcfg = cfg["models"][spec["cfg"]]
    rob = cfg["robustness"]
    severities = rob["severities"]
    conditions = rob[spec["conditions_key"]]
    imgsz = int(cfg["project"][spec["imgsz_key"]])
    seg = spec["task"] == "segment"
    names = {0: "pipe"}

    images_dir, labels_dir = _val_dirs(cfg, spec["subdir"])
    if not images_dir.is_dir():
        raise FileNotFoundError(
            f"Val images not found at {images_dir} — run scripts/prepare_data.py first."
        )
    n_val = sum(1 for p in images_dir.glob('*') if p.suffix.lower() in IMG_EXTS)

    from ultralytics import YOLO
    weights = REPO_ROOT / mcfg["weights"]
    model = YOLO(str(weights))

    print("=" * 70)
    print(f" Robustness sweep: {cli_name}  ({spec['cfg']}, {n_val} val images)")
    print(f" {len(conditions)} conditions x {len(severities)} severities at imgsz={imgsz}")
    print("=" * 70)

    results_dir = REPO_ROOT / "results" / "robustness"
    results_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        clean_yaml = degrade_val_set(images_dir, labels_dir, cli_name, None, 0.0,
                                     td / "clean", names)
        clean = _extract(model.val(data=str(clean_yaml), imgsz=imgsz,
                                   verbose=False, plots=False).results_dict, seg)
        print(f"  clean baseline: mAP50={clean['map50']:.3f}  mAP50-95={clean['map50_95']:.3f}")

        grid: list[dict] = []
        for cond in conditions:
            for sev in severities:
                work = td / f"{cond}_{sev}"
                y = degrade_val_set(images_dir, labels_dir, cli_name, cond, sev, work, names)
                rd = model.val(data=str(y), imgsz=imgsz, verbose=False, plots=False).results_dict
                row = {"condition": cond, "severity": sev, **_extract(rd, seg)}
                grid.append(row)
                drop = clean["map50"] - row["map50"] if row["map50"] is not None else 0.0
                print(f"  {cond:<20} sev={sev:<4} mAP50={row['map50']:.3f}  (drop {drop:+.3f})")
                shutil.rmtree(work, ignore_errors=True)

    out = {"modality": cli_name, "task": spec["task"], "imgsz": imgsz,
           "metric": "mask mAP50" if seg else "box mAP50",
           "clean": clean, "grid": grid}
    (results_dir / f"{cli_name}.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    _plot(out, results_dir / f"{cli_name}_robustness.png")
    print(f"\n  wrote results/robustness/{cli_name}.json + {cli_name}_robustness.png")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Per-modality degradation sweep (mAP vs severity).")
    ap.add_argument("--config", default="configs/model_config.yaml")
    ap.add_argument("--modality", required=True, choices=["rgb", "sonar", "both"])
    args = ap.parse_args()

    cfg = load_config(args.config)
    selection = ["rgb", "sonar"] if args.modality == "both" else [args.modality]
    for name in selection:
        run_modality(name, cfg)


if __name__ == "__main__":
    main()
