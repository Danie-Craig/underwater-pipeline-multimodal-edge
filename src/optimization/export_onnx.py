"""Export trained models to ONNX (§7, step 4, run on the Thunder VM).

ONNX is architecture-neutral, so the ``.onnx`` produced here is the artifact
that travels to the Jetson, where the TensorRT engine is built on-device.

This uses the Ultralytics export API directly and is runnable as-is once the
trained weights exist. Only the FP16/INT8 *engine* build is Jetson-side — see
``tensorrt_inference``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src import REPO_ROOT, load_config


def export_model(weights: str | Path, opset: int, imgsz: int, dynamic: bool, task: str) -> Path:
    """Export one Ultralytics model to ONNX and return the output path.

    We export in FP32 here on purpose: precision (FP16/INT8) is applied when
    the TensorRT engine is built on the Orin, not at ONNX-export time.
    """
    from ultralytics import YOLO

    model = YOLO(str(weights), task=task)
    out = model.export(
        format="onnx",
        opset=opset,
        imgsz=imgsz,
        dynamic=dynamic,
        half=False,        # keep ONNX in FP32; quantize at TRT build time
        simplify=True,
    )
    return Path(out)


def main() -> None:
    ap = argparse.ArgumentParser(description="Export YOLO11 models to ONNX (step 4).")
    ap.add_argument("--config", default="configs/model_config.yaml")
    ap.add_argument(
        "--models",
        nargs="+",
        default=["rgb_seg", "sonar_det"],
        help="which model keys from the config to export",
    )
    args = ap.parse_args()

    cfg = load_config(args.config)
    opt = cfg["optimization"]
    opset, dynamic = int(opt["onnx_opset"]), bool(opt["dynamic_batch"])

    onnx_dir = REPO_ROOT / "models" / "onnx"
    onnx_dir.mkdir(parents=True, exist_ok=True)

    for key in args.models:
        m = cfg["models"][key]
        imgsz = cfg["project"]["rgb_imgsz"] if key == "rgb_seg" else cfg["project"]["sonar_imgsz"]
        print(f"[export] {key}: {m['weights']}  →  ONNX (opset={opset}, imgsz={imgsz})")
        produced = export_model(m["weights"], opset, imgsz, dynamic, m["task"])
        target = REPO_ROOT / m["onnx"]
        if produced.resolve() != target.resolve():
            target.parent.mkdir(parents=True, exist_ok=True)
            produced.replace(target)
        print(f"[export] {key}: wrote {target.relative_to(REPO_ROOT)}")

    print(
        "\nNext: copy models/onnx/*.onnx to the Jetson and build the TensorRT "
        "engines there (see src/optimization/tensorrt_inference.py)."
    )


if __name__ == "__main__":
    main()
