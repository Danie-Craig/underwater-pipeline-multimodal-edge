#!/usr/bin/env python3
"""Environment check — run first on each machine (roadmap step 1 acceptance).

Works on both the Thunder VM (training/dev) and the Jetson (edge inference). It
reports the Python version, every dependency it can find (with versions),
CUDA/GPU visibility, the ONNX Runtime execution providers, and the TensorRT
version. It never crashes on a missing optional dep — it just marks it — and
exits non-zero only if the core scientific stack is incomplete.

    python scripts/verify_setup.py
"""

from __future__ import annotations

import importlib
import platform
import sys
from pathlib import Path

# Make ``src`` importable when run from anywhere (no install step needed).
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

OK, WARN, BAD = "\033[92m✓\033[0m", "\033[93m•\033[0m", "\033[91m✗\033[0m"

# group -> [(import_name, friendly_name), ...]
GROUPS: dict[str, list[tuple[str, str]]] = {
    "core": [
        ("numpy", "numpy"), ("scipy", "scipy"), ("cv2", "opencv"),
        ("yaml", "PyYAML"), ("pandas", "pandas"), ("matplotlib", "matplotlib"),
        ("filterpy", "filterpy"), ("tqdm", "tqdm"),
    ],
    "training (VM)": [
        ("torch", "torch"), ("torchvision", "torchvision"), ("ultralytics", "ultralytics"),
        ("onnx", "onnx"),
    ],
    "server inference": [("onnxruntime", "onnxruntime-gpu")],
    "edge inference (Jetson)": [("tensorrt", "tensorrt"), ("pycuda", "pycuda")],
}


def _version(mod) -> str:
    for attr in ("__version__", "version", "VERSION"):
        v = getattr(mod, attr, None)
        if isinstance(v, str):
            return v
    return "?"


def check_group(name: str, items: list[tuple[str, str]]) -> dict[str, bool]:
    print(f"\n{name}:")
    found: dict[str, bool] = {}
    for import_name, friendly in items:
        try:
            mod = importlib.import_module(import_name)
            print(f"  {OK} {friendly:<16} {_version(mod)}")
            found[import_name] = True
        except Exception as exc:  # noqa: BLE001 - want to keep going
            print(f"  {WARN} {friendly:<16} not found ({type(exc).__name__})")
            found[import_name] = False
    return found


def report_cuda() -> None:
    print("\nCUDA / GPU:")
    try:
        import torch

        if torch.cuda.is_available():
            print(f"  {OK} torch CUDA available — {torch.cuda.device_count()} device(s)")
            for i in range(torch.cuda.device_count()):
                print(f"      [{i}] {torch.cuda.get_device_name(i)}")
            print(f"      torch CUDA runtime: {torch.version.cuda}")
        else:
            print(f"  {WARN} torch present but CUDA not available (CPU-only?)")
    except Exception:
        print(f"  {WARN} torch not installed — skipping CUDA check")

    try:
        import onnxruntime as ort

        print(f"  {OK} ONNX Runtime providers: {', '.join(ort.get_available_providers())}")
    except Exception:
        print(f"  {WARN} onnxruntime not installed — skipping provider check")

    try:
        import tensorrt as trt

        print(f"  {OK} TensorRT {trt.__version__} (engine build/inference available)")
    except Exception:
        print(f"  {WARN} tensorrt not importable (expected only on the Jetson)")


def report_config() -> None:
    print("\nProject config:")
    try:
        from src import REPO_ROOT, load_config

        cfg = load_config()
        print(f"  {OK} loaded configs/model_config.yaml  (project: {cfg['project']['name']})")
        print(f"  {OK} repo root resolved: {REPO_ROOT}")
    except Exception as exc:  # noqa: BLE001
        print(f"  {BAD} could not load config: {exc}")


def main() -> int:
    print("=" * 64)
    print(" underwater-pipeline-multimodal-edge — environment check")
    print("=" * 64)
    print(f"Python {platform.python_version()} on {platform.system()} ({platform.machine()})")
    if sys.version_info < (3, 10):
        print(f"  {BAD} Python 3.10+ required.")

    results = {name: check_group(name, items) for name, items in GROUPS.items()}
    report_cuda()
    report_config()

    core_ok = all(results["core"].values())
    train_ok = core_ok and all(results["training (VM)"].values())
    edge_ok = core_ok and results["edge inference (Jetson)"].get("tensorrt", False)

    print("\n" + "-" * 64)
    print(f"  core stack        : {'READY' if core_ok else 'INCOMPLETE'}")
    print(f"  training (VM)     : {'READY' if train_ok else 'incomplete (fine on Jetson)'}")
    print(f"  edge (Jetson)     : {'READY' if edge_ok else 'incomplete (fine on the VM)'}")
    print("-" * 64)

    if not core_ok:
        print(f"\n{BAD} Core dependencies missing — see requirements*.txt.")
        return 1
    print(f"\n{OK} Environment looks good for this machine's role.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
