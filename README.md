# Multimodal Underwater Pipeline Inspection on the Edge

Fuse an **optical (RGB) segmentation** model with an **acoustic (side-scan sonar)
detection** model for submarine-pipeline inspection, accelerate the pipeline on
an **NVIDIA Jetson AGX Orin** with **ONNX → TensorRT**, and prove — with a clean
benchmark and robustness sweep — that the multimodal system is faster *and* more
reliable than either sensor alone.

> Train two lightweight perception models on real underwater-robot data
> (**SubPipe**), fuse them along the vehicle's INS trajectory, deploy and
> accelerate them on the edge, and show the fused track stays locked when either
> single modality would lose the pipe.

## Architecture

```
RGB camera   ──► [ RGB segmentation (YOLO11-seg) ] ──► pipe mask ──┐
                                                                    ├─► [ Late fusion + Kalman ] ─► fused pipe track
Side-scan    ──► [ Sonar detection (YOLO11) ]      ──► boxes ──────┘        (along the AUV trajectory)
sonar                                                               ▲
INS / nav    ───────────────────────────────────────── vehicle pose / along-track position
```

The camera and sonar do **not** share a viewpoint, so fusion is **track-level**
along the AUV's INS trajectory — not pixel-to-pixel. Each modality contributes a
position measurement when confident; the Kalman track coasts through dropouts.

## Repository layout

```
configs/        model_config.yaml — models, thresholds, fusion + Kalman, gates
data/subpipe/   SubPipe / SubPipeMini (gitignored)
models/         onnx/ (exports) + trt_cache/ (engines, built on the Jetson) — gitignored
results/        benchmark/ robustness/ fusion/ inference/
scripts/        verify_setup · run_inference · run_benchmark · evaluate_robustness
src/
  data/         loaders.py · augmentations.py (RGB + sonar degradations)
  inference/    rgb_segmenter.py · sonar_detector.py (multi-backend wrappers)
  fusion/       late_fusion.py · kalman_tracker.py
  optimization/ export_onnx.py · tensorrt_inference.py · benchmark.py
  viz/          overlay.py · video_writer.py
requirements.txt          (Thunder VM: training + dev)
requirements-jetson.txt   (Jetson: edge inference)
```

## Two machines, one bridge

```
Thunder VM: train + export ONNX ──┬─► push code ──► GitHub ──► git clone ──► Jetson
                                  └─► copy the .onnx ───────────────────────► Jetson: build TensorRT engine + benchmark
results ◄────────── pulled to your laptop to view
```

ONNX (architecture-neutral) carries each model from the VM to the Jetson; the
TensorRT engine is **built on the Orin** (engines are GPU/version-specific).
Code travels via GitHub. Everything uses plain **venv + pip + git** — no Docker.

### Quickstart — Thunder VM (training / dev)

```bash
python -m venv .venv && source .venv/bin/activate
pip install --upgrade pip && pip install -r requirements.txt
python scripts/verify_setup.py          # confirms CUDA + all imports
```

### Quickstart — Jetson AGX Orin (edge inference)

```bash
# JetPack 6.x · TensorRT 10.x · CUDA 12.6 — confirm exact versions first.
python3 -m venv --system-site-packages .venv   # so system `tensorrt` stays importable
source .venv/bin/activate
pip install --upgrade pip && pip install -r requirements-jetson.txt
# then install the Jetson-specific onnxruntime-gpu wheel (see that file's header)
python scripts/verify_setup.py
```

## Roadmap status

- [x] **0 · Foundation** — repo + scaffold (this commit)
- [ ] **1 · Working environment** — rent the VM, venv + deps, `verify_setup.py` green
- [ ] **2 · Data** — download SubPipeMini, build loaders, verify annotations
- [ ] **3 · Baselines** — train/fine-tune RGB-seg + sonar-det, record clean metrics
- [ ] **4 · Edge optimization** — ONNX → Jetson → TensorRT engine → full benchmark
- [ ] **5 · Robustness** — per-modality degradation sweep + failure analysis
- [ ] **6 · Fusion** — late/track-level fusion + Kalman; RGB-only vs sonar-only vs fused ablation
- [ ] **7 · Mitigations** — aug fine-tuning, frame filtering, cross-modal gating, shutter what-if
- [ ] **8 · Writeup** — benchmark + robustness tables, fusion ablation, demo video

> Functions tied to data/hardware raise `NotImplementedError` with a pointer to
> the step that implements them. The config, benchmark harness, augmentation
> bank, Kalman filter, ONNX export, and visualization are usable now.

## Deliverables

Edge benchmark table (fused-pipeline FPS on the Jetson under TensorRT FP16, with
server-vs-edge comparison) · per-modality robustness tables · the RGB-only vs
sonar-only vs fused ablation · a failure analysis · mitigation results incl. the
simulated shutter-speed what-if · an annotated demo video · a clean, reproducible
repo.

## Dataset & acknowledgements

- **SubPipe** — Álvarez-Tuñón et al., *SubPipe: A Submarine Pipeline Inspection
  Dataset for Segmentation and Visual-inertial Localization* (2024).
  `remaro-network/SubPipe-dataset` (also on Zenodo). Use **SubPipeMini** for dev.
- **Ultralytics YOLO11** — segmentation + detection models.
- **ONNX Runtime** and **NVIDIA TensorRT** — cross-platform + edge acceleration.
