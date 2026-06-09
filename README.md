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

- [x] **0 · Foundation** — repo + scaffold
- [x] **1 · Working environment** — Thunder A6000 VM, venv + deps, `verify_setup.py` green
- [~] **2 · Data** — SubPipeMini (seg) + SubPipeMini2 (sonar): loaders + SubPipe→YOLO prep + annotation verify (`scripts/inspect_dataset.py`, `scripts/prepare_data.py`)
- [x] **3 · Baselines** — train/fine-tune RGB-seg + sonar-det, record clean metrics
- [x] **4 · Edge optimization** — ONNX → Jetson → TensorRT engine → full benchmark
- [x] **5 · Robustness** — per-modality degradation sweep + failure analysis
- [ ] **6 · Fusion** — late/track-level fusion + Kalman; RGB-only vs sonar-only vs fused ablation
- [ ] **7 · Mitigations** — aug fine-tuning, frame filtering, cross-modal gating, shutter what-if
- [ ] **8 · Writeup** — benchmark + robustness tables, fusion ablation, demo video

> Functions tied to data/hardware raise `NotImplementedError` with a pointer to
> the step that implements them. The config, benchmark harness, augmentation
> bank, Kalman filter, ONNX export, and visualization are usable now.

## Robustness: complementary failure modes

Both models were put through a per-modality degradation sweep
(`scripts/evaluate_robustness.py`): every clean validation frame is corrupted by
a bank of physically motivated underwater conditions at four severities (0.25,
0.5, 0.75, 1.0), and the corrupted set is re-scored. RGB is reported as mask
mAP50 (21 val frames, 26 pipe instances), sonar as box mAP50 (181 val frames,
126 instances). The augmentation bank lives in `src/data/augmentations.py`, and
the full sweeps are pictured in `results/robustness/rgb_severity_overview.png`
and `sonar_severity_overview.png`. These are two independent SubPipeMini subsets,
so this measures each sensor on its own data, not joint behavior on one
synchronized scene; the fused ablation covers that.

### RGB (optical), mask mAP50, clean = 0.801

| condition (roughly most to least tolerated) | 0.25 | 0.50 | 0.75 | 1.00 |
|---|---|---|---|---|
| color_attenuation | 0.798 | 0.808 | 0.796 | 0.774 |
| motion_blur | 0.781 | 0.772 | 0.784 | 0.823 |
| low_light | 0.790 | 0.829 | 0.585 | 0.137 |
| turbidity_haze | 0.742 | 0.459 | 0.173 | 0.039 |
| overexposure | 0.745 | 0.436 | 0.000 | 0.000 |
| backscatter | 0.607 | 0.162 | 0.002 | 0.000 |
| sand_occlusion | 0.557 | 0.144 | 0.010 | 0.001 |
| gaussian_noise | 0.151 | 0.000 | 0.001 | 0.000 |

The camera tolerates a global color shift and, notably, motion blur: even a
clearly visible smear barely moves the score, because the pipe is one large solid
region whose bulk survives blurring. Everything else degrades, and several
conditions are catastrophic. Sensor noise is the sharpest cliff, a
quarter-strength gaussian already cutting the score to 0.15 and half-strength
erasing it, with strong backscatter, blown-out overexposure, heavy turbidity, and
occlusion following. Occlusion is the headline optical failure: once sediment
buries the pipe there is nothing left for the camera to segment.

### Sonar (acoustic), box mAP50, clean = 0.995

| condition (roughly most to least tolerated) | 0.25 | 0.50 | 0.75 | 1.00 |
|---|---|---|---|---|
| range_falloff | 0.995 | 0.995 | 0.995 | 0.985 |
| speckle_noise | 0.995 | 0.994 | 0.974 | 0.985 |
| reverberation_clutter | 0.970 | 0.938 | 0.887 | 0.822 |
| beam_dropout | 0.946 | 0.863 | 0.732 | 0.673 |
| motion_smear | 0.964 | 0.653 | 0.181 | 0.080 |

The acoustic model is far steadier. Multiplicative speckle and range falloff
leave it essentially untouched, and it declines only gradually under
reverberation clutter and dropped beams, never dropping below about 0.67 for
either. Its single catastrophic failure is along-track motion smear: as platform
motion drags the return across the waterfall the target washes out, and the score
falls from 0.96 at a light smear to 0.08 at the strongest.

### Why this motivates fusion

The two failure profiles are nearly disjoint, which is the quantitative case for
combining the sensors:

- The conditions that destroy the camera (turbidity, darkness, sensor noise,
  backscatter) are optical failures that an acoustic sensor is physically
  indifferent to. Across its own sweep the sonar never falls below about 0.67
  except under motion smear, while RGB collapses toward zero under several of
  those optical conditions.
- The one condition that destroys the sonar (along-track motion smear) is the one
  the camera shrugs off (motion blur, flat near 0.78), because a large solid pipe
  blob tolerates blur far better than a thin acoustic return tolerates smear.

Each sensor's worst case is therefore the other's comfortable regime. A
track-level fused estimate can lean on sonar when the water turns turbid, dark, or
backscattering, and on RGB when motion smears the acoustic return, holding the
pipe where either modality alone would lose it. Occlusion is the one axis we
cannot yet score jointly: a buried pipe defeats the camera, and acoustic
shadowing is physically the sort of cue that could still flag it, but confirming
that needs co-registered buried-pipe sonar, which these separate subsets do not
provide. It is a fusion hypothesis to test, not a measured result here.

Two caveats on reading the tables. The RGB validation set is small (21 frames),
so individual cells carry sampling noise: a couple of the robust conditions read
a hair above the clean baseline at high severity, which is variance, not a real
gain. And the degradations are synthetic approximations of real optical and
acoustic failure modes, tuned for plausibility rather than reproduced from field
data.

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
