#!/bin/bash
# ===========================================================================
#  Edge benchmark on the Jetson AGX Orin (TensorRT FP32 baseline vs FP16).
# ===========================================================================
# Builds a TensorRT engine from each committed ONNX model and benchmarks it
# with `trtexec`, the tool bundled with JetPack/TensorRT. Deliberately uses no
# PyTorch, no Ultralytics, no pip installs, and no git credentials, so it runs
# on a shared Jetson and leaves no trace: the ONNX files and the engines live
# in a temp directory that is removed on exit.
#
# Usage on the Jetson:
#   bash jetson_benchmark.sh
#
# It prints, for each model and precision, the throughput (qps) and the
# GPU-compute latency. Copy those numbers; we record them as the edge results
# and commit the JSON from the VM (nothing is pushed from the Jetson).
# ===========================================================================
set -e

REPO_RAW="https://github.com/Danie-Craig/underwater-pipeline-multimodal-edge/raw/main"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT          # engines + onnx wiped automatically

# Locate trtexec (on PATH, or the standard JetPack location).
TRTEXEC="$(command -v trtexec || true)"
[ -z "$TRTEXEC" ] && [ -x /usr/src/tensorrt/bin/trtexec ] && TRTEXEC=/usr/src/tensorrt/bin/trtexec
if [ -z "$TRTEXEC" ] || [ ! -x "$TRTEXEC" ]; then
  echo "ERROR: trtexec not found. It ships with JetPack/TensorRT, e.g."
  echo "       /usr/src/tensorrt/bin/trtexec"
  exit 1
fi
echo "Using trtexec: $TRTEXEC"

# Get the ONNX models: prefer local repo copies, else fetch from GitHub.
ONNX_DIR="models/onnx"
if [ ! -f "$ONNX_DIR/rgb_seg.onnx" ]; then
  echo "Fetching ONNX models from GitHub into temp dir..."
  ONNX_DIR="$WORK"
  wget -q -O "$WORK/rgb_seg.onnx"   "$REPO_RAW/models/onnx/rgb_seg.onnx"
  wget -q -O "$WORK/sonar_det.onnx" "$REPO_RAW/models/onnx/sonar_det.onnx"
fi

for MODEL in rgb_seg sonar_det; do
  for PREC in fp32 fp16; do
    FLAG=""
    [ "$PREC" = "fp16" ] && FLAG="--fp16"
    echo ""
    echo "==================== $MODEL / $PREC ===================="
    "$TRTEXEC" --onnx="$ONNX_DIR/$MODEL.onnx" $FLAG \
      --saveEngine="$WORK/${MODEL}_${PREC}.engine" \
      --warmUp=2000 --iterations=300 --avgRuns=300 \
      2>&1 | grep -E "Throughput:|Latency:|GPU Compute Time:" || true
  done
done

echo ""
echo "============================================================"
echo "Done. Engines were built in a temp dir and are now deleted."
echo "Copy the 'Throughput' and 'GPU Compute Time' lines above for"
echo "each model/precision; we'll record them as the edge results."
echo "============================================================"
