#!/bin/bash
# Use this instead of 'pip install -r requirements.txt' on the Thunder VM.
# Ultralytics pulls opencv-python (non-headless) as a transitive dependency,
# which fails on headless servers. We strip and replace it here.
set -e
pip install --upgrade pip
pip install -r requirements.txt
pip uninstall -y opencv-python opencv-python-headless   # remove both conflicting builds
pip install opencv-python-headless                      # reinstall only the headless build
echo "Done. Run: python scripts/verify_setup.py"