#!/usr/bin/env bash
set -euo pipefail

mkdir -p results

{
  echo "===== CUDA ENVIRONMENT ====="
  nvcc --version
  nvidia-smi
  echo
  echo "===== DATASET ====="
  python3 scripts/generate_dataset.py
  echo
  echo "===== BUILD ====="
  make clean
  make
  echo
  echo "===== CUDA EXECUTION ====="
  rm -f data/output/*.pgm
  ./cuda_image_processor \
    --input data/input \
    --output data/output \
    --threads 256 \
    --contrast 1.20 \
    --brightness 5
  echo
  echo "===== VALIDATION ====="
  python3 scripts/validate_output.py data/input data/output
} 2>&1 | tee results/execution_log.txt

python3 scripts/validate_output.py data/input data/output \
  > results/validation.txt
