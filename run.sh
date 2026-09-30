#!/usr/bin/env bash
set -euo pipefail

echo "=== CUDA Batch Image Processing ==="
echo

python3 scripts/generate_dataset.py
make clean
make
rm -f data/output/*.pgm

./cuda_image_processor \
  --input data/input \
  --output data/output \
  --threads 256 \
  --contrast 1.20 \
  --brightness 5

python3 scripts/validate_output.py data/input data/output
