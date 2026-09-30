#!/usr/bin/env python3
"""Validate that the expected number of output PGM images exists."""

from pathlib import Path
import sys

input_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/input")
output_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("data/output")

inputs = sorted(input_dir.glob("*.pgm"))
outputs = sorted(output_dir.glob("*.pgm"))

print(f"Input images : {len(inputs)}")
print(f"Output images: {len(outputs)}")

if not inputs:
    print("FAIL: no input images found")
    raise SystemExit(1)

if len(inputs) != len(outputs):
    print("FAIL: input/output image counts do not match")
    raise SystemExit(1)

missing = [path.name for path in inputs if not (output_dir / path.name).exists()]
if missing:
    print("FAIL: missing output files:")
    for name in missing[:10]:
        print(f"  {name}")
    raise SystemExit(1)

print("PASS: every input image has a corresponding output image.")
