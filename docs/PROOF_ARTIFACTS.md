# Proof of Execution Artifact Checklist

Run the project in the CUDA lab before submitting this package.

## 1. Build/GPU evidence

Save the terminal output from:

```bash
nvcc --version
nvidia-smi
make clean
make
```

Suggested filename:

```text
01_build_and_gpu.txt
```

## 2. CUDA batch execution evidence

Run:

```bash
./cuda_image_processor \
  --input data/input \
  --output data/output \
  --threads 256 \
  --contrast 1.20 \
  --brightness 5
```

Save the terminal output as:

```text
02_cuda_execution.txt
```

The output should show that 120 images were found and processed.

## 3. Validation evidence

Run:

```bash
python3 scripts/validate_output.py data/input data/output
```

Save the output as:

```text
03_validation.txt
```

It should show:

```text
Input images : 120
Output images: 120
PASS: every input image has a corresponding output image.
```

## 4. Before/after files

Copy a few representative pairs into the proof package and rename them clearly:

```text
before_image_000.pgm
after_image_000.pgm
before_image_060.pgm
after_image_060.pgm
```

## 5. Final ZIP

Create:

```text
proof_of_execution.zip
```

containing all logs and representative input/output files.

Only submit execution values that were actually produced by the CUDA lab.
