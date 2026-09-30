# GPU-Accelerated Batch Image Processing Using CUDA

## Project Overview

This project demonstrates GPU-accelerated batch image processing using NVIDIA CUDA. It processes a large batch of grayscale images with a custom CUDA kernel.

The project generates 120 images of 256 x 256 pixels. Each pixel is processed independently on the GPU using a CUDA thread. The kernel performs contrast enhancement and brightness adjustment.

## Features

- Custom CUDA kernel
- 120-image batch dataset
- 256 x 256 grayscale images
- Configurable CUDA threads per block
- Command-line arguments
- CUDA GPU timing
- Automated dataset generation
- Automated output validation
- Makefile and execution script

## Requirements

- NVIDIA GPU
- CUDA Toolkit / `nvcc`
- Python 3
- Linux environment

Check the environment:

```bash
nvcc --version
nvidia-smi
```

## Build and Run

Generate the 120-image dataset:

```bash
python3 scripts/generate_dataset.py
```

Compile:

```bash
make clean
make
```

Run the CUDA program:

```bash
./cuda_image_processor \
  --input data/input \
  --output data/output \
  --threads 256 \
  --contrast 1.20 \
  --brightness 5
```

Validate:

```bash
python3 scripts/validate_output.py data/input data/output
```

Or run the complete workflow:

```bash
./run.sh
```

## Command-Line Arguments

```text
--input DIR       Input PGM image directory
--output DIR      Output directory
--threads INT     CUDA threads per block
--contrast VALUE  Contrast multiplier
--brightness INT  Brightness adjustment
--help            Display help
```

## CUDA Algorithm

Each GPU thread processes one pixel:

1. Read the grayscale pixel.
2. Center its value around 128.
3. Apply the contrast multiplier.
4. Apply brightness adjustment.
5. Clamp the result between 0 and 255.
6. Write the processed pixel.

The project uses CUDA device memory and host/device memory transfers. CUDA events measure kernel execution time.

## Dataset

The generator creates:

- 120 images
- 256 x 256 pixels per image
- 65,536 pixels per image
- 7,864,320 pixels in the complete batch

## Repository Structure

```text
CUDA_Batch_Image_Processing_Ready/
├── src/
│   └── image_processing.cu
├── scripts/
│   ├── generate_dataset.py
│   └── validate_output.py
├── data/
│   ├── input/
│   └── output/
├── results/
├── docs/
│   └── PROOF_ARTIFACTS.md
├── Makefile
├── run.sh
├── README.md
└── .gitignore
```

## Execution Evidence

When executed in a CUDA laboratory environment, save the actual terminal output to:

```text
results/execution_log.txt
results/validation.txt
```

Do not manually invent GPU names or timing values. They should come from the actual CUDA execution.

## Project Description

This project implements GPU-accelerated batch image processing using NVIDIA CUDA. The objective is to process a large amount of image data using GPU computation instead of relying only on CPU processing.

A custom CUDA kernel performs contrast enhancement and brightness adjustment independently for every pixel. Since individual pixel calculations are independent, the workload is suitable for CUDA's parallel execution model. Each CUDA thread processes one image pixel.

The application generates a batch of 120 grayscale images, loads them from the input directory, allocates GPU memory, transfers image data from host memory to device memory, executes the CUDA kernel, transfers the results back to host memory, and saves the processed images.

The program accepts command-line arguments that control the input directory, output directory, CUDA threads per block, contrast, and brightness. A Makefile is provided for compilation and a shell script automates dataset generation, compilation, execution, and validation.

CUDA events are used to measure the GPU kernel execution time. The application also reports total processing time. This makes it possible to distinguish GPU computation from other operations such as file I/O and memory transfers.

A challenge in the project was demonstrating processing of a sufficiently large amount of data rather than a single image. The dataset generator therefore creates 120 deterministic images. Another challenge was correctly managing device memory during repeated image processing.

The project also includes an output validation script that verifies that every input image has a corresponding processed output. This provides evidence that the complete batch was processed successfully.

The main lesson learned is that independent image-pixel operations are naturally suited to GPU parallelism. The project also demonstrates that GPU kernel time and total execution time are different because data transfer and file operations contribute to overall application runtime.
