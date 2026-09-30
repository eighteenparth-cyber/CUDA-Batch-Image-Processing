#!/usr/bin/env python3
"""Generate a large batch of deterministic PGM test images."""

from pathlib import Path
import math
import random
import struct

WIDTH = 256
HEIGHT = 256
IMAGE_COUNT = 120
OUTPUT_DIR = Path("data/input")


def write_pgm(path: Path, pixels: bytes) -> None:
    with path.open("wb") as output:
        output.write(f"P5\n{WIDTH} {HEIGHT}\n255\n".encode("ascii"))
        output.write(pixels)


def make_image(index: int) -> bytes:
    random.seed(index)
    pixels = bytearray(WIDTH * HEIGHT)

    for y in range(HEIGHT):
        for x in range(WIDTH):
            gradient = (x + y + index * 7) % 256
            wave = int(35 * math.sin((x + index) / 15.0))
            noise = random.randint(-8, 8)
            value = max(0, min(255, gradient + wave + noise))
            pixels[y * WIDTH + x] = value

    return bytes(pixels)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for old_file in OUTPUT_DIR.glob("*.pgm"):
        old_file.unlink()

    for index in range(IMAGE_COUNT):
        path = OUTPUT_DIR / f"image_{index:03d}.pgm"
        write_pgm(path, make_image(index))

    print(f"Generated {IMAGE_COUNT} images in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
