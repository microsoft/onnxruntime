#!/usr/bin/env python3

import argparse
from pathlib import Path

from PIL import Image, ImageDraw

IMAGE_SIZE = 2048
TILE_SIZE = 128


def generate_image(output_path: Path) -> None:
    image = Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE))
    draw = ImageDraw.Draw(image)

    for y in range(0, IMAGE_SIZE, TILE_SIZE):
        for x in range(0, IMAGE_SIZE, TILE_SIZE):
            color = (
                (x // TILE_SIZE * 37 + y // TILE_SIZE * 17) % 256,
                (x // TILE_SIZE * 11 + y // TILE_SIZE * 43) % 256,
                (x // TILE_SIZE * 29 + y // TILE_SIZE * 23) % 256,
            )
            draw.rectangle(
                (x, y, x + TILE_SIZE - 1, y + TILE_SIZE - 1),
                fill=color,
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path, format="JPEG", quality=90, subsampling=0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a deterministic 2048x2048 JPEG test image.")
    parser.add_argument(
        "output",
        nargs="?",
        type=Path,
        default=Path(__file__).with_name("tensor-image.jpg"),
        help="Output path (default: tensor-image.jpg next to this script).",
    )
    args = parser.parse_args()
    generate_image(args.output)


if __name__ == "__main__":
    main()
