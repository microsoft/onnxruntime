#!/usr/bin/env python3
"""Save reproducible Gemma 4 processor inputs for CPU/EP QDQ comparisons."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image
from transformers import Gemma4ImageProcessorPil


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="New .npz file with model inputs and valid_mask.")
    parser.add_argument(
        "--model-only-output", type=Path,
        help="Also save a .npz with only model inputs for multi-output attention probes.",
    )
    parser.add_argument("--image", type=Path, help="Local RGB image; default: deterministic 1024x768 gradient.")
    parser.add_argument("--gradient-width", type=int, default=1024, help="Generated gradient width; default: 1024.")
    parser.add_argument("--gradient-height", type=int, default=768, help="Generated gradient height; default: 768.")
    parser.add_argument("--processor", default="google/gemma-4-E2B-it", help="Gemma 4 processor identifier.")
    return parser.parse_args()


def gradient_image(width: int, height: int) -> Image.Image:
    horizontal = np.linspace(0, 255, width, dtype=np.uint8)[None, :].repeat(height, axis=0)
    vertical = np.linspace(0, 255, height, dtype=np.uint8)[:, None].repeat(width, axis=1)
    return Image.fromarray(np.stack((horizontal, vertical, np.full_like(horizontal, 127)), axis=-1), "RGB")


def main() -> None:
    args = parse_args()
    output = args.output.resolve()
    if output.suffix.lower() != ".npz" or output.exists():
        raise ValueError(f"output must be a new .npz file: {output}")
    model_only = args.model_only_output.resolve() if args.model_only_output else None
    if model_only is not None and (
        model_only == output or model_only.suffix.lower() != ".npz" or model_only.exists()
    ):
        raise ValueError(f"model-only output must be a new, distinct .npz file: {model_only}")
    if args.image is not None and not args.image.is_file():
        raise ValueError(f"image does not exist: {args.image}")
    if args.gradient_width <= 0 or args.gradient_height <= 0:
        raise ValueError("gradient dimensions must be positive")
    if args.image is not None and (args.gradient_width != 1024 or args.gradient_height != 768):
        raise ValueError("--gradient-width and --gradient-height only apply without --image")
    if args.image is None:
        image = gradient_image(args.gradient_width, args.gradient_height)
    else:
        with Image.open(args.image) as source:
            image = source.convert("RGB")
    processor = Gemma4ImageProcessorPil.from_pretrained(args.processor)
    result = processor(images=image, return_tensors="np")
    pixels = np.asarray(result["pixel_values"])
    positions = np.asarray(result["image_position_ids"])
    valid_mask = positions[..., 0] != -1
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output, pixel_values=pixels, pixel_position_ids=positions, valid_mask=valid_mask)
    if model_only is not None:
        model_only.parent.mkdir(parents=True, exist_ok=True)
        np.savez(model_only, pixel_values=pixels, pixel_position_ids=positions)
        print(f"Saved model-only inputs: {model_only}")
    valid_count = np.count_nonzero(valid_mask)
    print(f"Saved {output}: patches={pixels.shape}, valid={valid_count}, padded={valid_mask.size - valid_count}")


if __name__ == "__main__":
    main()
