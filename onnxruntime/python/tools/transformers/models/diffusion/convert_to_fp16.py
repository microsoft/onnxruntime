# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
#
# This script converts stable diffusion onnx models from float to half (mixed) precision for GPU inference.
#
# Before running this script, you need convert checkpoint to float32 onnx models like the following
#    git clone https://github.com/huggingface/diffusers
#    cd diffusers
#    pip install -e .
#    huggingface-cli login
#    python3 scripts/convert_stable_diffusion_checkpoint_to_onnx.py --model_path runwayml/stable-diffusion-v1-5  --output_path ../stable-diffusion-v1-5
#
# Then you can use this script to convert them to float16 like the following:
#    pip3 install -U onnxruntime-gpu >= 1.14
#    python3 -m onnxruntime.transformers.models.diffusion.convert_to_fp16 -i ../stable-diffusion-v1-5 -o ../stable-diffusion-v1-5-fp16
# Note that float16 model is intended for CUDA Execution Provider. It might not run in CPU Execution Provider.

import argparse
import logging
import os
import shutil
import sys
from pathlib import Path

import coloredlogs

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from optimizer import optimize_model  # noqa: E402

logger = logging.getLogger(__name__)


def convert_to_fp16(source_dir: Path, target_dir: Path, overwrite: bool, use_external_data_format: bool):
    """Convert a model to float16

    Args:
        source_dir (Path): source directory
        target_dir (Path): target directory
        overwrite (bool): overwrite if exists
        use_external_data_format (bool): save model to two files: one for onnx graph, another for weights

    Raises:
        RuntimeError: input onnx model does not exist
        RuntimeError: output onnx model path existed
    """
    dirs_with_onnx = ["vae_encoder", "vae_decoder", "text_encoder", "safety_checker", "unet"]
    for name in dirs_with_onnx:
        onnx_model_path = source_dir / name / "model.onnx"

        if not os.path.exists(onnx_model_path):
            raise RuntimeError(f"input onnx model does not exist: {onnx_model_path}")

        num_heads = 0
        hidden_size = 0

        # Graph fusion before fp16 conversion, otherwise they cannot be fused later.
        # Right now, onnxruntime does not save >2GB model so we use script to optimize unet instead.
        m = optimize_model(
            str(onnx_model_path),
            model_type="unet",
            num_heads=num_heads,
            hidden_size=hidden_size,
            opt_level=0,
            optimization_options=None,
            use_gpu=False,
        )

        #  VAE-decoder in fp16 reduced quality thus we exclude it here
        if name != "vae_decoder":
            m.convert_float_to_float16(op_block_list=["RandomNormalLike", "Resize"])
        else:
            print("skip convert vae_decoder to fp16.")

        optimized_model_path = target_dir / name / "model.onnx"
        output_dir = optimized_model_path.parent
        if optimized_model_path.exists():
            if not overwrite:
                raise RuntimeError(f"output onnx model path existed: {optimized_model_path}")

        if output_dir.exists():
            shutil.rmtree(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        m.save_model_to_file(str(optimized_model_path), use_external_data_format=use_external_data_format)
        print(f"{onnx_model_path} => {optimized_model_path}")


def copy_extra(source_dir: Path, target_dir: Path, overwrite: bool):
    """Copy extra directory.

    Args:
        source_dir (Path): source directory
        target_dir (Path): target directory
        overwrite (bool): overwrite if exists

    Raises:
        RuntimeError: source path does not exist
        RuntimeError: output path exists but overwrite is false.
    """
    extra_dirs = ["scheduler", "tokenizer", "feature_extractor"]
    for name in extra_dirs:
        source_path = source_dir / name
        if not os.path.exists(source_path):
            raise RuntimeError(f"source path does not exist: {source_path}")

        target_path = target_dir / name
        if target_path.exists():
            if not overwrite:
                raise RuntimeError(f"output path existed: {target_path}")
            shutil.rmtree(target_path)

        shutil.copytree(source_path, target_path)
        print(f"{source_path} => {target_path}")

    extra_files = ["model_index.json"]
    for name in extra_files:
        source_path = source_dir / name
        if not os.path.exists(source_path):
            raise RuntimeError(f"source path does not exist: {source_path}")

        target_path = target_dir / name
        if target_path.exists():
            if not overwrite:
                raise RuntimeError(f"output path existed: {target_path}")
            os.remove(target_path)
        shutil.copyfile(source_path, target_path)
        print(f"{source_path} => {target_path}")


def parse_arguments():
    """Parse arguments

    Returns:
        Namespace: arguments
    """
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-i",
        "--input",
        required=True,
        type=str,
        help="Root of input directory of stable diffusion onnx pipeline with float32 models.",
    )

    parser.add_argument(
        "-o",
        "--output",
        required=True,
        type=str,
        help="Root of output directory of stable diffusion onnx pipeline with float16 models.",
    )

    parser.add_argument(
        "--overwrite",
        required=False,
        action="store_true",
        help="Overwrite exists files.",
    )
    parser.set_defaults(overwrite=False)

    parser.add_argument(
        "-e",
        "--use_external_data_format",
        required=False,
        action="store_true",
        help="Onnx model larger than 2GB need to use external data format.",
    )
    parser.set_defaults(use_external_data_format=False)

    args = parser.parse_args()
    return args


def main():
    coloredlogs.install(fmt="%(funcName)20s: %(message)s")
    args = parse_arguments()
    copy_extra(Path(args.input), Path(args.output), args.overwrite)
    convert_to_fp16(Path(args.input), Path(args.output), args.overwrite, args.use_external_data_format)


main()
