# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
#
# This script converts stable diffusion onnx models from float to half (mixed) precision for GPU inference.
#
# Before running this script, you need convert checkpoint to float32 onnx models like the following
#    export ONNX_ROOT=./sd_onnx
#    pip install -r requirements.txt
#    huggingface-cli login
#    wget https://raw.githubusercontent.com/huggingface/diffusers/v0.12.1/scripts/convert_stable_diffusion_checkpoint_to_onnx.py
#    python convert_stable_diffusion_checkpoint_to_onnx.py --model_path runwayml/stable-diffusion-v1-5  --output_path $ONNX_ROOT/stable-diffusion-v1-5-fp32
# Note that this script might not be compatible with older or newer version of diffusers.

# Then you can use this script to convert them to float16 like the following:
#    python optimize_pipeline.py -i $ONNX_ROOT/stable-diffusion-v1-5-fp32 -o $ONNX_ROOT/stable-diffusion-v1-5-fp16 --float16
# Or
#    python -m onnxruntime.transformers.models.stable_diffusion.optimize_pipeline -i $ONNX_ROOT/stable-diffusion-v1-5-fp32 -o $ONNX_ROOT/stable-diffusion-v1-5-fp16 --float16
#
# Note that output model is for CUDA Execution Provider. It might not run in CPU Execution Provider.
# Stable diffusion 2.1 model will get black images using float16 Attention. It is a known issue that we are working on.

import argparse
import logging
import os
import shutil
import sys
from pathlib import Path

import coloredlogs

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from fusion_options import FusionOptions
from optimizer import optimize_model  # noqa: E402

logger = logging.getLogger(__name__)


def optimize_sd_pipeline(
    source_dir: Path, target_dir: Path, overwrite: bool, use_external_data_format: bool, float16: bool
):
    """Optimize onnx models used in stable diffusion onnx pipeline and optionally convert to float16.

    Args:
        source_dir (Path): Root of input directory of stable diffusion onnx pipeline with float32 models.
        target_dir (Path): Root of output directory of stable diffusion onnx pipeline with optimized models.
        overwrite (bool): Overwrite files if exists.
        use_external_data_format (bool): save onnx model to two files: one for onnx graph, another for weights
        float16 (bool): use half precision

    Raises:
        RuntimeError: input onnx model does not exist
        RuntimeError: output onnx model path existed
    """
    dirs_with_onnx = ["unet", "vae_encoder", "vae_decoder", "text_encoder", "safety_checker"]
    for name in dirs_with_onnx:
        onnx_model_path = source_dir / name / "model.onnx"

        if not os.path.exists(onnx_model_path):
            message = f"input onnx model does not exist: {onnx_model_path}."
            if name not in ["safety_checker", "feature_extractor"]:
                raise RuntimeError(message)
            continue

        # Graph fusion before fp16 conversion, otherwise they cannot be fused later.
        # Right now, onnxruntime does not save >2GB model so we use script to optimize unet instead.
        logger.info(f"optimize {onnx_model_path}...")

        fusion_options = FusionOptions("unet")
        fusion_options.enable_packed_kv = float16

        m = optimize_model(
            str(onnx_model_path),
            model_type="unet",
            num_heads=0,  # will be deduced from graph
            hidden_size=0,  # will be deduced from graph
            opt_level=0,
            optimization_options=fusion_options,
            use_gpu=False,
        )

        if float16:
            logger.info("convert %s to float16 ...", name)
            m.convert_float_to_float16(op_block_list=["RandomNormalLike", "Resize", "GroupNorm"])

        optimized_model_path = target_dir / name / "model.onnx"
        output_dir = optimized_model_path.parent
        if optimized_model_path.exists():
            if not overwrite:
                raise RuntimeError(f"output onnx model path existed: {optimized_model_path}")

        if output_dir.exists():
            shutil.rmtree(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        m.save_model_to_file(str(optimized_model_path), use_external_data_format=use_external_data_format)
        logger.info("%s => %s", onnx_model_path, optimized_model_path)


def copy_extra_directory(source_dir: Path, target_dir: Path, overwrite: bool):
    """Copy extra directory that does not have onnx model

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
            message = f"source path does not exist: {source_path}"
            if name not in ["safety_checker", "feature_extractor"]:
                raise RuntimeError(message)
            continue

        target_path = target_dir / name
        if target_path.exists():
            if not overwrite:
                raise RuntimeError(f"output path existed: {target_path}")
            shutil.rmtree(target_path)

        shutil.copytree(source_path, target_path)
        logger.info("%s => %s", source_path, target_path)

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
        logger.info("%s => %s", source_path, target_path)


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
        help="Root of output directory of stable diffusion onnx pipeline with optimized models.",
    )

    parser.add_argument(
        "--float16",
        required=False,
        action="store_true",
        help="Output models of half or mixed precision.",
    )
    parser.set_defaults(float16=False)

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
        help="Onnx model larger than 2GB need to use external data format. "
        "Save onnx model to two files: one for onnx graph, another for large weights.",
    )
    parser.set_defaults(use_external_data_format=False)

    args = parser.parse_args()
    return args


def main():
    coloredlogs.install(fmt="%(funcName)20s: %(message)s")
    args = parse_arguments()
    copy_extra_directory(Path(args.input), Path(args.output), args.overwrite)
    optimize_sd_pipeline(
        Path(args.input), Path(args.output), args.overwrite, args.use_external_data_format, args.float16
    )


main()
