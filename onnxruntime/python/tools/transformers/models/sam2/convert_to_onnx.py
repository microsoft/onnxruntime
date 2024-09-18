# -------------------------------------------------------------------------
# Copyright (R) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import argparse
import os
import pathlib
import sys

import torch
from image_decoder import export_decoder_onnx, test_decoder_onnx
from image_encoder import export_image_encoder_onnx, test_image_encoder_onnx
from mask_decoder import export_mask_decoder_onnx, test_mask_decoder_onnx
from prompt_encoder import export_prompt_encoder_onnx, test_prompt_encoder_onnx
from sam2_demo import run_demo
from sam2_utils import build_sam2_model, get_decoder_onnx_path, get_image_encoder_onnx_path, setup_logger


def parse_arguments():
    parser = argparse.ArgumentParser(description="Export SAM2 models to ONNX")

    parser.add_argument(
        "--model_type",
        required=False,
        type=str,
        choices=["sam2_hiera_tiny", "sam2_hiera_small", "sam2_hiera_large", "sam2_hiera_base_plus"],
        default="sam2_hiera_large",
        help="The model type to export",
    )

    parser.add_argument(
        "--components",
        required=False,
        nargs="+",
        choices=["image_encoder", "mask_decoder", "prompt_encoder", "image_decoder"],
        default=["image_encoder", "image_decoder"],
        help="Type of ONNX models to export. "
        "Note that image_decoder is a combination of prompt_encoder and mask_decoder",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        help="The output directory for the ONNX models",
        default="sam2_onnx_models",
    )

    parser.add_argument(
        "--dynamic_batch_axes",
        required=False,
        default=False,
        action="store_true",
        help="Export image_encoder with dynamic batch axes",
    )

    parser.add_argument(
        "--multimask_output",
        required=False,
        default=False,
        action="store_true",
        help="Export mask_decoder or image_decoder with multimask_output",
    )

    parser.add_argument(
        "--disable_dynamic_multimask_via_stability",
        required=False,
        action="store_true",
        help="Disable mask_decoder dynamic_multimask_via_stability, and output first mask only."
        "This option will be ignored when multimask_output is True",
    )

    parser.add_argument(
        "--sam2_dir",
        required=False,
        type=str,
        default="./segment-anything-2",
        help="The directory of segment-anything-2 git repository",
    )

    parser.add_argument(
        "--overwrite",
        required=False,
        default=False,
        action="store_true",
        help="Overwrite onnx model file if exists.",
    )

    parser.add_argument(
        "--demo",
        required=False,
        default=False,
        action="store_true",
        help="Run demo with the exported ONNX models. Requires GPU.",
    )

    parser.add_argument(
        "--verbose",
        required=False,
        default=False,
        action="store_true",
        help="Print verbose information",
    )

    args = parser.parse_args()
    return args


def main():
    args = parse_arguments()

    checkpoints_dir = os.path.join(args.sam2_dir, "checkpoints")
    sam2_config_dir = os.path.join(args.sam2_dir, "sam2_configs")
    if not os.path.exists(args.sam2_dir):
        raise FileNotFoundError(f"{args.sam2_dir} does not exist. Please specify --sam2_dir correctly.")

    if not os.path.exists(checkpoints_dir):
        raise FileNotFoundError(f"{checkpoints_dir}/checkpoints does not exist. Please specify --sam2_dir correctly.")

    if not os.path.exists(sam2_config_dir):
        raise FileNotFoundError(f"{sam2_config_dir}/checkpoints does not exist. Please specify --sam2_dir correctly.")

    if not os.path.exists(os.path.join(checkpoints_dir, f"{args.model_type}.pt")):
        raise FileNotFoundError(
            f"{checkpoints_dir}/{args.model_type}.pt does not exist. Please run download_ckpts.sh under the checkpoints directory."
        )

    if args.sam2_dir not in sys.path:
        sys.path.append(args.sam2_dir)

    pathlib.Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    sam2_model = build_sam2_model(checkpoints_dir, args.model_type, device="cpu")

    for component in args.components:
        if component == "image_encoder":
            onnx_model_path = get_image_encoder_onnx_path(args.output_dir, args.model_type)
            if args.overwrite or not os.path.exists(onnx_model_path):
                export_image_encoder_onnx(sam2_model, onnx_model_path, args.dynamic_batch_axes, args.verbose)
                test_image_encoder_onnx(sam2_model, onnx_model_path, dynamic_batch_axes=False)

        elif component == "mask_decoder":
            onnx_model_path = os.path.join(args.output_dir, f"{args.model_type}_mask_decoder.onnx")
            if args.overwrite or not os.path.exists(onnx_model_path):
                export_mask_decoder_onnx(
                    sam2_model,
                    onnx_model_path,
                    args.multimask_output,
                    not args.disable_dynamic_multimask_via_stability,
                    args.verbose,
                )
                test_mask_decoder_onnx(
                    sam2_model,
                    onnx_model_path,
                    args.multimask_output,
                    not args.disable_dynamic_multimask_via_stability,
                )
        elif component == "prompt_encoder":
            onnx_model_path = os.path.join(args.output_dir, f"{args.model_type}_prompt_encoder.onnx")
            if args.overwrite or not os.path.exists(onnx_model_path):
                export_prompt_encoder_onnx(sam2_model, onnx_model_path)
                test_prompt_encoder_onnx(sam2_model, onnx_model_path)
        elif component == "image_decoder":
            onnx_model_path = get_decoder_onnx_path(args.output_dir, args.model_type, args.multimask_output)
            if args.overwrite or not os.path.exists(onnx_model_path):
                export_decoder_onnx(sam2_model, onnx_model_path, args.multimask_output)
                test_decoder_onnx(sam2_model, onnx_model_path, args.multimask_output)

    if args.demo and torch.cuda.is_available():
        # Export required ONNX models for demo if not already exported.
        onnx_model_path = get_image_encoder_onnx_path(args.output_dir, args.model_type)
        if not os.path.exists(onnx_model_path):
            export_image_encoder_onnx(sam2_model, onnx_model_path, args.dynamic_batch_axes, args.verbose)

        onnx_model_path = get_decoder_onnx_path(args.output_dir, args.model_type, True)
        if not os.path.exists(onnx_model_path):
            export_decoder_onnx(sam2_model, onnx_model_path, True)

        onnx_model_path = get_decoder_onnx_path(args.output_dir, args.model_type, False)
        if not os.path.exists(onnx_model_path):
            export_decoder_onnx(sam2_model, onnx_model_path, False)

        image_files = run_demo(checkpoints_dir, args.model_type, engine="ort", onnx_directory=args.output_dir)
        print("demo output files for ONNX Runtime:", image_files)

        # Get results from torch engine to compare.
        with torch.autocast("cuda", dtype=torch.bfloat16):
            image_files = run_demo(checkpoints_dir, args.model_type, engine="torch", onnx_directory=args.output_dir)
            print("demo output files for PyTorch:", image_files)


if __name__ == "__main__":
    setup_logger(verbose=False)
    with torch.no_grad():
        main()
