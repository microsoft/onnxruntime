# -------------------------------------------------------------------------
# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
"""
This converts GPT2 model to onnx. Examples:
(1) Convert pretrained model 'gpt2' to ONNX
   python convert_to_onnx.py -m gpt2 --output gpt2.onnx
(2) Convert pretrained model 'distilgpt2' to ONNX, and use optimizer to get float16 model.
   python convert_to_onnx.py -m distilgpt2 --output distilgpt2_fp16.onnx -o -p fp16
(3) Convert a model check point to ONNX, and run optimization and int8 quantization
   python convert_to_onnx.py -m ./my_model_checkpoint/ --output my_model_int8.onnx -o -p int8

"""

import os
import sys
import argparse
import coloredlogs
import logging
import torch
import numpy
import json
from pathlib import Path
from transformers import AutoConfig

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from t5_helper import PRETRAINED_T5_MODELS, T5Helper
from benchmark_helper import create_onnxruntime_session, setup_logger, prepare_environment, Precision
from quantize_helper import QuantizeHelper
from onnxruntime.quantization import quantize_dynamic

logger = logging.getLogger('')


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('-m',
                        '--model_name_or_path',
                        required=False,
                        default=PRETRAINED_T5_MODELS[0],
                        type=str,
                        help='Model path, or pretrained model name in the list: ' + ', '.join(PRETRAINED_T5_MODELS))

    parser.add_argument('--cache_dir',
                        required=False,
                        type=str,
                        default=os.path.join('.', 'cache_models'),
                        help='Directory to cache pre-trained models')

    parser.add_argument('--output',
                        required=False,
                        type=str,
                        default=os.path.join('.', 'onnx_models'),
                        help='Output directory')

    parser.add_argument('--use_past_state', required=False, action='store_true', help="use past state for inference")
    parser.set_defaults(use_past_state=False)

    parser.add_argument('-o',
                        '--optimize_onnx',
                        required=False,
                        action='store_true',
                        help='Use optimizer.py to optimize onnx model')
    parser.set_defaults(optimize_onnx=False)

    parser.add_argument('--use_gpu', required=False, action='store_true', help="use GPU for inference")
    parser.set_defaults(use_gpu=False)

    parser.add_argument("-p",
                        "--precision",
                        required=False,
                        type=Precision,
                        default=Precision.FLOAT32,
                        choices=list(Precision),
                        help="Precision of model to run. fp32 for full precision, fp16 for half precision")

    parser.add_argument('--verbose', required=False, action='store_true')
    parser.set_defaults(verbose=False)

    parser.add_argument('-e', '--use_external_data_format', required=False, action='store_true')
    parser.set_defaults(use_external_data_format=False)

    args = parser.parse_args()

    return args


def export_onnx_models(model_name_or_path, cache_dir, output_dir, use_past_state, use_gpu, use_external_data_format,
                       optimize_onnx, precision, verbose, overwrite:bool = False):
    device = torch.device("cuda:0" if use_gpu else "cpu")

    encoder, decoder = T5Helper.load_model(model_name_or_path, cache_dir, device, use_past_state)
    config = decoder.config

    if (not use_external_data_format) and (config.num_layers > 24):
        logger.info(f"Try use_external_data_format when model size > 2GB")

    output_paths = []
    for is_encoder in [False, True]:
        encoder_or_decoder = encoder if is_encoder else decoder
        filename_suffix = "_encoder" if is_encoder else "_decoder"
        if use_past_state and not is_encoder:
            filename_suffix += "_past"

        # Export encoder to ONNX
        onnx_path = T5Helper.get_onnx_path(output_dir,
                                           model_name_or_path,
                                           suffix=filename_suffix,
                                           new_folder=use_external_data_format)

        if overwrite or not os.path.exists(onnx_path):
            logger.info(f"Exporting ONNX model to {onnx_path}")
            T5Helper.export_onnx(encoder_or_decoder, device, onnx_path, verbose, use_external_data_format)
        else:
            logger.info(f"Skip exporting: existed ONNX model {onnx_path}")

        # Optimize encoder ONNX
        if optimize_onnx or precision != Precision.FLOAT32:
            output_path = T5Helper.get_onnx_path(output_dir,
                                                 model_name_or_path,
                                                 suffix=filename_suffix + "_" + str(precision),
                                                 new_folder=use_external_data_format)

            if overwrite or not os.path.exists(output_path):
                logger.info(f"Optimizing model to {output_path}")
                T5Helper.optimize_onnx(onnx_path, output_path, precision == Precision.FLOAT16, config.num_heads,
                                    config.hidden_size, use_external_data_format)
            else:
                logger.info(f"Skip optimizing: existed ONNX model {onnx_path}")
        else:
            output_path = onnx_path

        if precision == Precision.INT8:
            quant_path = T5Helper.get_onnx_path(output_dir,
                                        model_name_or_path,
                                        suffix=filename_suffix + "_quant_" + str(precision),
                                        new_folder=use_external_data_format)
            QuantizeHelper.quantize_onnx_model(output_path, quant_path)
            output_path = quant_path

        # Copy ONNX to output if necessary
        """
        if args.output.endswith('.onnx') and output_path != args.output and not use_external_data_format:
            import shutil
            shutil.move(output_path, args.output)
            output_path = args.output
        """

        output_paths.append(output_path)

    return output_paths


def main():
    args = parse_arguments()
    setup_logger(args.verbose)

    logger.info(f"Arguments:{args}")

    cache_dir = args.cache_dir
    output_dir = args.output if not args.output.endswith(".onnx") else os.path.dirname(args.output)
    prepare_environment(cache_dir, output_dir, args.use_gpu)

    if args.precision != Precision.FLOAT32:
        assert args.optimize_onnx, "fp16/int8 requires --optimize_onnx"

    if args.precision == Precision.FLOAT16:
        assert args.use_gpu, "fp16 requires --use_gpu"

    output_paths = export_onnx_models(args.model_name_or_path, cache_dir, output_dir, args.use_past_state, args.use_gpu,
                                      args.use_external_data_format, args.optimize_onnx, args.precision, args.verbose)

    logger.info(f"Done! Outputs: [output_paths]")


if __name__ == '__main__':
    main()
