# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

import os
import argparse
import logging
import torch
from transformers import AutoConfig
from gpt2_helper import Gpt2Helper, MyGPT2Model, MyGPT2LMHeadModel, MODEL_CLASSES, DEFAULT_TOLERANCE
from quantize_helper import QuantizeHelper
from benchmark_helper import create_onnxruntime_session, setup_logger, prepare_environment, Precision
logger = logging.getLogger('')


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('-m',
                        '--model_name_or_path',
                        required=True,
                        type=str,
                        help='Model path, or pretrained model name in the list: ' + ', '.join(['gpt2', 'distilgpt2']))

    parser.add_argument('--model_class',
                        required=False,
                        type=str,
                        default='GPT2LMHeadModel',
                        choices=list(MODEL_CLASSES.keys()),
                        help='Model type selected in the list: ' + ', '.join(MODEL_CLASSES.keys()))

    parser.add_argument('--cache_dir',
                        required=False,
                        type=str,
                        default=os.path.join('.', 'cache_models'),
                        help='Directory to cache pre-trained models')

    parser.add_argument('--output',
                        required=False,
                        type=str,
                        default=os.path.join('.', 'onnx_models'),
                        help='Output directory, or model path ends with .onnx')

    parser.add_argument('-o',
                        '--optimize_onnx',
                        required=False,
                        action='store_true',
                        help='Use optimizer.py to optimize onnx model')
    parser.set_defaults(optimize_onnx=False)

    parser.add_argument('--use_gpu', required=False, action='store_true', help="use GPU for inference")
    parser.set_defaults(use_gpu=False)

    parser.add_argument('--tolerance',
                        required=False,
                        type=float,
                        default=0,
                        help="the aboslute and relative tolerance for parity verification")

    parser.add_argument(
        "-p",
        "--precision",
        required=False,
        type=Precision,
        default=Precision.FLOAT32,
        choices=list(Precision),
        help="Precision of model to run. fp32 for full precision, fp16 for half precision, and int8 for quantization")

    parser.add_argument('--verbose', required=False, action='store_true')
    parser.set_defaults(verbose=False)

    args = parser.parse_args()

    return args


def main():
    args = parse_arguments()
    setup_logger(args.verbose)

    if args.tolerance == 0:
        args.tolerance = DEFAULT_TOLERANCE[args.precision]

    logger.info(f"Arguments:{args}")

    cache_dir = args.cache_dir
    output_dir = args.output if not args.output.endswith(".onnx") else os.path.dirname(args.output)
    prepare_environment(cache_dir, output_dir, args.use_gpu)

    if args.precision != Precision.FLOAT32:
        assert args.optimize_onnx, "fp16/int8 requires --optimize_onnx"

    if args.precision == Precision.FLOAT16:
        assert args.use_gpu, "fp16 requires --use_gpu"

    if args.precision == Precision.INT8:
        assert not args.use_gpu, "quantization only supports CPU"

    model_class = MODEL_CLASSES[args.model_class][0]
    model = model_class.from_pretrained(args.model_name_or_path, cache_dir=cache_dir)

    device = torch.device("cuda:0" if args.use_gpu else "cpu")
    model.eval().to(device)

    onnx_model_paths = Gpt2Helper.get_onnx_paths(output_dir, args.model_name_or_path, args.model_class)
    raw_onnx_model = args.output if args.output.endswith('.onnx') else onnx_model_paths["raw"]
    output_path = raw_onnx_model if (
        args.output.endswith('.onnx') or
        (args.precision == Precision.FLOAT32 and not args.optimize_onnx)) else onnx_model_paths[str(args.precision)]

    Gpt2Helper.export_onnx(model, device, raw_onnx_model, args.verbose)

    if args.optimize_onnx or args.precision != Precision.FLOAT32:
        Gpt2Helper.optimize_onnx(raw_onnx_model, output_path, args.precision == Precision.FLOAT16,
                                 model.config.num_attention_heads, model.config.hidden_size)

    if args.precision == Precision.INT8:
        logger.info("quantizing model...")
        QuantizeHelper.quantize_onnx_model(output_path, output_path)
        model = QuantizeHelper.quantize_torch_model(model)
        logger.info("finished quantizing model")

    session = create_onnxruntime_session(output_path, args.use_gpu, enable_optimizations=False, verbose=args.verbose)
    if session is not None:
        Gpt2Helper.test_parity(session,
                               model,
                               device,
                               args.precision == Precision.FLOAT16,
                               rtol=args.tolerance,
                               atol=args.tolerance)

    logger.info(f"Done. Output model: {output_path}")


if __name__ == '__main__':
    main()
