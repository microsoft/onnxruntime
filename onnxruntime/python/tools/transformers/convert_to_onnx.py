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
import argparse
import coloredlogs
import logging
import torch
import numpy
import json
from pathlib import Path
from packaging import version
from transformers import AutoConfig
from gpt2_helper import Gpt2Helper, MODEL_CLASSES, DEFAULT_TOLERANCE, PRETRAINED_GPT2_MODELS
from gpt2_tester import Gpt2Tester
from quantize_helper import QuantizeHelper
from benchmark_helper import create_onnxruntime_session, setup_logger, prepare_environment, Precision

logger = logging.getLogger('')


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('-m',
                        '--model_name_or_path',
                        required=True,
                        type=str,
                        help='Model path, or pretrained model name in the list: ' + ', '.join(PRETRAINED_GPT2_MODELS))

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

    parser.add_argument('--input_test_file',
                        '-i',
                        required=False,
                        type=str,
                        default='',
                        help='Path to the file with inputs to test with')

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

    parser.add_argument('-e', '--use_external_data_format', required=False, action='store_true')
    parser.set_defaults(use_external_data_format=False)

    args = parser.parse_args()

    return args


def main():
    from transformers import __version__ as transformers_version
    if version.parse(transformers_version) < version.parse("3.1.0"): # past_key_values name does not exist in 3.0.2 or older
        raise RuntimeError("This tool requires transformers 3.1.0 or later.")

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

    if args.use_external_data_format:
        assert not args.output.endswith('.onnx'), "output shall be a directory for --use_external_data_format"

    model_class = MODEL_CLASSES[args.model_class][0]
    config = AutoConfig.from_pretrained(args.model_name_or_path, cache_dir=cache_dir)
    model = model_class.from_pretrained(args.model_name_or_path, config=config, cache_dir=cache_dir)

    device = torch.device("cuda:0" if args.use_gpu else "cpu")
    model.eval().to(device)

    if (not args.use_external_data_format) and (config.n_layer > 24):
        logger.info(f"Try --use_external_data_format when model size > 2GB")

    onnx_model_paths = Gpt2Helper.get_onnx_paths(output_dir,
                                                 args.model_name_or_path,
                                                 args.model_class,
                                                 new_folder=args.use_external_data_format)

    raw_onnx_model = onnx_model_paths["raw"]

    logger.info(f"Exporting ONNX model to {raw_onnx_model}")
    use_padding = MODEL_CLASSES[args.model_class][2]
    Gpt2Helper.export_onnx(model,
                           device,
                           raw_onnx_model,
                           args.verbose,
                           args.use_external_data_format,
                           has_position_ids=use_padding,
                           has_attention_mask=use_padding)

    if args.optimize_onnx or args.precision != Precision.FLOAT32:
        output_path = onnx_model_paths[str(args.precision) if args.precision != Precision.INT8 else 'fp32']

        logger.info(f"Optimizing model to {output_path}")
        Gpt2Helper.optimize_onnx(raw_onnx_model, output_path, args.precision == Precision.FLOAT16,
                                 model.config.num_attention_heads, model.config.hidden_size,
                                 args.use_external_data_format)
    else:
        output_path = raw_onnx_model

    if args.precision == Precision.INT8:
        logger.info("quantizing model...")
        QuantizeHelper.quantize_onnx_model(output_path, onnx_model_paths['int8'], args.use_external_data_format)
        model = QuantizeHelper.quantize_torch_model(model)
        logger.info("finished quantizing model")
        output_path = onnx_model_paths['int8']

    if args.output.endswith('.onnx') and output_path != args.output and not args.use_external_data_format:
        import shutil
        shutil.move(output_path, args.output)
        output_path = args.output

    logger.info(f"Output path: {output_path}")

    session = create_onnxruntime_session(output_path, args.use_gpu, enable_all_optimization=True, verbose=args.verbose)
    if session is not None:
        Gpt2Helper.test_parity(session,
                               model,
                               device,
                               args.precision == Precision.FLOAT16,
                               rtol=args.tolerance,
                               atol=args.tolerance,
                               model_class=args.model_class,
                               has_position_ids=use_padding,
                               has_attention_mask=use_padding)

    if args.input_test_file:
        test_inputs = []
        # Each line of test file is a JSON string like:
        # {"input_ids": [[14698, 257, 1310, 13688, 319, 326]]}
        with open(args.input_test_file) as read_f:
            for _, line in enumerate(read_f):
                line = line.rstrip()
                data = json.loads(line)
                input_ids = torch.from_numpy(numpy.asarray(data["input_ids"], dtype=numpy.int64)).to(device)

                if use_padding:
                    if "attention_mask" in data:
                        numpy_float = numpy.float16 if args.precision == Precision.FLOAT16 else numpy.float32
                        attention_mask = torch.from_numpy(numpy.asarray(data["attention_mask"],
                                                                        dtype=numpy_float)).to(device)
                    else:
                        padding = -1
                        attention_mask = (
                            input_ids !=
                            padding).type(torch.float16 if args.precision == Precision.FLOAT16 else torch.float32)
                        input_ids.masked_fill_(input_ids == padding, 0)

                    if "position_ids" in data:
                        position_ids = torch.from_numpy(numpy.asarray(data["position_ids"],
                                                                      dtype=numpy.int64)).to(device)
                    else:
                        position_ids = (attention_mask.long().cumsum(-1) - 1)
                        position_ids.masked_fill_(position_ids < 0, 0)

                    inputs = {"input_ids": input_ids, "position_ids": position_ids, "attention_mask": attention_mask}
                else:
                    inputs = {"input_ids": input_ids}

                test_inputs.append(inputs)

        Gpt2Tester.test_generation(session,
                                   model,
                                   device,
                                   test_inputs,
                                   precision=args.precision,
                                   model_class=args.model_class,
                                   top_k=20,
                                   top_k_no_order=True,
                                   max_steps=24,
                                   max_inputs=0,
                                   verbose=args.verbose,
                                   save_test_data=3,
                                   save_test_data_dir=Path(output_path).parent)

    logger.info(f"Done. Output model: {output_path}")


if __name__ == '__main__':
    main()
