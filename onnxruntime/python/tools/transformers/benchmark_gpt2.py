# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
# This script benchmarks gpt2 model with past state.
# For gpt2 model without past state, use benchmark.py to measure performance.

import os
import sys
import numpy
import csv
from datetime import datetime
import psutil
import argparse
import logging
import torch
import onnx
from transformers import AutoConfig
from gpt2_helper import Gpt2Helper, MyGPT2Model, MyGPT2LMHeadModel, MODEL_CLASSES, DEFAULT_TOLERANCE
from quantize_helper import QuantizeHelper
from benchmark_helper import create_onnxruntime_session, setup_logger, prepare_environment, Precision

logger = logging.getLogger('')

PRETRAINED_MODELS = ['gpt2', 'distilgpt2']


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('-m',
                        '--model_name',
                        required=True,
                        type=str,
                        choices=PRETRAINED_MODELS,
                        help='Pretrained model selected in the list: ' + ', '.join(PRETRAINED_MODELS))

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

    parser.add_argument('--onnx_dir',
                        required=False,
                        type=str,
                        default=os.path.join('.', 'onnx_models'),
                        help='Directory to store onnx models')

    parser.add_argument('--test_times',
                        required=False,
                        default=100,
                        type=int,
                        help='Number of repeat times to get average inference latency.')

    parser.add_argument('-v', '--validate_onnx', required=False, action='store_true', help='Validate ONNX model')

    parser.add_argument('-o',
                        '--optimize_onnx',
                        required=False,
                        action='store_true',
                        help='Use optimizer.py to optimize onnx model')
    parser.set_defaults(optimize_onnx=False)

    parser.add_argument('--use_gpu', required=False, action='store_true', help="use GPU for inference")
    parser.set_defaults(use_gpu=False)

    parser.add_argument(
        "-p",
        "--precision",
        type=Precision,
        default=Precision.FLOAT32,
        choices=list(Precision),
        help="Precision of model to run. fp32 for full precision, fp16 for half precision, and int8 for quantization")

    parser.add_argument('--torchscript', required=False, action='store_true', help="use Torchscript")
    parser.set_defaults(torchscript=False)

    parser.add_argument('-b', '--batch_sizes', nargs='+', type=int, default=[1], help="batch size")

    parser.add_argument('-s',
                        '--past_sequence_lengths',
                        nargs='+',
                        type=int,
                        default=[8, 16, 32, 64, 128, 256],
                        help="past sequence lengths")

    parser.add_argument("-r", "--result_csv", required=False, default=None, help="CSV file for saving summary results.")

    parser.add_argument("--thread_num", required=False, type=int, default=-1, help="Threads to use")

    parser.add_argument('--verbose', required=False, action='store_true')
    parser.set_defaults(verbose=False)

    args = parser.parse_args()

    return args


def main():
    args = parse_arguments()
    setup_logger(args.verbose)

    logger.info(f"Arguments:{args}")
    if args.precision == Precision.FLOAT16:
        assert args.optimize_onnx and args.use_gpu, "fp16 requires --optimize_onnx --use_gpu"

    if args.precision == Precision.INT8:
        assert not args.use_gpu, "quantization only supports CPU"

    torch.set_num_threads(psutil.cpu_count(logical=True) if args.thread_num <= 0 else args.thread_num)
    print(torch.__config__.parallel_info())

    cache_dir = args.cache_dir
    output_dir = args.onnx_dir
    prepare_environment(cache_dir, output_dir, args.use_gpu)

    model_class = MODEL_CLASSES[args.model_class][0]

    config = AutoConfig.from_pretrained(args.model_name, torchscript=args.torchscript, cache_dir=cache_dir)
    model = model_class.from_pretrained(args.model_name, config=config, cache_dir=cache_dir)

    # This scirpt does not support float16 for PyTorch.
    #if args.float16:
    #    model.half()

    device = torch.device("cuda:0" if args.use_gpu else "cpu")
    model.to(device)

    onnx_model_paths = Gpt2Helper.get_onnx_paths(output_dir, args.model_name, args.model_class)

    onnx_model_path = onnx_model_paths["raw"]
    Gpt2Helper.export_onnx(model, device, onnx_model_path, args.verbose)

    if args.optimize_onnx or args.precision != Precision.FLOAT32:
        onnx_model_path = onnx_model_paths[str(args.precision)]
        Gpt2Helper.optimize_onnx(onnx_model_paths["raw"], onnx_model_path, args.precision == Precision.FLOAT16,
                                 model.config.num_attention_heads, model.config.hidden_size)

        if args.precision == Precision.INT8:
            logger.info("quantizing model...")
            QuantizeHelper.quantize_onnx_model(onnx_model_path, onnx_model_path)
            model = QuantizeHelper.quantize_torch_model(model)
            logger.info("finished quantizing model")

    if args.torchscript:
        model = Gpt2Helper.torchscript(model, config, device)

    session = create_onnxruntime_session(onnx_model_path,
                                         args.use_gpu,
                                         enable_all_optimization=False,
                                         num_threads=args.thread_num,
                                         verbose=args.verbose)
    if session is None:
        return

    # One word is generated for each inference. This length does not include that of past state.
    sequence_length = 1

    # Allocate output buffers for IO Binding
    max_output_shapes = Gpt2Helper.get_output_shapes(max(args.batch_sizes), max(args.past_sequence_lengths),
                                                     sequence_length, config, args.model_class)
    output_buffers = Gpt2Helper.get_output_buffers(max_output_shapes, device, args.precision == Precision.FLOAT16)

    csv_filename = args.result_csv or "benchmark_result_{}.csv".format(datetime.now().strftime("%Y%m%d-%H%M%S"))
    with open(csv_filename, mode="a", newline='') as csv_file:
        column_names = [
            "model_name", "model_class", "gpu", "precision", "optimizer", "torchscript", "batch_size",
            "past_sequence_length", "torch_latency", "ort_latency", "ort_io_latency"
        ]
        csv_writer = csv.DictWriter(csv_file, fieldnames=column_names)
        csv_writer.writeheader()

        for batch_size in args.batch_sizes:
            for past_sequence_length in args.past_sequence_lengths:
                logger.debug(f"Running test for batch_size={batch_size} past_sequence_length={past_sequence_length}...")
                dummy_inputs = Gpt2Helper.get_dummy_inputs(batch_size, past_sequence_length, sequence_length,
                                                           config.num_attention_heads, config.hidden_size,
                                                           config.n_layer, config.vocab_size, device,
                                                           args.precision == Precision.FLOAT16)
                output_shapes = Gpt2Helper.get_output_shapes(batch_size, past_sequence_length, sequence_length, config,
                                                             args.model_class)

                try:
                    outputs, torch_latency = Gpt2Helper.pytorch_inference(model, dummy_inputs, args.test_times)
                    ort_outputs, ort_latency = Gpt2Helper.onnxruntime_inference(session, dummy_inputs, args.test_times)
                    ort_io_outputs, ort_io_latency = Gpt2Helper.onnxruntime_inference_with_binded_io(
                        session, dummy_inputs, output_buffers, output_shapes, args.test_times)
                    if args.validate_onnx:
                        if Gpt2Helper.compare_outputs(outputs,
                                                      ort_outputs,
                                                      rtol=DEFAULT_TOLERANCE[args.precision],
                                                      atol=DEFAULT_TOLERANCE[args.precision]):
                            logger.info(
                                f'Pytorch and ONNX Runtime outputs are all close (tolerance={DEFAULT_TOLERANCE[args.precision]}).'
                            )
                        if Gpt2Helper.compare_outputs(outputs,
                                                      ort_io_outputs,
                                                      rtol=DEFAULT_TOLERANCE[args.precision],
                                                      atol=DEFAULT_TOLERANCE[args.precision]):
                            logger.info(
                                f'Pytorch and ONNX Runtime IO Binding outputs are all close (tolerance={DEFAULT_TOLERANCE[args.precision]}).'
                            )

                    logger.info(
                        f"batch_size={batch_size}, past_sequence_length={past_sequence_length}, torch_latency={torch_latency:.2f}, ort_latency={ort_latency:.2f}, ort_io_latency={ort_io_latency:.2f}"
                    )

                    row = {
                        "model_name": args.model_name,
                        "model_class": args.model_class,
                        "gpu": args.use_gpu,
                        "precision": args.precision,
                        "optimizer": args.optimize_onnx,
                        "torchscript": args.torchscript,
                        "batch_size": batch_size,
                        "past_sequence_length": past_sequence_length,
                        "torch_latency": f"{torch_latency:.2f}",
                        "ort_latency": f"{ort_latency:.2f}",
                        "ort_io_latency": f"{ort_io_latency:.2f}"
                    }
                    csv_writer.writerow(row)
                except:
                    logger.error(f"Exception", exc_info=True)

    logger.info(f"Results are saved to file {csv_filename}")


if __name__ == '__main__':
    main()
