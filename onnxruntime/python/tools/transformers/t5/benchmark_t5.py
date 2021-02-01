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
from transformers import T5Config
from convert_to_onnx import export_onnx_models
from t5_helper import T5EncoderHelper, T5DecoderHelper, T5Helper, IOBindingHelper, PRETRAINED_T5_MODELS, T5DecoderNoPastState

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from benchmark_helper import create_onnxruntime_session, setup_logger, prepare_environment, Precision

logger = logging.getLogger('')


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser()

    parser.add_argument('-m',
                        '--model_name_or_path',
                        required=False,
                        default=PRETRAINED_T5_MODELS[0],
                        type=str,
                        help='Model path, or pretrained model name selected in the list: ' +
                        ', '.join(PRETRAINED_T5_MODELS))

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

    parser.add_argument('--use_past_state', required=False, action='store_true', help="use past state for inference")
    parser.set_defaults(use_past_state=False)

    parser.add_argument('-v', '--validate_onnx', required=False, action='store_true', help='Validate ONNX model')
    parser.set_defaults(validate_onnx=False)

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

    parser.add_argument('--sequence_lengths',
                        nargs='+',
                        type=int,
                        default=[1],
                        help="sequence lengths (excluding past)")

    parser.add_argument('-s',
                        '--past_sequence_lengths',
                        nargs='+',
                        type=int,
                        default=[8, 16], #[8, 16, 32, 64, 128, 256],
                        help="past sequence lengths")

    parser.add_argument("-r", "--result_csv", required=False, default=None, help="CSV file for saving summary results.")

    parser.add_argument("--thread_num", required=False, type=int, default=-1, help="Threads to use")

    parser.add_argument('--include_copy_output_latency', required=False, action='store_true')
    parser.set_defaults(include_copy_output_latency=False)

    parser.add_argument('--verbose', required=False, action='store_true')
    parser.set_defaults(verbose=False)

    args = parser.parse_args(argv)

    return args


def main(args):
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

    device = torch.device("cuda:0" if args.use_gpu else "cpu")
    encoder, decoder = T5Helper.load_model(args.model_name_or_path, cache_dir, device, args.use_past_state)
    
    if args.torchscript:
        encoder = T5EncoderHelper.torchscript(encoder, device)
        decoder = T5EncoderHelper.torchscript(decoder, device)

    config = decoder.config
    use_external_data_format = (config.num_layers > 24)

    from convert_to_onnx import export_onnx_models
    output_paths = export_onnx_models(args.model_name_or_path, cache_dir, output_dir, args.use_past_state, args.use_gpu,
                                      use_external_data_format, args.optimize_onnx, args.precision, args.verbose, overwrite=False)

    decoder_session = create_onnxruntime_session(output_paths[0],
                                                 args.use_gpu,
                                                 enable_all_optimization=False,
                                                 num_threads=args.thread_num,
                                                 verbose=args.verbose)
    assert decoder_session is not None

    encoder_session = create_onnxruntime_session(output_paths[1],
                                                 args.use_gpu,
                                                 enable_all_optimization=False,
                                                 num_threads=args.thread_num,
                                                 verbose=args.verbose)
    assert decoder_session is not None

    # Allocate output buffers for IO Binding
    max_encoder_output_shapes = T5EncoderHelper.get_output_shapes(max(args.batch_sizes), max(args.sequence_lengths),
                                                                  config)
    encoder_output_buffers = IOBindingHelper.get_output_buffers(max_encoder_output_shapes, device,
                                                         args.precision == Precision.FLOAT16)

    # Test encoder
    """
    csv_filename = args.result_csv or "benchmark_t5_encoder_{}.csv".format(datetime.now().strftime("%Y%m%d-%H%M%S"))
    with open(csv_filename, mode="a", newline='') as csv_file:
        column_names = [
            "model_name", "model_class", "gpu", "precision", "optimizer", "torchscript", "batch_size",
            "sequence_length", "torch_latency", "onnxruntime_latency", "onnxruntime_io_binding_latency"
        ]
        csv_writer = csv.DictWriter(csv_file, fieldnames=column_names)
        csv_writer.writeheader()

        for batch_size in args.batch_sizes:
            for sequence_length in args.sequence_lengths:
                assert batch_size > 0 and sequence_length > 0
                logger.debug(
                    f"Running test for encoder with batch_size={batch_size} sequence_length={sequence_length} ...")

                encoder_inputs = T5EncoderHelper.random_inputs(batch_size, sequence_length, config.vocab_size, device)
                output_shapes = T5EncoderHelper.get_output_shapes(batch_size, sequence_length, config)

                try:
                    outputs, torch_latency = T5Helper.pytorch_inference(encoder, encoder_inputs, args.test_times)
                    ort_outputs, ort_latency = T5EncoderHelper.onnxruntime_inference(
                        encoder_session, encoder_inputs, args.test_times)
                    ort_io_outputs, ort_io_latency = T5EncoderHelper.onnxruntime_inference_with_binded_io(
                        encoder_session,
                        encoder_inputs,
                        encoder_output_buffers,
                        output_shapes,
                        args.test_times,
                        return_numpy=False,
                        include_copy_output_latency=args.include_copy_output_latency)

                    logger.info(
                        f"model=encoder, batch_size={batch_size}, sequence_length={sequence_length}, torch_latency={torch_latency:.2f}, onnxruntime_latency={ort_latency:.2f}, onnxruntime_io_binding_latency={ort_io_latency:.2f}"
                    )

                    row = {
                        "model_name": args.model_name_or_path,
                        "model_class": "encoder",  #TODO
                        "gpu": args.use_gpu,
                        "precision": args.precision,
                        "optimizer": args.optimize_onnx,
                        "torchscript": args.torchscript,
                        "batch_size": batch_size,
                        "sequence_length": sequence_length,
                        "torch_latency": f"{torch_latency:.2f}",
                        "onnxruntime_latency": f"{ort_latency:.2f}",
                        "onnxruntime_io_binding_latency": f"{ort_io_latency:.2f}"
                    }
                    csv_writer.writerow(row)
                except:
                    logger.error(f"Exception", exc_info=True)

    logger.info(f"Results are saved to file {csv_filename}")
    """

    # Test decoder
    config:T5Config = decoder.config
    max_decoder_output_shapes = T5DecoderHelper.get_output_shapes(max(args.batch_sizes),
                                                                  max(args.sequence_lengths),
                                                                  max(args.past_sequence_lengths),
                                                                  config,
                                                                  args.use_past_state)
    decoder_output_buffers = IOBindingHelper.get_output_buffers(max_decoder_output_shapes, device, args.precision == Precision.FLOAT16)

    csv_filename = args.result_csv or "benchmark_t5_decoder_{}.csv".format(datetime.now().strftime("%Y%m%d-%H%M%S"))
    with open(csv_filename, mode="a", newline='') as csv_file:
        column_names = [
            "model_name", "model_class", "gpu", "precision", "optimizer", "torchscript", "batch_size",
            "sequence_length", "past_sequence_length", "torch_latency", "onnxruntime_latency",
            "onnxruntime_io_binding_latency"
        ]
        csv_writer = csv.DictWriter(csv_file, fieldnames=column_names)
        csv_writer.writeheader()

        for batch_size in args.batch_sizes:
            for sequence_length in args.sequence_lengths:
                for past_sequence_length in args.past_sequence_lengths:
                    assert batch_size > 0 and sequence_length > 0 and past_sequence_length >= 0
                    logger.debug(
                        f"Running test for batch_size={batch_size} sequence_length={sequence_length} past_sequence_length={past_sequence_length}..."
                    )

                    decoder_inputs = T5DecoderHelper.random_inputs(decoder, batch_size, sequence_length, past_sequence_length, device)

                    output_shapes = T5DecoderHelper.get_output_shapes(batch_size, sequence_length, past_sequence_length, config, args.use_past_state)

                    try:
                        outputs, torch_latency = T5Helper.pytorch_inference(decoder, decoder_inputs, args.test_times)
                        ort_outputs, ort_latency = T5DecoderHelper.onnxruntime_inference(decoder_session, decoder_inputs, args.test_times)
                        ort_io_outputs, ort_io_latency = T5DecoderHelper.onnxruntime_inference_with_binded_io(
                            decoder_session,
                            decoder_inputs,
                            decoder_output_buffers,
                            output_shapes,
                            args.test_times,
                            return_numpy=False,
                            include_copy_output_latency=args.include_copy_output_latency)

                        logger.info(
                            f"model=decoder, batch_size={batch_size}, sequence_length={sequence_length}, past_sequence_length={past_sequence_length}, torch_latency={torch_latency:.2f}, onnxruntime_latency={ort_latency:.2f}, onnxruntime_io_binding_latency={ort_io_latency:.2f}"
                        )

                        row = {
                            "model_name": args.model_name_or_path,
                            "model_class": "decoder", #TODO
                            "gpu": args.use_gpu,
                            "precision": args.precision,
                            "optimizer": args.optimize_onnx,
                            "torchscript": args.torchscript,
                            "batch_size": batch_size,
                            "sequence_length": sequence_length,
                            "past_sequence_length": past_sequence_length,
                            "torch_latency": f"{torch_latency:.2f}",
                            "onnxruntime_latency": f"{ort_latency:.2f}",
                            "onnxruntime_io_binding_latency": f"{ort_io_latency:.2f}"
                        }
                        csv_writer.writerow(row)
                    except:
                        logger.error(f"Exception", exc_info=True)

    logger.info(f"Results are saved to file {csv_filename}")


if __name__ == '__main__':
    args = parse_arguments()
    setup_logger(args.verbose)
    main(args)
