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
from packaging import version
from transformers import AutoConfig
from gpt2_helper import Gpt2Helper, DEFAULT_TOLERANCE, PRETRAINED_GPT2_MODELS
from gpt2_beamsearch_helper import Gpt2HelperFactory, MODEL_CLASSES
from quantize_helper import QuantizeHelper
from benchmark_helper import create_onnxruntime_session, setup_logger, prepare_environment, Precision

logger = logging.getLogger('')


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser()

    parser.add_argument('-m',
                        '--model_name_or_path',
                        required=True,
                        type=str,
                        help='Model path, or pretrained model name selected in the list: ' +
                        ', '.join(PRETRAINED_GPT2_MODELS))

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
    parser.add_argument('--beam_size', type=int, default=4, help='Beam size if greedy/top-p/top-k sampling is needed')

    parser.add_argument('--sequence_lengths',
                        nargs='+',
                        type=int,
                        default=[1],
                        help="sequence lengths (excluding past)")

    parser.add_argument('-s',
                        '--past_sequence_lengths',
                        nargs='+',
                        type=int,
                        default=[8, 16, 32, 64, 128, 256],
                        help="past sequence lengths")

    parser.add_argument("-r", "--result_csv", required=False, default=None, help="CSV file for saving summary results.")

    parser.add_argument("--thread_num", required=False, type=int, default=-1, help="Threads to use")

    parser.add_argument('--include_copy_output_latency', required=False, action='store_true')
    parser.set_defaults(include_copy_output_latency=False)

    parser.add_argument('--verbose', required=False, action='store_true')
    parser.set_defaults(verbose=False)

    search_option_group = parser.add_argument_group("configurable one step search options")

    search_option_group.add_argument('--ignore_eos',
                                     type=bool,
                                     default=False,
                                     help='If ignore end of sentence token in model inference.')
    search_option_group.add_argument('--repetition_penalty',
                                     type=float,
                                     default=1,
                                     help='Positive. >1 to penalize and <1 to encorage.')
    search_option_group.add_argument('--temperature',
                                     type=float,
                                     default=1,
                                     help='Softmax temperature for output logits.')
    search_option_group.add_argument('--excluded_token_ids',
                                     required=False,
                                     nargs='+',
                                     type=float,
                                     help='A list of token ids to be excluded in inference.')
    search_option_group.add_argument('--length_penalty',
                                     type=float,
                                     default=1,
                                     help='Positive. >1 to penalize and <1 to encorage short sentence.')

    sampling_option_group = parser.add_argument_group("one step sampling options")
    sampling_option_group.add_argument('--do_sample',
                                       action='store_true',
                                       help='If to do sampling instead of beam search or greedy.')
    sampling_option_group.add_argument('--do_sample_top_p',
                                       type=float,
                                       default=0.95,
                                       help='Nuclear/top-p sampling accumulation probability.')
    sampling_option_group.add_argument('--do_sample_top_k', type=int, default=0, help='Use top-k if non-zero.')

    args = parser.parse_args(argv)

    return args


def main(args):
    from transformers import __version__ as transformers_version
    if version.parse(transformers_version) < version.parse(
            "3.1.0"):  # past_key_values name does not exist in 3.0.2 or older
        raise RuntimeError("This tool requires transformers 3.1.0 or later.")

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
    if args.model_class == "GPT2LMHeadModel_BeamSearchStep":
        model_type = "beam_search_step"
    elif args.model_class == "GPT2LMHeadModel_ConfigurableOneStepSearch":
        model_type = "configurable_one_step_search"
    else:
        model_type = "default"

    gpt2helper = Gpt2HelperFactory.create_helper(model_type)
    config = AutoConfig.from_pretrained(args.model_name_or_path, torchscript=args.torchscript, cache_dir=cache_dir)
    if model_type == 'beam_search_step':
        model = model_class.from_pretrained(args.model_name_or_path,
                                            config=config,
                                            batch_size=1,
                                            beam_size=args.beam_size,
                                            cache_dir=cache_dir)
    elif model_type == 'configurable_one_step_search':
        model = model_class.from_pretrained(args.model_name_or_path,
                                            config=config,
                                            batch_size=1,
                                            beam_size=args.beam_size,
                                            ignore_eos=args.ignore_eos,
                                            temperature=args.temperature,
                                            repetition_penalty=args.repetition_penalty,
                                            excluded_token_ids=args.excluded_token_ids,
                                            length_penalty=args.length_penalty,
                                            do_sample=args.do_sample,
                                            do_sample_top_p=args.do_sample_top_p,
                                            do_sample_top_k=args.do_sample_top_k,
                                            cache_dir=cache_dir)
    else:
        model = model_class.from_pretrained(args.model_name_or_path, config=config, cache_dir=cache_dir)

    # This scirpt does not support float16 for PyTorch.
    #if args.float16:
    #    model.half()

    device = torch.device("cuda:0" if args.use_gpu else "cpu")
    model.to(device)
    use_external_data_format = (config.n_layer > 24)  #TODO: find a way to check model size > 2GB
    onnx_model_paths = gpt2helper.get_onnx_paths(output_dir,
                                                 args.model_name_or_path,
                                                 args.model_class,
                                                 has_past=True,
                                                 new_folder=use_external_data_format)

    onnx_model_path = onnx_model_paths["raw"]
    use_padding = MODEL_CLASSES[args.model_class][2]
    gpt2helper.export_onnx(model,
                           device,
                           onnx_model_path,
                           args.verbose,
                           use_external_data_format,
                           has_position_ids=use_padding,
                           has_attention_mask=use_padding)

    if args.optimize_onnx or args.precision != Precision.FLOAT32:
        onnx_model_path = onnx_model_paths[str(args.precision) if args.precision != Precision.INT8 else 'fp32']
        gpt2helper.optimize_onnx(onnx_model_paths["raw"], onnx_model_path, args.precision == Precision.FLOAT16,
                                 model.config.num_attention_heads, model.config.hidden_size, use_external_data_format)

        if args.precision == Precision.INT8:
            logger.info("quantizing model...")
            QuantizeHelper.quantize_onnx_model(onnx_model_path, onnx_model_paths["int8"], use_external_data_format)
            model = QuantizeHelper.quantize_torch_model(model)
            logger.info("finished quantizing model")
            onnx_model_path = onnx_model_paths["int8"]

    if args.torchscript:
        model = gpt2helper.torchscript(model,
                                       config,
                                       device,
                                       has_position_ids=use_padding,
                                       has_attention_mask=use_padding)

    session = create_onnxruntime_session(onnx_model_path,
                                         args.use_gpu,
                                         enable_all_optimization=False,
                                         num_threads=args.thread_num,
                                         verbose=args.verbose)
    if session is None:
        return

    # Allocate output buffers for IO Binding
    if model_type == 'beam_search_step' or model_type == 'configurable_one_step_search':
        max_output_shapes = gpt2helper.get_output_shapes(max(args.batch_sizes), max(args.past_sequence_lengths),
                                                         max(args.past_sequence_lengths), max(args.sequence_lengths), 4,
                                                         0, config, args.model_class)
        output_buffers = gpt2helper.get_output_buffers(max_output_shapes, device, args.precision == Precision.FLOAT16)

    else:
        max_output_shapes = gpt2helper.get_output_shapes(max(args.batch_sizes), max(args.past_sequence_lengths),
                                                         max(args.sequence_lengths), config, args.model_class)
        output_buffers = gpt2helper.get_output_buffers(max_output_shapes, device, args.precision == Precision.FLOAT16)

    csv_filename = args.result_csv or "benchmark_result_{}.csv".format(datetime.now().strftime("%Y%m%d-%H%M%S"))
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
                    if model_type == 'beam_search_step' or model_type == 'configurable_one_step_search':
                        dummy_inputs = gpt2helper.get_dummy_inputs(batch_size,
                                                                   past_sequence_length,
                                                                   sequence_length,
                                                                   config.num_attention_heads,
                                                                   config.hidden_size,
                                                                   config.n_layer,
                                                                   config.vocab_size,
                                                                   device,
                                                                   float16=(args.precision == Precision.FLOAT16),
                                                                   has_position_ids=use_padding,
                                                                   has_attention_mask=use_padding)
                        output_shapes = gpt2helper.get_output_shapes(batch_size, past_sequence_length,
                                                                     past_sequence_length, sequence_length, 4, 0,
                                                                     config, args.model_class)
                    else:
                        dummy_inputs = gpt2helper.get_dummy_inputs(batch_size,
                                                                   past_sequence_length,
                                                                   sequence_length,
                                                                   config.num_attention_heads,
                                                                   config.hidden_size,
                                                                   config.n_layer,
                                                                   config.vocab_size,
                                                                   device,
                                                                   float16=(args.precision == Precision.FLOAT16),
                                                                   has_position_ids=use_padding,
                                                                   has_attention_mask=use_padding)
                        output_shapes = gpt2helper.get_output_shapes(batch_size, past_sequence_length, sequence_length,
                                                                     config, args.model_class)

                    try:
                        outputs, torch_latency = Gpt2Helper.pytorch_inference(model, dummy_inputs, args.test_times)
                        ort_outputs, ort_latency = Gpt2Helper.onnxruntime_inference(session, dummy_inputs,
                                                                                    args.test_times)
                        ort_io_outputs, ort_io_latency = gpt2helper.onnxruntime_inference_with_binded_io(
                            session,
                            dummy_inputs,
                            output_buffers,
                            output_shapes,
                            args.test_times,
                            return_numpy=False,
                            include_copy_output_latency=args.include_copy_output_latency)

                        if args.validate_onnx:
                            if gpt2helper.compare_outputs(outputs,
                                                          ort_outputs,
                                                          model_class,
                                                          rtol=DEFAULT_TOLERANCE[args.precision],
                                                          atol=DEFAULT_TOLERANCE[args.precision]):
                                logger.info(
                                    f'Pytorch and ONNX Runtime outputs are all close (tolerance={DEFAULT_TOLERANCE[args.precision]}).'
                                )

                            # Results of IO binding might be in GPU. Copy outputs to CPU for comparison.
                            copy_outputs = []
                            for output in ort_io_outputs:
                                copy_outputs.append(output.cpu().numpy())

                            if gpt2helper.compare_outputs(outputs,
                                                          copy_outputs,
                                                          model_class,
                                                          rtol=DEFAULT_TOLERANCE[args.precision],
                                                          atol=DEFAULT_TOLERANCE[args.precision]):
                                logger.info(
                                    f'Pytorch and ONNX Runtime IO Binding outputs are all close (tolerance={DEFAULT_TOLERANCE[args.precision]}).'
                                )

                        logger.info(
                            f"batch_size={batch_size}, sequence_length={sequence_length}, past_sequence_length={past_sequence_length}, torch_latency={torch_latency:.2f}, onnxruntime_latency={ort_latency:.2f}, onnxruntime_io_binding_latency={ort_io_latency:.2f}"
                        )

                        row = {
                            "model_name": args.model_name_or_path,
                            "model_class": args.model_class,
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
    return csv_filename


if __name__ == '__main__':
    args = parse_arguments()
    setup_logger(args.verbose)
    main(args)
