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
from gpt2_helper import DEFAULT_TOLERANCE, PRETRAINED_GPT2_MODELS
from gpt2_beamsearch_helper import Gpt2HelperFactory, MODEL_CLASSES
from gpt2_beamsearch_tester import Gpt2TesterFactory
from quantize_helper import QuantizeHelper
from benchmark_helper import create_onnxruntime_session, setup_logger, prepare_environment, Precision

logger = logging.getLogger('')


def parse_arguments(argv=None):
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

    parser.add_argument("-t",
                        "--test_cases",
                        required=False,
                        type=int,
                        default=1000,
                        help="Number of test cases for parity")

    parser.add_argument('--verbose', required=False, action='store_true')
    parser.set_defaults(verbose=False)

    parser.add_argument('-e', '--use_external_data_format', required=False, action='store_true')
    parser.set_defaults(use_external_data_format=False)
    parser.add_argument('--beam_size', type=int, default=4, help='Beam size if greedy/top-p/top-k sampling is needed')

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

    fp16_option_group = parser.add_argument_group(
        "float to float16 conversion parameters that works when \"--precision fp16\" is specified")

    fp16_option_group.add_argument('--keep_io_types',
                                   required=False,
                                   action='store_true',
                                   help='Use float32 for past inputs, present and logits outputs.')
    fp16_option_group.set_defaults(keep_io_types=False)

    fp16_option_group.add_argument('--io_block_list',
                                   nargs='+',
                                   default=[],
                                   help='List of inputs or outputs in float32 instead of float16')

    fp16_option_group.add_argument(
        '--op_block_list',
        nargs='+',
        default=[],
        help=
        'List of operators (like Attention Gather Add LayerNormalization FastGelu MatMul) to compute in float32 instead of float16.'
    )

    fp16_option_group.add_argument('--node_block_list',
                                   nargs='+',
                                   default=[],
                                   help='List of node names to compute in float32 instead of float16.')

    fp16_option_group.add_argument('--force_fp16_initializers',
                                   required=False,
                                   action='store_true',
                                   help='Convert all float initializers to float16.')
    fp16_option_group.set_defaults(force_fp16_initializers=False)

    args = parser.parse_args(argv)

    return args


def main(argv=None, experiment_name="", run_id=0, csv_filename="gpt2_parity_results.csv"):
    result = {}
    from transformers import __version__ as transformers_version
    if version.parse(transformers_version) < version.parse(
            "3.1.0"):  # past_key_values name does not exist in 3.0.2 or older
        raise RuntimeError("This tool requires transformers 3.1.0 or later.")

    args = parse_arguments(argv)
    setup_logger(args.verbose)

    if not experiment_name:
        import sys
        experiment_name = " ".join(argv if argv else sys.argv[1:])

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
    if args.model_class == "GPT2LMHeadModel_BeamSearchStep":
        model_type = "beam_search_step"
    elif args.model_class == "GPT2LMHeadModel_ConfigurableOneStepSearch":
        model_type = "configurable_one_step_search"
    else:
        model_type = "default"

    gpt2helper = Gpt2HelperFactory.create_helper(model_type)
    gpt2tester = Gpt2TesterFactory.create_tester(model_type)
    config = AutoConfig.from_pretrained(args.model_name_or_path, cache_dir=cache_dir)
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

    device = torch.device("cuda:0" if args.use_gpu else "cpu")
    model.eval().to(device)

    if (not args.use_external_data_format) and (config.n_layer > 24):
        logger.info(f"Try --use_external_data_format when model size > 2GB")

    onnx_model_paths = gpt2helper.get_onnx_paths(output_dir,
                                                 args.model_name_or_path,
                                                 args.model_class,
                                                 new_folder=args.use_external_data_format)

    raw_onnx_model = onnx_model_paths["raw"]

    logger.info(f"Exporting ONNX model to {raw_onnx_model}")
    use_padding = MODEL_CLASSES[args.model_class][2]

    gpt2helper.export_onnx(model,
                           device,
                           raw_onnx_model,
                           args.verbose,
                           args.use_external_data_format,
                           has_position_ids=use_padding,
                           has_attention_mask=use_padding)

    fp16_params = {"keep_io_types": args.keep_io_types}
    if args.io_block_list:
        fp16_params["keep_io_types"] = args.io_block_list
    if args.node_block_list:
        fp16_params["node_block_list"] = args.node_block_list
    if args.op_block_list:
        fp16_params["op_block_list"] = args.op_block_list
    if args.force_fp16_initializers:
        fp16_params["force_fp16_initializers"] = args.force_fp16_initializers

    is_io_float16 = (args.precision == Precision.FLOAT16 and not args.keep_io_types)

    if args.optimize_onnx or args.precision != Precision.FLOAT32:
        output_path = onnx_model_paths[str(args.precision) if args.precision != Precision.INT8 else 'fp32']

        logger.info(f"Optimizing model to {output_path}")
        gpt2helper.optimize_onnx(raw_onnx_model, output_path, args.precision == Precision.FLOAT16,
                                 model.config.num_attention_heads, model.config.hidden_size,
                                 args.use_external_data_format, **fp16_params)
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
    if args.model_class == "GPT2LMHeadModel" and session is not None:
        parity_result = gpt2helper.test_parity(session,
                                               model,
                                               device,
                                               is_io_float16,
                                               rtol=args.tolerance,
                                               atol=args.tolerance,
                                               model_class=args.model_class,
                                               has_position_ids=use_padding,
                                               has_attention_mask=use_padding,
                                               total_test_cases=args.test_cases,
                                               verbose=args.verbose)

        latency = gpt2helper.test_performance(session,
                                              model,
                                              device,
                                              is_io_float16,
                                              total_runs=100,
                                              use_io_binding=True,
                                              model_class=args.model_class,
                                              has_position_ids=use_padding,
                                              has_attention_mask=use_padding,
                                              batch_size=8,
                                              sequence_length=1,
                                              past_sequence_length=32)

        if args.precision == Precision.FLOAT16:
            logger.info(f"fp16 conversion parameters:{fp16_params}")

        # Write results to file
        import csv
        from onnxruntime import __version__ as ort_version
        latency_name = "average_latency(batch_size=8,sequence_length=1,past_sequence_length=32)"
        csv_file_existed = os.path.exists(csv_filename)
        with open(csv_filename, mode="a", newline='') as csv_file:
            column_names = [
                "experiment", "run_id", "model_name", "model_class", "gpu", "precision", "optimizer", "test_cases",
                "keep_io_types", "io_block_list", "op_block_list", "node_block_list", "force_fp16_initializers",
                "ORT_TRANSFORMER_OPTIONS", "ORT_CUDA_GEMM_OPTIONS", "onnxruntime", latency_name, "diff_50_percentile",
                "diff_90_percentile", "diff_95_percentile", "diff_99_percentile", "diff_pass_rate", "nan_rate",
                "top1_match_rate", "onnx_size_in_MB"
            ]
            csv_writer = csv.DictWriter(csv_file, fieldnames=column_names)
            if not csv_file_existed:
                csv_writer.writeheader()
            row = {
                "experiment": experiment_name,
                "run_id": run_id,
                "model_name": args.model_name_or_path,
                "model_class": args.model_class,
                "gpu": args.use_gpu,
                "precision": args.precision,
                "optimizer": args.optimize_onnx,
                "test_cases": args.test_cases,
                "keep_io_types": args.keep_io_types,
                "io_block_list": args.io_block_list,
                "op_block_list": args.op_block_list,
                "node_block_list": args.node_block_list,
                "force_fp16_initializers": args.force_fp16_initializers,
                "ORT_TRANSFORMER_OPTIONS": os.getenv('ORT_TRANSFORMER_OPTIONS'),
                "ORT_CUDA_GEMM_OPTIONS": os.getenv('ORT_CUDA_GEMM_OPTIONS'),
                "onnxruntime": ort_version,
                latency_name: f"{latency:.2f}",
                "diff_50_percentile": parity_result["max_diff_percentile_50"],
                "diff_90_percentile": parity_result["max_diff_percentile_90"],
                "diff_95_percentile": parity_result["max_diff_percentile_95"],
                "diff_99_percentile": parity_result["max_diff_percentile_99"],
                "diff_pass_rate": parity_result["diff_pass_rate"],
                "nan_rate": parity_result["nan_rate"],
                "top1_match_rate": parity_result["top1_match_rate"],
                "onnx_size_in_MB": "{}".format(int(os.path.getsize(output_path) / 1024 / 1024))
            }
            logger.info(f"result: {row}")
            result.update(row)
            csv_writer.writerow(row)

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
                        numpy_float = numpy.float16 if is_io_float16 else numpy.float32
                        attention_mask = torch.from_numpy(numpy.asarray(data["attention_mask"],
                                                                        dtype=numpy_float)).to(device)
                    else:
                        padding = -1
                        attention_mask = (input_ids != padding).type(torch.float16 if is_io_float16 else torch.float32)
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

                if model_type == "beam_search_step" or model_type == "configurable_one_step_search":
                    beam_select_idx = torch.zeros([1, input_ids.shape[0]]).long()

                    input_log_probs = torch.zeros([input_ids.shape[0], 1])
                    input_unfinished_sents = torch.ones([input_ids.shape[0], 1], dtype=torch.bool)
                    inputs.update({
                        "beam_select_idx": beam_select_idx,
                        "input_log_probs": input_log_probs,
                        "input_unfinished_sents": input_unfinished_sents,
                    })

                test_inputs.append(inputs)

        gpt2tester.test_generation(session,
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
    return result


if __name__ == '__main__':
    main()
