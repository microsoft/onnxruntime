# Copyright (c) Microsoft Corporation.  All rights reserved.
# Copyright 2018 The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Benchmarking the inference of pretrained transformer models.
    PyTorch/TorchScript benchmark is based on https://github.com/huggingface/transformers/blob/master/examples/benchmarks.py.
    One difference is that random input_ids is generated in this benchmark.

    For onnxruntime, this script will convert a pretrained model to ONNX, and optimize it when -o parameter is used.

    Example commands:
        Export all models to ONNX, optimize and validate them:
            python benchmark.py -b 0 -o -v -i 1 2 3
        Run OnnxRuntime on GPU for all models:
            python benchmark.py -g
        Run OnnxRuntime on GPU for all models with fp32 optimization:
            python benchmark.py -g -o
        Run OnnxRuntime on GPU with fp16 optimization:
            python benchmark.py -g -o -p "fp16"
        Run TorchScript on GPU for all models:
            python benchmark.py -e torchscript -g
        Run TorchScript on GPU for all models with fp16:
            python benchmark.py -e torchscript -g -p "fp16"
        Run ONNXRuntime and TorchScript on CPU for all models with quantization:
            python benchmark.py -e torchscript onnxruntime -p "int8" -o

    It is recommended to use run_benchmark.sh to launch benchmark.
"""

import argparse
import logging
import timeit
from datetime import datetime
import numpy

import os
import psutil
import onnx
from enum import Enum
from benchmark_helper import (create_onnxruntime_session, Precision, setup_logger, get_latency_result, output_details,
                              output_summary, output_fusion_statistics, inference_ort, inference_ort_with_io_binding,
                              allocateOutputBuffers)
from quantize_helper import QuantizeHelper
from onnx_exporter import create_onnxruntime_input, load_pretrained_model, export_onnx_model

logger = logging.getLogger('')

# List of pretrained models: https://huggingface.co/transformers/pretrained_models.html
# Pretrained model name to a tuple of input names, opset_version, use_external_data_format and optimization model type
MODELS = {
    "bert-base-cased": (["input_ids", "attention_mask", "token_type_ids"], 11, False, "bert"),
    "distilbert-base-uncased": (["input_ids", "attention_mask"], 11, False, "bert"),
    "roberta-base": (["input_ids", "attention_mask"], 11, False, "bert"),

    # No past state inputs for GPT models.
    "gpt2": (["input_ids"], 11, False, "gpt2"),  # no past state inputs & outputs
    "gpt2-large": (["input_ids"], 11, True, "gpt2"),  # Model>2GB. Need use_external_data_format=True to export it.
    "distilgpt2": (["input_ids"], 11, False, "gpt2"),  # no past state inputs & outputs

    #"openai-gpt": (["input_ids"], 11, False, "gpt2"),  # no past state inputs

    # Models uses Einsum, which need opset version 12 and PyTorch 1.5.0 or above.
    "albert-base-v2": (["input_ids"], 12, False, "bert"),
    #"xlnet-base-cased": (["input_ids"], 12, False, "bert"),

    #"xlm-mlm-en-2048": (["input_ids"], 11, True, "bert"),
}

cpu_count = psutil.cpu_count(logical=True)
# Set OMP environment variable before importing onnxruntime or torch.
if "OMP_NUM_THREADS" not in os.environ:
    os.environ["OMP_NUM_THREADS"] = str(cpu_count)

import torch
from transformers import (AutoConfig, AutoTokenizer, AutoModel, GPT2Model)


def run_onnxruntime(use_gpu, model_names, precision, batch_sizes, sequence_lengths, repeat_times, input_counts,
                    optimize_onnx, validate_onnx, cache_dir, onnx_dir, verbose, overwrite, disable_ort_io_binding,
                    use_raw_attention_mask, thread_num, model_fusion_statistics):
    import onnxruntime

    results = []
    if use_gpu and ('CUDAExecutionProvider' not in onnxruntime.get_available_providers()):
        logger.error(
            "Please install onnxruntime-gpu package instead of onnxruntime, and use a machine with GPU for testing gpu performance."
        )
        return results

    if (not use_gpu) and ('CUDAExecutionProvider' in onnxruntime.get_available_providers()):
        logger.warning("Please install onnxruntime package instead of onnxruntime-gpu to get best cpu performance.")

    for model_name in model_names:
        all_input_names = MODELS[model_name][0]
        for num_inputs in input_counts:
            if num_inputs > len(all_input_names):
                break

            input_names = all_input_names[:num_inputs]

            with torch.no_grad():
                onnx_model_file, is_valid_onnx_model, vocab_size, max_sequence_length = export_onnx_model(
                    model_name, MODELS[model_name][1], MODELS[model_name][2], MODELS[model_name][3], cache_dir,
                    onnx_dir, input_names, use_gpu, precision, optimize_onnx, validate_onnx, use_raw_attention_mask,
                    overwrite, model_fusion_statistics)
            if not is_valid_onnx_model:
                continue

            ort_session = create_onnxruntime_session(onnx_model_file,
                                                     use_gpu,
                                                     enable_all_optimization=True,
                                                     num_threads=thread_num)
            if ort_session is None:
                continue

            ort_output_names = [node_arg.name for node_arg in ort_session.get_outputs()]
            output_buffers = {"last_state": None, "pooler": None}
            device = "cuda" if use_gpu else "cpu"
            config = AutoConfig.from_pretrained(model_name, cache_dir=cache_dir)
            max_last_state_size = numpy.prod(
                [max(batch_sizes), max(sequence_lengths),
                 max(vocab_size, config.hidden_size)])
            max_pooler_size = numpy.prod([max(batch_sizes), config.hidden_size])
            for batch_size in batch_sizes:
                if batch_size <= 0:
                    continue
                for sequence_length in sequence_lengths:
                    if max_sequence_length is not None and sequence_length > max_sequence_length:
                        continue

                    ort_inputs = create_onnxruntime_input(vocab_size, batch_size, sequence_length, input_names)

                    result_template = {
                        "engine": "onnxruntime",
                        "version": onnxruntime.__version__,
                        "device": device,
                        "optimizer": optimize_onnx,
                        "precision": precision,
                        "io_binding": False,
                        "model_name": model_name,
                        "inputs": num_inputs,
                        "batch_size": batch_size,
                        "sequence_length": sequence_length,
                        "datetime": str(datetime.now()),
                    }
                    logger.info("Run onnxruntime on {} with input shape {}".format(model_name,
                                                                                   [batch_size, sequence_length]))
                    result = inference_ort(ort_session, ort_inputs, result_template, repeat_times, batch_size)
                    logger.info(result)
                    results.append(result)

                    if not disable_ort_io_binding:
                        logger.info("Run onnxruntime with io binding on {} with input shape {}".format(
                            model_name, [batch_size, sequence_length]))
                        # Get output sizes from a dummy ort run
                        ort_outputs = ort_session.run(ort_output_names, ort_inputs)
                        result = inference_ort_with_io_binding(ort_session, ort_inputs, result_template, repeat_times,
                                                               ort_output_names, ort_outputs, output_buffers,
                                                               max_last_state_size, max_pooler_size, batch_size, device)
                        logger.info(result)
                        results.append(result)

    return results


def run_pytorch(use_gpu, model_names, precision, batch_sizes, sequence_lengths, repeat_times, torchscript, cache_dir,
                verbose):
    results = []
    if use_gpu and not torch.cuda.is_available():
        logger.error("Please install PyTorch with Cuda, and use a machine with GPU for testing gpu performance.")
        return results

    torch.set_grad_enabled(False)

    for model_name in model_names:
        config = AutoConfig.from_pretrained(model_name, torchscript=torchscript, cache_dir=cache_dir)
        model = load_pretrained_model(model_name, config=config, cache_dir=cache_dir)
        tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
        max_input_size = tokenizer.max_model_input_sizes[model_name]
        logger.debug(f"Model {model}")
        logger.debug(f"Number of parameters {model.num_parameters()}")

        if precision == Precision.FLOAT16:
            model.half()

        device = torch.device("cuda:0" if use_gpu else "cpu")
        model.to(device)

        if precision == Precision.INT8:
            model = QuantizeHelper.quantize_torch_model(model)

        for batch_size in batch_sizes:
            if batch_size <= 0:
                continue

            for sequence_length in sequence_lengths:
                if max_input_size is not None and sequence_length > max_input_size:
                    continue

                logger.info("Run PyTorch on {} with input shape {}".format(model_name, [batch_size, sequence_length]))
                input_ids = torch.randint(low=0,
                                          high=config.vocab_size - 1,
                                          size=(batch_size, sequence_length),
                                          dtype=torch.long,
                                          device=device)
                try:
                    inference = torch.jit.trace(model, input_ids) if torchscript else model
                    inference(input_ids)

                    runtimes = timeit.repeat(lambda: inference(input_ids), repeat=repeat_times, number=1)

                    result = {
                        "engine": "torchscript" if torchscript else "torch",
                        "version": torch.__version__,
                        "device": "cuda" if use_gpu else "cpu",
                        "optimizer": "",
                        "precision": precision,
                        "io_binding": "",
                        "model_name": model_name,
                        "inputs": 1,
                        "batch_size": batch_size,
                        "sequence_length": sequence_length,
                        "datetime": str(datetime.now()),
                    }
                    result.update(get_latency_result(runtimes, batch_size))
                    logger.info(result)
                    results.append(result)
                except RuntimeError as e:
                    logger.exception(e)
                    torch.cuda.empty_cache()

    return results


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("-m",
                        "--models",
                        required=False,
                        nargs="+",
                        type=str,
                        default=["bert-base-cased", "roberta-base", "gpt2"],
                        choices=list(MODELS.keys()),
                        help="Pre-trained models in the list: " + ", ".join(MODELS.keys()))

    parser.add_argument("-e",
                        "--engines",
                        required=False,
                        nargs="+",
                        type=str,
                        default=['onnxruntime'],
                        choices=['onnxruntime', 'torch', 'torchscript'],
                        help="Engines to benchmark")

    parser.add_argument("-c",
                        "--cache_dir",
                        required=False,
                        type=str,
                        default=os.path.join('.', 'cache_models'),
                        help="Directory to cache pre-trained models")

    parser.add_argument("--onnx_dir",
                        required=False,
                        type=str,
                        default=os.path.join('.', 'onnx_models'),
                        help="Directory to store onnx models")

    parser.add_argument("-g", "--use_gpu", required=False, action="store_true", help="Run on cuda device")

    parser.add_argument(
        "-p",
        "--precision",
        type=Precision,
        default=Precision.FLOAT32,
        choices=list(Precision),
        help="Precision of model to run. fp32 for full precision, fp16 for half precision, and int8 for quantization")

    parser.add_argument("--verbose", required=False, action="store_true", help="Print more information")

    parser.add_argument("--overwrite", required=False, action="store_true", help="Overwrite existing models")

    parser.add_argument("-o",
                        "--optimize_onnx",
                        required=False,
                        action="store_true",
                        help="Use optimizer.py to optimize onnx model")

    parser.add_argument("-v", "--validate_onnx", required=False, action="store_true", help="Validate ONNX model")

    parser.add_argument("-f",
                        "--fusion_csv",
                        required=False,
                        default=None,
                        help="CSV file for saving summary results of graph optimization.")

    parser.add_argument("-d", "--detail_csv", required=False, default=None, help="CSV file for saving detail results.")

    parser.add_argument("-r", "--result_csv", required=False, default=None, help="CSV file for saving summary results.")

    parser.add_argument("-i",
                        "--input_counts",
                        required=False,
                        nargs="+",
                        default=[1],
                        type=int,
                        choices=[1, 2, 3],
                        help="Number of ONNX model inputs. Please use 1 for fair comparison with Torch or TorchScript.")

    parser.add_argument("-t",
                        "--test_times",
                        required=False,
                        default=100,
                        type=int,
                        help="Number of repeat times to get average inference latency.")

    parser.add_argument("-b", "--batch_sizes", nargs="+", type=int, default=[1])

    parser.add_argument("-s", "--sequence_lengths", nargs="+", type=int, default=[4, 8, 16, 32, 64, 128, 256])

    parser.add_argument('--disable_ort_io_binding',
                        required=False,
                        action='store_true',
                        help='Disable running ONNX Runtime with binded inputs and outputs. ')
    parser.set_defaults(disable_ort_io_binding=False)

    parser.add_argument('--use_raw_attention_mask',
                        required=False,
                        action='store_true',
                        help='Use raw attention mask in Attention operator for Bert models.')
    parser.set_defaults(use_raw_attention_mask=False)

    parser.add_argument("--thread_num", required=False, type=int, default=-1, help="Threads to use")

    args = parser.parse_args()
    return args


def main():
    args = parse_arguments()

    setup_logger(args.verbose)

    if args.precision == Precision.FLOAT16 and not args.use_gpu:
        logger.error("fp16 is for GPU only")
        return

    if args.precision == Precision.INT8 and args.use_gpu:
        logger.error("int8 is for CPU only")
        return

    logger.info(f"Arguments: {args}")

    if not os.path.exists(args.cache_dir):
        try:
            os.mkdir(args.cache_dir)
        except OSError:
            logger.error("Creation of the directory %s failed" % args.cache_dir)

    enable_torch = "torch" in args.engines
    enable_torchscript = "torchscript" in args.engines
    enable_onnxruntime = "onnxruntime" in args.engines

    results = []

    torch.set_num_threads(cpu_count if args.thread_num <= 0 else args.thread_num)
    logger.debug(torch.__config__.parallel_info())

    if enable_torch or enable_torchscript:
        if args.input_counts != [1]:
            logger.warning("--input_counts is not implemented for torch or torchscript engine.")

        if enable_torchscript:
            results += run_pytorch(args.use_gpu, args.models, args.precision, args.batch_sizes, args.sequence_lengths,
                                   args.test_times, True, args.cache_dir, args.verbose)

        if enable_torch:
            results += run_pytorch(args.use_gpu, args.models, args.precision, args.batch_sizes, args.sequence_lengths,
                                   args.test_times, False, args.cache_dir, args.verbose)

    model_fusion_statistics = {}
    if enable_onnxruntime:
        try:
            results += run_onnxruntime(args.use_gpu, args.models, args.precision, args.batch_sizes,
                                       args.sequence_lengths, args.test_times, args.input_counts, args.optimize_onnx,
                                       args.validate_onnx, args.cache_dir, args.onnx_dir, args.verbose, args.overwrite,
                                       args.disable_ort_io_binding, args.use_raw_attention_mask, args.thread_num,
                                       model_fusion_statistics)
        except:
            logger.error(f"Exception", exc_info=True)

    time_stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    if model_fusion_statistics:
        csv_filename = args.fusion_csv or f"benchmark_fusion_{time_stamp}.csv"
        output_fusion_statistics(model_fusion_statistics, csv_filename)

    if len(results) == 0:
        if args.batch_sizes != [0]:
            logger.warning("No any result avaiable.")
        return

    csv_filename = args.detail_csv or f"benchmark_detail_{time_stamp}.csv"
    output_details(results, csv_filename)

    csv_filename = args.result_csv or f"benchmark_summary_{time_stamp}.csv"
    output_summary(results, csv_filename, args)


if __name__ == "__main__":
    main()
