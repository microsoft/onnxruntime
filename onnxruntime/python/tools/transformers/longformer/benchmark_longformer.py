# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
#
# This script run benchmark of latency or peak memory usage of Longformer model inference.
#
# Please run convert_longformer_to_onnx.py to get onnx model before running this script.
# Tested with python 3.6, onnxruntime-gpu 1.7.0, PyTorch 1.7.1, transformers 4.3.2, CUDA 10.2.
#
# Example commands for exporting longformer base model in Linux or WSL:
#   cd ../torch_extensions
#   python setup.py install
#   cd ../longformer
#   python convert_longformer_to_onnx.py --model longformer-base-4096 --precision fp16 --optimize_onnx
#
# When there is no parameter, all avaiable tests (memory & latency) will run on the longformer-base-4096 pretrained model.
#    python benchmark_longformer.py
#
# Benchmark the latency (Exported onnx model is in the current directory):
#   python benchmark_longformer.py --model longformer-base-4096 --batch_sizes 1 --sequence_lengths 512 1024 2048 4096 --global_lengths 8 --onnx ./longformer-base-4096_fp16.onnx --validate_onnx -t 100
#
# Benchmark GPU peak memory:
#   export ORT_LONGFORMER_COMPACT_MEMORY=0
#   python benchmark_longformer.py --model longformer-base-4096 --batch_sizes 1 --sequence_lengths 4096 --global_lengths 8 --onnx_dir . --memory -t 10
#   export ORT_LONGFORMER_COMPACT_MEMORY=1
#   python benchmark_longformer.py --model longformer-base-4096 --batch_sizes 1 --sequence_lengths 4096 --global_lengths 8 --onnx_dir . --memory -t 10
# By default, compact memory kernel is not enabled. You need set an environment variable ORT_LONGFORMER_COMPACT_MEMORY=1 to enable it.

import timeit
from datetime import datetime
import csv
import argparse
import os
import sys
import torch
import onnxruntime
import numpy as np
import math

from longformer_helper import LongformerHelper, PRETRAINED_LONGFORMER_MODELS

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import benchmark_helper


def test_torch_latency(device, model, model_name, batch_sizes, sequence_lengths, global_lengths, test_times,
                       num_threads, verbose):
    if num_threads > 0:
        torch.set_num_threads(num_threads)

    results = []
    for batch_size in batch_sizes:
        for sequence_length in sequence_lengths:
            for global_length in global_lengths:
                print(f"batch_size={batch_size} sequence_length={sequence_length} global_length={global_length}...")
                inputs: LongforerInputs = LongformerHelper.get_dummy_inputs(batch_size, sequence_length, global_length,
                                                                            device)
                input_list = inputs.to_list()

                _ = model(*input_list)
                runtimes = timeit.repeat(lambda: model(*input_list), repeat=test_times, number=1)
                result = {
                    "engine": "torch",  #TODO: test torchscript
                    "version": torch.__version__,
                    "device": "cuda",
                    "optimizer": "",
                    "precision": "fp32",
                    "io_binding": "",
                    "model_name": model_name,
                    "description": model_name + "[torch]",
                    "inputs": 3,
                    "threads": num_threads,
                    "batch_size": batch_size,
                    "sequence_length": sequence_length,
                    "global_length": global_length,
                    "datetime": str(datetime.now()),
                    "memory": "?",
                }
                result.update(benchmark_helper.get_latency_result(runtimes, batch_size))

                print(result)
                results.append(result)
    return results


def test_parity(device, model, ort_session, batch_size, sequence_length, global_length, verbose=True):
    print(
        f"Comparing Torch and ORT outputs for batch_size={batch_size} sequence_length={sequence_length} global_length={global_length}..."
    )
    dummy_inputs: LongforerInputs = LongformerHelper.get_dummy_inputs(batch_size, sequence_length, global_length,
                                                                      device)
    ort_inputs = dummy_inputs.get_ort_inputs()
    ort_outputs = ort_session.run(None, ort_inputs)
    input_list = dummy_inputs.to_list()
    torch_outputs = model(*input_list)
    max_diff = np.amax(torch_outputs[0].cpu().numpy() - ort_outputs[0])
    print(f"last_state max diff = {max_diff}")
    if verbose and (math.isnan(max_diff) or max_diff > 0.001):
        print("torch last_state:", torch_outputs[0])
        print("ort last_state:", ort_outputs[0])
    return max_diff


def test_ort_latency(device,
                     model,
                     model_name,
                     description,
                     ort_session,
                     batch_sizes,
                     sequence_lengths,
                     global_lengths,
                     test_times,
                     num_threads,
                     optimizer=False,
                     precision='fp32',
                     validate_onnx=True,
                     disable_io_binding=False,
                     verbose=True):
    results = []
    for batch_size in batch_sizes:
        for sequence_length in sequence_lengths:
            for global_length in global_lengths:
                assert global_length <= model.config.attention_window[
                    0], "Limitation of current implementation: number of global token <= attention_window"
                print(
                    f"Testing batch_size={batch_size} sequence_length={sequence_length} global_length={global_length} optimizer={optimizer}, precision={precision} io_binding={not disable_io_binding}..."
                )
                dummy_inputs: LongforerInputs = LongformerHelper.get_dummy_inputs(batch_size, sequence_length,
                                                                                  global_length, device)

                # Run OnnxRuntime
                ort_inputs = dummy_inputs.get_ort_inputs()

                if verbose:
                    print(ort_inputs)

                # run one query for warm up
                ort_outputs = ort_session.run(None, ort_inputs)

                result_template = {
                    "model_name": model_name,
                    "description": description,
                    "inputs": 3,
                    "engine": "OnnxRuntime",
                    "version": onnxruntime.__version__,
                    "device": "cuda",
                    "precision": precision,
                    "optimizer": optimizer,
                    "threads": num_threads,
                    "batch_size": batch_size,
                    "sequence_length": sequence_length,
                    "global_length": global_length,
                    "test_times": test_times,
                    "datetime": str(datetime.now()),
                    "memory": "",
                }

                if not disable_io_binding:
                    max_last_state_size = max(batch_sizes) * max(sequence_lengths) * model.config.hidden_size
                    max_pooler_size = max(batch_sizes) * max(sequence_lengths)
                    result = benchmark_helper.inference_ort_with_io_binding(
                        ort_session,
                        ort_inputs,
                        result_template=result_template,
                        repeat_times=test_times,
                        ort_output_names=["last_state", "pooler"],
                        ort_outputs=ort_outputs,
                        output_buffers=[],
                        output_buffer_max_sizes=[max_last_state_size, max_pooler_size],
                        batch_size=batch_size,
                        device=device,
                        data_type=np.longlong,  #input data type
                    )
                else:
                    result = benchmark_helper.inference_ort(ort_session,
                                                            ort_inputs,
                                                            result_template=result_template,
                                                            repeat_times=test_times,
                                                            batch_size=batch_size)

                if validate_onnx:
                    max_diff = test_parity(device, model, ort_session, batch_size, sequence_length, global_length,
                                           verbose)
                    result["description"] += f"(max_diff={max_diff})"

                results.append(result)
    return results


def test_ort_memory(device, onnx_model_path, batch_size, sequence_length, global_length, test_times, num_threads):
    print(
        f"Testing memory for model={onnx_model_path}, batch_size={batch_size}, sequence_length={sequence_length}, global_length={global_length}, test_times={test_times}, num_threads={num_threads}"
    )

    def inference():
        session = benchmark_helper.create_onnxruntime_session(onnx_model_path,
                                                              use_gpu=True,
                                                              enable_all_optimization=True,
                                                              num_threads=num_threads)

        dummy_inputs: LongforerInputs = LongformerHelper.get_dummy_inputs(batch_size, sequence_length, global_length,
                                                                          device)
        ort_inputs = dummy_inputs.get_ort_inputs()
        for _ in range(test_times):
            ort_outputs = session.run(None, ort_inputs)

    memory_used = benchmark_helper.measure_memory(is_gpu=True, func=inference)

    return {
        "onnx_model": onnx_model_path,
        "batch_size": batch_size,
        "sequence_length": sequence_length,
        "global_length": global_length,
        "test_times": test_times,
        "num_threads": num_threads,
        "memory": memory_used
    }


def load_torch_model(model_name, device):
    torch_model_name_or_dir = PRETRAINED_LONGFORMER_MODELS[
        model_name] if model_name in PRETRAINED_LONGFORMER_MODELS else model_name

    from transformers import LongformerModel
    model = LongformerModel.from_pretrained(torch_model_name_or_dir)
    model.to(device)
    return model


def find_onnx_model(model_name, onnx_dir='.'):
    # Search onnx model in the following order: optimized fp16 model, optimized fp32 model, raw model
    # TODO: call convert_longformer_to_onnx to export onnx instead.
    import os.path
    onnx_model_path = os.path.join(onnx_dir, model_name + ".onnx")
    optimized_fp32_model = os.path.join(onnx_dir, model_name + "_fp32.onnx")
    optimized_fp16_model = os.path.join(onnx_dir, model_name + "_fp16.onnx")
    if os.path.isfile(optimized_fp16_model):
        onnx_model_path = optimized_fp16_model
    elif os.path.isfile(optimized_fp32_model):
        onnx_model_path = optimized_fp32_model
    return onnx_model_path


def test_memory(args, device):
    if len(args.batch_sizes) > 1:
        raise RuntimeError("For memory test, only one batch_size (-b) is allowed.")
    if len(args.sequence_lengths) > 1:
        raise RuntimeError("For memory test, only one sequence_length (-s) is allowed.")
    if len(args.global_lengths) > 1:
        raise RuntimeError("For memory test, only one global_length (-g) is allowed.")

    model_name = args.model
    onnx_model_path = find_onnx_model(model_name) if not args.onnx else args.onnx

    torch.cuda.empty_cache()
    return test_ort_memory(device, onnx_model_path, args.batch_sizes[0], args.sequence_lengths[0],
                           args.global_lengths[0], args.test_times, args.num_threads)


def test_ort(args, device):
    model_name = args.model

    onnx_model_path = find_onnx_model(model_name) if not args.onnx else args.onnx

    optimized = onnx_model_path.endswith("_fp16.onnx") or onnx_model_path.endswith("_fp32.onnx")
    precision = 'fp32' if not onnx_model_path.endswith("_fp16.onnx") else 'fp16'

    model = load_torch_model(model_name, device)

    num_threads = args.num_threads

    session = benchmark_helper.create_onnxruntime_session(onnx_model_path,
                                                          use_gpu=True,
                                                          enable_all_optimization=True,
                                                          num_threads=num_threads)
    if session is None:
        raise RuntimeError(f"Failed to create ORT sesssion from ONNX file {onnx_model_path}")

    description = onnx_model_path
    if (os.environ.get('ORT_LONGFORMER_COMPACT_MEMORY', '0') == "1"):
        description += "[compact_memory]"

    return test_ort_latency(device, model, model_name, description, session, args.batch_sizes, args.sequence_lengths,
                            args.global_lengths, args.test_times, num_threads, optimized, precision, args.validate_onnx,
                            args.disable_io_binding, args.verbose)


def test_torch(args, device):
    model = load_torch_model(args.model, device)
    return test_torch_latency(device, model, args.model, args.batch_sizes, args.sequence_lengths, args.global_lengths,
                              args.test_times, args.num_threads, args.verbose)


def test_latency(args, device):
    if "onnxruntime" == args.engine:
        return test_ort(args, device)
    elif "torch" == args.engine:
        return test_torch(args, device)

    raise RuntimeError("unknown engine " + args.engine)


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser()

    parser.add_argument("-m",
                        "--model",
                        required=False,
                        type=str,
                        default="longformer-base-4096",
                        help="Checkpoint directory or pre-trained model names in the list: " +
                        ", ".join(PRETRAINED_LONGFORMER_MODELS.keys()))

    parser.add_argument("-e",
                        "--engine",
                        required=False,
                        type=str,
                        default='onnxruntime',
                        choices=['onnxruntime', 'torch'],
                        help="Engine to benchmark.")

    parser.add_argument("-t",
                        "--test_times",
                        required=False,
                        default=1000,
                        type=int,
                        help="Number of repeat times to get average inference latency.")

    parser.add_argument("-b", "--batch_sizes", nargs="+", type=int, default=[1])

    # If --export_padding is not used in exporting onnx model, there is no padding in ONNX model so you will need padding inputs by yourself before running onnx model.
    # In that case, you can only test sequence length that is multiple of attention window size.
    parser.add_argument(
        "-s",
        "--sequence_lengths",
        nargs="+",
        type=int,
        default=[512, 1024, 2048, 4096],
        help=
        "Sequence lengths. It could have multiple values in latency test. If --export_padding is not used in exporting onnx model, sequence length shall be multiple of window size."
    )

    parser.add_argument("--onnx", required=False, type=str, default=None, help="Onnx model path")

    parser.add_argument("-g",
                        "--global_lengths",
                        nargs="+",
                        type=int,
                        default=[0],
                        help="Number of global tokens. It could have multiple values in latency test.")

    parser.add_argument("-n", "--num_threads", required=False, type=int, default=0, help="Threads to use.")

    parser.add_argument("-v",
                        "--validate_onnx",
                        required=False,
                        action="store_true",
                        help="Validate that ONNX model generates same output as PyTorch model.")

    parser.add_argument("--disable_io_binding", required=False, action="store_true", help="Do not use IO Binding.")

    parser.add_argument("--memory", required=False, action="store_true", help="Test memory usage instead of latency.")

    parser.add_argument("--verbose", required=False, action="store_true", help="Print more information.")

    args = parser.parse_args(argv)

    return args


def output_details(results, csv_filename):
    latency_results = [result for result in results if 'average_latency_ms' in result]
    if len(latency_results) == 0:
        print("No latency results for output.")
        return

    with open(csv_filename, mode="a", newline='') as csv_file:
        column_names = [
            "engine", "version", "device", "precision", "optimizer", "io_binding", "model_name", "inputs", "threads",
            "datetime", "test_times", "description", "batch_size", "sequence_length", "global_length", "memory", "QPS",
            "average_latency_ms", "latency_variance", "latency_90_percentile", "latency_95_percentile",
            "latency_99_percentile"
        ]

        csv_writer = csv.DictWriter(csv_file, fieldnames=column_names)
        csv_writer.writeheader()
        for result in latency_results:
            print(
                f"b={result['batch_size']}, s={result['sequence_length']}, g={result['global_length']}, latency={result['average_latency_ms']}ms, memory={result['memory']}MB {result['description']}"
            )
            csv_writer.writerow(result)
    print(f"Detail results are saved to csv file: {csv_filename}")


def run(args):
    if not torch.cuda.is_available():
        raise RuntimeError("Please install PyTorch with Cuda, and use a machine with GPU for testing gpu performance.")

    torch.set_grad_enabled(False)

    # set random seed manully to get deterministic results
    #benchmark_helper.set_random_seed(123)

    # Currently, the longformer attention operator could only run in GPU (no CPU implementation yet).
    device = torch.device('cuda:0')

    if args.memory:
        return test_memory(args, device)
    else:
        return test_latency(args, device)


def test_all():
    results = []
    test_times = 100
    sequence_lengths = [512, 1024, 2048, 4096]
    for model_name in ['longformer-base-4096']:
        for batch_size in [1]:
            for sequence_length in sequence_lengths:
                for global_length in [8]:
                    engine_name = 'torch'
                    args = parse_arguments(
                        f"-e {engine_name} -t {test_times} -b {batch_size} -s {sequence_length} -g {global_length} -t {test_times} -m {model_name}"
                        .split(' '))
                    results += run(args)

                    engine_name = 'onnxruntime'
                    onnx_paths = [f"{model_name}_fp32.onnx", f"{model_name}_fp16.onnx"]  # optimized models
                    for onnx_path in onnx_paths:
                        if os.path.exists(onnx_path):
                            for compact_memory in ["0", "1"]:
                                os.environ["ORT_LONGFORMER_COMPACT_MEMORY"] = compact_memory
                                print("ORT_LONGFORMER_COMPACT_MEMORY=", compact_memory)

                                args = parse_arguments(
                                    f"--disable_io_binding -e {engine_name} --onnx {onnx_path} -t {test_times} -b {batch_size} -s {sequence_length} -g {global_length} -t 10 -m {model_name} --memory"
                                    .split(' '))
                                memory_results = run(args)
                                print(memory_results)

                                args = parse_arguments(
                                    f"--disable_io_binding -e {engine_name} --onnx {onnx_path} -t {test_times} -b {batch_size} -s {sequence_length} -g {global_length} -t {test_times} -m {model_name} --validate_onnx"
                                    .split(' '))
                                latency_results = run(args)
                                if len(latency_results) == 1:
                                    latency_results[0]["memory"] = memory_results["memory"]

                                print(latency_results)

                                results += latency_results
    return results


if __name__ == "__main__":
    if len(sys.argv) > 1:
        args = parse_arguments()
        results = run(args)
    else:
        results = test_all()

    time_stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    csv_filename = f"benchmark_detail_{time_stamp}.csv"
    output_details(results, csv_filename)
