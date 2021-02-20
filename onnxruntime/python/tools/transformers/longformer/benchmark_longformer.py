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
#   python convert_longformer_to_onnx.py --model longformer-base-4096 --precision fp32 --optimize_onnx
#
# Benchmark the latency (Exported onnx model is in the current directory):
#   python benchmark_longformer.py --models longformer-base-4096 --batch_sizes 1 --sequence_lengths 512 1024 2048 4096 --global_lengths 8 --onnx_dir . --validate_onnx -t 100
#
# Benchmark GPU peak memory:
#   export ORT_LONGFORMER_COMPACT_MEMORY=0
#   python benchmark_longformer.py --models longformer-base-4096 --batch_sizes 1 --sequence_lengths 4096 --global_lengths 8 --onnx_dir . --memory -t 10
#   export ORT_LONGFORMER_COMPACT_MEMORY=1
#   python benchmark_longformer.py --models longformer-base-4096 --batch_sizes 1 --sequence_lengths 4096 --global_lengths 8 --onnx_dir . --memory -t 10
# By default, compact memory kernel is not enabled since it is slower. You need set an environment variable ORT_LONGFORMER_COMPACT_MEMORY=1 to enable it, which uses less memory in this test.

import timeit
from datetime import datetime
import csv
import argparse
import os
import sys
import torch
import onnxruntime
import numpy as np
import pprint
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
                    "inputs": 3,
                    "threads": num_threads,
                    "batch_size": batch_size,
                    "sequence_length": sequence_length,
                    "global_length": global_length,
                    "datetime": str(datetime.now()),
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


def test_ort_latency(device,
                     model,
                     model_name,
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

                pprint.pprint(result)
                results.append(result)

                if validate_onnx:
                    test_parity(device, model, ort_session, batch_size, sequence_length, global_length, verbose)

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

    benchmark_helper.measure_memory(is_gpu=True, func=inference)
    print("Memory test is done")


def test_all(args):
    # Currently, the longformer attention operator could only run in GPU (no CPU implementation yet).
    device = torch.device('cuda:0')

    results = []
    for model_name in args.models:
        # Here we run an example input
        from transformers import LongformerModel
        torch_model_name_or_dir = PRETRAINED_LONGFORMER_MODELS[model_name]
        model = LongformerModel.from_pretrained(torch_model_name_or_dir)  # pretrained model name or directory
        model.to(device)

        # Search onnx model in the following order: optimized fp16 model, optimized fp32 model, raw model
        # TODO: call convert_longformer_to_onnx to export onnx instead.
        import os.path
        optimized = False
        precision = 'fp32'
        onnx_model_path = os.path.join(args.onnx_dir, model_name + ".onnx")
        optimized_fp32_model = os.path.join(args.onnx_dir, model_name + "_fp32.onnx")
        optimized_fp16_model = os.path.join(args.onnx_dir, model_name + "_fp16.onnx")
        if os.path.isfile(optimized_fp16_model):
            onnx_model_path = optimized_fp16_model
            optimized = True
            precision = 'fp16'
        elif os.path.isfile(optimized_fp32_model):
            onnx_model_path = optimized_fp32_model
            optimized = True
        print("ONNX model path:", onnx_model_path)

        for num_threads in args.num_threads:
            if "torch" in args.engines:
                results += test_torch_latency(device, model, model_name, args.batch_sizes, args.sequence_lengths,
                                              args.global_lengths, args.test_times, num_threads, args.verbose)

            if "onnxruntime" in args.engines:
                if args.memory:
                    test_ort_memory(device, onnx_model_path, args.batch_sizes[0], args.sequence_lengths[0],
                                    args.global_lengths[0], args.test_times, num_threads)
                else:  # test latency
                    session = benchmark_helper.create_onnxruntime_session(onnx_model_path,
                                                                          use_gpu=True,
                                                                          enable_all_optimization=True,
                                                                          num_threads=num_threads)
                    if session is None:
                        raise RuntimeError(f"Failed to create ORT sesssion from ONNX file {onnx_model_path}")

                    results += test_ort_latency(device, model, model_name, session, args.batch_sizes,
                                                args.sequence_lengths, args.global_lengths, args.test_times,
                                                num_threads, optimized, precision, args.validate_onnx,
                                                args.disable_io_binding, args.verbose)
    return results


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser()

    parser.add_argument("-m",
                        "--models",
                        required=False,
                        nargs="+",
                        type=str,
                        default=["longformer-base-4096"],
                        help="Checkpoint directory or pre-trained model names in the list: " +
                        ", ".join(PRETRAINED_LONGFORMER_MODELS.keys()))

    parser.add_argument("-e",
                        "--engines",
                        required=False,
                        nargs="+",
                        type=str,
                        default=['onnxruntime'],
                        choices=['onnxruntime', 'torch'],
                        help="Engines to benchmark. For large model, recommend to test only one engine at a time.")

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

    parser.add_argument("--onnx_dir",
                        required=False,
                        type=str,
                        default=os.path.join('.', 'onnx_models'),
                        help="Directory to search onnx models.")

    parser.add_argument("-g",
                        "--global_lengths",
                        nargs="+",
                        type=int,
                        default=[0],
                        help="Number of global tokens. It could have multiple values in latency test.")

    parser.add_argument("-n",
                        "--num_threads",
                        required=False,
                        nargs="+",
                        type=int,
                        default=[0],
                        help="Threads to use. It could have multiple values in latency test.")

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


def output_summary(results, csv_filename, args):
    with open(csv_filename, mode="a", newline='') as csv_file:
        header_names = [
            "model_name", "inputs", "engine", "version", "device", "precision", "optimizer", "io_binding", "threads"
        ]
        data_names = []
        for batch_size in args.batch_sizes:
            for sequence_length in args.sequence_lengths:
                for global_length in args.global_lengths:
                    data_names.append(f"b{batch_size}_s{sequence_length}_g{global_length}")

        csv_writer = csv.DictWriter(csv_file, fieldnames=header_names + data_names)
        csv_writer.writeheader()
        for model in args.models:
            for input_count in [1, 2, 3]:
                for engine_name in args.engines:
                    for io_binding in [True, False, ""]:
                        for threads in args.num_threads:
                            row = {}
                            for result in results:
                                if result["model_name"] == model and result["inputs"] == input_count and \
                                   result["engine"] == engine_name and result["io_binding"] == io_binding and \
                                   result["threads"] == threads:
                                    headers = {k: v for k, v in result.items() if k in header_names}
                                    if not row:
                                        row.update(headers)
                                        row.update({k: "" for k in data_names})
                                    else:
                                        for k in header_names:
                                            assert row[k] == headers[k]
                                    b = result["batch_size"]
                                    s = result["sequence_length"]
                                    g = result["global_length"]
                                    row[f"b{b}_s{s}_g{g}"] = result["average_latency_ms"]
                            if row:
                                csv_writer.writerow(row)

    print(f"Summary results are saved to csv file: {csv_filename}")


def output_details(results, csv_filename):
    with open(csv_filename, mode="a", newline='') as csv_file:
        column_names = [
            "engine", "version", "device", "precision", "optimizer", "io_binding", "model_name", "inputs", "threads",
            "batch_size", "sequence_length", "global_length", "datetime", "test_times", "QPS", "average_latency_ms",
            "latency_variance", "latency_90_percentile", "latency_95_percentile", "latency_99_percentile"
        ]

        csv_writer = csv.DictWriter(csv_file, fieldnames=column_names)
        csv_writer.writeheader()
        for result in results:
            csv_writer.writerow(result)

    print(f"Detail results are saved to csv file: {csv_filename}")


def main(args):
    assert len(args.models) == 1, "run only one model at a time"

    if args.memory:
        if len(args.batch_sizes) > 1:
            raise RuntimeError("For memory test, only one batch_size (-b) is allowed.")
        if len(args.sequence_lengths) > 1:
            raise RuntimeError("For memory test, only one sequence_length (-s) is allowed.")
        if len(args.global_lengths) > 1:
            raise RuntimeError("For memory test, only one global_length (-g) is allowed.")
        if len(args.num_threads) > 1:
            raise RuntimeError("For memory test, only one value of --num_threads is allowed.")

    if not torch.cuda.is_available():
        raise RuntimeError("Please install PyTorch with Cuda, and use a machine with GPU for testing gpu performance.")

    torch.set_grad_enabled(False)

    # set random seed manully to get deterministic results
    #benchmark_helper.set_random_seed(123)

    all_results = test_all(args)

    time_stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    csv_filename = f"benchmark_detail_{time_stamp}.csv"
    output_details(all_results, csv_filename)

    csv_filename = f"benchmark_summary_{time_stamp}.csv"
    output_summary(all_results, csv_filename, args)


if __name__ == "__main__":
    args = parse_arguments()
    #args = parse_arguments("-e onnxruntime -t 1 -b 1 -s 4 -g 2 --onnx_dir . -t 1 -m longformer-random-tiny".split(' '))

    main(args)
