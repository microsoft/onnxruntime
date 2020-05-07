#-------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.
#--------------------------------------------------------------------------
""" Benchmarking the inference of BERT models from huggingface transformers
    Example commands:
        Run OnnxRuntime on GPU for all models:
            python benchmark_bert.py --use_gpu
        Run OnnxRuntime on CPU for all models:
            python benchmark_bert.py
        Run PyTorch and TorchScript on CPU for all models:
            python benchmark_bert.py --engines torch torchscript
        Run OnnxRuntime on the bert-base-cased model:
            python benchmark_bert.py --models bert-base-cased --batch_sizes 1 --sequence_lengths 128 --test_times 1000
"""

import argparse
import csv
import timeit
from datetime import datetime
import numpy
import sys
import os
import psutil
import traceback
from packaging import version

# List of pretrained models: https://huggingface.co/transformers/pretrained_models.html
# Pretrained model name to a tuple of input names, number of optional inputs
MODELS = {
    "bert-base-cased": (["input_ids", "attention_mask", "token_type_ids"], 2),
    "distilbert-base-uncased": (["input_ids", "attention_mask"], 1),
    "roberta-base": (["input_ids", "attention_mask"], 1),
}

cpu_count = psutil.cpu_count(logical=True)
# Set OMP environment variable before importing onnxruntime or torch.
if "OMP_NUM_THREADS" not in os.environ:
    os.environ["OMP_NUM_THREADS"] = str(cpu_count)

from transformers import (AutoConfig, AutoTokenizer, is_torch_available)

if is_torch_available():
    import torch
    from transformers import AutoModel


def create_onnxruntime_session(onnx_model_path, use_gpu):
    import onnxruntime
    sess_options = onnxruntime.SessionOptions()

    if (not use_gpu) and (version.parse(onnxruntime.__version__) < version.parse('1.3.0')):
        # Set intra_op_num_threads = 1 to enable OpenMP for onnxruntime 1.2.0 (cpu)
        # onnxruntime-gpu is not built with openmp so it is better to use default (0) or cpu_count instead.
        sess_options.intra_op_num_threads = 1

    execution_providers = ['CPUExecutionProvider'] if not use_gpu else ['CUDAExecutionProvider', 'CPUExecutionProvider']
    session = onnxruntime.InferenceSession(onnx_model_path, sess_options, providers=execution_providers)
    return session


def create_onnxruntime_input(vocab_size, batch_size, sequence_length, input_names):
    input_ids = numpy.random.randint(low=0, high=vocab_size - 1, size=(batch_size, sequence_length), dtype=numpy.int64)

    inputs = {'input_ids': input_ids}

    if "attention_mask" in input_names:
        attention_mask = numpy.ones([batch_size, sequence_length], dtype=numpy.int64)
        inputs['attention_mask'] = attention_mask

    if "token_type_ids" in input_names:
        segment_ids = numpy.zeros([batch_size, sequence_length], dtype=numpy.int64)
        inputs['token_type_ids'] = segment_ids

    return inputs

def get_input_names(model_name, num_inputs):
    inputs = MODELS[model_name][0]
    num_required_inputs = len(inputs) - MODELS[model_name][1]
    if num_inputs > len(inputs) or num_inputs < num_required_inputs:
        print(f"Model {model_name} allowed inputs: Min={num_required_inputs}, Max={len(inputs)}, Requested={num_inputs}.")
        return None

    return inputs[:num_inputs]

def filter_inputs(inputs, input_names):
    remaining_model_inputs = {}
    for input_name in input_names:
        remaining_model_inputs[input_name] = inputs[input_name]
    return remaining_model_inputs

def build_dynamic_axes(example_inputs, example_outputs):
    sequence_length = example_inputs["input_ids"].shape[-1]

    dynamic_axes = {key: {0: 'batch_size', 1: 'seq_len'} for key in example_inputs.keys()}

    assert isinstance(example_outputs, (list, tuple))
    outputs = example_outputs
    output_names = ['output_' + str(i + 1) for i in range(len(outputs))]
    for i, output_name in enumerate(output_names):
        dynamic_axes[output_name] = {0: 'batch_size'}
        dims = outputs[i].shape
        for j, dim in enumerate(dims):
            if dim == sequence_length:
                dynamic_axes[output_name].update({j: 'seq_len'})
    return dynamic_axes, output_names

def export_onnx_model(model_name, cache_dir, input_names):
    config = AutoConfig.from_pretrained(model_name, cache_dir=cache_dir)
    model = AutoModel.from_pretrained(model_name, config=config, cache_dir=cache_dir)
    model.cpu()

    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
    model_inputs = tokenizer.encode_plus("This is a sample input", return_tensors="pt")

    model_inputs = filter_inputs(model_inputs, input_names)

    onnx_model_filename = "{}_{}.onnx".format(model_name, str(len(input_names)))
    if not os.path.exists(onnx_model_filename):
        print("Exporting ONNX model to {}".format(onnx_model_filename))

        model_outputs = model(**model_inputs)

        dynamic_axes, output_names = build_dynamic_axes(model_inputs, model_outputs)

        torch.onnx.export(model=model,
                          args=tuple(model_inputs.values()),
                          f=onnx_model_filename,
                          input_names=list(model_inputs.keys()),
                          output_names=output_names,
                          dynamic_axes=dynamic_axes,
                          do_constant_folding=True,
                          opset_version=11)
    else:
        print(f"Skip export since model existed: {onnx_model_filename}")

    return onnx_model_filename, config.vocab_size, tokenizer.max_model_input_sizes[model_name]


def create_fp16_model(onnx_model_filename):
    optimized_model_filename = onnx_model_filename.replace(".onnx", "_fp16.onnx")
    if not os.path.exists(optimized_model_filename):
        import bert_model_optimization as bert_opt
        bert_model = bert_opt.optimize_model(onnx_model_filename, "bert", num_heads=12, hidden_size=768, opt_level=0)
        bert_model.convert_model_float32_to_float16()
        bert_model.save_model_to_file(optimized_model_filename)
    else:
        print(f"Skip fp16 optimization since model existed: {optimized_model_filename}")
    return optimized_model_filename


def run_onnxruntime(use_gpu, model_names, fp16, batch_sizes, sequence_lengths, repeat_times, num_inputs, cache_dir, verbose):
    import onnxruntime

    results = []
    if use_gpu and ('CUDAExecutionProvider' not in onnxruntime.get_available_providers()):
        print(
            "Please install onnxruntime-gpu package instead of onnxruntime, and use a machine with GPU for testing gpu performance."
        )
        return results

    if (not use_gpu) and ('CUDAExecutionProvider' in onnxruntime.get_available_providers()):
        print("Warning: Please install onnxruntime package instead of onnxruntime-gpu to get best cpu performance.")

    for model_name in model_names:
        input_names = get_input_names(model_name, num_inputs)
        if input_names is None:
            print(f"Skip model {model_name} which does not support {num_inputs} inputs")
            continue

        onnx_model_file, vocab_size, max_sequence_length = export_onnx_model(model_name, cache_dir, input_names)
        if fp16:
            onnx_model_file = create_fp16_model(onnx_model_file)

        for batch_size in batch_sizes:
            for sequence_length in sequence_lengths:
                if max_sequence_length is not None and sequence_length > max_sequence_length:
                    continue

                ort_session = create_onnxruntime_session(onnx_model_file, use_gpu)
                
                ort_input = create_onnxruntime_input(vocab_size, batch_size, sequence_length, input_names)
                ort_session.run(None, ort_input)

                if verbose:
                    print("Run onnxruntime on {} with input shape {}".format(model_name, [batch_size, sequence_length]))
                runtimes = timeit.repeat(lambda: ort_session.run(None, ort_input), number=1, repeat=repeat_times)
                latency_ms = sum(runtimes) / float(len(runtimes)) * 1000.0
                latency_variance = numpy.var(runtimes, dtype=numpy.float64) * 1000.0
                throughput = batch_size * (1000.0 / latency_ms)

                result = {
                    "runtime": "onnxruntime",
                    "version": onnxruntime.__version__,
                    "device": "cuda" if use_gpu else "cpu",
                    "fp16": fp16,
                    "model_name": model_name,
                    "inputs": num_inputs,
                    "batch_size": batch_size,
                    "sequence_length": sequence_length,
                    "average_latency_ms": "{:.2f}".format(latency_ms),
                    "latency_variance": "{:.2f}".format(latency_variance),
                    "latency_90_percentile": "{:.2f}".format(numpy.percentile(runtimes, 90) * 1000.0),
                    "latency_95_percentile": "{:.2f}".format(numpy.percentile(runtimes, 95) * 1000.0),
                    "latency_99_percentile": "{:.2f}".format(numpy.percentile(runtimes, 99) * 1000.0),
                    "QPS": "{:.2f}".format(throughput),
                }

                print(result)
                results.append(result)

    return results


def run_pytorch(use_gpu, model_names, fp16, batch_sizes, sequence_lengths, repeat_times, torchscript, cache_dir,
                verbose):
    results = []
    if use_gpu and not torch.cuda.is_available():
        print("Please install PyTorch with Cuda, and use a machine with GPU for testing gpu performance.")
        return results

    torch.set_num_threads(cpu_count)
    torch.set_grad_enabled(False)

    for model_name in model_names:
        config = AutoConfig.from_pretrained(model_name, torchscript=torchscript, cache_dir=cache_dir)
        model = AutoModel.from_pretrained(model_name, config=config, cache_dir=cache_dir)
        tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
        max_input_size = tokenizer.max_model_input_sizes[model_name]
        if verbose:
            print(f"Model {model}")
            print(f"Number of parameters {model.num_parameters()}")

        for batch_size in batch_sizes:
            if fp16:
                model.half()
            device = "cuda" if use_gpu else "cpu"
            model.to(device)

            for sequence_length in sequence_lengths:
                if max_input_size is not None and sequence_length > max_input_size:
                    continue

                if verbose:
                    print("Run PyTorch on {} with input shape {}".format(model_name, [batch_size, sequence_length]))

                input_ids = torch.randint(low=0,
                                          high=config.vocab_size - 1,
                                          size=(batch_size, sequence_length),
                                          dtype=torch.long,
                                          device=device)
                try:
                    if torchscript:
                        if verbose:
                            print("Tracing model with input shape {}".format(input_ids.shape))
                        inference = torch.jit.trace(model, input_ids)
                        inference(input_ids)
                    else:
                        inference = model
                        inference(input_ids)

                    runtimes = timeit.repeat(lambda: inference(input_ids), repeat=repeat_times, number=1)
                    latency_ms = sum(runtimes) / float(len(runtimes)) * 1000
                    latency_variance = numpy.var(runtimes, dtype=numpy.float64) * 1000.0
                    throughput = batch_size * (1000.0 / latency_ms)

                    result = {
                        "runtime": "torchscript" if torchscript else "torch",
                        "version": torch.__version__,
                        "device": device,
                        "fp16": fp16,
                        "model_name": model_name,
                        "inputs": 1,
                        "batch_size": batch_size,
                        "sequence_length": sequence_length,
                        "average_latency_ms": "{:.2f}".format(latency_ms),
                        "latency_variance": "{:.2f}".format(latency_variance),
                        "latency_90_percentile": "{:.2f}".format(numpy.percentile(runtimes, 90) * 1000),
                        "latency_95_percentile": "{:.2f}".format(numpy.percentile(runtimes, 95) * 1000),
                        "latency_99_percentile": "{:.2f}".format(numpy.percentile(runtimes, 99) * 1000),
                        "QPS": "{:.2f}".format(throughput),
                    }

                    print(result)
                    results.append(result)
                except RuntimeError as e:
                    print("Runtime error: {}".format(e))
                    torch.cuda.empty_cache()
    return results


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--models",
        required=False,
        nargs="+",
        type=str,
        default=list(MODELS.keys()),
        choices=list(MODELS.keys()),
        help="Pre-trained models in the list: " + ", ".join(MODELS.keys()))

    parser.add_argument("--engines",
                        required=False,
                        nargs="+",
                        type=str,
                        default=['onnxruntime'],
                        choices=['onnxruntime', 'torch', 'torchscript'],
                        help="Engines to benchmark")

    parser.add_argument("--cache_dir",
                        required=False,
                        type=str,
                        default="./cache_models",
                        help="Directory to cache pre-trained models")

    parser.add_argument("--use_gpu", required=False, action="store_true", help="Run on cuda device")

    parser.add_argument("--fp16", required=False, action="store_true", help="Use FP16 to accelerate inference")

    parser.add_argument("--verbose", required=False, action="store_true", help="Print more information")

    parser.add_argument("--csv_filename", required=False, default=None, help="CSV file for saving results.")

    parser.add_argument("--num_inputs", required=False,  default=1, type=int, choices=[1, 2, 3], help="Number of ONNX model inputs. Please use 1 for fair comparison with Torch or TorchScript.")

    parser.add_argument("--test_times",
                        required=False,
                        default=1000,
                        type=int,
                        help="Number of repeat times to get average inference latency.")

    parser.add_argument("--batch_sizes", nargs="+", type=int, default=[1, 2])

    parser.add_argument("--sequence_lengths", nargs="+", type=int, default=[8, 32, 128])

    args = parser.parse_args()
    return args


def main():
    args = parse_arguments()
    if args.fp16 and not args.use_gpu:
        print("--fp16 is for GPU only")
        args.fp16 = False

    print(f"Arguments: {args}")

    if not os.path.exists(args.cache_dir):
        try:
            os.mkdir(args.cache_dir)
        except OSError:
            print("Creation of the directory %s failed" % args.cache_dir)

    enable_torch = "torch" in args.engines
    enable_torchscript = "torchscript" in args.engines
    enable_onnxruntime = "onnxruntime" in args.engines

    results = []
    if enable_torch or enable_torchscript:
        if not is_torch_available():
            raise ImportError("Trying to run a PyTorch benchmark but PyTorch was not found in the environment.")
        
        if args.num_inputs != 1:
            print("warning: --num_inputs is not implemented for torch or torchscript engine: reset to 1.")

        if enable_torchscript:
            results += run_pytorch(args.use_gpu, args.models, args.fp16, args.batch_sizes, args.sequence_lengths,
                                   args.test_times, True, args.cache_dir, args.verbose)

        if enable_torch:
            results += run_pytorch(args.use_gpu, args.models, args.fp16, args.batch_sizes, args.sequence_lengths,
                                   args.test_times, False, args.cache_dir, args.verbose)

    if enable_onnxruntime:
        try:
            results += run_onnxruntime(args.use_gpu, args.models, args.fp16, args.batch_sizes, args.sequence_lengths,
                                       args.test_times, args.num_inputs, args.cache_dir, args.verbose)
        except:
            print(f"Exception:{traceback.format_exc()}")

    if len(results) == 0:
        print("No any result avaiable.")
        return

    csv_filename = args.csv_filename or "benchmark_result_{}.csv".format(datetime.now().strftime("%Y%m%d-%H%M%S"))
    with open(csv_filename, mode="a", newline='') as csv_file:
        column_names = [
            "runtime", "version", "device", "fp16", "model_name", "inputs", "batch_size", "sequence_length", "QPS", "average_latency_ms",
            "latency_variance", "latency_90_percentile", "latency_95_percentile", "latency_99_percentile"
        ]

        csv_writer = csv.DictWriter(csv_file, fieldnames=column_names)
        csv_writer.writeheader()
        for result in results:
            csv_writer.writerow(result)


if __name__ == "__main__":
    main()
