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
import coloredlogs
import csv
import timeit
from datetime import datetime
import numpy
import sys
import os
import psutil
import onnx
from enum import Enum
from packaging import version
from transformers.modeling_utils import Conv1D
from benchmark_helper import create_onnxruntime_session, Precision
from gpt2_helper import GPT2ModelNoPastState
from quantize_helper import QuantizeHelper

logger = logging.getLogger('')

# List of pretrained models: https://huggingface.co/transformers/pretrained_models.html
# Pretrained model name to a tuple of input names, opset_version, use_external_data_format and optimization model type
MODELS = {
    "bert-base-cased": (["input_ids", "attention_mask", "token_type_ids"], 11, False, "bert"),
    "distilbert-base-uncased": (["input_ids", "attention_mask"], 11, False, "bert"),
    "roberta-base": (["input_ids", "attention_mask"], 11, False, "bert"),

    # No past state inputs for GPT models.
    "gpt2": (["input_ids"], 11, False, "gpt2"),  # no past state inputs & outputs
    "distilgpt2": (["input_ids"], 11, False, "gpt2"),  # no past state inputs & outputs

    #"openai-gpt": (["input_ids"], 11, False, "gpt2"),  # no past state inputs

    # Models uses Einsum, which need opset version 12 and PyTorch 1.5.0 or above.
    "albert-base-v2": (["input_ids"], 12, False, "bert"),
    #"xlnet-base-cased": (["input_ids"], 12, False, "bert"),

    # Model>2GB. Need use_external_data_format=True to export it.
    #"xlm-mlm-en-2048": (["input_ids"], 11, True, "bert"),
    "gpt2-large": (["input_ids"], 11, True, "gpt2"),  # no past state inputs & outputs
}

cpu_count = psutil.cpu_count(logical=True)
# Set OMP environment variable before importing onnxruntime or torch.
if "OMP_NUM_THREADS" not in os.environ:
    os.environ["OMP_NUM_THREADS"] = str(cpu_count)

import torch
from transformers import (AutoConfig, AutoTokenizer, AutoModel, GPT2Model)


def load_pretrained_model(model_name, config, cache_dir):
    if model_name in ["gpt2", "distilgpt2", "gpt2-large"]:
        return GPT2ModelNoPastState.from_pretrained(model_name, config=config, cache_dir=cache_dir)
    return AutoModel.from_pretrained(model_name, config=config, cache_dir=cache_dir)


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


def filter_inputs(inputs, input_names):
    remaining_model_inputs = {}
    for input_name in input_names:
        remaining_model_inputs[input_name] = inputs[input_name]
    return remaining_model_inputs


def flatten(inputs):
    return [[flatten(i) for i in inputs] if isinstance(inputs, (list, tuple)) else inputs]


def update_flatten_list(inputs, res_list):
    for i in inputs:
        res_list.append(i) if not isinstance(i, (list, tuple)) else update_flatten_list(i, res_list)
    return res_list


def build_dynamic_axes(example_inputs, outputs_flatten):
    sequence_length = example_inputs["input_ids"].shape[-1]

    dynamic_axes = {key: {0: 'batch_size', 1: 'seq_len'} for key in example_inputs.keys()}

    output_names = ['output_' + str(i + 1) for i in range(len(outputs_flatten))]
    for i, output_name in enumerate(output_names):
        dynamic_axes[output_name] = {0: 'batch_size'}
        dims = outputs_flatten[i].shape
        for j, dim in enumerate(dims):
            if dim == sequence_length:
                dynamic_axes[output_name].update({j: 'seq_len'})
    return dynamic_axes, output_names


def validate_onnx_model(onnx_model_path, example_inputs, example_outputs_flatten, use_gpu, fp16):
    test_session = create_onnxruntime_session(onnx_model_path, use_gpu, enable_all_optimization=False)
    if test_session is None:
        logger.error(f"{onnx_model_path} is an invalid ONNX model")
        return False

    logger.info(f"{onnx_model_path} is a valid ONNX model")

    # Compare the inference result with PyTorch
    example_ort_inputs = {k: t.cpu().numpy() for k, t in example_inputs.items()}
    example_ort_outputs = test_session.run(None, example_ort_inputs)
    if len(example_outputs_flatten) != len(example_ort_outputs):
        logger.error(
            f"Number of output tensors expected {len(example_outputs_flatten)}, got {len(example_ort_outputs)}")
        return False

    for i in range(len(example_outputs_flatten)):
        abs_diff = numpy.amax(numpy.abs(example_ort_outputs[i] - example_outputs_flatten[i].cpu().numpy()))
        if abs_diff > 1e-4:
            logger.info(f"Max absolute diff={abs_diff} for output tensor {i}")

        rtol = 5e-02 if fp16 else 1e-4
        atol = 1e-01 if fp16 else 1e-4
        if not numpy.allclose(example_ort_outputs[i], example_outputs_flatten[i].cpu(), rtol=rtol, atol=atol):
            logger.error(f"Output tensor {i} is not close: rtol={rtol}, atol={atol}")
            return False

    logger.info(f"inference result of onnxruntime is validated on {onnx_model_path}")
    return True


model_fusion_statistics = {}


def get_onnx_file_path(onnx_dir: str, model_name: str, input_count: int, optimized_by_script: bool, use_gpu: bool,
                       precision: Precision, optimized_by_onnxruntime: bool):
    if not optimized_by_script:
        filename = f"{model_name}_{input_count}"
    else:
        device = "gpu" if use_gpu else "cpu"
        filename = f"{model_name}_{input_count}_{precision}_{device}"

    if optimized_by_onnxruntime:
        filename += f"_ort"

    use_external_data = MODELS[model_name][2]
    directory = os.path.join(onnx_dir, filename) if use_external_data else onnx_dir
    if not os.path.exists(directory):
        os.makedirs(directory)

    return os.path.join(directory, f"{filename}.onnx")


def optimize_onnx_model_by_ort(onnx_model_path, ort_model_path, use_gpu, overwrite):
    if overwrite or not os.path.exists(ort_model_path):
        from optimizer import optimize_by_onnxruntime, get_fusion_statistics
        # Use onnxruntime to optimize model, which will be saved to *_ort.onnx
        opt_model = optimize_by_onnxruntime(onnx_model_path,
                                            use_gpu=use_gpu,
                                            optimized_model_path=ort_model_path,
                                            opt_level=99)
        model_fusion_statistics[ort_model_path] = get_fusion_statistics(ort_model_path)
    else:
        logger.info(f"Skip optimization since model existed: {ort_model_path}")


def optimize_onnx_model(onnx_model_path, optimized_model_path, model_type, num_attention_heads, hidden_size, use_gpu,
                        fp16, use_raw_attention_mask, overwrite):
    if overwrite or not os.path.exists(optimized_model_path):
        from optimizer import optimize_model
        from onnx_model_bert import BertOptimizationOptions
        optimization_options = BertOptimizationOptions(model_type)
        if use_raw_attention_mask:
            optimization_options.use_raw_attention_mask()
        if fp16:
            optimization_options.enable_gelu_approximation = True

        # Use script to optimize model.
        # Use opt_level <= 1 for models to be converted to fp16, because some fused op (like FusedGemm) has only fp32 and no fp16.
        # It is better to be conservative so we use opt_level=0 here, in case MemcpyFromHost is added to the graph by OnnxRuntime.
        opt_model = optimize_model(onnx_model_path,
                                   model_type,
                                   num_heads=num_attention_heads,
                                   hidden_size=hidden_size,
                                   opt_level=0,
                                   optimization_options=optimization_options,
                                   use_gpu=use_gpu,
                                   only_onnxruntime=False)
        model_fusion_statistics[optimized_model_path] = opt_model.get_fused_operator_statistics()

        if fp16:
            opt_model.convert_model_float32_to_float16()
        opt_model.save_model_to_file(optimized_model_path)
    else:
        logger.info(f"Skip optimization since model existed: {optimized_model_path}")


def export_onnx_model(model_name, cache_dir, onnx_dir, input_names, use_gpu, precision, optimize_onnx, validate_onnx,
                      use_raw_attention_mask, overwrite):
    config = AutoConfig.from_pretrained(model_name, cache_dir=cache_dir)
    model = load_pretrained_model(model_name, config=config, cache_dir=cache_dir)
    model.cpu()

    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
    example_inputs = tokenizer.encode_plus("This is a sample input", return_tensors="pt")

    example_inputs = filter_inputs(example_inputs, input_names)

    example_outputs = model(**example_inputs)

    assert isinstance(example_outputs, (list, tuple))
    # Flatten is needed for gpt2 and distilgpt2.
    example_outputs_flatten = flatten(example_outputs)
    example_outputs_flatten = update_flatten_list(example_outputs_flatten, [])

    onnx_model_path = get_onnx_file_path(onnx_dir, model_name, len(input_names), False, use_gpu, precision, False)

    if overwrite or not os.path.exists(onnx_model_path):
        logger.info("Exporting ONNX model to {}".format(onnx_model_path))

        dynamic_axes, output_names = build_dynamic_axes(example_inputs, example_outputs_flatten)

        torch.onnx.export(model=model,
                          args=tuple(example_inputs.values()),
                          f=onnx_model_path,
                          input_names=list(example_inputs.keys()),
                          output_names=output_names,
                          example_outputs=example_outputs,
                          dynamic_axes=dynamic_axes,
                          do_constant_folding=True,
                          opset_version=MODELS[model_name][1],
                          use_external_data_format=MODELS[model_name][2])
    else:
        logger.info(f"Skip export since model existed: {onnx_model_path}")

    is_valid_onnx_model = True
    if validate_onnx:
        is_valid_onnx_model = validate_onnx_model(onnx_model_path, example_inputs, example_outputs_flatten, use_gpu,
                                                  False)

    if optimize_onnx or precision == Precision.FLOAT16 or precision == Precision.INT8:  # Use script (optimizer.py) to optimize
        model_type = MODELS[model_name][3]
        optimized_model_path = get_onnx_file_path(onnx_dir, model_name, len(input_names), True, use_gpu, precision,
                                                  False)
        optimize_onnx_model(onnx_model_path, optimized_model_path, model_type, config.num_attention_heads,
                            config.hidden_size, use_gpu, precision == Precision.FLOAT16, use_raw_attention_mask,
                            overwrite)

        onnx_model_path = optimized_model_path
        if validate_onnx:
            is_valid_onnx_model = validate_onnx_model(onnx_model_path, example_inputs, example_outputs_flatten, use_gpu,
                                                      precision == Precision.FLOAT16)

        if precision == Precision.INT8:
            logger.info(f"Quantizing model: {onnx_model_path}")
            QuantizeHelper.quantize_onnx_model(onnx_model_path, onnx_model_path)
            logger.info(f"Finished quantizing model: {onnx_model_path}")

    else:  # Use OnnxRuntime to optimize
        if is_valid_onnx_model:
            ort_model_path = get_onnx_file_path(onnx_dir, model_name, len(input_names), False, use_gpu, precision, True)
            optimize_onnx_model_by_ort(onnx_model_path, ort_model_path, use_gpu, overwrite)

    return onnx_model_path, is_valid_onnx_model, config.vocab_size, tokenizer.max_model_input_sizes[model_name]


def get_latency_result(runtimes, batch_size):
    latency_ms = sum(runtimes) / float(len(runtimes)) * 1000.0
    latency_variance = numpy.var(runtimes, dtype=numpy.float64) * 1000.0
    throughput = batch_size * (1000.0 / latency_ms)

    return {
        "test_times": len(runtimes),
        "latency_variance": "{:.2f}".format(latency_variance),
        "latency_90_percentile": "{:.2f}".format(numpy.percentile(runtimes, 90) * 1000.0),
        "latency_95_percentile": "{:.2f}".format(numpy.percentile(runtimes, 95) * 1000.0),
        "latency_99_percentile": "{:.2f}".format(numpy.percentile(runtimes, 99) * 1000.0),
        "average_latency_ms": "{:.2f}".format(latency_ms),
        "QPS": "{:.2f}".format(throughput),
    }


def inference_ort(ort_session, ort_inputs, result_template, repeat_times, batch_size):
    result = {}
    runtimes = timeit.repeat(lambda: ort_session.run(None, ort_inputs), number=1, repeat=repeat_times)
    result.update(result_template)
    result.update({"io_binding": False})
    result.update(get_latency_result(runtimes, batch_size))
    return result


def inference_ort_with_io_binding(ort_session, ort_inputs, result_template, repeat_times, ort_output_names, ort_outputs,
                                  output_buffers, max_last_state_size, max_pooler_size, batch_size, device):
    result = {}

    # Bind inputs and outputs to onnxruntime session
    io_binding = ort_session.io_binding()
    # Bind inputs to device
    for name in ort_inputs.keys():
        np_input = torch.from_numpy(ort_inputs[name]).to(device)
        io_binding.bind_input(name, np_input.device.type, 0, numpy.longlong, np_input.shape, np_input.data_ptr())
    has_pooler = True if len(ort_output_names) == 2 else False
    # Bind outputs buffers with the sizes needed if not allocated already
    if output_buffers["last_state"] is None:
        allocateOutputBuffers(output_buffers, max_last_state_size, max_pooler_size, device, has_pooler)
    last_state_buffer = output_buffers["last_state"]
    pooler_buffer = output_buffers["pooler"]
    io_binding.bind_output(ort_output_names[0], last_state_buffer.device.type, 0, numpy.float32, ort_outputs[0].shape,
                           last_state_buffer.data_ptr())
    if has_pooler:
        io_binding.bind_output(ort_output_names[1], pooler_buffer.device.type, 0, numpy.float32, ort_outputs[1].shape,
                               pooler_buffer.data_ptr())

    runtimes = timeit.repeat(lambda: ort_session.run_with_iobinding(io_binding), number=1, repeat=repeat_times)
    result.update(result_template)
    result.update({"io_binding": True})
    result.update(get_latency_result(runtimes, batch_size))
    return result


def allocateOutputBuffers(output_buffers, max_last_state_size, max_pooler_size, device, has_pooler=False):
    # Allocate output tensors with the largest test size needed. So the allocated memory can be reused
    # for each test run.
    # dummy last state
    if output_buffers["last_state"] is None:
        output_buffers["last_state"] = torch.empty(max_last_state_size, dtype=torch.float32, device=device)
    # create dummy pooler
    if output_buffers["pooler"] is None and has_pooler:
        output_buffers["pooler"] = torch.empty(max_pooler_size, dtype=torch.float32, device=device)


def run_onnxruntime(use_gpu, model_names, precision, batch_sizes, sequence_lengths, repeat_times, input_counts,
                    optimize_onnx, validate_onnx, cache_dir, onnx_dir, verbose, overwrite, disable_ort_io_binding,
                    use_raw_attention_mask, thread_num):
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
                    model_name, cache_dir, onnx_dir, input_names, use_gpu, precision, optimize_onnx, validate_onnx,
                    use_raw_attention_mask, overwrite)
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


def output_details(results, csv_filename):
    with open(csv_filename, mode="a", newline='') as csv_file:
        column_names = [
            "engine", "version", "device", "precision", "optimizer", "io_binding", "model_name", "inputs", "batch_size",
            "sequence_length", "datetime", "test_times", "QPS", "average_latency_ms", "latency_variance",
            "latency_90_percentile", "latency_95_percentile", "latency_99_percentile"
        ]

        csv_writer = csv.DictWriter(csv_file, fieldnames=column_names)
        csv_writer.writeheader()
        for result in results:
            csv_writer.writerow(result)

    logger.info(f"Detail results are saved to csv file: {csv_filename}")


def output_summary(results, csv_filename, args):
    with open(csv_filename, mode="a", newline='') as csv_file:
        header_names = ["model_name", "inputs", "engine", "version", "device", "precision", "optimizer", "io_binding"]
        data_names = []
        for batch_size in args.batch_sizes:
            for sequence_length in args.sequence_lengths:
                data_names.append(f"b{batch_size}_s{sequence_length}")

        csv_writer = csv.DictWriter(csv_file, fieldnames=header_names + data_names)
        csv_writer.writeheader()
        for model_name in args.models:
            for input_count in [1, 2, 3]:
                for engine_name in args.engines:
                    for io_binding in [True, False, ""]:
                        row = {}
                        for result in results:
                            if result["model_name"] == model_name and result["inputs"] == input_count and result[
                                    "engine"] == engine_name and result["io_binding"] == io_binding:
                                headers = {k: v for k, v in result.items() if k in header_names}
                                if not row:
                                    row.update(headers)
                                    row.update({k: "" for k in data_names})
                                else:
                                    for k in header_names:
                                        assert row[k] == headers[k]
                                b = result["batch_size"]
                                s = result["sequence_length"]
                                row[f"b{b}_s{s}"] = result["average_latency_ms"]
                        if row:
                            csv_writer.writerow(row)

    logger.info(f"Summary results are saved to csv file: {csv_filename}")


def output_fusion_statistics(model_fusion_statistics, csv_filename):
    from transformers import __version__ as transformers_version
    with open(csv_filename, mode="a", newline='') as csv_file:
        column_names = ["model_filename", "datetime", "transformers", "torch"] + list(
            next(iter(model_fusion_statistics.values())).keys())
        csv_writer = csv.DictWriter(csv_file, fieldnames=column_names)
        csv_writer.writeheader()
        for key in model_fusion_statistics.keys():
            model_fusion_statistics[key]["datetime"] = str(datetime.now())
            model_fusion_statistics[key]["transformers"] = transformers_version
            model_fusion_statistics[key]["torch"] = torch.__version__
            model_fusion_statistics[key]["model_filename"] = key
            csv_writer.writerow(model_fusion_statistics[key])
    logger.info(f"Fusion statistics is saved to csv file: {csv_filename}")


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


def setup_logger(verbose):
    if verbose:
        coloredlogs.install(level='DEBUG', fmt='[%(filename)s:%(lineno)s - %(funcName)20s()] %(message)s')
    else:
        coloredlogs.install(fmt='%(message)s')
        logging.getLogger("transformers").setLevel(logging.WARNING)


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
    print(torch.__config__.parallel_info())

    if enable_torch or enable_torchscript:
        if args.input_counts != [1]:
            logger.warning("--input_counts is not implemented for torch or torchscript engine.")

        if enable_torchscript:
            results += run_pytorch(args.use_gpu, args.models, args.precision, args.batch_sizes, args.sequence_lengths,
                                   args.test_times, True, args.cache_dir, args.verbose)

        if enable_torch:
            results += run_pytorch(args.use_gpu, args.models, args.precision, args.batch_sizes, args.sequence_lengths,
                                   args.test_times, False, args.cache_dir, args.verbose)

    if enable_onnxruntime:
        try:
            results += run_onnxruntime(args.use_gpu, args.models, args.precision, args.batch_sizes,
                                       args.sequence_lengths, args.test_times, args.input_counts, args.optimize_onnx,
                                       args.validate_onnx, args.cache_dir, args.onnx_dir, args.verbose, args.overwrite,
                                       args.disable_ort_io_binding, args.use_raw_attention_mask, args.thread_num)
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
