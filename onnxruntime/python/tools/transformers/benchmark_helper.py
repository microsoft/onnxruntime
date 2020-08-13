# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

import os
import sys
import csv
import numpy
import time
import timeit
from datetime import datetime
import argparse
import logging
import coloredlogs
import torch
import onnx
from enum import Enum
from packaging import version

logger = logging.getLogger(__name__)


class Precision(Enum):
    FLOAT32 = 'fp32'
    FLOAT16 = 'fp16'
    INT8 = 'int8'

    def __str__(self):
        return self.value


def create_onnxruntime_session(onnx_model_path, use_gpu, enable_all_optimization=True, num_threads=-1, verbose=False):
    session = None
    try:
        from onnxruntime import SessionOptions, InferenceSession, GraphOptimizationLevel, __version__ as onnxruntime_version
        sess_options = SessionOptions()

        if enable_all_optimization:
            sess_options.graph_optimization_level = GraphOptimizationLevel.ORT_ENABLE_ALL
        else:
            sess_options.graph_optimization_level = GraphOptimizationLevel.ORT_ENABLE_BASIC

        if num_threads > 0:
            sess_options.intra_op_num_threads = num_threads
            logger.debug(f"Session option: intra_op_num_threads={sess_options.intra_op_num_threads}")
        elif (not use_gpu) and (version.parse(onnxruntime_version) < version.parse('1.3.0')):
            # Set intra_op_num_threads = 1 to enable OpenMP for onnxruntime 1.2.0 (cpu)
            # onnxruntime-gpu is not built with openmp so it is better to use default (0) or cpu_count instead.
            sess_options.intra_op_num_threads = 1

        if verbose:
            sess_options.log_severity_level = 0

        logger.debug(f"Create session for onnx model: {onnx_model_path}")
        execution_providers = ['CPUExecutionProvider'
                               ] if not use_gpu else ['CUDAExecutionProvider', 'CPUExecutionProvider']
        session = InferenceSession(onnx_model_path, sess_options, providers=execution_providers)
    except:
        logger.error(f"Exception", exc_info=True)

    return session


def setup_logger(verbose=True):
    if verbose:
        coloredlogs.install(level='DEBUG', fmt='[%(filename)s:%(lineno)s - %(funcName)20s()] %(message)s')
    else:
        coloredlogs.install(fmt='%(message)s')
        logging.getLogger("transformers").setLevel(logging.WARNING)


def prepare_environment(cache_dir, output_dir, use_gpu):
    if cache_dir and not os.path.exists(cache_dir):
        os.makedirs(cache_dir)

    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    import onnxruntime
    if use_gpu:
        assert 'CUDAExecutionProvider' in onnxruntime.get_available_providers(
        ), "Please install onnxruntime-gpu package to test GPU inference."

    import transformers
    logger.info(f'PyTorch Version:{torch.__version__}')
    logger.info(f'Transformers Version:{transformers.__version__}')
    logger.info(f'Onnxruntime Version:{onnxruntime.__version__}')

    from packaging import version
    assert version.parse(torch.__version__) >= version.parse('1.4.0')
    assert version.parse(transformers.__version__) >= version.parse('2.11.0')
    assert version.parse(onnxruntime.__version__) >= version.parse('1.4.0')


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
