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
from onnx_exporter import create_onnxruntime_input, load_pretrained_model, export_onnx_model_from_pt, export_onnx_model_from_tf

logger = logging.getLogger('')

from huggingface_models import MODELS, MODEL_CLASSES

cpu_count = psutil.cpu_count(logical=False)

# Set OMP environment variable before importing onnxruntime or torch.
if "OMP_NUM_THREADS" not in os.environ:
    os.environ["OMP_NUM_THREADS"] = str(cpu_count)

import torch
from transformers import (AutoConfig, AutoTokenizer, AutoModel, GPT2Model)

# ORT benchmark
def run_onnxruntime(use_gpu, precision, num_threads, batch_sizes, sequence_lengths,
                    repeat_times, optimize_onnx, verbose,
                    disable_ort_io_binding, model_source):
    model_name = 'Nuance_test'
    onnx_model_file = "/bert_ort/wy/Transformers/nuance_test/onnxruntime/python/tools/transformers/onnx_models/model_result.onnx"
    import onnxruntime

    results = []
    if use_gpu and ('CUDAExecutionProvider' not in onnxruntime.get_available_providers()):
        logger.error(
            "Please install onnxruntime-gpu package instead of onnxruntime, and use a machine with GPU for testing gpu performance."
        )
        return results

    if (not use_gpu) and ('CUDAExecutionProvider' in onnxruntime.get_available_providers()):
        logger.warning("Please install onnxruntime package instead of onnxruntime-gpu to get best cpu performance.")

    ort_session = create_onnxruntime_session(onnx_model_file,
                                             use_gpu,
                                             enable_all_optimization=True,
                                             num_threads=num_threads,
                                             verbose=verbose)
    device = "cuda" if use_gpu else "cpu"
    for batch_size in batch_sizes:
        if batch_size <= 0:
            continue
        for sequence_length in sequence_lengths:
            input_value_type = numpy.int64 if 'pt' in model_source else numpy.int32
            #--sequence_length 16 --input_ids "input_ids_tensor:0" --input_mask "external_feature_multihot_vectors_tensor:0"
            ort_inputs = create_onnxruntime_input(512, batch_size, sequence_length, "input_names",
                                                  input_value_type)
            result_template = {
                "engine": "onnxruntime",
                "version": onnxruntime.__version__,
                "device": device,
                "optimizer": optimize_onnx,
                "precision": precision,
                "io_binding": not disable_ort_io_binding,
                "model_name": model_name,
                "inputs": 1,
                "threads": num_threads,
                "batch_size": batch_size,
                "sequence_length": sequence_length,
                "datetime": str(datetime.now()),
            }
            logger.info("Run onnxruntime on {} with input shape {}".format(model_name,
                                                                           [batch_size, sequence_length]))
            result = inference_ort(ort_session, ort_inputs, result_template, repeat_times, batch_size)
            logger.info(result)
            results.append(result)

    return results

def main():
    results = []
    for num_threads in range(24):
        print("thread_num:" + str(num_threads + 1))
        results += run_onnxruntime(False, Precision.FLOAT32, num_threads + 1,
                                   [1], [32, 64, 128], 1000, True, False, True, 'pt')

    csv_filename = f"benchmark_detail_1.csv"
    output_details(results, csv_filename)

if __name__ == "__main__":
    main()