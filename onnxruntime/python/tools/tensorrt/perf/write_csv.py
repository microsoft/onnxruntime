# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""
Functions for creating CSV files for benchmarking or validation runs.
"""

import csv
import json
import os

import pandas as pd
from perf_utils import (
    avg_ending,
    cpu,
    cuda,
    cuda_fp16,
    memory_ending,
    model_title,
    op_metrics_columns,
    ort_provider_list,
    percentile_ending,
    provider_list,
    second,
    second_session_ending,
    session_ending,
    standalone_trt,
    standalone_trt_fp16,
    table_headers,
    trt,
    trt_fp16,
)


def output_details(results, csv_filename):
    need_write_header = True
    if os.path.exists(csv_filename):
        need_write_header = False

    with open(csv_filename, mode="a", newline="", encoding="utf-8") as csv_file:
        column_names = [
            "engine",
            "version",
            "device",
            "fp16",
            "io_binding",
            "graph_optimizations",
            "enable_cache",
            "model_name",
            "inputs",
            "batch_size",
            "sequence_length",
            "datetime",
            "test_times",
            "QPS",
            "average_latency_ms",
            "latency_variance",
            "latency_90_percentile",
            "latency_95_percentile",
            "latency_99_percentile",
        ]

        csv_writer = csv.DictWriter(csv_file, fieldnames=column_names)
        if need_write_header:
            csv_writer.writeheader()
        for result in results:
            csv_writer.writerow(result)


def output_fail(model_to_fail_ep, csv_filename):

    with open(csv_filename, mode="w", newline="", encoding="utf-8") as csv_file:
        column_names = ["model", "ep", "error type", "error message"]

        csv_writer = csv.DictWriter(csv_file, fieldnames=column_names)
        csv_writer.writeheader()

        for model, model_info in model_to_fail_ep.items():
            for exec_provider, ep_info in model_info.items():
                result = {}
                result["model"] = model
                result["ep"] = exec_provider
                result["error type"] = ep_info["error_type"]
                result["error message"] = ep_info["error_message"]
                csv_writer.writerow(result)


def add_status_dict(status_dict, model_name, exec_provider, status):
    if model_name not in status_dict:
        status_dict[model_name] = {}
    status_dict[model_name][exec_provider] = status


def build_status(status_dict, results, is_fail):

    if is_fail:
        for model, model_info in results.items():
            for exec_provider, _ in model_info.items():
                model_name = model
                status = "Fail"
                add_status_dict(status_dict, model_name, exec_provider, status)
    else:
        for model, value in results.items():
            for exec_provider, _ in value.items():
                model_name = model
                status = "Pass"
                add_status_dict(status_dict, model_name, exec_provider, status)

    return status_dict


def output_status(results, csv_filename):

    need_write_header = True
    if os.path.exists(csv_filename):
        need_write_header = False

    with open(csv_filename, mode="a", newline="", encoding="utf-8") as csv_file:
        column_names = table_headers

        csv_writer = csv.writer(csv_file)

        if need_write_header:
            csv_writer.writerow(column_names)

        cpu_status = ""
        cuda_fp32_status = ""
        trt_fp32_status = ""
        standalone_fp32_status = ""
        cuda_fp16_status = ""
        trt_fp16_status = ""
        standalone_fp16_status = ""

        for model_name, ep_dict in results.items():
            for exec_provider, status in ep_dict.items():
                if exec_provider == cpu:
                    cpu_status = status
                elif exec_provider == cuda:
                    cuda_fp32_status = status
                elif exec_provider == trt:
                    trt_fp32_status = status
                elif exec_provider == standalone_trt:
                    standalone_fp32_status = status
                elif exec_provider == cuda_fp16:
                    cuda_fp16_status = status
                elif exec_provider == trt_fp16:
                    trt_fp16_status = status
                elif exec_provider == standalone_trt_fp16:
                    standalone_fp16_status = status
                else:
                    continue

            row = [
                model_name,
                cpu_status,
                cuda_fp32_status,
                trt_fp32_status,
                standalone_fp32_status,
                cuda_fp16_status,
                trt_fp16_status,
                standalone_fp16_status,
            ]
            csv_writer.writerow(row)


def output_specs(info, ep_option_overrides, csv_filename):
    cpu_version = info["cpu_info"][2]
    gpu_version = info["gpu_info"][0]
    tensorrt_version = info["trt"] + " , *All ORT-TRT and TRT are run in Mixed Precision mode (Fp16 and Fp32)."
    cuda_version = info["cuda"]
    cudnn_version = info["cudnn"]
    ep_options = json.dumps(ep_option_overrides)

    table = pd.DataFrame(
        {
            ".": [1, 2, 3, 4, 5, 6],
            "Spec": ["CPU", "GPU", "TensorRT", "CUDA", "CuDNN", "EPOptionOverrides"],
            "Version": [
                cpu_version,
                gpu_version,
                tensorrt_version,
                cuda_version,
                cudnn_version,
                ep_options,
            ],
        }
    )
    table.to_csv(csv_filename, index=False)


def output_session_creation(results, csv_filename):
    need_write_header = True
    if os.path.exists(csv_filename):
        need_write_header = False

    with open(csv_filename, mode="a", newline="", encoding="utf-8") as csv_file:
        session_1 = [p + session_ending for p in ort_provider_list]
        session_2 = [p + second_session_ending for p in ort_provider_list]
        column_names = [model_title] + session_1 + session_2
        csv_writer = csv.writer(csv_file)

        csv_writer = csv.writer(csv_file)

        if need_write_header:
            csv_writer.writerow(column_names)

        cpu_time = ""
        cuda_fp32_time = ""
        trt_fp32_time = ""
        cuda_fp16_time = ""
        trt_fp16_time = ""
        cpu_time_2 = ""
        cuda_fp32_time_2 = ""
        trt_fp32_time_2 = ""
        cuda_fp16_time_2 = ""
        trt_fp16_time_2 = ""

        for model_name, ep_dict in results.items():
            for ep_key, sess_time in ep_dict.items():
                if ep_key == cpu:
                    cpu_time = sess_time
                elif ep_key == cuda:
                    cuda_fp32_time = sess_time
                elif ep_key == trt:
                    trt_fp32_time = sess_time
                elif ep_key == cuda_fp16:
                    cuda_fp16_time = sess_time
                elif ep_key == trt_fp16:
                    trt_fp16_time = sess_time
                if ep_key == cpu + second:
                    cpu_time_2 = sess_time
                elif ep_key == cuda + second:
                    cuda_fp32_time_2 = sess_time
                elif ep_key == trt + second:
                    trt_fp32_time_2 = sess_time
                elif ep_key == cuda_fp16 + second:
                    cuda_fp16_time_2 = sess_time
                elif ep_key == trt_fp16 + second:
                    trt_fp16_time_2 = sess_time
                else:
                    continue

            row = [
                model_name,
                cpu_time,
                cuda_fp32_time,
                trt_fp32_time,
                cuda_fp16_time,
                trt_fp16_time,
                cpu_time_2,
                cuda_fp32_time_2,
                trt_fp32_time_2,
                cuda_fp16_time_2,
                trt_fp16_time_2,
            ]
            csv_writer.writerow(row)


def output_latency(results, csv_filename):
    need_write_header = True
    if os.path.exists(csv_filename):
        need_write_header = False

    with open(csv_filename, mode="a", newline="", encoding="utf-8") as csv_file:
        column_names = [model_title]
        for provider in provider_list:
            column_names.append(provider + avg_ending)
            column_names.append(provider + percentile_ending)
            if cpu not in provider:
                column_names.append(provider + memory_ending)

        csv_writer = csv.writer(csv_file)

        if need_write_header:
            csv_writer.writerow(column_names)

        for key, value in results.items():
            cpu_average = ""
            if cpu in value and "average_latency_ms" in value[cpu]:
                cpu_average = value[cpu]["average_latency_ms"]

            cpu_90_percentile = ""
            if cpu in value and "latency_90_percentile" in value[cpu]:
                cpu_90_percentile = value[cpu]["latency_90_percentile"]

            cuda_average = ""
            if cuda in value and "average_latency_ms" in value[cuda]:
                cuda_average = value[cuda]["average_latency_ms"]

            cuda_90_percentile = ""
            if cuda in value and "latency_90_percentile" in value[cuda]:
                cuda_90_percentile = value[cuda]["latency_90_percentile"]

            cuda_memory = ""
            if cuda in value and "memory" in value[cuda]:
                cuda_memory = value[cuda]["memory"]

            trt_average = ""
            if trt in value and "average_latency_ms" in value[trt]:
                trt_average = value[trt]["average_latency_ms"]

            trt_90_percentile = ""
            if trt in value and "latency_90_percentile" in value[trt]:
                trt_90_percentile = value[trt]["latency_90_percentile"]

            trt_memory = ""
            if trt in value and "memory" in value[trt]:
                trt_memory = value[trt]["memory"]

            standalone_trt_average = ""
            if standalone_trt in value and "average_latency_ms" in value[standalone_trt]:
                standalone_trt_average = value[standalone_trt]["average_latency_ms"]

            standalone_trt_90_percentile = ""
            if standalone_trt in value and "latency_90_percentile" in value[standalone_trt]:
                standalone_trt_90_percentile = value[standalone_trt]["latency_90_percentile"]

            standalone_trt_memory = ""
            if standalone_trt in value and "memory" in value[standalone_trt]:
                standalone_trt_memory = value[standalone_trt]["memory"]

            cuda_fp16_average = ""
            if cuda_fp16 in value and "average_latency_ms" in value[cuda_fp16]:
                cuda_fp16_average = value[cuda_fp16]["average_latency_ms"]

            cuda_fp16_memory = ""
            if cuda_fp16 in value and "memory" in value[cuda_fp16]:
                cuda_fp16_memory = value[cuda_fp16]["memory"]

            cuda_fp16_90_percentile = ""
            if cuda_fp16 in value and "latency_90_percentile" in value[cuda_fp16]:
                cuda_fp16_90_percentile = value[cuda_fp16]["latency_90_percentile"]

            trt_fp16_average = ""
            if trt_fp16 in value and "average_latency_ms" in value[trt_fp16]:
                trt_fp16_average = value[trt_fp16]["average_latency_ms"]

            trt_fp16_90_percentile = ""
            if trt_fp16 in value and "latency_90_percentile" in value[trt_fp16]:
                trt_fp16_90_percentile = value[trt_fp16]["latency_90_percentile"]

            trt_fp16_memory = ""
            if trt_fp16 in value and "memory" in value[trt_fp16]:
                trt_fp16_memory = value[trt_fp16]["memory"]

            standalone_trt_fp16_average = ""
            if standalone_trt_fp16 in value and "average_latency_ms" in value[standalone_trt_fp16]:
                standalone_trt_fp16_average = value[standalone_trt_fp16]["average_latency_ms"]

            standalone_trt_fp16_90_percentile = ""
            if standalone_trt_fp16 in value and "latency_90_percentile" in value[standalone_trt_fp16]:
                standalone_trt_fp16_90_percentile = value[standalone_trt_fp16]["latency_90_percentile"]

            standalone_trt_fp16_memory = ""
            if standalone_trt_fp16 in value and "memory" in value[standalone_trt_fp16]:
                standalone_trt_fp16_memory = value[standalone_trt_fp16]["memory"]

            row = [
                key,
                cpu_average,
                cpu_90_percentile,
                cuda_average,
                cuda_90_percentile,
                cuda_memory,
                trt_average,
                trt_90_percentile,
                trt_memory,
                standalone_trt_average,
                standalone_trt_90_percentile,
                standalone_trt_memory,
                cuda_fp16_average,
                cuda_fp16_90_percentile,
                cuda_fp16_memory,
                trt_fp16_average,
                trt_fp16_90_percentile,
                trt_fp16_memory,
                standalone_trt_fp16_average,
                standalone_trt_fp16_90_percentile,
                standalone_trt_fp16_memory,
            ]
            csv_writer.writerow(row)


def get_operator_metrics_rows(model, input_ep, event_category, operator_metrics):
    """
    Returns a list of rows to append to the 'operator metrics' CSV file.
    Each row contains metrics (e.g., num_instances, duration, etc.) for a particular operator (e.g, Conv, Add, etc.) used
    in the specified model/input_ep.

    :param model: The name of the model (e.g., zfnet512-9).
    :param input_ep: The name of the input EP (e.g., ORT-CUDAFp32, ORT-TRTFp32, etc.).
    :param event_category: The event category (i.e., "Node" or "Kernel").
    :param operator_metrics: A dictionary that maps on ORT execution provider to a dictionary of operator metrics.

        Ex: {
                "CPUExecutionProvider" : {
                    "Conv": {"num_instances": 20, "total_dur": 200, "min_dur": 10, "max_dur": 20},
                    "Add": {"num_instances": 22, "total_dur": ... }
                },
                "CUDAExecutionProvider": { ... },
                "TensorrtExecutionProvider: { ... }
            }

    :return: A list of rows containing operator metrics for the specified model and input_ep.
        Ex: [
            {
                "model_name": "zfnet512-9",
                "input_ep": "ORT-CUDAFp32",
                "operator": "Conv",
                "assigned_ep": "CUDAExecutionProvider",
                "event_category": "Node",
                "num_instances": 333,
                "total_dur": 12345,
                "min_dur": 31,
                "max_dur": 6003
            },
            { ... },
            ...
        ]
    """

    rows = []

    for assigned_ep, operators in operator_metrics.items():
        for op_name, metrics in operators.items():
            row = {
                "model_name": model,
                "input_ep": input_ep,
                "operator": op_name,
                "assigned_ep": assigned_ep,
                "event_category": event_category,
                "num_instances": metrics["num_instances"],
                "total_dur": metrics["total_dur"],
                "min_dur": metrics["min_dur"],
                "max_dur": metrics["max_dur"],
            }

            rows.append(row)

    return rows


def output_metrics(model_to_op_metrics, csv_filename):
    """
    Writes every model's operator metrics to a CSV file.

    :param model_to_op_metrics: A dictionary that maps a model to operator metrics for each input EP.

        Ex: {
                "zfnet512-9": {
                    "ORT-CPUFp32": { ... },
                    "ORT-CUDAFp32": { ... },
                    "ORT-TRTFp32": { ... }
                },
                "resnet101-v1-7": {
                    "ORT-CPUFp32": { ... },
                    "ORT-CUDAFp32": { ... },
                    "ORT-TRTFp32": { ... }
                },
                ...
            }

    :param csv_filename: The name of the CSV file to write.
    """

    with open(csv_filename, mode="w", newline="", encoding="utf-8") as csv_file:
        column_names = [c[1] for c in op_metrics_columns]
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(column_names)

        results = []
        for model, ep_info in model_to_op_metrics.items():
            for input_ep in [cuda, trt, cuda_fp16, trt_fp16]:
                if input_ep in ep_info:
                    rows = get_operator_metrics_rows(model, input_ep, "Node", ep_info[input_ep]["nodes"])
                    results.extend(rows)

                    rows = get_operator_metrics_rows(model, input_ep, "Kernel", ep_info[input_ep]["kernels"])
                    results.extend(rows)

        for value in results:
            row = [value[c[0]] for c in op_metrics_columns]
            csv_writer.writerow(row)
