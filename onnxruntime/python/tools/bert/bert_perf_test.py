#-------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.
#--------------------------------------------------------------------------

# This tool measures the inference performance of onnxruntime or onnxruntime-gpu python package on Bert model.

# The input model shall have exactly three inputs. The model is either fully optimized (with EmbedLayerNormalization node),
# or with reasonable input names (one input name has 'mask' substring, another has 'token' or 'segment' substring).
# See get_bert_inputs function in bert_test_data.py for more information.

# Example command to run test on batch_size 1 and 2 for a model on GPU:
#   python bert_perf_test.py --model bert.onnx --batch_size 1 2 --sequence_length 128 --use_gpu --samples 1000 --test_times 1

import sys
import argparse
import os
from pathlib import Path
import timeit
import statistics
import psutil
import csv
import numpy as np
import random
from datetime import datetime
import multiprocessing
from bert_test_data import get_bert_inputs, generate_test_data


def create_session(model_path, use_gpu, intra_op_num_threads, graph_optimization_level=None):
    # Import onnxruntime shall be after OpenMP environment variable setting.
    # So we put the import in function to delay importing instead of top of this script.
    import onnxruntime

    if use_gpu and ('CUDAExecutionProvider' not in onnxruntime.get_available_providers()):
        print(
            "Warning: Please install onnxruntime-gpu package instead of onnxruntime, and use a machine with GPU for testing gpu performance."
        )
    elif (not use_gpu) and ('CUDAExecutionProvider' in onnxruntime.get_available_providers()):
        print("Warning: Please install onnxruntime package instead of onnxruntime-gpu to get best cpu performance.")

    if intra_op_num_threads is None and graph_optimization_level is None:
        session = onnxruntime.InferenceSession(model_path)
    else:
        execution_providers = ['CPUExecutionProvider'
                              ] if not use_gpu else ['CUDAExecutionProvider', 'CPUExecutionProvider']

        sess_options = onnxruntime.SessionOptions()
        sess_options.execution_mode = onnxruntime.ExecutionMode.ORT_SEQUENTIAL

        if graph_optimization_level is None:
            sess_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
        elif graph_optimization_level == 0:
            sess_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL
        elif graph_optimization_level == 1:
            sess_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_BASIC
        elif graph_optimization_level == 2:
            sess_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
        elif graph_optimization_level == 99:
            sess_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
        else:
            sess_options.graph_optimization_level = graph_optimization_level

        if intra_op_num_threads is not None:
            sess_options.intra_op_num_threads = intra_op_num_threads

        session = onnxruntime.InferenceSession(model_path, sess_options, providers=execution_providers)

    if use_gpu:
        assert 'CUDAExecutionProvider' in session.get_providers()
    return session


def onnxruntime_inference(session, all_inputs, output_names, warmup=True):
    if warmup and len(all_inputs) > 0:
        # Use a random input as warm up.
        session.run(output_names, random.choice(all_inputs))

    results = []
    latency_list = []
    for test_case_id, inputs in enumerate(all_inputs):
        start_time = timeit.default_timer()
        result = session.run(output_names, inputs)
        latency = timeit.default_timer() - start_time
        results.append(result)
        latency_list.append(latency)
    return results, latency_list


def get_contiguous_inputs(all_inputs):
    """
    Convert input to be contiguous.
    """
    contiguous_inputs = []

    start_time = timeit.default_timer()
    for test_case_id, inputs in enumerate(all_inputs):
        real_inputs = {}
        for key, value in inputs.items():
            real_inputs[key] = np.ascontiguousarray(value)
        contiguous_inputs.append(real_inputs)
    latency = timeit.default_timer() - start_time

    average_latency_ms = latency / len(contiguous_inputs) * 1000
    return contiguous_inputs, average_latency_ms


def to_string(model_path, session, test_setting):
    sess_options = session.get_session_options()
    option = "model={}".format(os.path.basename(model_path))
    option += ",graph_optimization_level={},intra_op_num_threads={}".format(sess_options.graph_optimization_level,
                                                                            sess_options.intra_op_num_threads).replace(
                                                                                'GraphOptimizationLevel.ORT_', '')
    option += ",OMP_NUM_THREADS={}".format(os.environ["OMP_NUM_THREADS"] if "OMP_NUM_THREADS" in os.environ else "")
    option += ",OMP_WAIT_POLICY={}".format(os.environ["OMP_WAIT_POLICY"] if "OMP_WAIT_POLICY" in os.environ else "")
    option += ",{}".format(test_setting)
    return option


def setup_openmp_environ(omp_num_threads, omp_wait_policy):
    if omp_num_threads is None:
        if "OMP_NUM_THREADS" in os.environ:
            del os.environ["OMP_NUM_THREADS"]
    else:
        os.environ["OMP_NUM_THREADS"] = str(omp_num_threads)

    if omp_wait_policy is None:
        if "OMP_WAIT_POLICY" in os.environ:
            del os.environ["OMP_WAIT_POLICY"]
    else:
        assert omp_wait_policy in ["ACTIVE", "PASSIVE"]
        os.environ["OMP_WAIT_POLICY"] = omp_wait_policy


def run_one_test(perf_results, model_path, all_inputs, batch_size, sequence_length, use_gpu, test_cases, test_times,
                 contiguous, intra_op_num_threads, omp_num_threads, omp_wait_policy, no_warmup, opt_level,
                 extra_latency):
    # Environment variable shall be set before import onnxruntime.
    setup_openmp_environ(omp_num_threads, omp_wait_policy)

    test_setting = "batch_size={},sequence_length={},test_cases={},test_times={},contiguous={},use_gpu={},warmup={}".format(
        batch_size, sequence_length, test_cases, test_times, contiguous, use_gpu, not no_warmup)

    session = create_session(model_path, use_gpu, intra_op_num_threads, opt_level)
    output_names = [output.name for output in session.get_outputs()]

    key = to_string(model_path, session, test_setting)
    if key in perf_results:
        print("skip duplicated test:", key)
        return

    print("Running test:", key)

    all_latency_list = []
    for i in range(test_times):
        results, latency_list = onnxruntime_inference(session, all_inputs, output_names, not no_warmup)
        all_latency_list.extend(latency_list)

    # latency in miliseconds
    latency_ms = np.array(all_latency_list) * 1000 + extra_latency

    average_latency = statistics.mean(latency_ms)
    latency_50 = np.percentile(latency_ms, 50)
    latency_75 = np.percentile(latency_ms, 75)
    latency_90 = np.percentile(latency_ms, 90)
    latency_95 = np.percentile(latency_ms, 95)
    latency_99 = np.percentile(latency_ms, 99)
    throughput = batch_size * (1000.0 / average_latency)

    perf_results[key] = (average_latency, latency_50, latency_75, latency_90, latency_95, latency_99, throughput)

    print("Average latency = {} ms, Throughput = {} QPS".format(format(average_latency, '.2f'),
                                                                format(throughput, '.2f')))


def launch_test(perf_results, model_path, all_inputs, batch_size, sequence_length, use_gpu, test_cases, test_times,
                contiguous, intra_op_num_threads, omp_num_threads, omp_wait_policy, no_warmup, opt_level,
                extra_latency):
    process = multiprocessing.Process(target=run_one_test,
                                      args=(perf_results, model_path, all_inputs, batch_size, sequence_length, use_gpu,
                                            test_cases, test_times, contiguous, intra_op_num_threads, omp_num_threads,
                                            omp_wait_policy, no_warmup, opt_level, extra_latency))
    process.start()
    process.join()


def run_perf_tests(perf_results, model_path, batch_size, sequence_length, use_gpu, test_cases, test_times, contiguous,
                   all_inputs, test_all, no_warmup, opt_level, extra_latency):
    cpu_count = psutil.cpu_count(logical=False)
    logical_cores = psutil.cpu_count(logical=True)

    # Test a setting without any setting as baseline 1.
    launch_test(perf_results, model_path, all_inputs, batch_size, sequence_length, use_gpu, test_cases, test_times,
                contiguous, None, None, None, no_warmup, opt_level, extra_latency)

    if not use_gpu:
        # For CPU: intra_op_num_threads = 1, omp_num_threads=None, omp_wait_policy=None
        # Another setting without environment variable as baseline 2.
        launch_test(perf_results, model_path, all_inputs, batch_size, sequence_length, use_gpu, test_cases, test_times,
                    contiguous, 1, None, None, no_warmup, opt_level, extra_latency)
    else:
        # For GPU, we test two more settings by default:
        # (1) intra_op_num_threads = 1, omp_num_threads=cpu_count, omp_wait_policy=PASSIVE
        # (2) intra_op_num_threads = logical_cores, omp_num_threads=1, omp_wait_policy=ACTIVE
        launch_test(perf_results, model_path, all_inputs, batch_size, sequence_length, use_gpu, test_cases, test_times,
                    contiguous, 1, cpu_count, 'PASSIVE', no_warmup, opt_level, extra_latency)

        launch_test(perf_results, model_path, all_inputs, batch_size, sequence_length, use_gpu, test_cases, test_times,
                    contiguous, logical_cores, 1, 'ACTIVE', no_warmup, opt_level, extra_latency)

    # GPU latency is not sensitive to these settings. No need to test many combinations.
    # Skip remaining settings for GPU without --all flag.
    if use_gpu and not test_all:
        return

    candidates = list(set([1, logical_cores, cpu_count]))

    for intra_op_num_threads in candidates:
        for omp_num_threads in candidates:
            # skip settings that are very slow
            if intra_op_num_threads == 1 and omp_num_threads == 1 and logical_cores != 1:
                continue

            # When logical and physical cores are not the same, there are many combinations.
            # Remove some settings are not good normally.
            if logical_cores > cpu_count:
                if omp_num_threads == logical_cores and intra_op_num_threads != 1:
                    continue
                if intra_op_num_threads == logical_cores and omp_num_threads != 1:
                    continue

            if not test_all:
                if intra_op_num_threads != 1 and omp_num_threads != 1:
                    continue

            for omp_wait_policy in ['ACTIVE', 'PASSIVE']:
                launch_test(perf_results, model_path, all_inputs, batch_size, sequence_length, use_gpu, test_cases,
                            test_times, contiguous, intra_op_num_threads, omp_num_threads, omp_wait_policy, no_warmup,
                            opt_level, extra_latency)


def run_performance(perf_results, model_path, batch_size, sequence_length, use_gpu, test_cases, test_times, seed,
                    verbose, inclusive, test_all, no_warmup, opt_level, input_ids_name, segment_ids_name,
                    input_mask_name):

    input_ids, segment_ids, input_mask = get_bert_inputs(model_path, input_ids_name, segment_ids_name, input_mask_name)

    # Do not generate random mask for performance test.
    print(f"Generating {test_cases} samples for batch_size={batch_size} sequence_length={sequence_length}")
    all_inputs = generate_test_data(batch_size,
                                    sequence_length,
                                    test_cases,
                                    seed,
                                    verbose,
                                    input_ids,
                                    segment_ids,
                                    input_mask,
                                    random_mask_length=False)

    contiguous = False
    run_perf_tests(perf_results,
                   model_path,
                   batch_size,
                   sequence_length,
                   use_gpu,
                   test_cases,
                   test_times,
                   contiguous,
                   all_inputs,
                   test_all,
                   no_warmup,
                   opt_level,
                   extra_latency=0)

    # only test contiguous array when the --all flag is set.
    if not test_all:
        return

    # Convert inputs to contiguous array, which could improve inference performance
    all_inputs, contiguous_latency = get_contiguous_inputs(all_inputs)
    print("Extra latency for converting inputs to contiguous: {} ms".format(format(contiguous_latency, '.2f')))

    contiguous = True
    run_perf_tests(perf_results,
                   model_path,
                   batch_size,
                   sequence_length,
                   use_gpu,
                   test_cases,
                   test_times,
                   contiguous,
                   all_inputs,
                   test_all,
                   no_warmup,
                   opt_level,
                   extra_latency=contiguous_latency if inclusive else 0)


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, type=str, help="bert onnx model path")

    parser.add_argument('--batch_size',
                        required=True,
                        type=int,
                        nargs="+",
                        help="batch size of input. Allow one or multiple values in the range of [1, 128].")

    parser.add_argument('--sequence_length', required=True, type=int, help="maximum sequence length of input")

    parser.add_argument('--samples', required=False, type=int, default=10, help="number of samples to be generated")

    parser.add_argument('--test_times',
                        required=False,
                        type=int,
                        default=0,
                        help="number of times to run per sample. By default, the value is 1000 / samples")

    parser.add_argument(
        '--opt_level',
        required=False,
        type=int,
        choices=[0, 1, 2, 99],
        default=99,
        help="onnxruntime optimization level: 0 - disable all, 1 - basic, 2 - extended, 99 - enable all.")

    parser.add_argument('--seed',
                        required=False,
                        type=int,
                        default=3,
                        help="random seed. Use the same seed to make sure test data is same in multiple tests.")

    parser.add_argument('--verbose', required=False, action='store_true', help="print verbose information")
    parser.set_defaults(verbose=False)

    parser.add_argument('--use_gpu', required=False, action='store_true', help="use GPU")
    parser.set_defaults(use_gpu=False)

    parser.add_argument('--inclusive',
                        required=False,
                        action='store_true',
                        help="include the latency of converting array to contiguous")
    parser.set_defaults(inclusive=False)

    parser.add_argument('--all', required=False, action='store_true', help="test all candidate settings")
    parser.set_defaults(all=False)

    parser.add_argument('--no_warmup', required=False, action='store_true', help="do not use one sample for warm-up.")
    parser.set_defaults(no_warmup=False)

    parser.add_argument('--input_ids', required=False, type=str, default=None, help="input name for input ids")
    parser.add_argument('--segment_ids', required=False, type=str, default=None, help="input name for segment ids")
    parser.add_argument('--input_mask', required=False, type=str, default=None, help="input name for attention mask")

    args = parser.parse_args()
    return args


def main():
    args = parse_arguments()

    if args.test_times == 0:
        args.test_times = max(1, int(1000 / args.samples))

    manager = multiprocessing.Manager()
    perf_results = manager.dict()

    batch_size_set = set(args.batch_size)
    if not min(batch_size_set) >= 1 and max(batch_size_set) <= 128:
        raise Exception("batch_size not in range [1, 128]")

    for batch_size in batch_size_set:
        run_performance(perf_results, args.model, batch_size, args.sequence_length, args.use_gpu, args.samples,
                        args.test_times, args.seed, args.verbose, args.inclusive, args.all, args.no_warmup,
                        args.opt_level, args.input_ids, args.segment_ids, args.input_mask)

    # Sort the results so that the first one has smallest latency.
    sorted_results = sorted(perf_results.items(), reverse=False, key=lambda x: x[1])

    summary_file = os.path.join(
        Path(args.model).parent,
        "perf_results_{}_B{}_S{}_{}.txt".format('GPU' if args.use_gpu else 'CPU',
                                                "-".join([str(x) for x in sorted(list(batch_size_set))]),
                                                args.sequence_length,
                                                datetime.now().strftime("%Y%m%d-%H%M%S")))
    with open(summary_file, 'w+', newline='') as tsv_file:
        tsv_writer = csv.writer(tsv_file, delimiter='\t', lineterminator='\n')
        headers = None
        for (key, perf_result) in sorted_results:
            params = key.split(',')
            if headers is None:
                headers = [
                    "Latency(ms)", "Latency_P50", "Latency_P75", "Latency_P90", "Latency_P95", "Latency_P99",
                    "Throughput(QPS)"
                ]
                headers.extend([x.split('=')[0] for x in params])
                tsv_writer.writerow(headers)

            values = [format(x, '.2f') for x in perf_result]
            values.extend([x.split('=')[1] for x in params])
            tsv_writer.writerow(values)

    print("Test summary is saved to", summary_file)


if __name__ == "__main__":
    # work around for AnaConda Jupyter. See https://stackoverflow.com/questions/45720153/python-multiprocessing-error-attributeerror-module-main-has-no-attribute
    __spec__ = None

    main()
