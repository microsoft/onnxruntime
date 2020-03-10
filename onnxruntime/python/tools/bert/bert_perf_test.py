#-------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.
#--------------------------------------------------------------------------

# This tool measures the inference performance of onnxruntime or onnxruntime-gpu python package on Bert model.

import sys
import argparse
import os
from pathlib import Path
import timeit
import statistics
import psutil
import csv
import numpy as np
from datetime import datetime
import multiprocessing
from bert_test_data import get_bert_inputs, generate_test_data

def create_session(model_path, use_gpu, intra_op_num_threads, graph_optimization_level=None):
    # Import onnxruntime shall be after OpenMP environment variable setting.
    # So we put the import in function to delay importing instead of top of this script.
    import onnxruntime

    if use_gpu and ('CUDAExecutionProvider' not in onnxruntime.get_available_providers()):
        print("Warning: Please install onnxruntime-gpu package instead of onnxruntime, and use a machine with GPU for testing gpu performance.")
    elif (not use_gpu) and ('CUDAExecutionProvider' in onnxruntime.get_available_providers()):
        print("Warning: Please install onnxruntime package instead of onnxruntime-gpu to get best cpu performance.")

    if intra_op_num_threads is None and graph_optimization_level is None:
        session = onnxruntime.InferenceSession(model_path)
    else:
        execution_providers = ['CPUExecutionProvider'] if not use_gpu else ['CUDAExecutionProvider', 'CPUExecutionProvider']
        sess_options = onnxruntime.SessionOptions()
        sess_options.execution_mode = onnxruntime.ExecutionMode.ORT_SEQUENTIAL
        if graph_optimization_level is None:
            sess_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
        else:
            sess_options.graph_optimization_level = graph_optimization_level
        sess_options.intra_op_num_threads = intra_op_num_threads
        session = onnxruntime.InferenceSession(model_path, sess_options, providers=execution_providers)

    if use_gpu:
        assert 'CUDAExecutionProvider' in session.get_providers()
    return session

def onnxruntime_inference(session, all_inputs, output_names):
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
    option += ",graph_optimization_level={},intra_op_num_threads={}".format(sess_options.graph_optimization_level, sess_options.intra_op_num_threads).replace('GraphOptimizationLevel.ORT_', '')
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

def run_one_test(latency_results, model_path, all_inputs, batch_size, sequence_length, use_gpu, test_cases, test_times, contiguous, intra_op_num_threads, omp_num_threads, omp_wait_policy, extra_latency):
    # Environment variable shall be set before import onnxruntime.
    setup_openmp_environ(omp_num_threads, omp_wait_policy)

    test_setting =  "batch_size={},sequence_length={},test_cases={},test_times={},contiguous={},use_gpu={}".format(batch_size,sequence_length,test_cases,test_times,contiguous,use_gpu)

    session = create_session(model_path, use_gpu, intra_op_num_threads)
    output_names = [output.name for output in session.get_outputs()]

    key = to_string(model_path, session, test_setting)
    print("Running test:", key)

    all_latency_list = []
    for i in range(test_times):
        results, latency_list = onnxruntime_inference(session, all_inputs, output_names)
        all_latency_list.extend(latency_list)

    average_latency = statistics.mean(all_latency_list) * 1000 + extra_latency
    print("Average latency is {} ms".format(format(average_latency, '.2f')))
    latency_results[key] = average_latency



def launch_test(latency_results, model_path, all_inputs, batch_size, sequence_length, use_gpu, test_cases, test_times, contiguous, intra_op_num_threads, omp_num_threads, omp_wait_policy, extra_latency):
    process = multiprocessing.Process(target=run_one_test, args=(latency_results, model_path, all_inputs, batch_size, sequence_length, use_gpu, test_cases, test_times, contiguous, intra_op_num_threads, omp_num_threads, omp_wait_policy, extra_latency))
    process.start()
    process.join()

def run_perf_tests(average_latency, model_path, batch_size, sequence_length, use_gpu, test_cases, test_times, contiguous, all_inputs, test_all, extra_latency):
    # Test a setting without any setting as baseline 1.
    launch_test(average_latency, model_path, all_inputs, batch_size, sequence_length, use_gpu, test_cases, test_times, contiguous, None, None, None, extra_latency)

    candidates = list(set([1, psutil.cpu_count(logical=True), psutil.cpu_count(logical=False)]))

    for intra_op_num_threads in candidates:
        # Test a setting without environment variable as baseline 2.
        if intra_op_num_threads == 1:
            launch_test(average_latency, model_path, all_inputs, batch_size, sequence_length, use_gpu, test_cases, test_times, contiguous, intra_op_num_threads, None, None, extra_latency)

        for omp_num_threads in candidates:
            # skip settings that are very slow
            if intra_op_num_threads == 1 and omp_num_threads == 1 and psutil.cpu_count(logical=True) != 1:
                continue

            # When logical and physical cores are not the same, there are many combinations.
            # Remove some settings are not good normally.
            if psutil.cpu_count(logical=True) > psutil.cpu_count(logical=False):
                if omp_num_threads == psutil.cpu_count(logical=True) and intra_op_num_threads != 1:
                    continue
                if intra_op_num_threads == psutil.cpu_count(logical=True) and omp_num_threads != 1:
                    continue

            if not test_all:
                if intra_op_num_threads != 1 and omp_num_threads != 1:
                    continue

            for omp_wait_policy in ['ACTIVE', 'PASSIVE']:
                launch_test(average_latency, model_path, all_inputs, batch_size, sequence_length, use_gpu, test_cases, test_times, contiguous, intra_op_num_threads, omp_num_threads, omp_wait_policy, extra_latency)

def run_performance(average_latency, model_path, batch_size, sequence_length, use_gpu, test_cases, test_times, seed, verbose, inclusive, test_all):
    # Try deduce input names from model.
    input_ids, segment_ids, input_mask = get_bert_inputs(model_path)

    # Do not generate random mask for performance test.
    print("generating test data...")
    all_inputs = generate_test_data(batch_size, sequence_length, test_cases, seed, verbose, input_ids, segment_ids, input_mask, random_mask_length=False)

    contiguous = False
    run_perf_tests(average_latency, model_path, batch_size, sequence_length, use_gpu, test_cases, test_times, contiguous, all_inputs, test_all, extra_latency=0)

    # only test contiguous array when the --all flag is set.
    if not test_all:
        return

    # Convert inputs to contiguous array, which could improve inference performance
    all_inputs, contiguous_latency = get_contiguous_inputs(all_inputs)
    print("Extra latency for converting inputs to contiguous: {} ms".format(format(contiguous_latency, '.2f')))

    contiguous = True
    run_perf_tests(average_latency, model_path, batch_size, sequence_length, use_gpu, test_cases, test_times, contiguous, all_inputs, test_all, extra_latency=contiguous_latency if inclusive else 0)

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, type=str,
                        help="bert onnx model path")

    parser.add_argument('--batch_size', required=True, type=int,
                        help="batch size of input")

    parser.add_argument('--sequence_length',  required=True, type=int,
                        help="maximum sequence length of input")

    parser.add_argument('--samples',  required=False, type=int, default=10,
                        help="number of samples to be generated")

    parser.add_argument('--test_times',  required=False, type=int, default=0,
                        help="number of times to run per sample. By default, the value is 1000 / samples")

    parser.add_argument('--seed',  required=False, type=int, default=3,
                        help="random seed. Use the same seed to make sure test data is same in multiple tests.")

    parser.add_argument('--verbose', required=False, action='store_true', help="print verbose information")
    parser.set_defaults(verbose=False)

    parser.add_argument('--use_gpu', required=False, action='store_true', help="use GPU")
    parser.set_defaults(use_gpu=False)

    parser.add_argument('--inclusive', required=False, action='store_true', help="include the latency of converting array to contiguous")
    parser.set_defaults(inclusive=False)

    parser.add_argument('--all', required=False, action='store_true', help="test all candidate settings")
    parser.set_defaults(all=False)

    args = parser.parse_args()
    return args

def main():
    args = parse_arguments()

    if args.test_times == 0:
        args.test_times = max(1, int(1000 / args.samples))

    manager = multiprocessing.Manager()
    return_dict = manager.dict()
    run_performance(return_dict, args.model, args.batch_size, args.sequence_length, args.use_gpu, args.samples, args.test_times, args.seed, args.verbose, args.inclusive, args.all)

    average_latency = return_dict
    if average_latency is None:
        return

    # Sort the results so that the first one has smallest latency.
    sorted_results = sorted(return_dict.items(), reverse=False, key=lambda x: x[1])
 
    summary_file = os.path.join(Path(args.model).parent, "perf_results_{}_B{}_S{}_{}.txt".format('GPU' if args.use_gpu else 'CPU', args.batch_size, args.sequence_length, datetime.now().strftime("%Y%m%d-%H%M%S")))
    with open(summary_file, 'w+', newline='') as tsv_file:
        tsv_writer = csv.writer(tsv_file, delimiter='\t', lineterminator='\n')
        headers = None
        for (key, latency) in sorted_results:
            params = key.split(',')
            if headers is None:
                headers = ["Latency(ms)", "Throughput(QPS)"]
                headers.extend([x.split('=')[0] for x in params])
                tsv_writer.writerow(headers)

            throughput = args.batch_size * (1000 / latency)
            values = [format(latency, '.2f'), format(throughput, '.2f')]

            values.extend([x.split('=')[1] for x in params])
            tsv_writer.writerow(values)

    print("Test summary is saved to", summary_file)

if __name__ == "__main__":
    main()
