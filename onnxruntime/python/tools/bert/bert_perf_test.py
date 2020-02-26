#-------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.
#--------------------------------------------------------------------------

# This tool measures the inference performance of onnxruntime or onnxruntime-gpu python package on Bert model.

import sys
import argparse
import os
import onnxruntime
from pathlib import Path
import timeit
import statistics
import psutil
import csv
import numpy as np
from datetime import datetime
from compare_bert_results import get_bert_inputs, create_session, generate_test_data, onnxruntime_inference

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

def run_one_test(latency_results, model_path, all_inputs, batch_size, sequence_length, use_gpu, test_cases, test_times, use_openmp, contiguous, num_threads, wait_policy):
    test_setting =  "batch_size={},sequence_length={},test_cases={},test_times={},contiguous={},use_gpu={}".format(batch_size,sequence_length,test_cases,test_times,contiguous,use_gpu)

    graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = create_session(model_path, use_gpu, use_openmp, graph_optimization_level, num_threads, wait_policy)
    output_names = [output.name for output in session.get_outputs()]

    key = to_string(model_path, session, test_setting)
    print("Running test:", key)

    all_latency_list = []
    for i in range(test_times):
        results, latency_list = onnxruntime_inference(session, all_inputs, output_names)
        all_latency_list.extend(latency_list)

    average_latency = statistics.mean(all_latency_list) * 1000
    print("Average latency is {} ms".format(format(average_latency, '.2f')))
    latency_results[key] = average_latency

def run_perf_tests(average_latency, model_path, batch_size, sequence_length, use_gpu, test_cases, test_times, seed, verbose, contiguous, input_ids, segment_ids, input_mask, all_inputs):

    run_one_test(average_latency, model_path, all_inputs, batch_size, sequence_length, use_gpu, test_cases, test_times, use_openmp=False, contiguous=contiguous, num_threads=psutil.cpu_count(logical=True), wait_policy='ACTIVE')

    # onnxruntime-gpu package is not built with OpenMP, so skip openmp test for gpu.
    if not use_gpu:
        run_one_test(average_latency, model_path, all_inputs, batch_size, sequence_length, use_gpu, test_cases, test_times, use_openmp=True, contiguous=contiguous, num_threads=psutil.cpu_count(logical=True), wait_policy='PASSIVE')
        run_one_test(average_latency, model_path, all_inputs, batch_size, sequence_length, use_gpu, test_cases, test_times, use_openmp=True, contiguous=contiguous, num_threads=psutil.cpu_count(logical=True), wait_policy='ACTIVE')

        if psutil.cpu_count(logical=True) != psutil.cpu_count(logical=False):
            run_one_test(average_latency, model_path, all_inputs, batch_size, sequence_length, use_gpu, test_cases, test_times, use_openmp=True, contiguous=contiguous, num_threads=psutil.cpu_count(logical=False), wait_policy='ACTIVE')
            run_one_test(average_latency, model_path, all_inputs, batch_size, sequence_length, use_gpu, test_cases, test_times, use_openmp=True, contiguous=contiguous, num_threads=psutil.cpu_count(logical=False), wait_policy='PASSIVE')

    return average_latency

def run_performance(average_latency, model_path, batch_size, sequence_length, use_gpu, test_cases, test_times, seed, verbose):
    # Try deduce input names from model.
    input_ids, segment_ids, input_mask = get_bert_inputs(model_path)

    # Do not generate random mask for performance test.
    print("generating test data...")
    all_inputs = generate_test_data(batch_size, sequence_length, test_cases, seed, verbose, input_ids, segment_ids, input_mask, random_mask_length=False)

    contiguous = False
    run_perf_tests(average_latency, model_path, batch_size, sequence_length, use_gpu, test_cases, test_times, seed, verbose, contiguous, input_ids, segment_ids, input_mask, all_inputs)

    # Convert inputs to contiguous array, which could improve inference performance
    all_inputs, contiguous_latency = get_contiguous_inputs(all_inputs)
    print("Extra latency for converting inputs to contiguous: {} ms".format(format(contiguous_latency, '.2f')))

    contiguous = True
    run_perf_tests(average_latency, model_path, batch_size, sequence_length, use_gpu, test_cases, test_times, seed, verbose, contiguous, input_ids, segment_ids, input_mask, all_inputs)

    return contiguous_latency

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, type=str,
                        help="bert onnx model path")

    parser.add_argument('--batch_size', required=True, type=int,
                        help="batch size of input")

    parser.add_argument('--sequence_length',  required=True, type=int,
                        help="maximum sequence length of input")

    parser.add_argument('--samples',  required=False, type=int, default=1,
                        help="number of samples to be generated")

    parser.add_argument('--test_times',  required=False, type=int, default=0,
                        help="number of times to run per sample")

    parser.add_argument('--seed',  required=False, type=int, default=3,
                        help="random seed")

    parser.add_argument('--verbose', required=False, action='store_true', help="print verbose information")
    parser.set_defaults(verbose=False)

    parser.add_argument('--use_gpu', required=False, action='store_true', help="use GPU")
    parser.set_defaults(use_gpu=False)

    parser.add_argument('--inclusive', required=False, action='store_true', help="include the latency of converting array to contiguous")
    parser.set_defaults(inclusive=False)

    args = parser.parse_args()
    return args

def main():
    args = parse_arguments()

    if args.test_times == 0:
        args.test_times = max(1, int(1000 / args.samples))

    if args.use_gpu and ('CUDAExecutionProvider' not in onnxruntime.get_available_providers()):
        print("Please install onnxruntime-gpu package instead of onnxruntime, and use a machine with GPU for testing gpu performance.")
        return
    elif (not args.use_gpu) and ('CUDAExecutionProvider' in onnxruntime.get_available_providers()):
        print("Warning: Please install onnxruntime package instead of onnxruntime-gpu to get best cpu performance.")

    average_latency = {}
    contiguous_latency = run_performance(average_latency, args.model, args.batch_size, args.sequence_length, args.use_gpu, args.samples, args.test_times, args.seed, args.verbose)

    if average_latency is None:
        return

    summary_file = os.path.join(Path(args.model).parent, "perf_results_{}_B{}_S{}_{}.txt".format('GPU' if args.use_gpu else 'CPU', args.batch_size, args.sequence_length, datetime.now().strftime("%Y%m%d-%H%M%S")))
    with open(summary_file, 'w+', newline='') as tsv_file:
        tsv_writer = csv.writer(tsv_file, delimiter='\t', lineterminator='\n')
        headers = None
        for key, latency in average_latency.items():
            params = key.split(',')
            if headers is None:
                headers = ["Latency(ms)", "Throughput(QPS)"]
                headers.extend([x.split('=')[0] for x in params])
                tsv_writer.writerow(headers)

            # include the extra latency of array conversion if required.
            if args.inclusive and 'contiguous=True' in params:
                latency += contiguous_latency

            throughput = args.batch_size * (1000 / latency)
            values = [format(latency, '.2f'), format(throughput, '.2f')]

            values.extend([x.split('=')[1] for x in params])
            tsv_writer.writerow(values)

    print("Test summary is saved to", summary_file)

if __name__ == "__main__":
    main()