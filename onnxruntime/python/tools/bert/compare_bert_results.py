#-------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.
#--------------------------------------------------------------------------

# It is a tool to compare the inference results of the original model and optimized model.

import sys
import argparse
import numpy as np
import os
import onnxruntime
import random
from pathlib import Path
import statistics
import onnx
import onnx.utils
import psutil
import csv
import timeit
from datetime import datetime
from onnx import ModelProto, TensorProto, numpy_helper
from OnnxModel import OnnxModel
from bert_model_optimization import optimize_by_onnxruntime
from bert_test_data import get_bert_inputs, generate_test_data, output_test_data
from bert_perf_test import create_session, onnxruntime_inference

def run_model(baseline_model, all_inputs, use_gpu, use_openmp, graph_optimization_level):
    session = create_session(baseline_model, use_gpu, use_openmp, graph_optimization_level, num_threads=psutil.cpu_count(logical=True), wait_policy='ACTIVE')
    output_names = [output.name for output in session.get_outputs()]
    results, latency_list = onnxruntime_inference(session, all_inputs, output_names)
    return results, latency_list, output_names

def compare(baseline_results, treatment_results, verbose, rtol=1e-3, atol=1e-4):
    # Validate the output of baseline and treatment, to make sure the results are similar.
    diff_count = 0
    max_rel_diff = 0
    max_abs_diff = 0
    for test_case_id, results in enumerate(baseline_results):
        case_passed = True
        for i in range(len(results)):
            treatment_first_output = treatment_results[test_case_id][0].tolist()
            rel_diff = np.amax(np.abs((treatment_results[test_case_id][0] - results[0]) / results[0]))
            abs_diff = np.amax(np.abs(treatment_results[test_case_id][0] - results[0]))
            max_rel_diff = max(max_rel_diff, rel_diff)
            max_abs_diff = max(max_abs_diff, abs_diff)
            if verbose:
                print("case {} output {}".format(test_case_id, i))
                print("baseline={}\ntreatment={}".format(results[0].tolist(), treatment_first_output))
                print("rel_diff={} abs_diff={}".format(rel_diff, abs_diff))
            if not np.allclose(results[0].tolist(), treatment_first_output, rtol=rtol, atol=atol):
                if case_passed:
                    case_passed = False
                    diff_count += 1

    if diff_count == 0:
        print("100% passed for {} random inputs given thresholds (rtol={}, atol={}).".format(len(baseline_results), rtol, atol))
    else:
        print("{} out of {} results not passed for thresholds (rtol={}, atol={}).".format(diff_count, len(baseline_results), rtol, atol))

    print("maximum absolute difference={}".format(max_abs_diff))

    print("maximum relative difference={}".format(max_rel_diff))

def run_test(baseline_model, optimized_model, output_dir, batch_size, sequence_length, use_gpu, test_cases, seed, use_openmp, verbose, rtol, atol):
    # Try deduce input names from optimized model.
    input_ids, segment_ids, input_mask = get_bert_inputs(optimized_model)

    # Use random mask length for accuracy test. It might introduce slight inflation in latency reported in this script.
    all_inputs = generate_test_data(batch_size, sequence_length, test_cases, seed, verbose, input_ids, segment_ids, input_mask, random_mask_length=True)

    baseline_results, baseline_latency, output_names = run_model(baseline_model, all_inputs, use_gpu, use_openmp, onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL)
    if verbose:
        print("baseline average latency: {} ms".format(statistics.mean(baseline_latency) * 1000))

    if output_dir is not None:
        for i, inputs in enumerate(all_inputs):
            output_test_data(output_dir, i, inputs)

    treatment_results, treatment_latency, treatment_output_names = run_model(optimized_model, all_inputs, use_gpu, use_openmp, onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL)
    if verbose:
        print("treatment average latency: {} ms".format(statistics.mean(treatment_latency) * 1000))

    # Validate the output of baseline and treatment, to make sure the results are similar.
    compare(baseline_results, treatment_results, verbose, rtol, atol)

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--baseline_model', required=True, type=str,
                        help="baseline onnx model path")

    parser.add_argument('--optimized_model', required=False, type=str, default=None,
                        help="optimized model for the baseline model. They shall have same inputs. If it is None, an optimized model will be generated using OnnxRuntime.")

    parser.add_argument('--output_dir', required=False, type=str, default=None,
                        help="output test data path. If not specified, test data will not be saved.")

    parser.add_argument('--batch_size', required=True, type=int,
                        help="batch size of input")

    parser.add_argument('--sequence_length',  required=True, type=int,
                        help="maximum sequence length of input")

    parser.add_argument('--rtol',  required=False, type=float, default=1e-3,
                        help="relative tolerance")

    parser.add_argument('--atol',  required=False, type=float, default=1e-4,
                        help="absolute tolerance")

    parser.add_argument('--samples',  required=False, type=int, default=100,
                        help="number of test cases to be generated")

    parser.add_argument('--seed',  required=False, type=int, default=3,
                        help="random seed")

    parser.add_argument('--use_gpu',  required=False, action='store_true', help="use GPU")
    parser.set_defaults(use_gpu=False)

    parser.add_argument('--no_openmp', required=False, action='store_true', help="do not use openmp")
    parser.set_defaults(no_openmp=False)

    parser.add_argument('--verbose', required=False, action='store_true', help="print verbose information")
    parser.set_defaults(verbose=False)

    args = parser.parse_args()
    return args

def main():
    args = parse_arguments()

    optimized_model = optimize_by_onnxruntime(args.baseline_model, args.use_gpu) if (args.optimized_model is None) else args.optimized_model

    if args.use_gpu and ('CUDAExecutionProvider' not in onnxruntime.get_available_providers()):
        print("Please install onnxruntime-gpu package instead of onnxruntime, and use a machine with GPU for testing gpu.")

    if args.output_dir is not None:
        # create the output directory if not existed
        path = Path(args.output_dir)
        path.mkdir(parents=True, exist_ok=True)

    run_test(
        args.baseline_model,
        optimized_model,
        args.output_dir,
        args.batch_size,
        args.sequence_length,
        args.use_gpu,
        args.samples,
        args.seed,
        not args.no_openmp,
        args.verbose,
        args.rtol,
        args.atol)

if __name__ == "__main__":
    main()
