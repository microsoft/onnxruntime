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

from dataclasses import dataclass


@dataclass
class TestSetting:
    batch_size: int
    sequence_length: int
    test_cases: int
    test_times: int
    contiguous: bool
    use_gpu: bool
    warmup: bool
    omp_num_threads: int
    omp_wait_policy: str
    intra_op_num_threads: int
    seed: int
    verbose: bool
    contiguous: bool
    inclusive: bool
    extra_latency: float = 0

    def get_setting(self) -> str:
        return f"batch_size={self.batch_size},sequence_length={self.sequence_length},test_cases={self.test_cases},test_times={self.test_times},contiguous={self.contiguous},use_gpu={self.use_gpu},warmup={self.warmup}"

    def check(self, intra_op_threads, omp_threads, omp_policy) -> bool:
        if intra_op_threads is None:
            if self.intra_op_num_threads is not None and self.intra_op_num_threads > 0:
                return False
        else:
            assert intra_op_threads > 0
            if not (self.intra_op_num_threads is None or self.intra_op_num_threads == intra_op_threads):
                return False

        if omp_threads is None:
            if self.omp_num_threads is not None and self.omp_num_threads > 0:
                return False
        else:
            assert omp_threads > 0
            if not (self.omp_num_threads is None or self.omp_num_threads == omp_threads):
                return False

        if self.omp_wait_policy is not None:
            if omp_policy != self.omp_wait_policy:
                return False

        return True


@dataclass
class ModelSetting:
    model_path: str
    input_ids_name: str
    segment_ids_name: str
    input_mask_name: str
    opt_level: int


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
    option += ",{}".format(test_setting.get_setting())
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
        assert omp_wait_policy in ["ACTIVE", "PASSIVE"], f"{omp_wait_policy} is not a valid policy"
        os.environ["OMP_WAIT_POLICY"] = omp_wait_policy


def run_one_test(model_setting, test_setting, perf_results, all_inputs, intra_op_num_threads, omp_num_threads,
                 omp_wait_policy):
    # Environment variable shall be set before import onnxruntime.
    setup_openmp_environ(omp_num_threads, omp_wait_policy)

    session = create_session(model_setting.model_path, test_setting.use_gpu, intra_op_num_threads,
                             model_setting.opt_level)
    output_names = [output.name for output in session.get_outputs()]

    key = to_string(model_setting.model_path, session, test_setting)
    if key in perf_results:
        print("skip duplicated test:", key)
        return

    print("Running test:", key)

    all_latency_list = []
    for i in range(test_setting.test_times):
        results, latency_list = onnxruntime_inference(session, all_inputs, output_names, test_setting.warmup)
        all_latency_list.extend(latency_list)

    # latency in miliseconds
    latency_ms = np.array(all_latency_list) * 1000 + test_setting.extra_latency

    average_latency = statistics.mean(latency_ms)
    latency_50 = np.percentile(latency_ms, 50)
    latency_75 = np.percentile(latency_ms, 75)
    latency_90 = np.percentile(latency_ms, 90)
    latency_95 = np.percentile(latency_ms, 95)
    latency_99 = np.percentile(latency_ms, 99)
    throughput = test_setting.batch_size * (1000.0 / average_latency)

    perf_results[key] = (average_latency, latency_50, latency_75, latency_90, latency_95, latency_99, throughput)

    print("Average latency = {} ms, Throughput = {} QPS".format(format(average_latency, '.2f'),
                                                                format(throughput, '.2f')))


def launch_test(model_setting, test_setting, perf_results, all_inputs, intra_op_num_threads, omp_num_threads,
                omp_wait_policy):
    if not test_setting.check(intra_op_num_threads, omp_num_threads, omp_wait_policy):
        return

    process = multiprocessing.Process(target=run_one_test,
                                      args=(model_setting, test_setting, perf_results, all_inputs, intra_op_num_threads,
                                            omp_num_threads, omp_wait_policy))
    process.start()
    process.join()


def run_perf_tests(model_setting, test_setting, perf_results, test_all, all_inputs):
    cpu_count = psutil.cpu_count(logical=False)
    logical_cores = psutil.cpu_count(logical=True)

    candidate_threads = list(set([1, logical_cores, cpu_count]))

    if (test_setting.intra_op_num_threads is not None) or (test_setting.omp_num_threads is not None):

        if test_setting.intra_op_num_threads is not None:
            intra_op_threads = [test_setting.intra_op_num_threads]
        else:
            intra_op_threads = [None] + candidate_threads

        if test_setting.omp_num_threads is not None:
            omp_threads = [test_setting.omp_num_threads]
        else:
            omp_threads = [None] + candidate_threads

        if test_setting.omp_wait_policy is not None:
            omp_policies = [test_setting.omp_wait_policy]
        else:
            omp_policies = [None, 'PASSIVE', 'ACTIVE']

        for it in intra_op_threads:
            for ot in omp_threads:
                for op in omp_policies:
                    launch_test(model_setting, test_setting, perf_results, all_inputs, it, ot, op)
        return

    # Test a setting without any setting as baseline 1.
    launch_test(model_setting, test_setting, perf_results, all_inputs, None, None, None)

    if not test_setting.use_gpu:
        # For CPU: intra_op_num_threads = 1, omp_num_threads=None, omp_wait_policy=None
        # Another setting without environment variable as baseline 2.
        launch_test(model_setting, test_setting, perf_results, all_inputs, 1, None, None)
    else:
        # For GPU, we test two more settings by default:
        # (1) intra_op_num_threads = 1, omp_num_threads=cpu_count, omp_wait_policy=PASSIVE
        # (2) intra_op_num_threads = logical_cores, omp_num_threads=1, omp_wait_policy=ACTIVE
        launch_test(model_setting, test_setting, perf_results, all_inputs, 1, cpu_count, 'PASSIVE')

        launch_test(model_setting, test_setting, perf_results, all_inputs, logical_cores, 1, 'ACTIVE')

    # GPU latency is not sensitive to these settings. No need to test many combinations.
    # Skip remaining settings for GPU without --all flag.
    if test_setting.use_gpu and not test_all:
        return

    for intra_op_num_threads in candidate_threads:
        for omp_num_threads in candidate_threads:
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
                launch_test(model_setting, test_setting, perf_results, all_inputs, intra_op_num_threads,
                            omp_num_threads, omp_wait_policy)


def run_performance(model_setting, test_setting, perf_results, test_all):
    input_ids, segment_ids, input_mask = get_bert_inputs(model_setting.model_path, model_setting.input_ids_name,
                                                         model_setting.segment_ids_name, model_setting.input_mask_name)

    # Do not generate random mask for performance test.
    print(
        f"Generating {test_setting.test_cases} samples for batch_size={test_setting.batch_size} sequence_length={test_setting.sequence_length}"
    )
    all_inputs = generate_test_data(test_setting.batch_size,
                                    test_setting.sequence_length,
                                    test_setting.test_cases,
                                    test_setting.seed,
                                    test_setting.verbose,
                                    input_ids,
                                    segment_ids,
                                    input_mask,
                                    random_mask_length=False)
    if test_setting.contiguous:
        all_inputs, contiguous_latency = get_contiguous_inputs(all_inputs)
        print("Extra latency for converting inputs to contiguous: {} ms".format(format(contiguous_latency, '.2f')))
        test_setting.extra_latency = contiguous_latency if test_setting.inclusive else 0

    run_perf_tests(model_setting, test_setting, perf_results, test_all, all_inputs)


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

    parser.add_argument('--omp_num_threads',
                        required=False,
                        type=int,
                        default=None,
                        help=">0, set OMP_NUM_THREADS value. 0, do not set")

    parser.add_argument('--intra_op_num_threads',
                        required=False,
                        type=int,
                        default=None,
                        help=">=0, set intra_op_num_threads")

    parser.add_argument('--omp_wait_policy',
                        required=False,
                        type=str,
                        default=None,
                        choices=['ACTIVE', 'PASSIVE'],
                        help="OMP_WAIT_POLICY")

    parser.add_argument('--contiguous', required=False, action='store_true', help="contiguous input")
    parser.set_defaults(contiguous=False)

    parser.add_argument('--no_warmup', required=False, action='store_true', help="do not use one sample for warm-up.")
    parser.set_defaults(no_warmup=False)

    parser.add_argument('--input_ids_name', required=False, type=str, default=None, help="input name for input ids")
    parser.add_argument('--segment_ids_name', required=False, type=str, default=None, help="input name for segment ids")
    parser.add_argument('--input_mask_name',
                        required=False,
                        type=str,
                        default=None,
                        help="input name for attention mask")

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

    model_setting = ModelSetting(args.model, args.input_ids_name, args.segment_ids_name, args.input_mask_name,
                                 args.opt_level)

    for batch_size in batch_size_set:
        test_setting = TestSetting(
            batch_size,
            args.sequence_length,
            args.samples,
            args.test_times,
            None,  #contiguous
            args.use_gpu,
            not args.no_warmup,
            args.omp_num_threads,
            args.omp_wait_policy,
            args.intra_op_num_threads,
            args.seed,
            args.verbose,
            args.contiguous,
            args.inclusive)
        print("test setting", test_setting)
        run_performance(model_setting, test_setting, perf_results, args.all)

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
