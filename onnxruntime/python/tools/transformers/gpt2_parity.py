# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

# This script uses different configurations in mixed precision conversion for GPT-2 model, and
# measures the inference latency, top 1 match rate (compared to PyTorch FP32 model) and ONNX model size.
# It outputs a csv file with Mann-Whitney U test and T-Test on each pair of experiments, where
# pvalue < 0.05 means two experiments have significant difference on top 1 match rate.
# User could use this script to select the best mixed precision model according to these metrics.

from convert_to_onnx import main, get_latency_name
import os
import argparse
import logging
from gpt2_helper import PRETRAINED_GPT2_MODELS, Gpt2Helper
from benchmark_helper import setup_logger
from onnx_model import OnnxModel
import onnx
import csv
import datetime
import scipy.stats
import torch

logger = logging.getLogger('')


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser()

    parser.add_argument('-m',
                        '--model_name_or_path',
                        required=True,
                        type=str,
                        help='Model path, or pretrained model name in the list: ' + ', '.join(PRETRAINED_GPT2_MODELS))

    parser.add_argument('--csv',
                        required=False,
                        type=str,
                        default='gpt2_parity_results.csv',
                        help='path of csv file to save the result')

    parser.add_argument('--test_cases', required=False, type=int, default=500, help="number of test cases per run")

    parser.add_argument('--runs', required=False, type=int, default=40, help="number of repeated runs")

    parser.add_argument('--use_gpu', required=False, action='store_true', help="use GPU for inference")
    parser.set_defaults(use_gpu=False)

    parser.add_argument('--all', required=False, action='store_true', help="run all combinations of mixed precision")
    parser.set_defaults(all=False)

    parser.add_argument('-e', '--use_external_data_format', required=False, action='store_true')
    parser.set_defaults(use_external_data_format=False)

    parser.add_argument('--verbose', required=False, action='store_true')
    parser.set_defaults(verbose=False)

    parser.add_argument('--skip_test',
                        required=False,
                        action='store_true',
                        help="do not run test, and only rank experiments based on existing csv file")
    parser.set_defaults(skip_test=False)

    args = parser.parse_args(argv)

    return args


class ParityTask:
    def __init__(self, test_cases, total_runs, csv_path):
        self.total_runs = total_runs
        self.test_cases = test_cases
        self.csv_path = csv_path
        self.results = []
        self.run_id = 0

    def run(self, argv, experiment_name):
        start_time = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
        run_id = f"{start_time}_{self.run_id}"
        self.run_id += 1

        try:
            result = main(argv + ["-t", f"{self.test_cases}", "-r", f"{self.total_runs}"],
                          experiment_name=experiment_name,
                          run_id=run_id,
                          csv_filename=self.csv_path)
        except:
            logger.exception(f"Failed to run experiment {experiment_name}")

        if result:
            self.results.append(result)


def load_results_from_csv(csv_path):
    rows = []
    import csv
    with open(csv_path, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            rows.append(row)
    return rows


def score(row):
    """Scoring function based on 3 metrics. The larger score is better."""
    latency_in_ms = float(row[get_latency_name()])
    top1_match_rate = float(row["top1_match_rate"])
    onnx_size_in_MB = float(row["onnx_size_in_MB"])
    # A simple scoring function: cost of 0.1ms latency ~ 0.1% match rate ~ 100MB size
    return (top1_match_rate * 1000 - latency_in_ms * 10 - onnx_size_in_MB / 100)


def print_wins(wins, rows, test_name):
    print()
    print("*" * 10)

    row_map = {}
    for row in rows:
        row_map[row["run_id"]] = row

    sorted_wins = dict(sorted(wins.items(), key=lambda item: (item[1], score(row_map[item[0]])), reverse=True))
    logger.debug(f"{test_name} Wins:{sorted_wins}")
    logger.info(f"Based on {test_name} wins and a scoring function, the ranking:")

    rank = 0
    previous_value = -1
    count = 0
    for key, value in sorted_wins.items():
        if value != previous_value:
            rank = count
        previous_value = value
        count += 1

        for row in rows:
            if row["run_id"] == key:
                logger.info(
                    "{:02d}: WINs={:02d}, run_id={}, latency={:5.2f} top1_match={:.4f} size={}_MB experiment={} {}".
                    format(
                        rank, value, key, float(row[get_latency_name()]), float(row["top1_match_rate"]),
                        row["onnx_size_in_MB"], row["experiment"], " (Half2 Disabled)" if
                        (row['ORT_CUDA_GEMM_OPTIONS'] == "4" and "Half2" not in row["experiment"]) else ""))
                break


def run_significance_test(rows, output_csv_path):
    """Run U test and T test.
    """
    utest_wins = {}
    ttest_wins = {}
    for row in rows:
        run_id = row["run_id"]
        utest_wins[run_id] = 0
        ttest_wins[run_id] = 0

    with open(output_csv_path, 'w', newline='') as csvfile:
        column_names = [
            'model_name', 'run_id_1', 'experiment_1', 'top1_match_rate_1', 'run_id_2', 'experiment_2',
            'top1_match_rate_2', 'U_statistic', 'U_pvalue', "T_statistic", "T_pvalue"
        ]

        writer = csv.DictWriter(csvfile, fieldnames=column_names)
        writer.writeheader()

        required_match_columns = ["model_name", "test_cases", "runs"]
        num_results = len(rows)
        for i in range(num_results - 1):
            result1 = rows[i]

            for j in range(i + 1, num_results, 1):
                result2 = rows[j]

                all_matched = True
                for column in required_match_columns:
                    if (result1[column] != result2[column]):
                        all_matched = False
                        break
                if not all_matched:
                    continue

                if isinstance(result1["top1_match_rate_per_run"], str):
                    import json
                    a = json.loads(result1["top1_match_rate_per_run"])
                    b = json.loads(result2["top1_match_rate_per_run"])
                else:
                    a = result1["top1_match_rate_per_run"]
                    b = result2["top1_match_rate_per_run"]

                try:
                    utest_statistic, utest_pvalue = scipy.stats.mannwhitneyu(
                        a, b, use_continuity=True, alternative="two-sided"
                    )  #TODO: shall we use one-sided: less or greater according to "top1_match_rate"
                except ValueError:  #ValueError: All numbers are identical in mannwhitneyu
                    utest_statistic = None
                    utest_pvalue = None
                ttest_statistic, ttest_pvalue = scipy.stats.ttest_ind(a, b, axis=None, equal_var=True)

                if utest_pvalue < 0.05:
                    if float(result1["top1_match_rate"]) > float(result2["top1_match_rate"]):
                        utest_wins[result1["run_id"]] += 1
                    else:
                        utest_wins[result2["run_id"]] += 1

                if ttest_pvalue < 0.05:
                    if float(result1["top1_match_rate"]) > float(result2["top1_match_rate"]):
                        ttest_wins[result1["run_id"]] += 1
                    else:
                        ttest_wins[result2["run_id"]] += 1

                row = {
                    'model_name': result1["model_name"],
                    'run_id_1': result1["run_id"],
                    'experiment_1': result1["experiment"],
                    'top1_match_rate_1': float(result1["top1_match_rate"]),
                    "run_id_2": result2["run_id"],
                    "experiment_2": result2["experiment"],
                    'top1_match_rate_2': float(result2["top1_match_rate"]),
                    'U_statistic': utest_statistic,
                    'U_pvalue': utest_pvalue,
                    'T_statistic': ttest_statistic,
                    'T_pvalue': ttest_pvalue
                }

                writer.writerow(row)
    logger.info(f"U-Test and T-Test results are output to {output_csv_path}")
    print_wins(utest_wins, rows, "U-Test")
    print_wins(ttest_wins, rows, "T-Test")


def get_last_matmul_node_name(raw_onnx_model: str):
    model = onnx.load(raw_onnx_model)
    onnx_model = OnnxModel(model)
    output_name_to_node = onnx_model.output_name_to_node()

    assert model.graph.output[0].name in output_name_to_node
    node = output_name_to_node[model.graph.output[0].name]
    if node.op_type == "MatMul":
        logger.info(f"Found last MatMul node for logits: {node.name}")
        return node.name

    logger.warning(f"Failed to find MatMul node for logits. Found {node.op_type} of node {node.name}")
    return None


def get_mixed_precision_parameters(args, last_matmul_node_name, op_block_list):
    model = args.model_name_or_path
    parameters = f"-m {model} -o --use_gpu -p fp16".split()
    if args.use_external_data_format:
        parameters.append("--use_external_data_format")
    parameters += ["--io_block_list", "logits", "--node_block_list", last_matmul_node_name]

    if op_block_list:
        parameters.extend(["--op_block_list"] + op_block_list)

    return parameters


def run_candidate(task: ParityTask, args, last_matmul_node_name, op_block_list=["FastGelu", "LayerNormalization"]):
    parameters = get_mixed_precision_parameters(args, last_matmul_node_name, op_block_list)
    op_block_list_str = ','.join(sorted(op_block_list))
    name_suffix = " (Half2 Disabled)" if os.getenv('ORT_CUDA_GEMM_OPTIONS') == "4" else ""
    if op_block_list:
        name = f"Mixed precision baseline + {op_block_list_str} in FP32{name_suffix}"
    else:
        name = f"Mixed precision baseline (logits output and last MatMul node {last_matmul_node_name} in FP32){name_suffix}"
    task.run(parameters, name)


def get_baselines(args):
    model = args.model_name_or_path
    fp32_baseline = f"-m {model} -o -p fp32".split()
    if args.use_gpu:
        fp32_baseline.append("--use_gpu")
    if args.use_external_data_format:
        fp32_baseline.append("--use_external_data_format")

    fp16_baseline = f"-m {model} -o --use_gpu -p fp16".split()
    if args.use_external_data_format:
        fp16_baseline.append("--use_external_data_format")

    return fp32_baseline, fp16_baseline


def get_all_operators():
    """All operators in the optimized model"""
    return "Attention Gather Add LayerNormalization FastGelu MatMul".split()


def run_tuning_step0(task, fp16_baseline):
    """Step 0 is to check which operator in FP16 causes most loss"""
    fp32_logits = ["--io_block_list", "logits"]
    task.run(fp16_baseline + fp32_logits, "FP16 except logits")

    fp32_io = ["--keep_io_types"]
    task.run(fp16_baseline + fp32_io, "Graph I/O FP32, Other FP16")

    op_list = get_all_operators()
    #task.run(fp16_baseline + fp32_io + ["--op_block_list"] + [o for o in op_list], "Everthing in FP32")

    # Only weights in FP16
    task.run(fp16_baseline + fp32_io + ["--op_block_list"] + [o for o in op_list] + ['--force_fp16_initializers'],
             "FP32 except weights in FP16")

    for op in op_list:
        op_block_list = ["--op_block_list"] + [o for o in op_list if o != op]
        task.run(fp16_baseline + fp32_io + op_block_list, f"FP32 except {op} in FP16")


def run_tuning_step1(task, mixed_precision_baseline):
    """Step 1 is to figure out which operator in FP32 could benefit most"""
    for op in get_all_operators():
        op_block_list = ["--op_block_list", op]
        task.run(mixed_precision_baseline + op_block_list, f"Mixed precision baseline + {op} in FP32")


def run_tuning_step2(task, mixed_precision_baseline):
    """Assumed that you have run step 1 to figure out that Logits FP32 and Add FP32 is important,
    Step 2 is to figure out a combination of two operators (one is Add from step one) to get better result
    """
    for op in get_all_operators():
        if op not in ['Add']:
            op_block_list = ["--op_block_list", 'Add', op]
            task.run(mixed_precision_baseline + op_block_list, f"Mixed precision baseline + Add,{op} in FP32")


def run_parity_disable_half2(task: ParityTask, args):
    onnx_model_paths = Gpt2Helper.get_onnx_paths('onnx_models',
                                                 args.model_name_or_path,
                                                 new_folder=args.use_external_data_format,
                                                 remove_existing=[])
    last_matmul_node_name = get_last_matmul_node_name(onnx_model_paths["raw"])
    run_candidate(task, args, last_matmul_node_name, op_block_list=[])
    run_candidate(task, args, last_matmul_node_name, op_block_list=["Add"])
    run_candidate(task, args, last_matmul_node_name, op_block_list=["LayerNormalization", "Add"])


def run_parity(task: ParityTask, args):
    onnx_model_paths = Gpt2Helper.get_onnx_paths('onnx_models',
                                                 args.model_name_or_path,
                                                 new_folder=args.use_external_data_format,
                                                 remove_existing=[])

    fp32_baseline, fp16_baseline = get_baselines(args)

    task.run(fp32_baseline, "FP32 baseline")

    # The following tests for fp16 requires GPU
    if not args.use_gpu:
        logger.info("skip mixed precision since --use_gpu is not specified")
        return

    task.run(fp16_baseline, "FP16 baseline")

    last_matmul_node_name = get_last_matmul_node_name(onnx_model_paths["raw"])

    # Mixed precision baseline
    run_candidate(task, args, last_matmul_node_name, op_block_list=[])

    # Result from tuning step 1
    run_candidate(task, args, last_matmul_node_name, op_block_list=["Add"])

    if args.all:
        run_tuning_step0(task, fp16_baseline)
        mixed_precision_baseline = get_mixed_precision_parameters(args, last_matmul_node_name, op_block_list=[])
        run_tuning_step1(task, mixed_precision_baseline)
        run_tuning_step2(task, mixed_precision_baseline)
    else:
        run_candidate(task, args, last_matmul_node_name, op_block_list=["LayerNormalization", "Add"])
        run_candidate(task, args, last_matmul_node_name, op_block_list=["FastGelu", "Add"])

    # Run a few good candidates
    run_candidate(task, args, last_matmul_node_name, op_block_list=["FastGelu", "LayerNormalization", "Add"])
    run_candidate(task, args, last_matmul_node_name, op_block_list=["FastGelu", "LayerNormalization", "Add", "Gather"])
    run_candidate(task, args, last_matmul_node_name, \
                  op_block_list=["FastGelu", "LayerNormalization", "Add", "Gather", "MatMul"])


if __name__ == '__main__':
    args = parse_arguments()
    setup_logger(args.verbose)

    if args.test_cases < 100 or args.runs < 20 or args.test_cases * args.runs < 10000:
        logger.warning(
            "Not enough test cases or runs to get stable results or test significance. Recommend test_cases >= 100, runs >= 20, test_cases * runs >= 10000."
        )

    task = ParityTask(args.test_cases, args.runs, args.csv)

    if not args.skip_test:
        if (os.getenv('ORT_CUDA_GEMM_OPTIONS') == "4" and args.use_gpu):
            assert torch.cuda.get_device_capability(
            )[0] >= 7, "half2 kernel is not avaiable in current GPU device. Please set environment variable ORT_CUDA_GEMM_OPTIONS=0 or use supported GPU like V100 or T4"
            run_parity_disable_half2(task, args)
        else:
            run_parity(task, args)

    try:
        rows = load_results_from_csv(task.csv_path)
    except:
        logger.exception(f"Failed to load csv {task.csv_path}")
        rows = task.results

    logger.info("Start running significance tests...")
    summary_csv = task.csv_path.replace('.csv', ".stats.csv")
    run_significance_test(rows, summary_csv)
