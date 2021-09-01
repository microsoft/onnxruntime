# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

from convert_to_onnx import main
import os
import argparse
import logging
from gpt2_helper import PRETRAINED_GPT2_MODELS
from benchmark_helper import setup_logger

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

    parser.add_argument('--runs',
                        required=False,
                        type=int,
                        default=5,
                        help="number of repeated runs to get median value of each metric")

    parser.add_argument('--use_gpu', required=False, action='store_true', help="use GPU for inference")
    parser.set_defaults(use_gpu=False)

    parser.add_argument('--all', required=False, action='store_true', help="run all combinations of mixed precision")
    parser.set_defaults(all=False)

    parser.add_argument('-e', '--use_external_data_format', required=False, action='store_true')
    parser.set_defaults(use_external_data_format=False)

    parser.add_argument('--verbose', required=False, action='store_true')
    parser.set_defaults(verbose=False)

    args = parser.parse_args(argv)

    return args


class ParityTask:
    def __init__(self, total_runs, csv_path):
        self.total_runs = total_runs
        self.csv_path = csv_path
        self.latency_name = "average_latency(batch_size=8,sequence_length=1,past_sequence_length=32)"
        self.metric_names = [
            self.latency_name, "diff_50_percentile", "diff_90_percentile", "diff_95_percentile", "diff_99_percentile",
            "diff_pass_rate", "nan_rate", "top1_match_rate", "onnx_size_in_MB"
        ]

    def run(self, argv, name):
        results = []
        experiment_name = name
        for i in range(self.total_runs):
            try:
                result = main(argv, experiment_name=experiment_name, run_id=i, csv_filename=self.csv_path)
            except:
                logger.error(f"Failed to run experiment{experiment_name}")
                continue
            if result:
                results.append(result)

        if len(results) == 0:
            return

        # Calculate median value per metric
        all_results = {}
        for name in self.metric_names:
            all_results[name] = []

        for result in results:
            for name in self.metric_names:
                if name in result:
                    all_results[name].append(result[name])

        import statistics
        median_result = results[0]
        for name in self.metric_names:
            median_result[name] = statistics.median(all_results[name])

        self.save_result(median_result)

    def save_result(self, result):
        import csv
        csv_filename = self.csv_path

        csv_file_existed = os.path.exists(csv_filename)
        with open(csv_filename, mode="a", newline='') as csv_file:
            column_names = [
                "experiment", "run_id", "model_name", "model_class", "gpu", "precision", "optimizer", "test_cases",
                "keep_io_types", "io_block_list", "op_block_list", "node_block_list", "force_fp16_initializers",
                "ORT_TRANSFORMER_OPTIONS", "ORT_CUDA_GEMM_OPTIONS", "onnxruntime"
            ] + self.metric_names

            csv_writer = csv.DictWriter(csv_file, fieldnames=column_names)
            if not csv_file_existed:
                csv_writer.writeheader()

            row = {}
            for name in column_names:
                row[name] = result[name]

            row["run_id"] = "median"

            csv_writer.writerow(row)
            logger.info(f"result saved to {csv_filename}: {row}")


def run_parity(args):
    task = ParityTask(args.runs, args.csv)

    model = args.model_name_or_path
    fp32_baseline = f"-m {model} -o -p fp32".split()
    if args.use_gpu:
        fp32_baseline.append("--use_gpu")

    if args.use_external_data_format:
        fp32_baseline.append("--use_external_data_format")

    task.run(fp32_baseline, "fp32 baseline")

    # The following tests for fp16 requires GPU
    if not args.use_gpu:
        logger.info("skip mixed precision since --use_gpu is not specified")
        return

    baseline = f"-m {model} -o --use_gpu -p fp16".split()
    if args.use_external_data_format:
        baseline.append("--use_external_data_format")
    task.run(baseline, "fp16 baseline")

    if not args.all:
        logger.info("skip remaining combinations since --all is not specified")
        return

    fp32_logits = ["--io_block_list", "logits"]
    task.run(baseline + fp32_logits, "fp16 except logits")

    fp32_io = ["--keep_io_types"]
    task.run(baseline + fp32_io, "Graph I/O FP32, Other FP16")

    op_list = "Attention Gather Add LayerNormalization FastGelu MatMul".split()
    task.run(baseline + fp32_io + ["--op_block_list"] + [o for o in op_list], "Everthing in FP32")

    for op in op_list:
        op_block_list = ["--op_block_list"] + [o for o in op_list if o != op]
        task.run(baseline + fp32_io + op_block_list, f"FP32 except {op} in fp16")

    for op in op_list:
        op_block_list = ["--op_block_list", op]
        task.run(baseline + op_block_list, f"FP16 except {op} in fp32")

    op_block_list = ["--op_block_list", "LayerNormalization", "FastGelu"]
    task.run(baseline + op_block_list, f"FP16 except LayerNormalization and FastGelu in fp32")

    task.run(baseline + op_block_list + fp32_logits, f"FP16 except logits, LayerNormalization and FastGelu in fp32")


if __name__ == '__main__':
    args = parse_arguments()
    setup_logger(args.verbose)

    run_parity(args)
