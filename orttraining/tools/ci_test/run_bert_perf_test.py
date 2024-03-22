#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import argparse
import json
import os
import subprocess
import sys
from collections import namedtuple

SCRIPT_DIR = os.path.realpath(os.path.dirname(__file__))


def parse_args():
    parser = argparse.ArgumentParser(description="Runs BERT performance tests.")
    parser.add_argument("--binary_dir", required=True, help="Path to the ORT binary directory.")
    parser.add_argument("--training_data_root", required=True, help="Path to the training data root directory.")
    parser.add_argument("--model_root", required=True, help="Path to the model root directory.")
    parser.add_argument(
        "--gpu_sku",
        choices=["V100_16G", "MI100_32G"],
        default="V100_16G",
        required=False,
        help="GPU model (e.g. V100_16G, MI100_32G).",
    )
    return parser.parse_args()


# using the same params from "GitHub Master Merge Schedule" in OneNotes
def main():
    args = parse_args()

    Config = namedtuple(
        "Config", ["use_mixed_precision", "max_seq_length", "batch_size", "max_predictions_per_seq", "expected_perf"]
    )
    configs = {}
    configs["V100_16G"] = [
        Config(True, 128, 76, 20, -1.0),
        Config(True, 512, 11, 80, -1.0),
        Config(False, 128, 39, 20, -1.0),
        Config(False, 512, 6, 80, -1.0),
    ]

    configs["MI100_32G"] = [
        Config(True, 128, 128, 20, 240),
    ]

    # run BERT training
    for c in configs[args.gpu_sku]:
        model = "bert-large-uncased_L_24_H_1024_A_16_V_30528_S_512_Dp_0.1_optimized_layer_norm_opset12"
        precision_prefix = "fp16" if c.use_mixed_precision else "fp32"
        print(
            "######## testing name - "
            + ("fp16-" if c.use_mixed_precision else "fp32-")
            + str(c.max_seq_length)
            + " ##############"
        )
        cmds = [
            os.path.join(args.binary_dir, "onnxruntime_training_bert"),
            "--model_name",
            os.path.join(args.model_root, f"nv/bert-large/{model}"),
            "--train_data_dir",
            os.path.join(args.training_data_root, str(c.max_seq_length), "books_wiki_en_corpus/train"),
            "--test_data_dir",
            os.path.join(args.training_data_root, str(c.max_seq_length), "books_wiki_en_corpus/test"),
            "--train_batch_size",
            str(c.batch_size),
            "--mode",
            "train",
            "--max_seq_length",
            str(c.max_seq_length),
            "--num_train_steps",
            "640",
            "--display_loss_steps",
            "5",
            "--optimizer",
            "Lamb",
            "--learning_rate",
            "3e-3",
            "--warmup_ratio",
            "0.2843",
            "--warmup_mode",
            "Poly",
            "--gradient_accumulation_steps",
            "1",
            "--max_predictions_per_seq",
            str(c.max_predictions_per_seq),
            "--lambda",
            "0",
            "--use_nccl",
            "--perf_output_dir",
            os.path.join(SCRIPT_DIR, "results"),
        ]

        if c.use_mixed_precision:
            cmds.append("--use_mixed_precision"),
            cmds.append("--allreduce_in_fp16"),

        subprocess.run(cmds).check_returncode()  # noqa: PLW1510
        if c.expected_perf > 0.0:
            json_filename = (
                f"onnxruntime_perf_metrics_{model}.onnx_bert_{precision_prefix}_{c.max_seq_length}_Lamb.json"
            )
            with open(os.path.join(SCRIPT_DIR, "results", json_filename)) as json_file:
                results = json.load(json_file)
                assert results["EndToEndThroughput"] > 0.98 * c.expected_perf

    return 0


if __name__ == "__main__":
    sys.exit(main())
