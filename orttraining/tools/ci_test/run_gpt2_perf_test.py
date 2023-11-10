#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import argparse
import os
import subprocess
import sys
from collections import namedtuple

SCRIPT_DIR = os.path.realpath(os.path.dirname(__file__))


def parse_args():
    parser = argparse.ArgumentParser(description="Runs GPT-2 performance tests.")
    parser.add_argument("--binary_dir", required=True, help="Path to the ORT binary directory.")
    parser.add_argument("--training_data_root", required=True, help="Path to the training data root directory.")
    parser.add_argument("--model_root", required=True, help="Path to the model root directory.")
    return parser.parse_args()


# TODO - review to finalize params
def main():
    args = parse_args()

    Config = namedtuple("Config", ["use_mixed_precision", "max_seq_length", "batch_size"])
    configs = [Config(True, 1024, 1), Config(False, 1024, 1)]

    # run GPT-2 training
    for c in configs:
        print(
            "######## testing name - "
            + ("fp16-" if c.use_mixed_precision else "fp32-")
            + str(c.max_seq_length)
            + " ##############"
        )
        cmds = [
            os.path.join(args.binary_dir, "onnxruntime_training_gpt2"),
            "--model_name",
            os.path.join(
                args.model_root,
                "megatron-gpt2_hidden-size-1024_num-layers-24_vocab-size-50257_num-attention-heads-16_max-position-embeddings-1024_optimized_opset12",
            ),
            "--train_data_dir",
            os.path.join(args.training_data_root, "train"),
            "--test_data_dir",
            os.path.join(args.training_data_root, "test"),
            "--train_batch_size",
            str(c.batch_size),
            "--mode",
            "train",
            "--max_seq_length",
            str(c.max_seq_length),
            "--num_train_steps",
            "640",
            "--gradient_accumulation_steps",
            "1",
            "--perf_output_dir",
            os.path.join(SCRIPT_DIR, "results"),
        ]

        if c.use_mixed_precision:
            cmds.append("--use_mixed_precision"),

        subprocess.run(cmds).check_returncode()

    return 0


if __name__ == "__main__":
    sys.exit(main())
