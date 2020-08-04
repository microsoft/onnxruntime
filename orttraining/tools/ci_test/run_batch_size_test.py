#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import argparse
import collections
import subprocess
import sys
import os


def parse_args():
    parser = argparse.ArgumentParser(description="Runs a BERT batch size test.")
    parser.add_argument("--binary_dir", required=True, help="Path to the ORT binary directory.")
    parser.add_argument("--model_root", required=True, help="Path to the model root directory.")
    return parser.parse_args()


def main():
    args = parse_args()

    Config = collections.namedtuple("Config", ["enable_mixed_precision", "sequence_length", "max_batch_size"])
    configs = [
        Config(True, 128, 66),
        Config(True, 512, 10),
        Config(False, 128, 33),
        Config(False, 512, 5),
    ]

    # run BERT training
    for config in configs:
        print("##### testing name - {}-{} #####".format("fp16" if config.enable_mixed_precision else "fp32",
                                                        config.sequence_length))
        cmds = [
            os.path.join(args.binary_dir, "onnxruntime_training_bert"),
            "--model_name", os.path.join(
                args.model_root,
                "nv/bert-large/bert-large-uncased_L_24_H_1024_A_16_V_30528_S_512_Dp_0.1_optimized_layer_norm_opset12"),
            "--train_batch_size", str(config.max_batch_size),
            "--mode", "perf",
            "--max_seq_length", str(config.sequence_length),
            "--num_train_steps", "10",
            "--display_loss_steps", "5",
            "--optimizer", "adam",
            "--learning_rate", "5e-4",
            "--warmup_ratio", "0.1",
            "--warmup_mode", "Linear",
            "--gradient_accumulation_steps", "1",
            "--max_predictions_per_seq=20",
            "--allreduce_in_fp16",
            "--lambda", "0",
            "--use_nccl",
            "--seed", "42",
            "--enable_grad_norm_clip=false",
        ]

        if config.enable_mixed_precision:
            cmds.append("--use_mixed_precision"),

        subprocess.run(cmds, timeout=120).check_returncode()

    return 0


if __name__ == "__main__":
    sys.exit(main())
