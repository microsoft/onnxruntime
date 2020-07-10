#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import argparse
import subprocess
import sys
import os
from collections import namedtuple

SCRIPT_DIR = os.path.realpath(os.path.dirname(__file__))

def parse_args():
  parser = argparse.ArgumentParser(description="Runs BERT performance tests.")
  parser.add_argument("--binary_dir", required=True,
                      help="Path to the ORT binary directory.")
  parser.add_argument("--training_data_root", required=True,
                      help="Path to the training data root directory.")
  parser.add_argument("--model_root", required=True,
                      help="Path to the model root directory.")
  return parser.parse_args()

# using the same params from "GitHub Master Merge Schedule" in OneNotes
def main():
    args = parse_args()

    Config = namedtuple('Config', ['use_mixed_precision', 'max_seq_length', 'batch_size', 'max_predictions_per_seq'])
    configs = [
        Config(True, 128, 64, 20),
        Config(True, 512, 10, 80),
        Config(False, 128, 33, 20),
        Config(False, 512, 5, 80)
    ]

    # run BERT training
    for c in configs:
        print("######## testing name - " + ('fp16-' if c.use_mixed_precision else 'fp32-') + str(c.max_seq_length) + " ##############")
        cmds = [
            os.path.join(args.binary_dir, "onnxruntime_training_bert"),
            "--model_name", os.path.join(
                args.model_root, "nv/bert-large/bert-large-uncased_L_24_H_1024_A_16_V_30528_S_512_Dp_0.1_optimized_layer_norm_opset12"),
            "--train_data_dir", os.path.join(
                args.training_data_root, str(c.max_seq_length), "books_wiki_en_corpus/train"),
            "--test_data_dir", os.path.join(
                args.training_data_root, str(c.max_seq_length), "books_wiki_en_corpus/test"),
            "--train_batch_size", str(c.batch_size),
            "--mode", "train",
            "--max_seq_length", str(c.max_seq_length),
            "--num_train_steps", "640",
            "--display_loss_steps", "5",
            "--optimizer", "Lamb",
            "--learning_rate", "3e-3",
            "--warmup_ratio", "0.2843",
            "--warmup_mode", "Poly",
            "--gradient_accumulation_steps", "1",
            "--max_predictions_per_seq", str(c.max_predictions_per_seq),
            "--lambda", "0",
            "--use_nccl",
            "--perf_output_dir", os.path.join(SCRIPT_DIR, "results"), 
        ]

        if c.use_mixed_precision: 
            cmds.append("--use_mixed_precision"),
            cmds.append("--allreduce_in_fp16"),

        subprocess.run(cmds).check_returncode()

    return 0

if __name__ == "__main__":
    sys.exit(main())
