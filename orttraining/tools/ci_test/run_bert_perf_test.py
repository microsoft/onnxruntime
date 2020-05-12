#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import argparse
import subprocess
import sys
import os

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

    matrix = { # enable mixed-precision, sequence length, max batch size
        "fp16-128": [True, 128, 66, 20],
        "fp16-512": [True, 512, 10, 80],
        "fp32-128": [False, 128, 33, 20],
        "fp32-512": [False, 512, 5, 80]}

    # run BERT training
    for m in matrix:
        print("######## testing name - " + m + " ##############")
        cmds = [
            os.path.join(args.binary_dir, "onnxruntime_training_bert"),
            "--model_name", os.path.join(
                args.model_root, "nv/bert-large/bert-large-uncased_L_24_H_1024_A_16_V_30528_S_512_Dp_0.1_optimized_layer_norm"),
            "--train_data_dir", os.path.join(
                args.training_data_root, str(matrix[m][1]), "books_wiki_en_corpus/train"),
            "--test_data_dir", os.path.join(
                args.training_data_root, str(matrix[m][1]), "books_wiki_en_corpus/test"),
            "--train_batch_size", str(matrix[m][2]),
            "--mode", "train",
            "--max_seq_length", str(matrix[m][1]),
            "--num_train_steps", "100",
            "--display_loss_steps", "5",
            "--optimizer", "Lamb",
            "--learning_rate", "3e-3",
            "--warmup_ratio", "0.2843",
            "--warmup_mode", "Poly",
            "--gradient_accumulation_steps", "1",
            "--max_predictions_per_seq", str(matrix[m][3]),
            "--lambda", "0",
            "--use_nccl",
            "--perf_output_dir", os.path.join(SCRIPT_DIR, "results"), 
        ]

        if matrix[m][0]: 
            cmds.append("--use_mixed_precision"),
            cmds.append("--allreduce_in_fp16"),

        subprocess.run(cmds).check_returncode()

    return 0

if __name__ == "__main__":
  sys.exit(main())
