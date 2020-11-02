#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import argparse
import subprocess
import sys
import tempfile
import os

from compare_results import compare_results_files, Comparisons

SCRIPT_DIR = os.path.realpath(os.path.dirname(__file__))

def parse_args():
  parser = argparse.ArgumentParser(description="Runs a BERT convergence test.")
  parser.add_argument("--binary_dir", required=True,
                      help="Path to the ORT binary directory.")
  parser.add_argument("--training_data_root", required=True,
                      help="Path to the training data root directory.")
  parser.add_argument("--model_root", required=True,
                      help="Path to the model root directory.")
  return parser.parse_args()

def main():
  args = parse_args()

  with tempfile.TemporaryDirectory() as output_dir:
    convergence_test_output_path = os.path.join(
        output_dir, "convergence_test_out.csv")

    # run BERT training
    subprocess.run([
        os.path.join(args.binary_dir, "onnxruntime_training_bert"),
        "--model_name", os.path.join(
            args.model_root, "nv/bert-base/bert-base-uncased_L_12_H_768_A_12_V_30528_S_512_Dp_0.1_optimized_layer_norm_opset12"),
        "--train_data_dir", os.path.join(
            args.training_data_root, "128/books_wiki_en_corpus/train"),
        "--test_data_dir", os.path.join(
            args.training_data_root, "128/books_wiki_en_corpus/test"),
        "--train_batch_size", "64",
        "--mode", "train",
        "--num_train_steps", "800",
        "--display_loss_steps", "5",
        "--optimizer", "adam",
        "--learning_rate", "5e-4",
        "--warmup_ratio", "0.1",
        "--warmup_mode", "Linear",
        "--gradient_accumulation_steps", "16",
        "--max_predictions_per_seq=20",
        "--use_mixed_precision",
        "--allreduce_in_fp16",
        "--lambda", "0",
        "--use_nccl",
        "--convergence_test_output_file", convergence_test_output_path,
        "--seed", "42",
        "--enable_grad_norm_clip=false",
    ]).check_returncode()

    # verify output
    comparison_result = compare_results_files(
        expected_results_path=os.path.join(
            SCRIPT_DIR, "results", "bert_base.convergence.baseline.csv"),
        actual_results_path=convergence_test_output_path,
        field_comparisons={
            "step": Comparisons.eq(),
            "total_loss": Comparisons.float_le(1e-3),
            "mlm_loss": Comparisons.float_le(1e-3),
            "nsp_loss": Comparisons.float_le(1e-3),
        })

    return 0 if comparison_result else 1

if __name__ == "__main__":
  sys.exit(main())
