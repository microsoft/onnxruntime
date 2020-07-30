#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import argparse
import subprocess
import sys
import tempfile
import os
import re
import csv

from compare_results import compare_results_files, Comparisons

SCRIPT_DIR = os.path.realpath(os.path.dirname(__file__))

def parse_args():
  parser = argparse.ArgumentParser(description="Runs a BERT convergence test on multi GPU.")
  parser.add_argument("--binary_dir", default="/build/Linux/RelWithDebInfo", #required=True,
                      help="Path to the ORT binary directory.")
  parser.add_argument("--training_data_root", default="/bert_data", #required=True,
                      help="Path to the training data root directory.")
  parser.add_argument("--model_root", default="/bert_ort/bert_models/", #required=True,
                      help="Path to the model root directory.")
  return parser.parse_args()

def main():
  args = parse_args()

  with tempfile.TemporaryDirectory() as output_dir:
    convergence_test_output_path = os.path.join(
        output_dir, "multi_gpu_test_out.csv")
    convergence_test_output_path = "multi_gpu_test_out.csv"
    multigpu_test_output_path = "log-mem-many-p3-s128-b1-a16"
    # run BERT training
    subprocess.run([
        #"/bert_ort/openmpi/bin/mpirun",
        "mpirun",
        "-n", "2",
        "--tag-output",
        "-merge-stderr-to-stdout",
        "--output-filename", multigpu_test_output_path,
        #os.path.join(args.binary_dir, "onnxruntime_training_bert"),
        "./build/Linux/RelWithDebInfo/onnxruntime_training_bert",
        "--model_name", os.path.join(
            #args.model_root, "nv/bert-base/bert-base-uncased_L_12_H_768_A_12_V_30528_S_512_Dp_0.1_optimized_layer_norm_opset12"),
            args.model_root, "nv/bert-large/bert-large-uncased_L_24_H_1024_A_16_V_30528_S_512_Dp_0.1_optimized_layer_norm"),
        "--train_data_dir", os.path.join(
            args.training_data_root, "128/books_wiki_en_corpus/train"),
        "--test_data_dir", os.path.join(
            args.training_data_root, "128/books_wiki_en_corpus/test"),
        #"--train_batch_size", "64",
        "--train_batch_size", "1",
        "--mode", "train",
        #"--num_train_steps", "800",
        "--num_train_steps", "32",
        #"--display_loss_steps", "5",
        "--display_loss_steps", "1",
        #"--optimizer", "adam",
        "--optimizer", "lamb",
        #"--learning_rate", "5e-4",
        "--learning_rate", "0.006",
        #"--warmup_ratio", "0.1",
        "--warmup_ratio", "0",
        "--warmup_mode", "Linear",
        "--gradient_accumulation_steps", "16",
        #"--max_predictions_per_seq=20",
        #"--use_mixed_precision",
        #"--allreduce_in_fp16",
        #"--lambda", "0",
        "--use_nccl",
        #"--convergence_test_output_file", convergence_test_output_path,
        "--seed", "42",
        #"--enable_grad_norm_clip=false",
        "--data_parallel_size", "2"
    ]).check_returncode()

    expected_results_path = os.path.join(
            SCRIPT_DIR, "results", "multi_gpu_baseline","2_nodes")
    actual_results_path = os.path.join(multigpu_test_output_path, "1")
    # for each node of result files
    for dir in os.listdir(actual_results_path):
        actual_node_result_file = os.path.join(actual_results_path, dir, "stdout")
        # save a .csv file for comparison with baseline.csv
        with open (actual_node_result_file, "r") as actual_file:
            actual_node_result_csv = os.path.join(actual_results_path, dir, "result.csv")
            with open(actual_node_result_csv, 'w', newline='') as csvfile:
                csvfile.write("step,examples,total_loss,mlm_loss,nsp_loss\n")
                for line in actual_file:
                    if "#examples: 1, total_loss:" in line:
                        line = line[20:]
                        line = re.sub(r', [a-zA-Z_#]*\:', ',', line)
                        csvfile.write(line)
            #compare with baseline.csv
            comparison_result = compare_results_files(
                expected_results_path=os.path.join(
                    expected_results_path, dir, "baseline.csv"),
                actual_results_path=actual_node_result_csv,
                field_comparisons={
                    "step": Comparisons.eq(),
                    "total_loss": Comparisons.float_le(1e-3),
                    "mlm_loss": Comparisons.float_le(1e-3),
                    "nsp_loss": Comparisons.float_le(1e-3),
                })
            if not comparison_result:
                return 1
    return 0

if __name__ == "__main__":
  sys.exit(main())

