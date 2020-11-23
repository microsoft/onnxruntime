#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import sys
import argparse

from _test_commons import run_subprocess

import logging

logging.basicConfig(
    format="%(asctime)s %(name)s [%(levelname)s] - %(message)s",
    level=logging.DEBUG)
log = logging.getLogger("DistributedTests")


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cwd", help="Path to the current working directory")
    return parser.parse_args()


def run_checkpoint_tests(cwd, log):
    log.debug('Running: Checkpoint tests')

    command = [sys.executable, 'orttraining_test_checkpoint.py']

    run_subprocess(command, cwd=cwd, log=log).check_returncode()


def run_checkpointing_aggregation_tests(cwd):
    log.info("Running multi-GPU checkpointing tests.")

    import torch
    ngpus = torch.cuda.device_count()

    # generate checkpoint files required in orttraining_test_checkpoint_aggregation.py
    run_subprocess(['mpirun', '-n', str(ngpus), '-x', 'NCCL_DEBUG=INFO', sys.executable,
                    'orttrainer_bert_toy_onnx_ckpt_gen.py'], cwd=cwd)

    run_subprocess([sys.executable, '-m', 'pytest', '-sv', 'orttraining_test_checkpoint_aggregation.py'], cwd=cwd)


def main():
    import torch
    ngpus = torch.cuda.device_count()

    if ngpus < 2:
        raise RuntimeError("Cannot run distributed tests with less than 2 gpus.")

    args = parse_arguments()
    cwd = args.cwd

    log.info("Running distributed tests pipeline")

    run_checkpoint_tests(cwd, log)
    run_checkpointing_aggregation_tests(cwd)

    return 0


if __name__ == "__main__":
    sys.exit(main())
