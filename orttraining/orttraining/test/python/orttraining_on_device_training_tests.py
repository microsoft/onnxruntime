#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import argparse
import logging
import os
import sys

from _test_commons import run_subprocess

logging.basicConfig(format="%(asctime)s %(name)s [%(levelname)s] - %(message)s", level=logging.DEBUG)
log = logging.getLogger("OnDeviceTrainingTests")


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cwd", help="Path to the current working directory")
    return parser.parse_args()


def run_onnxblock_tests(cwd, log):
    log.debug("Running: onnxblock tests")

    command = [sys.executable, "-m", "pytest", "-sv", "orttraining_test_onnxblock.py"]

    run_subprocess(command, cwd=cwd, log=log).check_returncode()


def run_onnxruntime_test_all_ctest(cwd, log, filter):
    command = [os.path.join(cwd, "onnxruntime_test_all"), f"--gtest_filter={filter}"]

    run_subprocess(command, cwd=cwd, log=log).check_returncode()


def run_training_api_tests(cwd, log):
    log.debug("Running: TrainingApi tests")

    run_onnxruntime_test_all_ctest(cwd, log, "TrainingApiTest*")


def run_checkpoint_api_tests(cwd, log):
    log.debug("Running: TrainingApi tests")

    run_onnxruntime_test_all_ctest(cwd, log, "CheckpointApiTest*")


def main():
    args = parse_arguments()
    cwd = args.cwd

    log.info("Running ortmodule tests pipeline")

    run_onnxblock_tests(cwd, log)

    run_training_api_tests(cwd, log)

    run_checkpoint_api_tests(cwd, log)

    return 0


if __name__ == "__main__":
    sys.exit(main())
