#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

# This file contains calls to the tests for ONNXBlock (front end tooling for ort training apis) and training apis.
# The tests are run in a separate process to avoid testing the entire ort suite of tests yet again (since they
# are covered in other pipelines) using the gtest filter.

import argparse
import logging
import os
import sys

from _test_commons import run_subprocess

logging.basicConfig(format="%(asctime)s %(name)s [%(levelname)s] - %(message)s", level=logging.DEBUG)
log = logging.getLogger("TrainingAPIsTests")


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cwd", help="Path to the current working directory")
    return parser.parse_args()


def run_training_apis_python_api_tests(cwd, log):
    """Runs the tests for ort training apis."""

    log.debug("Running: ort training api tests")

    command = [sys.executable, "-m", "pytest", "-sv", "orttraining_test_python_bindings.py"]

    run_subprocess(command, cwd=cwd, log=log).check_returncode()


def run_onnxblock_tests(cwd, log):
    """Runs the offline tooling tests for ort training apis."""

    log.debug("Running: onnxblock tests")

    command = [sys.executable, "-m", "pytest", "-sv", "orttraining_test_onnxblock.py"]

    run_subprocess(command, cwd=cwd, log=log).check_returncode()


def run_onnxruntime_test_all_ctest(cwd, log, filter):
    """Calls onnxruntime_test_all gtest executable with the given filter."""

    command = [os.path.join(cwd, "onnxruntime_test_all"), f"--gtest_filter={filter}"]

    run_subprocess(command, cwd=cwd, log=log).check_returncode()


def run_training_api_tests(cwd, log):
    """Runs the onnxruntime_test_all executable with the TrainingApiTest* gtest filter."""

    log.debug("Running: TrainingApi and TrainingCApi tests")

    run_onnxruntime_test_all_ctest(cwd, log, "TrainingApiTest*")
    run_onnxruntime_test_all_ctest(cwd, log, "TrainingCApiTest*")


def run_checkpoint_api_tests(cwd, log):
    """Runs the onnxruntime_test_all executable with the CheckpointApiTest* gtest filter."""

    log.debug("Running: TrainingApi tests")

    run_onnxruntime_test_all_ctest(cwd, log, "CheckpointApiTest*")


def main():
    args = parse_arguments()
    cwd = args.cwd

    log.info("Running ortmodule tests pipeline")

    run_onnxblock_tests(cwd, log)

    run_training_apis_python_api_tests(cwd, log)

    run_training_api_tests(cwd, log)

    run_checkpoint_api_tests(cwd, log)

    return 0


if __name__ == "__main__":
    sys.exit(main())
