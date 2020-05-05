#!/usr/bin/env python3

# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import argparse
import os
import subprocess
import sys

SCRIPT_DIR = os.path.realpath(os.path.dirname(__file__))
REPO_DIR = os.path.normpath(os.path.join(SCRIPT_DIR, "..", ".."))
PYTHON_TEST_DIR = os.path.join(REPO_DIR, "onnxruntime", "test", "python")

def run_python_tests_with_wheel(wheel_path):
    if not os.path.exists(wheel_path):
        raise ValueError("Wheel file not found: {}".format(wheel_path))

    def run(*args, check=True):
        print("Running command:", args)
        return subprocess.run([*args], check=check)

    def pip(*args):
        run(sys.executable, "-m", "pip", *args)

    # install wheel
    pip("install", wheel_path)

    training_enabled = \
        run(sys.executable,
            "-c", "from onnxruntime import TrainingSession",
            check=False).returncode == 0

    # install requirements
    pip("install", "pytest")

    if training_enabled:
        pip("install", "--pre", "torch", "torchvision",
            "-f", "https://download.pytorch.org/whl/nightly/cu101/torch_nightly.html")

    # run tests
    test_paths = [
        os.path.join(PYTHON_TEST_DIR, "onnxruntime_test_python.py"),
    ]

    if training_enabled:
        test_paths += [
            os.path.join(PYTHON_TEST_DIR, "onnxruntime_test_training_unit_tests.py"),
            os.path.join(PYTHON_TEST_DIR, "onnxruntime_test_ort_trainer.py"),
        ]

    for test_path in test_paths:
        run(sys.executable, "-m", "pytest", test_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Installs the specified onnxruntime wheel and test dependencies and runs Python tests.")
    parser.add_argument("--wheel-path", required=True, help="Path to the wheel file.")

    args = parser.parse_args()

    run_python_tests_with_wheel(args.wheel_path)
