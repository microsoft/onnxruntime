#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import argparse
import logging
import os
import sys

from _test_commons import run_subprocess

logging.basicConfig(format="%(asctime)s %(name)s [%(levelname)s] - %(message)s", level=logging.DEBUG)
log = logging.getLogger("ORTModuleTests")


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cwd", help="Path to the current working directory")
    parser.add_argument("--mnist", help="Path to the mnist data directory", type=str, default=None)
    parser.add_argument("--bert_data", help="Path to the bert data directory", type=str, default=None)
    parser.add_argument(
        "--transformers_cache", help="Path to the transformers model cache directory", type=str, default=None
    )
    return parser.parse_args()


def get_env_with_transformers_cache(transformers_cache):
    return {"TRANSFORMERS_CACHE": transformers_cache} if transformers_cache else {}


def run_ortmodule_api_tests(cwd, log, transformers_cache):
    log.debug("Running: ORTModule-API tests")

    env = get_env_with_transformers_cache(transformers_cache)

    command = [sys.executable, "-m", "pytest", "-sv", "orttraining_test_ortmodule_api.py"]

    run_subprocess(command, cwd=cwd, log=log, env=env).check_returncode()


def run_ortmodule_ops_tests(cwd, log, transformers_cache):
    log.debug("Running: ORTModule-OPS tests")

    env = get_env_with_transformers_cache(transformers_cache)

    command = [sys.executable, "-m", "pytest", "-sv", "orttraining_test_ortmodule_onnx_ops.py"]

    run_subprocess(command, cwd=cwd, log=log, env=env).check_returncode()


def run_ortmodule_fallback_tests(cwd, log, transformers_cache):
    log.debug("Running: ORTModule-API tests")

    env = get_env_with_transformers_cache(transformers_cache)

    command = [sys.executable, "-m", "pytest", "-sv", "orttraining_test_ortmodule_fallback.py"]

    run_subprocess(command, cwd=cwd, log=log, env=env).check_returncode()


def run_ortmodule_poc_net(cwd, log, no_cuda, data_dir):
    log.debug(f"Running: ORTModule POCNet for MNIST with --no-cuda arg {no_cuda}.")

    command = [sys.executable, "orttraining_test_ortmodule_poc.py"]
    if no_cuda:
        command.extend(["--no-cuda", "--epochs", str(3)])

    if data_dir:
        command.extend(["--data-dir", data_dir])

    run_subprocess(command, cwd=cwd, log=log).check_returncode()


def run_ortmodule_torch_lightning(cwd, log, data_dir):
    log.debug("Running: ORTModule PyTorch Lightning sample .")

    command = [
        sys.executable,
        "orttraining_test_ortmodule_torch_lightning_basic.py",
        "--train-steps=470",
        "--epochs=2",
        "--batch-size=256",
    ]

    if data_dir:
        command.extend(["--data-dir", data_dir])

    run_subprocess(command, cwd=cwd, log=log).check_returncode()


def run_ortmodule_hf_bert_for_sequence_classification_from_pretrained(cwd, log, no_cuda, data_dir, transformers_cache):
    log.debug(f"Running: ORTModule HuggingFace BERT for sequence classification with --no-cuda arg {no_cuda}.")

    env = get_env_with_transformers_cache(transformers_cache)

    command = [sys.executable, "orttraining_test_ortmodule_bert_classifier.py"]
    if no_cuda:
        command.extend(["--no-cuda", "--epochs", str(3)])

    if data_dir:
        command.extend(["--data-dir", data_dir])

    run_subprocess(command, cwd=cwd, log=log, env=env).check_returncode()


def run_ortmodule_custom_autograd_tests(cwd, log):
    log.debug("Running: ORTModule-Custom AutoGrad Functions tests")

    command = [sys.executable, "-m", "pytest", "-sv", "orttraining_test_ortmodule_autograd.py"]

    run_subprocess(command, cwd=cwd, log=log).check_returncode()


def run_ortmodule_hierarchical_ortmodule_tests(cwd, log):
    log.debug("Running: ORTModule-Hierarchical model tests")

    command = [sys.executable, "-m", "pytest", "-sv", "orttraining_test_hierarchical_ortmodule.py"]

    run_subprocess(command, cwd=cwd, log=log).check_returncode()


def run_ortmodule_experimental_json_config_tests(cwd, log):
    log.debug("Running: ORTModule Experimental Load Config tests")

    command = [sys.executable, "-m", "pytest", "-sv", "orttraining_test_ortmodule_experimental_json_config.py"]

    run_subprocess(command, cwd=cwd, log=log).check_returncode()


def run_experimental_gradient_graph_tests(cwd, log):
    log.debug("Running: Experimental Gradient Graph Export Tests")

    command = [sys.executable, "-m", "pytest", "-sv", "orttraining_test_experimental_gradient_graph.py"]

    run_subprocess(command, cwd=cwd, log=log).check_returncode()


def run_data_sampler_tests(cwd, log):
    log.debug("Running: Data sampler tests")

    command = [sys.executable, "-m", "pytest", "-sv", "orttraining_test_sampler.py"]

    run_subprocess(command, cwd=cwd, log=log).check_returncode()


def run_hooks_tests(cwd, log):
    log.debug("Running: Data hooks tests")

    command = [sys.executable, "-m", "pytest", "-sv", "orttraining_test_ortmodule_hooks.py"]

    run_subprocess(command, cwd=cwd, log=log).check_returncode()


def run_utils_tests(cwd, log):
    log.debug("Running: Utils tests")

    command = [sys.executable, "-m", "pytest", "-sv", "orttraining_test_utilities.py"]

    run_subprocess(command, cwd=cwd, log=log).check_returncode()


def run_pytorch_export_contrib_ops_tests(cwd, log):
    log.debug("Running: PyTorch Export Contrib Ops Tests")

    command = [sys.executable, "-m", "pytest", "-sv", "test_pytorch_export_contrib_ops.py"]

    run_subprocess(command, cwd=cwd, log=log).check_returncode()


def main():
    args = parse_arguments()
    cwd = args.cwd

    log.info("Running ortmodule tests pipeline")

    run_ortmodule_api_tests(cwd, log, transformers_cache=args.transformers_cache)

    run_ortmodule_ops_tests(cwd, log, transformers_cache=args.transformers_cache)

    run_ortmodule_poc_net(cwd, log, no_cuda=False, data_dir=args.mnist)

    if os.getenv("ORTMODULE_DISABLE_CPU_TRAINING_TEST", "0") != "1":
        run_ortmodule_poc_net(cwd, log, no_cuda=True, data_dir=args.mnist)

    run_ortmodule_hf_bert_for_sequence_classification_from_pretrained(
        cwd, log, no_cuda=False, data_dir=args.bert_data, transformers_cache=args.transformers_cache
    )

    if os.getenv("ORTMODULE_DISABLE_CPU_TRAINING_TEST", "0") != "1":
        run_ortmodule_hf_bert_for_sequence_classification_from_pretrained(
            cwd, log, no_cuda=True, data_dir=args.bert_data, transformers_cache=args.transformers_cache
        )

    run_ortmodule_torch_lightning(cwd, log, args.mnist)

    run_ortmodule_custom_autograd_tests(cwd, log)

    run_ortmodule_experimental_json_config_tests(cwd, log)

    run_ortmodule_fallback_tests(cwd, log, args.transformers_cache)

    run_ortmodule_hierarchical_ortmodule_tests(cwd, log)

    run_data_sampler_tests(cwd, log)

    run_hooks_tests(cwd, log)

    run_utils_tests(cwd, log)

    run_experimental_gradient_graph_tests(cwd, log)

    # TODO(bmeswani): Enable this test once it can run with latest pytorch
    # run_pytorch_export_contrib_ops_tests(cwd, log)

    return 0


if __name__ == "__main__":
    sys.exit(main())
