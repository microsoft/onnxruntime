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
log = logging.getLogger("ORTModuleTests")


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cwd", help="Path to the current working directory")
    parser.add_argument("--mnist", help="Path to the mnist data directory", type=str, default=None)
    parser.add_argument("--bert_data", help="Path to the bert data directory", type=str, default=None)
    return parser.parse_args()


def run_ortmodule_api_tests(cwd, log):
    log.debug('Running: ORTModule-API tests')

    command = [sys.executable, '-m', 'pytest', '-sv', 'orttraining_test_ortmodule_api.py']

    run_subprocess(command, cwd=cwd, log=log).check_returncode()


def run_ortmodule_poc_net(cwd, log, no_cuda, data_dir):
    log.debug('Running: ORTModule POCNet for MNIST with --no-cuda arg {}.'.format(no_cuda))

    command = [sys.executable, 'orttraining_test_ortmodule_poc.py']
    if no_cuda:
        command.extend(['--no-cuda', '--epochs', str(3)])

    if data_dir:
        command.extend(['--data_dir', data_dir])

    run_subprocess(command, cwd=cwd, log=log).check_returncode()


def run_ort_module_hf_bert_for_sequence_classification_from_pretrained(cwd, log, no_cuda, data_dir):
    log.debug('Running: ORTModule HuggingFace BERT for sequence classification with --no-cuda arg {}.'.format(no_cuda))

    command = [sys.executable, 'orttraining_test_ortmodule_bert_classifier.py']
    if no_cuda:
        command.extend(['--no-cuda', '--epochs', str(3)])

    if data_dir:
        command.extend(['--data_dir', data_dir])

    run_subprocess(command, cwd=cwd, log=log).check_returncode()


def main():
    args = parse_arguments()
    cwd = args.cwd

    log.info("Running ortmodule tests pipeline")

    run_ortmodule_api_tests(cwd, log)

    run_ortmodule_poc_net(cwd, log, no_cuda=False, data_dir=args.mnist)

    run_ortmodule_poc_net(cwd, log, no_cuda=True, data_dir=args.mnist)

    run_ort_module_hf_bert_for_sequence_classification_from_pretrained(cwd, log, no_cuda=False, data_dir=args.bert_data)

    run_ort_module_hf_bert_for_sequence_classification_from_pretrained(cwd, log, no_cuda=True, data_dir=args.bert_data)

    return 0


if __name__ == "__main__":
    sys.exit(main())
