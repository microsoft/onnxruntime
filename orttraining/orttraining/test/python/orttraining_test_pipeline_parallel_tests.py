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
log = logging.getLogger("Build")


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cwd", help="cwd")
    return parser.parse_args()


def main():
    import torch
    ngpus = torch.cuda.device_count()

    # TODO: currently the CI machine only has 4 GPUs for parallel tests.
    # Fill in more pipeline partition options when the machine has different GPUs counts.
    if ngpus != 4:
        return 0

    log.info("Running distributed computation tests.")

    command = ['mpirun',  '-n', '4', 'python', './orttraining_test_parallel_train_simple_model.py',
               '-s', '10',
               '-n', '2']

    args = parse_arguments()
    cwd = args.cwd
    log.debug('RUN: ' + ' '.join(command))
    run_subprocess(command, cwd=cwd, log=log)

    return 0


if __name__ == "__main__":
    sys.exit(main())
