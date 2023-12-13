#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import argparse
import logging
import sys

from _test_commons import run_subprocess

logging.basicConfig(format="%(asctime)s %(name)s [%(levelname)s] - %(message)s", level=logging.DEBUG)
log = logging.getLogger("ORTModuleDistributedTests")


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cwd", help="Path to the current working directory")
    parser.add_argument("--mnist", help="Path to the mnist data directory", type=str, default=None)
    return parser.parse_args()


def run_ortmodule_deepspeed_zero_stage_1_tests(cwd, log, data_dir):
    log.debug("Running: ORTModule deepspeed zero stage 1 tests")

    command = [
        "deepspeed",
        "orttraining_test_ortmodule_deepspeed_zero_stage_1.py",
        "--deepspeed_config",
        "orttraining_test_ortmodule_deepspeed_zero_stage_1_config.json",
    ]

    if data_dir:
        command.extend(["--data-dir", data_dir])

    run_subprocess(command, cwd=cwd, log=log).check_returncode()


def run_pytorch_ddp_tests(cwd, log):
    log.debug("Running: ORTModule Pytorch DDP tests")

    command = [sys.executable, "orttraining_test_ortmodule_pytorch_ddp.py", "--use_ort_module"]

    run_subprocess(command, cwd=cwd, log=log).check_returncode()


def run_ortmodule_deepspeed_pipeline_parallel_tests(cwd, log):
    log.debug("Running: ORTModule deepspeed pipeline parallel tests")

    command = [
        "deepspeed",
        "orttraining_test_ortmodule_deepspeed_pipeline_parallel.py",
        "--deepspeed_config",
        "orttraining_test_ortmodule_deepspeed_pipeline_parallel_config.json",
    ]

    run_subprocess(command, cwd=cwd, log=log).check_returncode()


def run_ortmodule_fairscale_sharded_optimizer_tests(cwd, log, data_dir):
    log.debug("Running: ORTModule fairscale sharded optimizer tests")
    command = [
        "python3",
        "orttraining_test_ortmodule_fairscale_sharded_optimizer.py",
        "--use_sharded_optimizer",
        "--use_ortmodule",
    ]
    if data_dir:
        command.extend(["--data-dir", data_dir])

    run_subprocess(command, cwd=cwd, log=log).check_returncode()


def run_distributed_cache_test(cwd, log):
    log.debug("Running: ORTModule Cache Test")

    command = [
        "torchrun",
        "--nproc_per_node",
        "2",
        "orttraining_test_ortmodule_cache.py",
    ]

    run_subprocess(command, cwd=cwd, log=log).check_returncode()


def main():
    args = parse_arguments()
    cwd = args.cwd

    log.info("Running ortmodule tests pipeline")

    run_pytorch_ddp_tests(cwd, log)

    run_ortmodule_deepspeed_zero_stage_1_tests(cwd, log, args.mnist)

    run_ortmodule_deepspeed_pipeline_parallel_tests(cwd, log)
    run_ortmodule_fairscale_sharded_optimizer_tests(cwd, log, args.mnist)

    run_distributed_cache_test(cwd, log)
    return 0


if __name__ == "__main__":
    sys.exit(main())
