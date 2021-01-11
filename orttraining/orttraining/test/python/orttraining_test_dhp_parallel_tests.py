#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import logging
import os
import sys
import torch
from _test_commons import run_subprocess

logging.basicConfig(
    format="%(asctime)s %(name)s [%(levelname)s] - %(message)s",
    level=logging.DEBUG)
log = logging.getLogger('DistributedTests')

# This function should be used to call all DxHxP parallel test scripts.
def main():
    ngpus = torch.cuda.device_count()

    # Declare test scripts for parallel tests.
    # New test scripts should be added to "dhp_parallel" folder.
    distributed_test_files = [os.path.join('dhp_parallel',
                                           'orttraining_test_parallel_train_simple_model.py'),
                              os.path.join('dhp_parallel',
                                           'orttraining_test_parallel_train_simple_model_fp16.py')]
    # parallel_test_process_number[i] is the number of processes needed to run distributed_test_files[i].
    distributed_test_process_counts = [4, 4]

    log.info('Running parallel training tests.')
    for test_file, process_count in zip(distributed_test_files, distributed_test_process_counts):
        if ngpus < process_count:
            log.error(
                'Machine Configuration Error. More GPUs are needed to run ' + test_file)
            return 1
        log.debug('RUN: ' + test_file)

        command = ['mpirun', '-n',
                   str(process_count), sys.executable, test_file]

        # The current working directory is set in
        # onnxruntime/orttraining/orttraining/test/python/orttraining_distributed_tests.py
        run_subprocess(command, cwd=os.getcwd()).check_returncode()

    return 0


if __name__ == '__main__':
    sys.exit(main())
