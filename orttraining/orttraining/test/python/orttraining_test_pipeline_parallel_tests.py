#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import sys
import torch
import logging
from _test_commons import _distributed_run

logging.basicConfig(
    format="%(asctime)s %(name)s [%(levelname)s] - %(message)s",
    level=logging.DEBUG)
log = logging.getLogger('DistributedTests')

# This function should be used to call all DxHxP parallel test scripts.
def main():
    ngpus = torch.cuda.device_count()

    # Test scripts for parallel tests.
    distributed_test_files = ['./orttraining_test_parallel_train_simple_model.py']
    # parallel_test_process_number[i] is the number of processes needed to run distributed_test_files[i].
    distributed_test_process_counts = [4]

    log.info('Running parallel training tests.')
    for test_file, process_count in zip(distributed_test_files, distributed_test_process_counts):
        if ngpus < process_count:
            log.info('SKIP: ' + test_file)
            continue
        log.debug('RUN: ' + test_file)
        _distributed_run(test_file, None, None, process_count)

    return 0


if __name__ == '__main__':
    sys.exit(main())
