#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
import os
import sys

from _test_commons import _distributed_run

allreduce_adasum_execution_file = "orttraining_test_allreduce_adasum.py"

_distributed_run(allreduce_adasum_execution_file, "test_single_precision_adasum_on_gpu")
_distributed_run(allreduce_adasum_execution_file, "test_half_precision_adasum_on_gpu")
