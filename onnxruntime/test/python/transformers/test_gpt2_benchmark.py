#!/usr/bin/env python
# coding: utf-8
# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

import logging
import os
import unittest

import coloredlogs
import pytest
from parity_utilities import find_transformers_source

if find_transformers_source(sub_dir_paths=["models", "gpt2"]):
    from benchmark_gpt2 import main, parse_arguments
else:
    from onnxruntime.transformers.models.gpt2.benchmark_gpt2 import main, parse_arguments


class TestGpt2(unittest.TestCase):
    def setUp(self):
        from onnxruntime import get_available_providers

        self.test_cuda = "CUDAExecutionProvider" in get_available_providers()

    def run_benchmark_gpt2(self, arguments: str):
        args = parse_arguments(arguments.split())
        csv_filename = main(args)
        self.assertIsNotNone(csv_filename)
        self.assertTrue(os.path.exists(csv_filename))

    @pytest.mark.slow
    def test_gpt2_stage1(self):
        self.run_benchmark_gpt2("-m gpt2 --precision fp32 -v -b 1 --sequence_lengths 8 -s 0 --stage 1")

    @pytest.mark.slow
    def test_gpt2_fp32(self):
        self.run_benchmark_gpt2("-m gpt2 --precision fp32 -v -b 1 --sequence_lengths 2 -s 3")

    @pytest.mark.slow
    def test_gpt2_fp16(self):
        if self.test_cuda:
            self.run_benchmark_gpt2("-m gpt2 --precision fp16 -o -b 1 --sequence_lengths 2 -s 3 --use_gpu")

    @pytest.mark.slow
    def test_gpt2_int8(self):
        self.run_benchmark_gpt2("-m gpt2 --precision int8 -o  -b 1 --sequence_lengths 2 -s 3")


if __name__ == "__main__":
    coloredlogs.install(fmt="%(message)s")
    logging.getLogger("transformers").setLevel(logging.ERROR)
    unittest.main()
