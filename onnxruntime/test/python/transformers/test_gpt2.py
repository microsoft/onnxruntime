#!/usr/bin/env python
# coding: utf-8
# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

import unittest
import os
import logging
import coloredlogs
import pytest

from parity_utilities import find_transformers_source
if find_transformers_source():
    from benchmark_gpt2 import parse_arguments, main
else:
    from onnxruntime.transformers.benchmark_gpt2 import parse_arguments, main


class TestGpt2(unittest.TestCase):
    def setUp(self):
        from onnxruntime import get_available_providers
        self.test_cuda = 'CUDAExecutionProvider' in get_available_providers()

    def run_benchmark_gpt2(self, arguments: str):
        args = parse_arguments(arguments.split())
        csv_filename = main(args)
        self.assertTrue(os.path.exists(csv_filename))

    @pytest.mark.slow
    def test_gpt2_fp32(self):
        self.run_benchmark_gpt2('-m gpt2 --precision fp32 -v -b 1 -s 128')

    @pytest.mark.slow
    def test_gpt2_fp16(self):
        if self.test_cuda:
            self.run_benchmark_gpt2('-m gpt2 --precision fp16 -o -b 1 -s 128 --use_gpu')

    @pytest.mark.slow
    def test_gpt2_int8(self):
        self.run_benchmark_gpt2('-m gpt2 --precision int8 -o -b 1 -s 128')

    @pytest.mark.slow
    def test_gpt2_beam_search_step_fp32(self):
        self.run_benchmark_gpt2('-m gpt2 --model_class=GPT2LMHeadModel_BeamSearchStep --precision fp32 -v -b 1 -s 128')

    @pytest.mark.slow
    def test_gpt2_beam_search_step_fp16(self):
        if self.test_cuda:
            self.run_benchmark_gpt2(
                '-m gpt2 --model_class=GPT2LMHeadModel_BeamSearchStep --precision fp16 -o -b 1 -s 128 --use_gpu')

    @pytest.mark.slow
    def test_gpt2_beam_search_step_int8(self):
        self.run_benchmark_gpt2('-m gpt2 --model_class=GPT2LMHeadModel_BeamSearchStep --precision int8 -o -b 1 -s 128')

    @pytest.mark.slow
    def test_gpt2_configurable_one_step_search_fp32(self):
        self.run_benchmark_gpt2(
            '-m gpt2 --model_class=GPT2LMHeadModel_ConfigurableOneStepSearch --precision fp32 -v -b 1 -s 128')

    @pytest.mark.slow
    def test_gpt2_configurable_one_step_search_fp16(self):
        if self.test_cuda:
            self.run_benchmark_gpt2(
                '-m gpt2 --model_class=GPT2LMHeadModel_ConfigurableOneStepSearch --precision fp16 -o -b 1 -s 128 --use_gpu'
            )

    @pytest.mark.slow
    def test_gpt2_configurable_one_step_search_int8(self):
        self.run_benchmark_gpt2(
            '-m gpt2 --model_class=GPT2LMHeadModel_ConfigurableOneStepSearch --precision int8 -o -b 1 -s 128')


if __name__ == '__main__':
    coloredlogs.install(fmt='%(message)s')
    logging.getLogger("transformers").setLevel(logging.ERROR)
    unittest.main()
