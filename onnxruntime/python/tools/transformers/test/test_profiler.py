#!/usr/bin/env python
# coding: utf-8
# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

# For live logging, use the command: pytest -o log_cli=true --log-cli-level=DEBUG

import unittest
import os
import onnx
import onnxruntime
import pytest

from test_optimizer import _get_test_model_path


class TestBertProfiler(unittest.TestCase):
    def run_profile(self, arguments: str):
        from profiler import parse_arguments, run
        args = parse_arguments(arguments.split())
        results = run(args)
        self.assertTrue(len(results) > 1)

    def test_profiler_gpu(self):
        input_model_path = _get_test_model_path('bert_keras_squad')
        if 'CUDAExecutionProvider' in onnxruntime.get_available_providers():
            self.run_profile(f'--model {input_model_path} --batch_size 1 --sequence_length 7 --use_gpu')

    def test_profiler_cpu(self):
        input_model_path = _get_test_model_path('bert_keras_squad')
        self.run_profile(f'--model {input_model_path} --batch_size 1 --sequence_length 7 --dummy_inputs default')


if __name__ == '__main__':
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

    unittest.main()
