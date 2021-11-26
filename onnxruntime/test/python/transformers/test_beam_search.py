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
import pytest

class TestBeamSearch(unittest.TestCase):
    def setUp(self):
        from onnxruntime import get_available_providers
        self.test_cuda = 'CUDAExecutionProvider' in get_available_providers()

    def run_beam_search(self, arguments: str):
        from onnxruntime.transformers.convert_beam_search import main as run
        return run(arguments.split())

    @pytest.mark.slow
    def test_profiler_cpu(self):
        gpt2_onnx_path = os.path.join('.', 'onnx_models', 'gpt2_past_fp32_shape.onnx')
        beam_search_onnx_path = os.path.join('.', 'onnx_models', 'gpt2_beam_search_v1.onnx')
        result = self.run_beam_search(f'-m gpt2 --gpt2_onnx {gpt2_onnx_path} --output {beam_search_onnx_path} --output_sequences_score --repetition_penalty 2.0 --run_baseline')
        self.assertTrue(result, "ORT and PyTorch is expected to have same result, but current result is different")        

if __name__ == '__main__':
    unittest.main()
