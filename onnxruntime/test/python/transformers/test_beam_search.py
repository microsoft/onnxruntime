#!/usr/bin/env python
# coding: utf-8
# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

import unittest
import os
import pytest

from parity_utilities import find_transformers_source
if find_transformers_source():
    from convert_beam_search import main as run
else:
    from onnxruntime.transformers.convert_beam_search import main as run


class TestBeamSearch(unittest.TestCase):
    def setUp(self):
        #TODO: use a smaller model and enable tests in CI pipeline
        self.model_name = "gpt2"
        self.gpt2_onnx_path = os.path.join('.', 'onnx_models', 'gpt2_past_fp32_shape.onnx')
        self.beam_search_onnx_path = os.path.join('.', 'onnx_models', 'gpt2_beam_search.onnx')
        self.cpu_params = f'-m {self.model_name} --gpt2_onnx {self.gpt2_onnx_path} --output {self.beam_search_onnx_path} --output_sequences_score --repetition_penalty 2.0'

    def run_beam_search(self, arguments: str, sentences=None):
        return run(arguments.split(), sentences=sentences)

    @pytest.mark.slow
    def test_cpu(self):
        result = self.run_beam_search(self.cpu_params + " --num_return_sequences 2",
                                      sentences=["The product is released"])
        os.remove(self.gpt2_onnx_path)
        os.remove(self.beam_search_onnx_path)
        self.assertTrue(result["parity"], "ORT and PyTorch result is different")

    @pytest.mark.slow
    def test_early_stopping(self):
        result = self.run_beam_search(self.cpu_params + " --early_stopping")
        os.remove(self.gpt2_onnx_path)
        os.remove(self.beam_search_onnx_path)
        self.assertTrue(result["parity"], "ORT and PyTorch result is different")

    @pytest.mark.slow
    def test_temperature(self):
        result = self.run_beam_search(self.cpu_params + " --temperature 0.5")
        os.remove(self.gpt2_onnx_path)
        os.remove(self.beam_search_onnx_path)
        self.assertTrue(result["parity"], "ORT and PyTorch result is different")

    @pytest.mark.slow
    def test_length_penalty(self):
        result = self.run_beam_search(self.cpu_params + " --length_penalty 0.5")
        os.remove(self.gpt2_onnx_path)
        os.remove(self.beam_search_onnx_path)
        self.assertTrue(result["parity"], "ORT and PyTorch result is different")

    @pytest.mark.slow
    def test_no_repeat_ngram(self):
        for ngram_size in [1, 2]:
            result = self.run_beam_search(self.cpu_params + f' --no_repeat_ngram_size {ngram_size}')
            os.remove(self.gpt2_onnx_path)
            os.remove(self.beam_search_onnx_path)
            self.assertTrue(result["parity"], "ORT and PyTorch result is different")


if __name__ == '__main__':
    unittest.main()
