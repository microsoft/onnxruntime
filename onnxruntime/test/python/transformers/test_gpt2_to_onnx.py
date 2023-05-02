# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

import logging
import unittest

import coloredlogs
import pytest
from parity_utilities import find_transformers_source

from onnxruntime import get_available_providers

if find_transformers_source(sub_dir_paths=["models", "gpt2"]):
    from models.gpt2.convert_to_onnx import main as gpt2_to_onnx
else:
    from onnxruntime.transformers.models.gpt2.convert_to_onnx import main as gpt2_to_onnx


class TestGpt2ConvertToOnnx(unittest.TestCase):
    """Test convert_to_onnx.py of GPT-2 model."""

    def setUp(self):
        self.test_cuda = "CUDAExecutionProvider" in get_available_providers()

    def run_gpt2_to_onnx(self, arguments: str, stage: int):
        result = gpt2_to_onnx(arguments.split())
        self.assertIsNotNone(result)
        self.assertTrue(isinstance(result, dict))
        self.assertIn("top1_match_rate", result)
        self.assertGreater(result["top1_match_rate"], 0.98)
        self.assertIn("stage", result)
        self.assertEqual(result["stage"], stage)
        if stage == 1:
            self.assertIn("average_latency(batch_size=8,sequence_length=32,past_sequence_length=0)", result)
        else:
            self.assertIn("average_latency(batch_size=8,sequence_length=1,past_sequence_length=32)", result)

    @pytest.mark.slow
    def test_stage1(self):
        """Test stage 1 onnx model"""
        if self.test_cuda:
            self.run_gpt2_to_onnx("-m distilgpt2 -p fp32 --use_gpu -s 1 -t 100 --overwrite", 1)

    @pytest.mark.slow
    def test_stage2(self):
        """Test stage 2 onnx model"""
        if self.test_cuda:
            self.run_gpt2_to_onnx("-m distilgpt2 -p fp32 --use_gpu -s 2 -t 100 --overwrite", 2)

    @pytest.mark.slow
    def test_auto_mixed_precision(self):
        """Test mixed preicison onnx model"""
        if self.test_cuda:
            self.run_gpt2_to_onnx("-m distilgpt2 -p fp32 --use_gpu -p fp16 -o -a -t 100 --overwrite", 0)


if __name__ == "__main__":
    coloredlogs.install(fmt="%(message)s")
    logging.getLogger("transformers").setLevel(logging.ERROR)
    unittest.main()
