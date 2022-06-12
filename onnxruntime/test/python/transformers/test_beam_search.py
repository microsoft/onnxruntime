#!/usr/bin/env python
# coding: utf-8
# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

import os
import unittest

import pytest
import torch
from parity_utilities import find_transformers_source

from onnxruntime import get_available_providers

if find_transformers_source() and find_transformers_source(["models", "t5"]):
    from benchmark_helper import Precision
    from convert_beam_search import main as run
    from models.t5.convert_to_onnx import export_onnx_models as export_t5_onnx_models
else:
    from onnxruntime.transformers.benchmark_helper import Precision
    from onnxruntime.transformers.convert_beam_search import main as run
    from onnxruntime.transformers.models.t5.convert_to_onnx import export_onnx_models as export_t5_onnx_models


class TestBeamSearchGpt(unittest.TestCase):
    """Test BeamSearch for GPT-2 model"""

    def setUp(self):
        self.model_name = "gpt2"
        self.gpt2_onnx_path = os.path.join(".", "onnx_models", "gpt2_past_fp32_shape.onnx")
        self.beam_search_onnx_path = os.path.join(".", "onnx_models", "gpt2_beam_search.onnx")
        self.default_arguments = [
            f"-m {self.model_name}",
            f"--decoder_onnx {self.gpt2_onnx_path}",
            f"--output {self.beam_search_onnx_path}",
            "--output_sequences_score",
            "--repetition_penalty 2.0",
        ]
        self.sentences = [
            "The product is released",
            "I enjoy walking in the park",
            "Test best way to invest",
        ]
        self.enable_cuda = torch.cuda.is_available() and "CUDAExecutionProvider" in get_available_providers()
        self.remove_onnx_files()

    def tearDown(self):
        self.remove_onnx_files()

    def remove_onnx_files(self):
        if os.path.exists(self.gpt2_onnx_path):
            os.remove(self.gpt2_onnx_path)

        if os.path.exists(self.beam_search_onnx_path):
            os.remove(self.beam_search_onnx_path)

    def run_beam_search(self, extra_arguments: str, sentences=None):
        arguments = " ".join(self.default_arguments + [extra_arguments]).split()

        # Test CPU
        result = run(arguments, sentences=self.sentences if sentences is None else sentences)
        self.assertTrue(result["parity"], f"ORT and PyTorch result is different on CPU for arguments {arguments}")

        # Test GPU
        if self.enable_cuda:
            if "--use_gpu" not in arguments:
                arguments.append("--use_gpu")
            result = run(arguments, sentences=self.sentences if sentences is None else sentences)
            self.assertTrue(result["parity"], f"ORT and PyTorch result is different on GPU for arguments {arguments}")

        os.remove(self.beam_search_onnx_path)

    @pytest.mark.slow
    def test_return_sequences(self):
        for return_sequences in [1, 2]:
            self.run_beam_search(f"--num_return_sequences {return_sequences}")

    @pytest.mark.slow
    def test_early_stopping(self):
        self.run_beam_search("--early_stopping")

    # @pytest.mark.slow
    # def test_temperature(self):
    #     self.run_beam_search("--temperature 0.5")

    @pytest.mark.slow
    def test_length_penalty(self):
        for length_penalty in [0.5, 2.0]:
            self.run_beam_search(f"--length_penalty {length_penalty}")

    @pytest.mark.slow
    def test_no_repeat_ngram(self):
        for ngram_size in [1, 2]:
            self.run_beam_search(f"--no_repeat_ngram_size {ngram_size}")


class TestBeamSearchT5(unittest.TestCase):
    """Test BeamSearch for T5 model"""

    def setUp(self):
        self.model_name = "t5-small"
        self.decoder_onnx_path = os.path.join(".", "onnx_models", "t5-small_decoder.onnx")
        self.encoder_onnx_path = os.path.join(".", "onnx_models", "t5-small_encoder_decoder_init.onnx")
        self.beam_search_onnx_path = os.path.join(".", "onnx_models", "t5_small_beam_search.onnx")
        self.default_arguments = [
            f"-m {self.model_name}",
            "--model_type t5",
            f"--decoder_onnx {self.decoder_onnx_path}",
            f"--encoder_decoder_init_onnx {self.encoder_onnx_path}",
            f"--output {self.beam_search_onnx_path}",
            "--output_sequences_score",
            "--repetition_penalty 2.0",
        ]

        self.enable_cuda = torch.cuda.is_available() and "CUDAExecutionProvider" in get_available_providers()

        export_t5_onnx_models(
            self.model_name,
            os.path.join(".", "cache_models"),
            os.path.join(".", "onnx_models"),
            use_gpu=False,
            use_external_data_format=False,
            optimize_onnx=False,
            precision=Precision.FLOAT32,
            verbose=False,
            use_decoder_start_token=False,
            merge_encoder_and_decoder_init=True,
            overwrite=True,
            disable_auto_mixed_precision=False,
            use_int32_inputs=True,
        )

        self.sentences = [
            "translate English to French: The product is released",
            "summarize: research continues to show that pets bring real health benefits to their owners."
            + "Having a dog around can lead to lower levels of stress for both adults and kids.",
        ]

        if os.path.exists(self.beam_search_onnx_path):
            os.remove(self.beam_search_onnx_path)

    def tearDown(self):
        self.remove_onnx_files()

    def remove_onnx_files(self):
        if os.path.exists(self.beam_search_onnx_path):
            os.remove(self.beam_search_onnx_path)

        if os.path.exists(self.decoder_onnx_path):
            os.remove(self.decoder_onnx_path)

        if os.path.exists(self.encoder_onnx_path):
            os.remove(self.encoder_onnx_path)

    def run_beam_search(self, extra_arguments: str, sentences=None):
        arguments = " ".join(self.default_arguments + [extra_arguments]).split()

        # Test CPU
        result = run(arguments, sentences=self.sentences if sentences is None else sentences)
        self.assertTrue(result["parity"], f"ORT and PyTorch result is different on CPU for arguments {arguments}")

        # Test GPU
        if self.enable_cuda:
            if "--use_gpu" not in arguments:
                arguments.append("--use_gpu")
            result = run(arguments, sentences=self.sentences if sentences is None else sentences)
            self.assertTrue(result["parity"], f"ORT and PyTorch result is different on GPU for arguments {arguments}")

        os.remove(self.beam_search_onnx_path)

    @pytest.mark.slow
    def test_return_sequences(self):
        for return_sequences in [1, 2]:
            self.run_beam_search(f"--num_return_sequences {return_sequences}")

    @pytest.mark.slow
    def test_early_stopping(self):
        self.run_beam_search("--early_stopping")

    # @pytest.mark.slow
    # def test_temperature(self):
    #     self.run_beam_search("--temperature 0.5")

    @pytest.mark.slow
    def test_length_penalty(self):
        for length_penalty in [0.5, 2.0]:
            self.run_beam_search(f"--length_penalty {length_penalty}")

    @pytest.mark.slow
    def test_no_repeat_ngram(self):
        for ngram_size in [1, 2]:
            self.run_beam_search(f"--no_repeat_ngram_size {ngram_size}")


if __name__ == "__main__":
    unittest.main()
