#!/usr/bin/env python
# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

import os
import shutil
import unittest

import onnx
import pytest
import torch
from parity_utilities import find_transformers_source

from onnxruntime import get_available_providers

if find_transformers_source() and find_transformers_source(["models", "t5"]):
    from benchmark_helper import Precision
    from convert_generation import main as run
    from models.t5.convert_to_onnx import export_onnx_models as export_t5_onnx_models
    from models.whisper.convert_to_onnx import main as run_whisper
else:
    from onnxruntime.transformers.benchmark_helper import Precision
    from onnxruntime.transformers.convert_generation import main as run
    from onnxruntime.transformers.models.t5.convert_to_onnx import export_onnx_models as export_t5_onnx_models
    from onnxruntime.transformers.models.whisper.convert_to_onnx import main as run_whisper


def has_cuda_environment():
    return torch.cuda.is_available() and "CUDAExecutionProvider" in get_available_providers()


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
            "--repetition_penalty 2.0",
        ]
        self.sentences = [
            "The product is released",
            "I enjoy walking in the park",
            "Test best way to invest",
            # "The AI community building the future",
            # "The selloff in tech shares deepened",
            # "Abortion rights take center stage",
        ]
        self.enable_cuda = has_cuda_environment()
        self.remove_onnx_files()

    def tearDown(self):
        self.remove_onnx_files()

    def remove_onnx_files(self):
        if os.path.exists(self.gpt2_onnx_path):
            os.remove(self.gpt2_onnx_path)

        if os.path.exists(self.beam_search_onnx_path):
            os.remove(self.beam_search_onnx_path)

    def check_for_init_decoder_attr(self, model_path: str):
        init_decoder_found = False
        gpt2_beam_search_onnx_model = onnx.load(model_path)
        graph_proto = gpt2_beam_search_onnx_model.graph
        for node in graph_proto.node:
            if node.op_type == "BeamSearch" or node.op_type == "GreedySearch":
                for attr in node.attribute:
                    if attr.name == "init_decoder":
                        init_decoder_found = True
                        break

        self.assertTrue(init_decoder_found)

    def run_beam_search(self, extra_arguments: str, sentences=None, append_arguments=True, is_greedy=False):
        if append_arguments:
            arguments = " ".join([*self.default_arguments, extra_arguments]).split()
        else:
            arguments = extra_arguments.split()

        if is_greedy:
            arguments.extend(["--num_beams", "1", "--num_return_sequences", "1"])
        else:
            arguments.extend(["--output_sequences_score"])

        # Test CPU
        result = run(arguments, sentences=self.sentences if sentences is None else sentences)
        self.assertTrue(result["parity"], f"ORT and PyTorch result is different on CPU for arguments {arguments}")
        # (CPU) Check for the presence of the "init_decoder" attribute
        self.check_for_init_decoder_attr(self.beam_search_onnx_path)

        # Test GPU
        if self.enable_cuda:
            if "--use_gpu" not in arguments:
                arguments.append("--use_gpu")
            result = run(arguments, sentences=self.sentences if sentences is None else sentences)
            self.assertTrue(result["parity"], f"ORT and PyTorch result is different on GPU for arguments {arguments}")

            # (GPU) Check for the presence of the "init_decoder" attribute
            self.check_for_init_decoder_attr(self.beam_search_onnx_path)

        os.remove(self.beam_search_onnx_path)

    @pytest.mark.slow
    def test_return_sequences(self):
        for return_sequences in [1, 2]:
            self.run_beam_search(f"--num_return_sequences {return_sequences} --output_sequences_score")

    @pytest.mark.slow
    def test_early_stopping(self):
        self.run_beam_search("--early_stopping --output_sequences_score")

    @pytest.mark.slow
    def test_length_penalty(self):
        for length_penalty in [0.5, 2.0]:
            self.run_beam_search(f"--length_penalty {length_penalty} --output_sequences_score")

    @pytest.mark.slow
    def test_no_repeat_ngram(self):
        for ngram_size in [1, 2]:
            self.run_beam_search(f"--no_repeat_ngram_size {ngram_size} --output_sequences_score")

    @pytest.mark.slow
    def test_greedy_search(self):
        self.run_beam_search("", is_greedy=True)

    @pytest.mark.slow
    def test_greedy_search_past_present_share_buffer(self):
        if self.enable_cuda:
            self.run_beam_search("--past_present_share_buffer --use_gpu", is_greedy=True)

    @pytest.mark.slow
    def test_greedy_search_past_present_share_buffer_fp16(self):
        if self.enable_cuda:
            self.run_beam_search("--past_present_share_buffer --use_gpu -p fp16", is_greedy=True)

    @pytest.mark.slow
    def test_greedy_search_use_decoder_masked_self_attention(self):
        if self.enable_cuda:
            self.run_beam_search(
                "--past_present_share_buffer --use_decoder_masked_self_attention --use_gpu", is_greedy=True
            )

    @pytest.mark.slow
    def test_greedy_search_use_decoder_masked_self_attention_fp16(self):
        if self.enable_cuda:
            self.run_beam_search(
                "--past_present_share_buffer --use_decoder_masked_self_attention --use_gpu -p fp16", is_greedy=True
            )

    @pytest.mark.slow
    def test_greedy_search_float16(self):
        # TODO: investigate fp16 parity issue for greedy/beam search with repetition_penalty != 1.0
        if self.enable_cuda:
            self.run_beam_search("--repetition_penalty 1.0 --use_gpu -p fp16", is_greedy=True)

    @pytest.mark.slow
    def test_beam_search_use_decoder_masked_self_attention(self):
        if self.enable_cuda:
            self.run_beam_search("--past_present_share_buffer --use_decoder_masked_self_attention --use_gpu")

    @pytest.mark.slow
    def test_beam_search_use_decoder_masked_self_attention_fp16(self):
        if self.enable_cuda:
            self.run_beam_search("--past_present_share_buffer --use_decoder_masked_self_attention --use_gpu -p fp16")

    @pytest.mark.slow
    def test_external_data(self):
        self.run_beam_search(
            f"-m gpt2 --output_sequences_score -e --output {self.beam_search_onnx_path}",
            sentences=None,
            append_arguments=False,
        )


def get_tiny_t5_model_dir():
    """Get the path to the tiny T5 model directory."""
    # This function is used to get the path to the tiny T5 model directory.
    # It is used in the TestBeamSearchT5 and TestBeamSearchT5Fp16 classes.

    # Path relative to the build\Release directory, where transformers test is launched in pipeline.
    tiny_model_dir = os.path.join(
        "testdata",
        "transformers",
        "tiny_t5",
    )
    if os.path.exists(tiny_model_dir):
        return os.path.normpath(tiny_model_dir)

    # The path is relative to the current file's directory.
    tiny_model_dir = os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
        "testdata",
        "transformers",
        "tiny_t5",
    )
    return os.path.normpath(tiny_model_dir)


use_tiny_model = True


class TestBeamSearchT5(unittest.TestCase):
    """Test BeamSearch for T5 model with fp32 in CPU"""

    @classmethod
    def setUpClass(cls):
        tiny_model_dir = get_tiny_t5_model_dir()
        model_name = "tiny_t5" if use_tiny_model and os.path.exists(tiny_model_dir) else "t5-small"
        cls.model_name = tiny_model_dir if model_name == "tiny_t5" else "t5-small"
        cls.decoder_onnx_path = os.path.join(".", "t5_onnx_models", f"{model_name}_decoder.onnx")
        cls.encoder_onnx_path = os.path.join(".", "t5_onnx_models", f"{model_name}_encoder.onnx")
        cls.beam_search_onnx_path = os.path.join(".", "t5_onnx_models", f"{model_name}_beam_search.onnx")
        cls.default_arguments = [
            f"-m {cls.model_name}",
            "--model_type t5",
            f"--decoder_onnx {cls.decoder_onnx_path}",
            f"--encoder_decoder_init_onnx {cls.encoder_onnx_path}",
            f"--output {cls.beam_search_onnx_path}",
            "--output_sequences_score",
            "--repetition_penalty 2.0",
        ]

        # Remove onnx files if existed for any reason.
        cls.remove_onnx_files()

        # This is in class setup so that we only export t5 model once.
        paths = export_t5_onnx_models(
            cls.model_name,
            os.path.join(".", "cache_models"),
            os.path.join(".", "t5_onnx_models"),
            use_gpu=False,
            use_external_data_format=False,
            optimize_onnx=False,
            precision=Precision.FLOAT32.value,
            verbose=False,
            use_decoder_start_token=False,
            overwrite=True,
            disable_auto_mixed_precision=False,
            use_int32_inputs=True,
        )
        assert len(paths) == 2

        cls.sentences = [
            "translate English to French: The product is released",
            "summarize: research continues to show that pets bring real health benefits to their owners. Having a dog around can lead to lower levels of stress for both adults and kids.",
        ]

    @classmethod
    def remove_onnx_files(cls, beam_search_onnx_only: bool = False):
        if os.path.exists(cls.beam_search_onnx_path):
            os.remove(cls.beam_search_onnx_path)
        if os.path.exists(cls.beam_search_onnx_path + ".data"):
            os.remove(cls.beam_search_onnx_path + ".data")

        if not beam_search_onnx_only:
            if os.path.exists(cls.encoder_onnx_path):
                os.remove(cls.encoder_onnx_path)
            if os.path.exists(cls.decoder_onnx_path):
                os.remove(cls.decoder_onnx_path)

    @classmethod
    def tearDownClass(cls):
        # cls.remove_onnx_files()
        pass

    def setUp(self):
        pass

    def tearDown(self):
        # self.remove_onnx_files(beam_search_onnx_only=True)
        pass

    def run_beam_search(self, extra_arguments: str):
        arguments = " ".join([*self.default_arguments, extra_arguments]).split()

        # Test CPU
        result = run(arguments)
        self.assertTrue(result["parity"], f"ORT and PyTorch result is different on CPU for arguments {arguments}")

    def test_return_sequences(self):
        for return_sequences in [1, 2]:
            self.run_beam_search(f"--num_return_sequences {return_sequences}")

    def test_early_stopping(self):
        self.run_beam_search("--early_stopping")

    def test_length_penalty(self):
        for length_penalty in [0.5, 2.0]:
            self.run_beam_search(f"--length_penalty {length_penalty}")

    def test_no_repeat_ngram(self):
        for ngram_size in [1, 2]:
            self.run_beam_search(f"--no_repeat_ngram_size {ngram_size}")

    def test_custom_attention_mask(self):
        self.run_beam_search("--custom_attention_mask")

    def test_external_data(self):
        self.run_beam_search("-e")


@unittest.skipUnless(
    has_cuda_environment(),
    "skip since there is no cuda environment.",
)
class TestBeamSearchT5Fp16(unittest.TestCase):
    """Test BeamSearch for T5 model with fp16 in GPU"""

    @classmethod
    def setUpClass(cls):
        tiny_model_dir = get_tiny_t5_model_dir()
        tiny_model_dir = os.path.normpath(tiny_model_dir)
        cls.model_name = "tiny_t5" if use_tiny_model and os.path.exists(tiny_model_dir) else "t5-small"
        cls.model_id = tiny_model_dir if cls.model_name == "tiny_t5" else "t5-small"
        cls.beam_search_onnx_path = os.path.join(".", "onnx_models", f"{cls.model_name}_beam_search_fp16.onnx")
        cls.default_arguments = [
            f"-m {cls.model_id}",
            "--model_type t5",
            f"--output {cls.beam_search_onnx_path}",
            "--min_length 2",
            "--max_length 16",
            "--use_gpu",
            "-p fp16",
        ]

        cls.sentences = [
            "translate English to French: The product is released",
            "summarize: research continues to show that pets bring real health benefits to their owners. Having a dog around can lead to lower levels of stress for both adults and kids.",
        ]

        cls.remove_onnx_files()

    @classmethod
    def remove_onnx_files(cls):
        model_name = cls.model_name
        for file in [
            f"{model_name}_beam_search_fp16.onnx",
            f"{model_name}_encoder.onnx",
            f"{model_name}_encoder_fp16.onnx",
            f"{model_name}_decoder.onnx",
            f"{model_name}_decoder_fp16.onnx",
        ]:
            if os.path.exists(os.path.join(".", "onnx_models", file)):
                os.remove(os.path.join(".", "onnx_models", file))
            if os.path.exists(os.path.join(".", "onnx_models", file + ".data")):
                os.remove(os.path.join(".", "onnx_models", file + ".data"))

    def setUp(self):
        pass

    def tearDown(self):
        self.remove_onnx_files()

    def check_encoder_fusion(self):
        model_name = self.model_name
        onnx_path = os.path.join(".", "onnx_models", f"{model_name}_encoder_fp16.onnx")

        model = onnx.load_model(onnx_path, format=None, load_external_data=True)
        from onnxruntime.transformers.onnx_model import OnnxModel

        onnx_model = OnnxModel(model)
        op_counters = onnx_model.get_operator_statistics()
        print("encoder ops", op_counters)

        expected_node_count = {
            "RelativePositionBias": 1,
            "SimplifiedLayerNormalization": 5 if use_tiny_model else 13,
            "Attention": 2 if use_tiny_model else 6,
        }
        for key, value in expected_node_count.items():
            self.assertIn(key, op_counters, f"Expected {key} to be in op_counters")
            self.assertEqual(op_counters[key], value, f"Expected {key} to be {value}, but got {op_counters[key]}")

    def check_decoder_fusion(self):
        model_name = self.model_name
        onnx_path = os.path.join(".", "onnx_models", f"{model_name}_decoder_fp16.onnx")

        model = onnx.load_model(onnx_path, format=None, load_external_data=True)
        from onnxruntime.transformers.onnx_model import OnnxModel

        onnx_model = OnnxModel(model)
        op_counters = onnx_model.get_operator_statistics()
        print("decoder ops", op_counters)

        expected_node_count = {
            "RelativePositionBias": 1,
            "SimplifiedLayerNormalization": 7 if use_tiny_model else 19,
            "MultiHeadAttention": 4 if use_tiny_model else 12,
        }
        for key, value in expected_node_count.items():
            self.assertIn(key, op_counters, f"Expected {key} to be in op_counters")
            self.assertEqual(op_counters[key], value, f"Expected {key} to be {value}, but got {op_counters[key]}")

    def run_beam_search(self, extra_arguments: str):
        arguments = " ".join([*self.default_arguments, extra_arguments]).split()
        result = run(arguments)
        self.assertTrue(result["parity"], f"ORT and PyTorch result is different on GPU for arguments {arguments}")

    def test_return_sequences(self):
        for return_sequences in [1, 2]:
            self.run_beam_search(f"--num_return_sequences {return_sequences}")

    def test_early_stopping(self):
        self.run_beam_search("--early_stopping")

    def test_length_penalty(self):
        for length_penalty in [0.5, 2.0]:
            self.run_beam_search(f"--length_penalty {length_penalty}")

    def test_no_repeat_ngram(self):
        for ngram_size in [1, 2]:
            self.run_beam_search(f"--no_repeat_ngram_size {ngram_size}")

    def test_external_data(self):
        self.run_beam_search("-e")

        # Ensure fusion is done correctly.
        self.check_encoder_fusion()
        self.check_decoder_fusion()


class TestBeamSearchWhisper(unittest.TestCase):
    """Test BeamSearch for Whisper"""

    def setUp(self):
        self.model_name = "openai/whisper-tiny"
        self.pytorch_folder = "cache_models"
        self.onnx_folder = "onnx_models"
        self.decoder_onnx_path = os.path.join(".", self.onnx_folder, "whisper-tiny_decoder.onnx")
        self.encoder_onnx_path = os.path.join(".", self.onnx_folder, "whisper-tiny_encoder.onnx")
        self.beam_search_onnx_path = os.path.join(".", self.onnx_folder, "whisper-tiny_beamsearch.onnx")
        self.enable_cuda = has_cuda_environment()

        self.base_arguments = [
            "-m",
            self.model_name,
            "--output",
            self.onnx_folder,
            "--use_external_data_format",
        ]
        self.fp32_cpu_arguments = [
            "--precision",
            "fp32",
            "--optimize_onnx",
        ]
        self.fp16_cuda_arguments = [
            "--precision",
            "fp16",
            "--provider",
            "cuda",
            "--optimize_onnx",
            "--use_gpu",
        ]
        self.int8_cpu_arguments = [
            "--precision",
            "int8",
            "--quantize_embedding_layer",
        ]

    def tearDown(self):
        pytorch_dir = os.path.join(".", self.pytorch_folder)
        if os.path.exists(pytorch_dir):
            shutil.rmtree(pytorch_dir)
        onnx_dir = os.path.join(".", self.onnx_folder)
        if os.path.exists(onnx_dir):
            shutil.rmtree(onnx_dir)

    def remove_onnx_files(self):
        if os.path.exists(self.beam_search_onnx_path):
            os.remove(self.beam_search_onnx_path)
            os.remove(self.beam_search_onnx_path + ".data")

        if os.path.exists(self.decoder_onnx_path):
            os.remove(self.decoder_onnx_path)
            os.remove(self.decoder_onnx_path + ".data")

        if os.path.exists(self.encoder_onnx_path):
            os.remove(self.encoder_onnx_path)
            os.remove(self.encoder_onnx_path + ".data")

    def run_export(self, arguments):
        max_diff = run_whisper(arguments)
        self.assertTrue(os.path.exists(self.beam_search_onnx_path), "Whisper model was not exported")
        self.remove_onnx_files()
        self.assertTrue(max_diff == 0, f"ORT and PyTorch have a parity mismatch of {max_diff}")

    def run_configs(self, optional_arguments):
        # FP32 CPU
        arguments = self.base_arguments + self.fp32_cpu_arguments + optional_arguments
        self.run_export(arguments)

        if self.enable_cuda:
            # FP16 CUDA
            arguments = self.base_arguments + self.fp16_cuda_arguments + optional_arguments
            self.run_export(arguments)

        # INT8 CPU
        arguments = self.base_arguments + self.int8_cpu_arguments + optional_arguments
        if "--model_impl" not in arguments:
            self.run_export(arguments)

    @pytest.mark.slow
    def test_required_args(self):
        optional_args = []
        self.run_configs(optional_args)

    @pytest.mark.slow
    def test_forced_decoder_ids(self):
        decoder_input_ids = ["--use_forced_decoder_ids"]
        self.run_configs(decoder_input_ids)

    @pytest.mark.slow
    def test_logits_processor(self):
        logits_processor = ["--use_logits_processor"]
        self.run_configs(logits_processor)

    @pytest.mark.slow
    def test_cross_qk_overall(self):
        cross_qk_input_args = [
            "--use_vocab_mask",
            "--use_prefix_vocab_mask",
            "--use_forced_decoder_ids",
            "--use_logits_processor",
            "--collect_cross_qk",
            "--extra_decoding_ids",
        ]
        cross_qk_output_args = [
            "--output_cross_qk",
            "--output_no_speech_probs",
        ]
        self.run_configs(cross_qk_input_args + cross_qk_output_args)

    @pytest.mark.slow
    def test_openai_impl_whisper(self):
        optional_args = ["--model_impl", "openai"]
        self.run_configs(optional_args)


if __name__ == "__main__":
    unittest.main()
