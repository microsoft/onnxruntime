#!/usr/bin/env python
# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

# For live logging, use the command: pytest -o log_cli=true --log-cli-level=DEBUG

import shutil
import unittest

import pytest
import torch
from model_loader import get_fusion_test_model, get_test_data_path
from onnx import TensorProto, load_model
from parity_utilities import find_transformers_source
from transformers import is_tf_available

if find_transformers_source():
    from benchmark_helper import ConfigModifier, OptimizerInfo, Precision
    from fusion_options import FusionOptions
    from huggingface_models import MODELS
    from onnx_exporter import export_onnx_model_from_pt, export_onnx_model_from_tf
    from onnx_model import OnnxModel
    from optimizer import optimize_model
else:
    from onnxruntime.transformers.benchmark_helper import ConfigModifier, OptimizerInfo, Precision
    from onnxruntime.transformers.fusion_options import FusionOptions
    from onnxruntime.transformers.huggingface_models import MODELS
    from onnxruntime.transformers.onnx_exporter import export_onnx_model_from_pt, export_onnx_model_from_tf
    from onnxruntime.transformers.onnx_model import OnnxModel
    from onnxruntime.transformers.optimizer import optimize_model

TEST_MODELS = {
    "bert_keras_0": (
        "models",
        "TFBertForSequenceClassification_1.onnx",
    ),  # bert_mrpc_tensorflow2.1_opset10
    "bert_keras_squad": (
        "models",
        "TFBertForQuestionAnswering.onnx",
    ),  # bert_squad_tensorflow2.1_keras2onnx_opset11
    "gpt2_past": ("models", "gpt2_past.onnx"),  # gpt2_pytorch1.5_opset11
    "gpt2_past_mask": ("FUSION", "gpt2_past_mask_one_layer.onnx"),
    "multiple_embed": ("FUSION", "embed_layer_norm_multiple.onnx"),
    "bert_tf2onnx_0": ("models", "bert_tf2onnx_0.onnx"),
}


def _get_test_model_path(name):
    sub_dir, file = TEST_MODELS[name]
    if sub_dir == "FUSION":
        return get_fusion_test_model(file)
    else:
        return get_test_data_path(sub_dir, file)


class TestModelOptimization(unittest.TestCase):
    def verify_node_count(self, onnx_model, expected_node_count, test_name):
        for op_type, count in expected_node_count.items():
            if len(onnx_model.get_nodes_by_op_type(op_type)) != count:
                print(f"Counters is not expected in test: {test_name}")
                for op, counter in expected_node_count.items():
                    print(f"{op}: {len(onnx_model.get_nodes_by_op_type(op))} expected={counter}")

                self.assertEqual(len(onnx_model.get_nodes_by_op_type(op_type)), count)

    # test huggingface pytorch model
    def _test_optimizer_on_huggingface_model(
        self,
        model_name,
        expected_fusion_result_list,
        inputs_count=1,
        validate_model=True,
    ):
        # Remove cached model so that CI machine has enough space. Do not remove cache models in dev machine.
        if not find_transformers_source():
            shutil.rmtree("./cache_models", ignore_errors=True)
        shutil.rmtree("./onnx_models", ignore_errors=True)

        # expect fusion result list have the following keys
        # EmbedLayerNormalization, Attention, Gelu, FastGelu, BiasGelu, LayerNormalization, SkipLayerNormalization
        model_fusion_statistics = {}

        input_names = MODELS[model_name][0]

        config_modifier = ConfigModifier(None)
        fusion_options = None
        model_class = "AutoModel"
        with torch.no_grad():
            _, is_valid_onnx_model, _, _ = export_onnx_model_from_pt(
                model_name,
                MODELS[model_name][1],  # opset version
                MODELS[model_name][2],  # use_external_data_format
                MODELS[model_name][3],  # optimization model type
                model_class,
                config_modifier,
                "./cache_models",
                "./onnx_models",
                input_names[:inputs_count],
                False,
                Precision.FLOAT32,
                OptimizerInfo.BYSCRIPT,
                True,
                True,
                True,
                model_fusion_statistics,
                fusion_options,
            )

        if validate_model:
            self.assertEqual(is_valid_onnx_model, True)

        expected_node_count = {
            "EmbedLayerNormalization": expected_fusion_result_list[0],
            "Attention": expected_fusion_result_list[1],
            "Gelu": expected_fusion_result_list[2],
            "FastGelu": expected_fusion_result_list[3],
            "BiasGelu": expected_fusion_result_list[4],
            "LayerNormalization": expected_fusion_result_list[5],
            "SkipLayerNormalization": expected_fusion_result_list[6],
        }

        for value in model_fusion_statistics.values():
            actual_node_count = value

        for op_type, count in expected_node_count.items():
            if op_type not in actual_node_count or actual_node_count[op_type] != count:
                print(f"expected: {expected_node_count} got {actual_node_count}")
                self.assertTrue(False)

    def test_gpt2_past(self):
        for enable_skip_layer_norm_fusion in [False, True]:
            input_path = _get_test_model_path("gpt2_past")

            options = FusionOptions("gpt2")
            options.enable_skip_layer_norm = enable_skip_layer_norm_fusion

            model = optimize_model(
                input_path,
                "gpt2",
                num_heads=2,
                hidden_size=4,
                optimization_options=options,
            )

            expected_node_count = {
                "EmbedLayerNormalization": 0,
                "Attention": 12,
                "Gelu": 0,
                "FastGelu": 12,
                "BiasGelu": 0,
                # First LayerNorm is never fused to SkipLayerNorm as it doesn't meet the requirements
                "LayerNormalization": 25 if not enable_skip_layer_norm_fusion else 1,
                "SkipLayerNormalization": 0 if not enable_skip_layer_norm_fusion else 24,
            }
            self.verify_node_count(model, expected_node_count, "test_gpt2_past")

    def test_gpt2_past_fp16(self):
        input_model_path = _get_test_model_path("gpt2_past")
        model = OnnxModel(load_model(input_model_path, format=None, load_external_data=True))
        model.convert_float_to_float16(keep_io_types=False, use_symbolic_shape_infer=False)
        for input in model.graph().input[1:]:
            self.assertEqual(input.type.tensor_type.elem_type, TensorProto.FLOAT16)
        for output in model.graph().output:
            self.assertEqual(output.type.tensor_type.elem_type, TensorProto.FLOAT16)

    def test_gpt2_past_mask(self):
        for enable_skip_layer_norm_fusion in [False, True]:
            input_path = _get_test_model_path("gpt2_past_mask")

            options = FusionOptions("gpt2")
            options.enable_skip_layer_norm = enable_skip_layer_norm_fusion

            model = optimize_model(
                input_path,
                "gpt2",
                num_heads=2,
                hidden_size=4,
                optimization_options=options,
            )

            expected_node_count = {
                "EmbedLayerNormalization": 1,
                "Attention": 1,
                "Gelu": 0,
                "FastGelu": 1,
                "BiasGelu": 0,
                "LayerNormalization": 1 if not enable_skip_layer_norm_fusion else 0,
                "SkipLayerNormalization": 0 if not enable_skip_layer_norm_fusion else 1,
            }
            self.verify_node_count(model, expected_node_count, "test_gpt2_past_mask")

    def test_multiple_embed(self):
        input_model_path = _get_test_model_path("multiple_embed")
        model = optimize_model(input_model_path, "bert", num_heads=2, hidden_size=4)
        expected_node_count = {
            "EmbedLayerNormalization": 2,
            "Attention": 2,
            "Gelu": 0,
            "FastGelu": 0,
            "BiasGelu": 0,
            "LayerNormalization": 0,
            "SkipLayerNormalization": 0,
        }
        self.verify_node_count(model, expected_node_count, "test_multiple_embed")

    def test_embed_layer_norm_fusion(self):
        onnx_files = []
        for i in [3, 8, 9]:
            onnx_files.append(f"embed_layer_norm_format{i}.onnx")
            onnx_files.append(f"embed_layer_norm_format{i}_opset13.onnx")
        onnx_files.append("embed_layer_norm_format3_no_cast.onnx")
        onnx_files.append("embed_layer_norm_format3_no_cast_opset13.onnx")

        options = FusionOptions("bert")
        options.use_raw_attention_mask(False)

        for file in onnx_files:
            input_model_path = get_fusion_test_model(file)
            model = optimize_model(input_model_path, "bert", optimization_options=options)
            expected_node_count = {
                "EmbedLayerNormalization": 1,
                "Attention": 1,
                "ReduceSum": 0,
            }
            self.verify_node_count(model, expected_node_count, file)

    @pytest.mark.slow
    def test_huggingface_bert_fusion_1(self):
        self._test_optimizer_on_huggingface_model("bert-base-uncased", [1, 12, 0, 0, 12, 0, 24], inputs_count=1)

    @pytest.mark.slow
    def test_huggingface_bert_fusion_2(self):
        self._test_optimizer_on_huggingface_model("bert-base-uncased", [1, 12, 0, 0, 12, 0, 24], inputs_count=2)

    @pytest.mark.slow
    def test_huggingface_bert_fusion_3(self):
        self._test_optimizer_on_huggingface_model("bert-base-uncased", [1, 12, 0, 0, 12, 0, 24], inputs_count=3)

    @pytest.mark.slow
    def test_huggingface_openaigpt_fusion(self):
        self._test_optimizer_on_huggingface_model("openai-gpt", [0, 12, 0, 12, 0, 0, 24])

    @pytest.mark.slow
    @unittest.skip("skip failed fusion test of gpt-2 on PyTorch 1.12 and transformers 4.18. TODO: fix it")
    def test_huggingface_gpt2_fusion(self):
        self._test_optimizer_on_huggingface_model("gpt2", [0, 12, 0, 12, 0, 25, 0])

    @pytest.mark.slow
    @unittest.skip("skip failed fusion test of xlm on PyTorch 1.12 and transformers 4.18. TODO: fix it")
    def test_huggingface_xlm_fusion(self):
        self._test_optimizer_on_huggingface_model("xlm-mlm-ende-1024", [0, 6, 0, 0, 6, 0, 13])

    @pytest.mark.slow
    def test_huggingface_roberta_fusion(self):
        self._test_optimizer_on_huggingface_model("roberta-base", [0, 12, 0, 0, 12, 1, 24])

    @pytest.mark.slow
    def test_huggingface_distillbert_fusion(self):
        self._test_optimizer_on_huggingface_model("distilbert-base-uncased", [1, 6, 0, 0, 6, 0, 12], inputs_count=1)
        self._test_optimizer_on_huggingface_model("distilbert-base-uncased", [1, 6, 0, 0, 6, 0, 12], inputs_count=2)

    @pytest.mark.slow
    @unittest.skip("skip failed fusion test of camembert on PyTorch 1.12 and transformers 4.18. TODO: fix it")
    def test_huggingface_camembert_fusion(self):
        self._test_optimizer_on_huggingface_model("camembert-base", [0, 12, 0, 0, 12, 1, 24], validate_model=False)

    @pytest.mark.slow
    @unittest.skip("skip failed fusion test of albert on PyTorch 1.12 and transformers 4.18. TODO: fix it")
    def test_huggingface_albert_fusion(self):
        self._test_optimizer_on_huggingface_model("albert-base-v1", [0, 12, 0, 0, 12, 1, 24])

    @pytest.mark.slow
    @unittest.skip("skip fusion test of t5 since it is not implemented yet")
    def test_huggingface_t5_fusion(self):
        self._test_optimizer_on_huggingface_model("t5-small", [0, 0, 0, 0, 0, 0, 0])

    @pytest.mark.slow
    def test_huggingface_xlmroberta_fusion(self):
        self._test_optimizer_on_huggingface_model("xlm-roberta-base", [0, 12, 0, 0, 12, 1, 24])

    @pytest.mark.slow
    @unittest.skip("skip failed fusion test of flaubert on PyTorch 1.12 and transformers 4.18. TODO: fix it")
    def test_huggingface_flaubert_fusion(self):
        self._test_optimizer_on_huggingface_model(
            "flaubert/flaubert_base_cased",
            [0, 12, 0, 0, 12, 0, 25],
            validate_model=False,
        )
        self._test_optimizer_on_huggingface_model(
            "flaubert/flaubert_small_cased",
            [0, 6, 0, 0, 6, 12, 1],
            validate_model=False,
        )

    @pytest.mark.slow
    @unittest.skip("skip failed fusion test of dialogpt on PyTorch 1.12 and transformers 4.18. TODO: fix it")
    def test_huggingface_dialogpt_fusion(self):
        self._test_optimizer_on_huggingface_model("microsoft/DialoGPT-small", [0, 12, 0, 12, 0, 25, 0])

    @pytest.mark.slow
    def test_huggingface_bart_fusion(self):
        self._test_optimizer_on_huggingface_model("facebook/bart-base", [0, 0, 0, 0, 12, 2, 30])

    @pytest.mark.slow
    def test_huggingface_vit_fusion(self):
        self._test_optimizer_on_huggingface_model("google/vit-base-patch16-224", [0, 11, 0, 0, 12, 1, 24])


@unittest.skipUnless(is_tf_available(), "skip TestBertOptimizationTF since tensorflow is not available")
class TestTensorflowModelOptimization(unittest.TestCase):
    def setUp(self):
        try:
            import tf2onnx  # noqa: F401
        except ImportError:
            self.skipTest("skip TestBertOptimizationTF since tf2onnx not installed")

    def _test_optimizer_on_tf_model(self, model_name, expected_fusion_result_list, inputs_count, validate_model=True):
        # Remove cached model so that CI machine has enough space. Do not remove cache models in dev machine.
        if not find_transformers_source():
            shutil.rmtree("./cache_models", ignore_errors=True)
        shutil.rmtree("./onnx_models", ignore_errors=True)

        # expect fusion result list have the following keys
        # EmbedLayerNormalization, Attention, Gelu, FastGelu, BiasGelu, LayerNormalization, SkipLayerNormalization
        model_fusion_statistics = {}
        print("testing mode ", model_name)
        print("testing input number = ", inputs_count)
        input_names = MODELS[model_name][0]

        config_modifier = ConfigModifier(None)
        fusion_options = None
        model_class = "AutoModel"
        with torch.no_grad():
            _, is_valid_onnx_model, _, _ = export_onnx_model_from_tf(
                model_name,
                MODELS[model_name][1],  # opset version
                MODELS[model_name][2],  # use_external_data_format
                MODELS[model_name][3],  # optimization model
                model_class,
                config_modifier,
                "./cache_models",
                "./onnx_models",
                input_names[:inputs_count],
                False,
                Precision.FLOAT32,
                True,
                True,
                True,
                True,
                model_fusion_statistics,
                fusion_options,
            )

        onnx_model = next(iter(model_fusion_statistics.keys()))
        fusion_result_list = list(model_fusion_statistics[onnx_model].values())

        if validate_model:
            self.assertEqual(is_valid_onnx_model, True)
        self.assertEqual(fusion_result_list, expected_fusion_result_list)

    @pytest.mark.slow
    def test_huggingface_bert_base_cased_from_tf2onnx_1(self):
        self._test_optimizer_on_tf_model("bert-base-cased", [0, 12, 0, 0, 0, 0, 25], 1)

    @pytest.mark.slow
    def test_huggingface_bert_base_cased_from_tf2onnx_2(self):
        self._test_optimizer_on_tf_model("bert-base-cased", [0, 12, 0, 0, 0, 0, 25], 2)

    @pytest.mark.slow
    def test_huggingface_bert_base_cased_from_tf2onnx_3(self):
        self._test_optimizer_on_tf_model("bert-base-cased", [0, 12, 0, 0, 0, 0, 25], 3)

    @pytest.mark.slow
    def test_huggingface_distilgpt2_from_tf2onnx(self):
        self._test_optimizer_on_tf_model("distilgpt2", [0, 0, 0, 0, 0, 12, 1], 1)

    @pytest.mark.slow
    def test_huggingface_albert_from_tf2onnx(self):
        self._test_optimizer_on_tf_model("albert-base-v1", [0, 0, 0, 0, 0, 0, 25], 1)

    @pytest.mark.slow
    def test_huggingface_gpt2_from_tf2onnx(self):
        self._test_optimizer_on_tf_model("gpt2", [0, 0, 0, 0, 0, 24, 1], 1, validate_model=False)

    @pytest.mark.slow
    def test_huggingface_roberta_from_tf2onnx(self):
        self._test_optimizer_on_tf_model("roberta-base", [0, 12, 0, 0, 0, 0, 25], 1, validate_model=False)

    @pytest.mark.slow
    def test_huggingface_distilbert_from_tf2onnx(self):
        self._test_optimizer_on_tf_model("distilbert-base-uncased", [0, 0, 0, 0, 0, 0, 13], 1, validate_model=False)

    @pytest.mark.slow
    def test_huggingface_xlm_from_tf2onnx(self):
        self._test_optimizer_on_tf_model("xlm-mlm-ende-1024", [0, 0, 0, 0, 0, 1, 12], 1, validate_model=False)


if __name__ == "__main__":
    unittest.main()
