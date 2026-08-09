#!/usr/bin/env python
# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

# For live logging, use the following command:
#   pytest -o log_cli=true --log-cli-level=DEBUG test_optimizer.py

import unittest

from model_loader import get_fusion_test_model, get_test_data_path
from onnx import TensorProto, load_model
from parity_utilities import find_transformers_source

if find_transformers_source():
    from fusion_options import FusionOptions
    from onnx_model import OnnxModel
    from optimizer import optimize_model
else:
    from onnxruntime.transformers.fusion_options import FusionOptions
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


if __name__ == "__main__":
    unittest.main()
