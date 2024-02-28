#!/usr/bin/env python
# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

import math
import unittest
from pathlib import Path

import numpy as np
import onnx

from onnxruntime.quantization.execution_providers.qnn import qnn_preprocess_model
from onnxruntime.quantization.quant_utils import model_has_external_data, ms_domain


class TestQnnPreprocessModel(unittest.TestCase):
    def build_model(self, shape, scale_val, bias_val):
        """
        Build a model that supports 3 kinds of fusions:
        - Erf sequence to Gelu
        - ReduceL2 sequence to LpNormalization
        - ReduceMean sequence to LayerNormalization
        """
        root_inp = onnx.helper.make_tensor_value_info("root", onnx.TensorProto.FLOAT, shape)
        output = onnx.helper.make_tensor_value_info("output", onnx.TensorProto.FLOAT, shape)

        # Erf sequence
        one_const = onnx.numpy_helper.from_array(np.array(1.0, dtype=np.float32), "one_const")
        half_const = onnx.numpy_helper.from_array(np.array(0.5, dtype=np.float32), "half_const")
        root2_const = onnx.numpy_helper.from_array(np.array(math.sqrt(2.0), dtype=np.float32), "root2_const")

        e_mul0_node = onnx.helper.make_node("Mul", ["root", "half_const"], ["e_mul0_out"])
        e_div_node = onnx.helper.make_node("Div", ["root", "root2_const"], ["e_div_out"])
        e_erf_node = onnx.helper.make_node("Erf", ["e_div_out"], ["e_erf_out"])
        e_add_node = onnx.helper.make_node("Add", ["e_erf_out", "one_const"], ["e_add_out"])
        e_mul1_node = onnx.helper.make_node("Mul", ["e_add_out", "e_mul0_out"], ["erf_seq_output"])

        # ReduceL2 sequence
        axes_const = onnx.numpy_helper.from_array(np.array([-1], dtype=np.int64), "axes_const")
        eps_const = onnx.numpy_helper.from_array(np.array(1e-12, dtype=np.float32), "eps_const")
        shape_const = onnx.numpy_helper.from_array(np.array(list(shape), dtype=np.int64), "shape_const")

        l2_rl2_node = onnx.helper.make_node("ReduceL2", ["erf_seq_output", "axes_const"], ["l2_rl2_out"], keepdims=1)
        l2_clip_node = onnx.helper.make_node("Clip", ["l2_rl2_out", "eps_const"], ["l2_clip_out"])
        l2_expand_node = onnx.helper.make_node("Expand", ["l2_clip_out", "shape_const"], ["l2_expand_out"])
        l2_div_node = onnx.helper.make_node("Div", ["erf_seq_output", "l2_expand_out"], ["l2_seq_output"])

        # ReduceMean sequence
        scale_const = onnx.numpy_helper.from_array(np.array(scale_val, dtype=np.float32), "scale_const")
        bias_const = onnx.numpy_helper.from_array(np.array(bias_val, dtype=np.float32), "bias_const")
        two_const = onnx.numpy_helper.from_array(np.array(2.0, dtype=np.float32), "two_const")

        m_rm0_node = onnx.helper.make_node("ReduceMean", ["l2_seq_output", "axes_const"], ["m_rm0_out"])
        m_sub_node = onnx.helper.make_node("Sub", ["l2_seq_output", "m_rm0_out"], ["m_sub_out"])
        m_pow_node = onnx.helper.make_node("Pow", ["m_sub_out", "two_const"], ["m_pow_out"])
        m_rm1_node = onnx.helper.make_node("ReduceMean", ["m_pow_out", "axes_const"], ["m_rm1_out"])
        m_add0_node = onnx.helper.make_node("Add", ["m_rm1_out", "eps_const"], ["m_add0_out"])
        m_sqrt_node = onnx.helper.make_node("Sqrt", ["m_add0_out"], ["m_sqrt_out"])
        m_div_node = onnx.helper.make_node("Div", ["m_sub_out", "m_sqrt_out"], ["m_div_out"])
        m_mul_node = onnx.helper.make_node("Mul", ["m_div_out", "scale_const"], ["m_mul_out"])
        m_add1_node = onnx.helper.make_node("Add", ["m_mul_out", "bias_const"], ["output"])

        graph = onnx.helper.make_graph(
            [
                e_mul0_node,
                e_div_node,
                e_erf_node,
                e_add_node,
                e_mul1_node,
                l2_rl2_node,
                l2_clip_node,
                l2_expand_node,
                l2_div_node,
                m_rm0_node,
                m_sub_node,
                m_pow_node,
                m_rm1_node,
                m_add0_node,
                m_sqrt_node,
                m_div_node,
                m_mul_node,
                m_add1_node,
            ],
            "qnn_f32_model",
            [root_inp],
            [output],
            initializer=[
                one_const,
                half_const,
                root2_const,
                axes_const,
                eps_const,
                shape_const,
                scale_const,
                bias_const,
                two_const,
            ],
        )
        opset_imports = [
            onnx.helper.make_opsetid("", 18),
        ]
        model = onnx.helper.make_model(graph, opset_imports=opset_imports)
        return onnx.shape_inference.infer_shapes(model)

    def test_all_fusions(self):
        """
        Test calling qnn_preprocess_model() with a model that supports all 3 fusions.
        """
        model = self.build_model((1, 2, 3), [2.0, 2.0, 2.0], [1.0, 1.0, 1.0])
        onnx.save_model(model, "model.onnx")
        modified = qnn_preprocess_model("model.onnx", "model.qnn_pp.onnx", fuse_layernorm=True)

        self.assertTrue(modified)

        fused_model = onnx.load_model("model.qnn_pp.onnx")

        # 3 fused Ops: Gelu, LpNorm, LayerNorm
        self.assertEqual(len(fused_model.graph.node), 3)
        expected_op_types = {"Gelu", "LpNormalization", "LayerNormalization"}
        for node in fused_model.graph.node:
            self.assertIn(node.op_type, expected_op_types)

        # Should have added "com.microsoft" opset import because we added a Gelu.
        ms_domain_opset = next((opset for opset in fused_model.opset_import if opset.domain == ms_domain), None)
        self.assertIsNotNone(ms_domain_opset)
        self.assertEqual(ms_domain_opset.version, 1)

    def test_external_data(self):
        """
        Test calling qnn_preprocess_model() with a model that uses external data.
        The new preprocessed model should also have external data.
        """
        model = self.build_model((1, 2, 3), [2.0, 2.0, 2.0], [1.0, 1.0, 1.0])
        onnx.save_model(
            model,
            "model.onnx",
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location="weights.bin",
            size_threshold=0,
        )
        modified = qnn_preprocess_model(
            "model.onnx",
            "model.qnn_pp.onnx",
            fuse_layernorm=True,
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            external_data_location="weights2.bin",
            external_data_size_threshold=0,
        )

        self.assertTrue(modified)

        # Model should still have external data.
        self.assertTrue(model_has_external_data(Path("model.qnn_pp.onnx")))

        fused_model = onnx.load_model("model.qnn_pp.onnx", load_external_data=False)

        # 3 fused Ops: Gelu, LpNorm, LayerNorm
        self.assertEqual(len(fused_model.graph.node), 3)
        expected_op_types = {"Gelu", "LpNormalization", "LayerNormalization"}
        for node in fused_model.graph.node:
            self.assertIn(node.op_type, expected_op_types)


if __name__ == "__main__":
    unittest.main()
