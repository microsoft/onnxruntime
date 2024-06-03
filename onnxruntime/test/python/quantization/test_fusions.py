#!/usr/bin/env python
# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

import math
import unittest

import numpy as np
import onnx

import onnxruntime
from onnxruntime.quantization.execution_providers.qnn.fusion_lpnorm import FusionLpNormalization
from onnxruntime.quantization.fusions import FusionGelu, FusionLayerNormalization
from onnxruntime.quantization.onnx_model import ONNXModel


class TestFusions(unittest.TestCase):
    def check_fused_model_correctness(self, orig_model, fused_model, inputs, rtol=1e-7, atol=0):
        """
        Checks that the output of the fused model matches the output of the original model.
        """
        orig_session = onnxruntime.InferenceSession(orig_model.SerializeToString(), providers=["CPUExecutionProvider"])
        orig_results = orig_session.run(None, inputs)

        fused_session = onnxruntime.InferenceSession(
            fused_model.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        fused_results = fused_session.run([], inputs)

        self.assertEqual(len(orig_results), len(fused_results), "Number of outputs for fused model differs")
        for idx, expected_output in enumerate(orig_results):
            actual_output = fused_results[idx]
            np.testing.assert_allclose(
                expected_output,
                actual_output,
                rtol=rtol,
                atol=atol,
                err_msg=f"Fused model output {idx} differs",
            )

    def build_erf_sequence_1_model(self, shape):
        """
        Erf sequence that fuses into Gelu:
           +-------Mul(0.5)---------------------+
           |                                    |
           |                                    v
        [root] --> Div -----> Erf  --> Add --> Mul -->
                  (B=1.4142...)       (1)

        This method builds 2 of these Erf sequences:

        [root] -> ERF_SEQUENCE1 -> ERF_SEQUENCE2 -> output
        """
        root_inp = onnx.helper.make_tensor_value_info("root", onnx.TensorProto.FLOAT, shape)
        output = onnx.helper.make_tensor_value_info("output", onnx.TensorProto.FLOAT, shape)
        one_const = onnx.numpy_helper.from_array(np.array(1.0, dtype=np.float32), "one_const")
        half_const = onnx.numpy_helper.from_array(np.array(0.5, dtype=np.float32), "half_const")
        root2_const = onnx.numpy_helper.from_array(np.array(math.sqrt(2.0), dtype=np.float32), "root2_const")

        # First Erf sequence
        mul0_node = onnx.helper.make_node("Mul", ["root", "half_const"], ["mul0_out"])
        div_node = onnx.helper.make_node("Div", ["root", "root2_const"], ["div_out"])
        erf_node = onnx.helper.make_node("Erf", ["div_out"], ["erf_out"])
        add_node = onnx.helper.make_node("Add", ["erf_out", "one_const"], ["add_out"])
        mul1_node = onnx.helper.make_node("Mul", ["add_out", "mul0_out"], ["seq1_output"])

        # Second Erf sequence
        mul0_node_dup = onnx.helper.make_node("Mul", ["seq1_output", "half_const"], ["mul0_out_dup"])
        div_node_dup = onnx.helper.make_node("Div", ["seq1_output", "root2_const"], ["div_out_dup"])
        erf_node_dup = onnx.helper.make_node("Erf", ["div_out_dup"], ["erf_out_dup"])
        add_node_dup = onnx.helper.make_node("Add", ["erf_out_dup", "one_const"], ["add_out_dup"])
        mul1_node_dup = onnx.helper.make_node("Mul", ["add_out_dup", "mul0_out_dup"], ["output"])

        graph = onnx.helper.make_graph(
            [
                mul0_node,
                div_node,
                erf_node,
                add_node,
                mul1_node,
                mul0_node_dup,
                div_node_dup,
                erf_node_dup,
                add_node_dup,
                mul1_node_dup,
            ],
            "two_erf_sequences",
            [root_inp],
            [output],
            initializer=[one_const, half_const, root2_const],
        )
        opset_imports = [
            onnx.helper.make_opsetid("", 18),
            onnx.helper.make_opsetid("com.microsoft", 1),
        ]
        model = onnx.helper.make_model(graph, opset_imports=opset_imports)
        return ONNXModel(model)

    def build_erf_sequence_2_model(self, shape):
        """
           +------------------------------------+
           |                                    |
           |                                    v
        [root] --> Div -----> Erf  --> Add --> Mul -->Mul -->
                  (B=1.4142...)       (1)            (0.5)

        """
        root_inp = onnx.helper.make_tensor_value_info("root", onnx.TensorProto.FLOAT, shape)
        output = onnx.helper.make_tensor_value_info("output", onnx.TensorProto.FLOAT, shape)
        one_const = onnx.numpy_helper.from_array(np.array(1.0, dtype=np.float32), "one_const")
        half_const = onnx.numpy_helper.from_array(np.array(0.5, dtype=np.float32), "half_const")
        root2_const = onnx.numpy_helper.from_array(np.array(math.sqrt(2.0), dtype=np.float32), "root2_const")

        div_node = onnx.helper.make_node("Div", ["root", "root2_const"], ["div_out"])
        erf_node = onnx.helper.make_node("Erf", ["div_out"], ["erf_out"])
        add_node = onnx.helper.make_node("Add", ["erf_out", "one_const"], ["add_out"])
        mul0_node = onnx.helper.make_node("Mul", ["add_out", "root"], ["mul0_out"])
        mul1_node = onnx.helper.make_node("Mul", ["mul0_out", "half_const"], ["output"])

        graph = onnx.helper.make_graph(
            [div_node, erf_node, add_node, mul0_node, mul1_node],
            "erf_sequence_2",
            [root_inp],
            [output],
            initializer=[one_const, half_const, root2_const],
        )
        opset_imports = [
            onnx.helper.make_opsetid("", 18),
            onnx.helper.make_opsetid("com.microsoft", 1),
        ]
        model = onnx.helper.make_model(graph, opset_imports=opset_imports)
        return ONNXModel(model)

    def build_erf_sequence_3_model(self, shape):
        """
           +------------------------------------------+
           |                                          |
           |                                          v
        [root] --> Div -----> Erf  --> Add --> Mul -->Mul
                  (B=1.4142...)       (A=1)   (A=0.5)

        """
        root_inp = onnx.helper.make_tensor_value_info("root", onnx.TensorProto.FLOAT, shape)
        output = onnx.helper.make_tensor_value_info("output", onnx.TensorProto.FLOAT, shape)
        one_const = onnx.numpy_helper.from_array(np.array(1.0, dtype=np.float32), "one_const")
        half_const = onnx.numpy_helper.from_array(np.array(0.5, dtype=np.float32), "half_const")
        root2_const = onnx.numpy_helper.from_array(np.array(math.sqrt(2.0), dtype=np.float32), "root2_const")

        div_node = onnx.helper.make_node("Div", ["root", "root2_const"], ["div_out"])
        erf_node = onnx.helper.make_node("Erf", ["div_out"], ["erf_out"])
        add_node = onnx.helper.make_node("Add", ["erf_out", "one_const"], ["add_out"])
        mul0_node = onnx.helper.make_node("Mul", ["add_out", "half_const"], ["mul0_out"])
        mul1_node = onnx.helper.make_node("Mul", ["mul0_out", "root"], ["output"])

        graph = onnx.helper.make_graph(
            [div_node, erf_node, add_node, mul0_node, mul1_node],
            "erf_sequence_3",
            [root_inp],
            [output],
            initializer=[one_const, half_const, root2_const],
        )
        opset_imports = [
            onnx.helper.make_opsetid("", 18),
            onnx.helper.make_opsetid("com.microsoft", 1),
        ]
        model = onnx.helper.make_model(graph, opset_imports=opset_imports)
        return ONNXModel(model)

    def build_erf_sequence_4_model(self, shape):
        """
           +----------------------------------------------+
           |                                              |
           |                                              v
        [root] --> Mul -----> Erf    -->   Add --> Mul -->Mul
                   (A=0.7071067690849304)  (B=1)  (B=0.5)

        """
        root_inp = onnx.helper.make_tensor_value_info("root", onnx.TensorProto.FLOAT, shape)
        output = onnx.helper.make_tensor_value_info("output", onnx.TensorProto.FLOAT, shape)
        one_const = onnx.numpy_helper.from_array(np.array(1.0, dtype=np.float32), "one_const")
        half_const = onnx.numpy_helper.from_array(np.array(0.5, dtype=np.float32), "half_const")
        frac_const = onnx.numpy_helper.from_array(np.array(0.7071067690849304, dtype=np.float32), "frac_const")

        mul0_node = onnx.helper.make_node("Mul", ["root", "frac_const"], ["mul0_out"])
        erf_node = onnx.helper.make_node("Erf", ["mul0_out"], ["erf_out"])
        add_node = onnx.helper.make_node("Add", ["erf_out", "one_const"], ["add_out"])
        mul1_node = onnx.helper.make_node("Mul", ["add_out", "half_const"], ["mul1_out"])
        mul2_node = onnx.helper.make_node("Mul", ["mul1_out", "root"], ["output"])

        graph = onnx.helper.make_graph(
            [mul0_node, erf_node, add_node, mul1_node, mul2_node],
            "erf_sequence_4",
            [root_inp],
            [output],
            initializer=[one_const, half_const, frac_const],
        )
        opset_imports = [
            onnx.helper.make_opsetid("", 18),
            onnx.helper.make_opsetid("com.microsoft", 1),
        ]
        model = onnx.helper.make_model(graph, opset_imports=opset_imports)
        return ONNXModel(model)

    def build_reduce_mean_sequence_model(self, shape, scale_val, bias_val, axis=-1):
        """
            +----------------------+
            |                      |
            |                      v
        [Root] --> ReduceMean -->  Sub  --> Pow --> ReduceMean --> Add --> Sqrt --> Div --> Mul --> Add
                   (axis=2 or -1)  |      (Y=2)   (axis=2 or -1)  (E-6 or E-12 or 0) ^       ^       ^
                                   |                                                 |       |       |
                                   +-------------------------------------------------+    [Scale]  [Bias]
        """
        root_inp = onnx.helper.make_tensor_value_info("root", onnx.TensorProto.FLOAT, shape)
        output = onnx.helper.make_tensor_value_info("output", onnx.TensorProto.FLOAT, shape)
        scale_const = onnx.numpy_helper.from_array(np.array(scale_val, dtype=np.float32), "scale_const")
        bias_const = onnx.numpy_helper.from_array(np.array(bias_val, dtype=np.float32), "bias_const")
        axes_const = onnx.numpy_helper.from_array(np.array([axis], dtype=np.int64), "axes_const")
        two_const = onnx.numpy_helper.from_array(np.array(2.0, dtype=np.float32), "two_const")
        eps_const = onnx.numpy_helper.from_array(np.array(1.0e-8, dtype=np.float32), "eps_const")

        rm0_node = onnx.helper.make_node("ReduceMean", ["root", "axes_const"], ["rm0_out"])
        sub_node = onnx.helper.make_node("Sub", ["root", "rm0_out"], ["sub_out"])
        pow_node = onnx.helper.make_node("Pow", ["sub_out", "two_const"], ["pow_out"])
        rm1_node = onnx.helper.make_node("ReduceMean", ["pow_out", "axes_const"], ["rm1_out"])
        add0_node = onnx.helper.make_node("Add", ["rm1_out", "eps_const"], ["add0_out"])
        sqrt_node = onnx.helper.make_node("Sqrt", ["add0_out"], ["sqrt_out"])
        div_node = onnx.helper.make_node("Div", ["sub_out", "sqrt_out"], ["div_out"])
        mul_node = onnx.helper.make_node("Mul", ["div_out", "scale_const"], ["mul_out"])
        add1_node = onnx.helper.make_node("Add", ["mul_out", "bias_const"], ["output"])

        graph = onnx.helper.make_graph(
            [rm0_node, sub_node, pow_node, rm1_node, add0_node, sqrt_node, div_node, mul_node, add1_node],
            "reduce_mean_sequence",
            [root_inp],
            [output],
            initializer=[scale_const, bias_const, axes_const, two_const, eps_const],
        )
        opset_imports = [
            onnx.helper.make_opsetid("", 18),
        ]
        model = onnx.helper.make_model(graph, opset_imports=opset_imports)
        return ONNXModel(model)

    def build_reduce_l2_sequence_model(self, shape, epsilon_val, axis=-1):
        """
        [root] --> ReduceL2 -----> Clip  --> Expand ----> Div -->
           |      (axis=-1)    (min=epsilon) (shape=root)  ^
           |   (keepdims=True)                             |
           |                                               |
           +-----------------------------------------------+
        """
        root_inp = onnx.helper.make_tensor_value_info("root", onnx.TensorProto.FLOAT, shape)
        output = onnx.helper.make_tensor_value_info("output", onnx.TensorProto.FLOAT, shape)
        axes_const = onnx.numpy_helper.from_array(np.array([axis], dtype=np.int64), "axes_const")
        eps_const = onnx.numpy_helper.from_array(np.array(epsilon_val, dtype=np.float32), "eps_const")
        shape_const = onnx.numpy_helper.from_array(np.array(list(shape), dtype=np.int64), "shape_const")

        rl2_node = onnx.helper.make_node("ReduceL2", ["root", "axes_const"], ["rl2_out"], keepdims=1)
        clip_node = onnx.helper.make_node("Clip", ["rl2_out", "eps_const"], ["clip_out"])
        expand_node = onnx.helper.make_node("Expand", ["clip_out", "shape_const"], ["expand_out"])
        div_node = onnx.helper.make_node("Div", ["root", "expand_out"], ["output"])

        graph = onnx.helper.make_graph(
            [rl2_node, clip_node, expand_node, div_node],
            "reducel2_sequence",
            [root_inp],
            [output],
            initializer=[axes_const, eps_const, shape_const],
        )
        opset_imports = [
            onnx.helper.make_opsetid("", 18),
        ]
        model = onnx.helper.make_model(graph, opset_imports=opset_imports)
        return ONNXModel(model)

    def test_fuse_erf_to_gelu_1(self):
        shape = (1, 2, 3)
        model = self.build_erf_sequence_1_model(shape)
        orig_model = onnx.ModelProto()
        orig_model.CopyFrom(model.model)

        # Check that fusion simplified model to 2 Gelu nodes.
        modified = FusionGelu(model).apply()
        self.assertTrue(modified)
        self.assertEqual(len(model.model.graph.node), 2)

        gelu_node_0 = model.model.graph.node[0]
        gelu_node_1 = model.model.graph.node[1]
        self.assertEqual(gelu_node_0.op_type, "Gelu")
        self.assertEqual(gelu_node_1.op_type, "Gelu")

        self.assertTrue(gelu_node_0.name)
        self.assertTrue(gelu_node_1.name)
        self.assertNotEqual(gelu_node_0.name, gelu_node_1.name)  # Generated names should not be equal

        # Check that fusion is equivalent to original Erf model.
        inputs = {"root": np.ones(shape, dtype=np.float32)}
        self.check_fused_model_correctness(orig_model, model.model, inputs)

    def test_fuse_erf_to_gelu_2(self):
        shape = (1, 2, 3)
        model = self.build_erf_sequence_2_model(shape)
        orig_model = onnx.ModelProto()
        orig_model.CopyFrom(model.model)

        # Check that fusion simplified model to 1 Gelu node.
        modified = FusionGelu(model).apply()
        self.assertTrue(modified)
        self.assertEqual(len(model.model.graph.node), 1)

        gelu_node = model.model.graph.node[0]
        self.assertEqual(gelu_node.op_type, "Gelu")
        self.assertTrue(gelu_node.name)

        # Check that fusion is equivalent to original Erf model.
        inputs = {"root": np.ones(shape, dtype=np.float32)}
        self.check_fused_model_correctness(orig_model, model.model, inputs)

    def test_fuse_erf_to_gelu_3(self):
        shape = (1, 2, 3)
        model = self.build_erf_sequence_3_model(shape)
        orig_model = onnx.ModelProto()
        orig_model.CopyFrom(model.model)

        # Check that fusion simplified model to 1 Gelu node.
        modified = FusionGelu(model).apply()
        self.assertTrue(modified)
        self.assertEqual(len(model.model.graph.node), 1)

        gelu_node = model.model.graph.node[0]
        self.assertEqual(gelu_node.op_type, "Gelu")
        self.assertTrue(gelu_node.name)

        # Check that fusion is equivalent to original Erf model.
        inputs = {"root": np.ones(shape, dtype=np.float32)}
        self.check_fused_model_correctness(orig_model, model.model, inputs)

    def test_fuse_erf_to_gelu_4(self):
        shape = (1, 2, 3)
        model = self.build_erf_sequence_4_model(shape)
        orig_model = onnx.ModelProto()
        orig_model.CopyFrom(model.model)

        # Check that fusion simplified model to 1 Gelu node.
        modified = FusionGelu(model).apply()
        self.assertTrue(modified)
        self.assertEqual(len(model.model.graph.node), 1)

        gelu_node = model.model.graph.node[0]
        self.assertEqual(gelu_node.op_type, "Gelu")
        self.assertTrue(gelu_node.name)

        # Check that fusion is equivalent to original Erf model.
        inputs = {"root": np.ones(shape, dtype=np.float32)}
        self.check_fused_model_correctness(orig_model, model.model, inputs)

    def test_fuse_reduce_l2_to_lpnorm(self):
        shape = (1, 2, 3)
        model = self.build_reduce_l2_sequence_model(shape, 1e-12, axis=-1)
        orig_model = onnx.ModelProto()
        orig_model.CopyFrom(model.model)

        # Check that fusion simplified model to 1 LpNormalization node.
        modified = FusionLpNormalization(model).apply()
        self.assertTrue(modified)
        self.assertEqual(len(model.model.graph.node), 1)

        lpnorm_node = model.model.graph.node[0]
        self.assertEqual(lpnorm_node.op_type, "LpNormalization")
        self.assertTrue(lpnorm_node.name)

        # LpNorm's p attribute should be set to 2
        p_attr = next(attr for attr in lpnorm_node.attribute if attr.name == "p")
        self.assertEqual(p_attr.i, 2)

    def test_fuse_reduce_mean_to_layer_norm(self):
        shape = (1, 2, 3)
        model = self.build_reduce_mean_sequence_model(shape, [2.0, 2.0, 2.0], [1.0, 1.0, 1.0], axis=-1)
        orig_model = onnx.ModelProto()
        orig_model.CopyFrom(model.model)

        # Check that fusion simplified model to 1 LayerNormalization node.
        modified = FusionLayerNormalization(model).apply()
        self.assertTrue(modified)
        self.assertEqual(len(model.model.graph.node), 1)

        layer_norm_node = model.model.graph.node[0]
        self.assertEqual(layer_norm_node.op_type, "LayerNormalization")
        self.assertTrue(layer_norm_node.name)

        # Check that fused model is equivalent to original model.
        inputs = {"root": np.ones(shape, dtype=np.float32)}
        self.check_fused_model_correctness(orig_model, model.model, inputs)


if __name__ == "__main__":
    unittest.main()
