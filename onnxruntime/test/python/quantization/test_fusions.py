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
from onnxruntime.quantization.fusions import FusionGelu
from onnxruntime.quantization.onnx_model import ONNXModel


class TestFusions(unittest.TestCase):
    def check_fused_model_correctness(self, orig_model, fused_model, inputs, rtol=1e-7, atol=0):
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
           +-------Mul(0.5)---------------------+
           |                                    |
           |                                    v
        [root] --> Div -----> Erf  --> Add --> Mul -->
                  (B=1.4142...)       (1)

        """
        root_inp = onnx.helper.make_tensor_value_info("root", onnx.TensorProto.FLOAT, shape)
        output = onnx.helper.make_tensor_value_info("output", onnx.TensorProto.FLOAT, shape)
        one_const = onnx.numpy_helper.from_array(np.array(1.0, dtype=np.float32), "one_const")
        half_const = onnx.numpy_helper.from_array(np.array(0.5, dtype=np.float32), "half_const")
        root2_const = onnx.numpy_helper.from_array(np.array(math.sqrt(2.0), dtype=np.float32), "root2_const")

        mul0_node = onnx.helper.make_node("Mul", ["root", "half_const"], ["mul0_out"])
        div_node = onnx.helper.make_node("Div", ["root", "root2_const"], ["div_out"])
        erf_node = onnx.helper.make_node("Erf", ["div_out"], ["erf_out"])
        add_node = onnx.helper.make_node("Add", ["erf_out", "one_const"], ["add_out"])
        mul1_node = onnx.helper.make_node("Mul", ["add_out", "mul0_out"], ["output"])

        graph = onnx.helper.make_graph(
            [mul0_node, div_node, erf_node, add_node, mul1_node],
            "elf_sequence_1",
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
            "elf_sequence_2",
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
            "elf_sequence_3",
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
            "elf_sequence_4",
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

    def test_fuse_erf_to_gelu_1(self):
        shape = (1, 2, 3)
        model = self.build_erf_sequence_1_model(shape)
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


if __name__ == "__main__":
    unittest.main()
