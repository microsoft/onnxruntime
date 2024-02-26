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

from onnxruntime.quantization.fusions import FusionGelu
from onnxruntime.quantization.onnx_model import ONNXModel


class TestFusions(unittest.TestCase):
    def build_erf_sequence_1_model(self):
        """
           +-------Mul(0.5)---------------------+
           |                                    |
           |                                    v
        [root] --> Div -----> Erf  --> Add --> Mul -->
                  (B=1.4142...)       (1)

        """
        shape = (1, 2, 3)
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
        model = onnx.helper.make_model(graph, opset_imports=[onnx.helper.make_opsetid("", 18)])
        return ONNXModel(model)

    def build_erf_sequence_2_model(self):
        """
           +------------------------------------+
           |                                    |
           |                                    v
        [root] --> Div -----> Erf  --> Add --> Mul -->Mul -->
                  (B=1.4142...)       (1)            (0.5)

        """
        shape = (1, 2, 3)
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
        model = onnx.helper.make_model(graph, opset_imports=[onnx.helper.make_opsetid("", 18)])
        return ONNXModel(model)

    def build_erf_sequence_3_model(self):
        """
           +------------------------------------------+
           |                                          |
           |                                          v
        [root] --> Div -----> Erf  --> Add --> Mul -->Mul
                  (B=1.4142...)       (A=1)   (A=0.5)

        """
        shape = (1, 2, 3)
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
        model = onnx.helper.make_model(graph, opset_imports=[onnx.helper.make_opsetid("", 18)])
        return ONNXModel(model)

    def build_erf_sequence_4_model(self):
        """
           +----------------------------------------------+
           |                                              |
           |                                              v
        [root] --> Mul -----> Erf    -->   Add --> Mul -->Mul
                   (A=0.7071067690849304)  (B=1)  (B=0.5)

        """
        shape = (1, 2, 3)
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
        model = onnx.helper.make_model(graph, opset_imports=[onnx.helper.make_opsetid("", 18)])
        return ONNXModel(model)

    def test_fuse_erf_to_gelu_1(self):
        model = self.build_erf_sequence_1_model()
        modified = FusionGelu(model).apply()

        self.assertTrue(modified)
        self.assertEqual(len(model.model.graph.node), 1)

        gelu_node = model.model.graph.node[0]
        self.assertEqual(gelu_node.op_type, "Gelu")
        self.assertTrue(gelu_node.name)

    def test_fuse_erf_to_gelu_2(self):
        model = self.build_erf_sequence_2_model()
        modified = FusionGelu(model).apply()

        self.assertTrue(modified)
        self.assertEqual(len(model.model.graph.node), 1)

        gelu_node = model.model.graph.node[0]
        self.assertEqual(gelu_node.op_type, "Gelu")
        self.assertTrue(gelu_node.name)

    def test_fuse_erf_to_gelu_3(self):
        model = self.build_erf_sequence_3_model()
        modified = FusionGelu(model).apply()

        self.assertTrue(modified)
        self.assertEqual(len(model.model.graph.node), 1)

        gelu_node = model.model.graph.node[0]
        self.assertEqual(gelu_node.op_type, "Gelu")
        self.assertTrue(gelu_node.name)

    def test_fuse_erf_to_gelu_4(self):
        model = self.build_erf_sequence_4_model()
        modified = FusionGelu(model).apply()

        self.assertTrue(modified)
        self.assertEqual(len(model.model.graph.node), 1)

        gelu_node = model.model.graph.node[0]
        self.assertEqual(gelu_node.op_type, "Gelu")
        self.assertTrue(gelu_node.name)


if __name__ == "__main__":
    unittest.main()
