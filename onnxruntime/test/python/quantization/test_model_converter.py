import unittest
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
from op_test_utils import check_model_correctness, check_op_type_count

from onnxruntime.quantization.fp16_converter import FP16Converter


def generate_input_initializer(tensor_shape, tensor_dtype, input_name):
    """
    Helper function to generate initializers for test inputs
    """
    tensor = np.random.normal(0, 0.3, tensor_shape).astype(tensor_dtype)
    init = numpy_helper.from_array(tensor, input_name)
    return init


class TestONNXModel(unittest.TestCase):
    @staticmethod
    def construct_conv_model():
        #       input
        #      /    \
        #     /      \
        #  Conv(1)    |
        #     |       |
        #    Relu  Conv(2)
        #     |      |
        #      \    /
        #       Add
        #        |
        #       (output)
        initializers = []
        input = helper.make_tensor_value_info("input", TensorProto.FLOAT, [4, 2, 8, 8])
        output = helper.make_tensor_value_info("output", TensorProto.FLOAT, [4, 2, 8, 8])
        initializers.append(generate_input_initializer([2, 2, 1, 1], np.float32, "W1"))
        initializers.append(generate_input_initializer([2, 2, 1, 1], np.float32, "W2"))
        initializers.append(generate_input_initializer([2], np.float32, "B"))
        conv_node_1 = onnx.helper.make_node("Conv", ["input", "W1", "B"], ["Conv1_O"], name="Conv1")
        conv_node_2 = onnx.helper.make_node("Conv", ["input", "W2", "B"], ["Conv2_O"], name="Conv2")
        relu_node = onnx.helper.make_node("Relu", ["Conv1_O"], ["Relu_O"], name="Relu")
        add_node = onnx.helper.make_node("Add", ["Relu_O", "Conv2_O"], ["output"], name="Add")
        graph = helper.make_graph(
            [conv_node_1, relu_node, conv_node_2, add_node],
            "onnx_model_test",
            [input],
            [output],
            initializer=initializers,
        )
        return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])

    def construct_test(self, op: str):
        np.random.seed(1)
        model_fp32_path = "pre_converter_{}.fp32.onnx".format(op)
        model_fp16_path = "post_converter_{}.fp16.onnx".format(op)
        model = self.construct_conv_model()

        converter = FP16Converter()
        converter.set_model(model)
        converter.export_model_to_path(Path(model_fp32_path))
        op_count = get_op_count_from_model(op, model)
        fp32_nodes = {"Cast": 0, op: op_count}
        check_op_type_count(self, model_fp32_path, **fp32_nodes)

        converter.convert_op(op)
        converter.export_model_to_path(Path(model_fp16_path))

        fp16_model = converter.get_model()
        fp16_op_count = get_op_count_from_model(op, fp16_model)
        fp16_nodes = {"Cast": 2 * fp16_op_count, op: op_count}
        check_op_type_count(self, model_fp16_path, **fp16_nodes)
        check_model_correctness(
            self,
            model_fp32_path,
            model_fp16_path,
            {"input": np.random.rand(4, 2, 8, 8).astype(np.float32)},
        )

    def test_conv_model_converter(self):
        self.construct_test("Conv")


def get_op_count_from_model(op, model):
    return len([node for node in list(model.graph.node) if node.op_type == op])


if __name__ == "__main__":
    unittest.main()
