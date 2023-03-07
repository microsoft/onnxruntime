import os
import unittest

import numpy as np
import onnx
import requests
from onnx import TensorProto, helper, numpy_helper
from onnxruntime.quantization.fp16_converter_simple import FP16ConverterSimple

from op_test_utils import check_model_correctness, check_op_type_count

from onnxruntime.quantization.fp16_converter import FP16Converter
from onnxruntime.transformers.float16 import convert_float_to_float16


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
        test_input = helper.make_tensor_value_info("input", TensorProto.FLOAT, [4, 2, 8, 8])
        test_output = helper.make_tensor_value_info("output", TensorProto.FLOAT, [4, 2, 8, 8])
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
            [test_input],
            [test_output],
            initializer=initializers,
        )
        return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])

    @staticmethod
    def construc_matmul_model_with_init():
        #    (input)                 (input)
        #       |                       |
        #   Transpose               Transpose
        #       |                       |
        #       \     (init)  ===>   Cast(1)) (init)
        #        \      /                \      /
        #         MatMul                  MatMul
        #           |                       |
        #           |                     Cast(2)
        #           |                       |
        #        (output)                (output)

        initializers = []
        test_input = helper.make_tensor_value_info("input", TensorProto.FLOAT, [4, 2])
        test_ouput = helper.make_tensor_value_info("output", TensorProto.FLOAT, [2, 2])
        initializers.append(generate_input_initializer([4, 2], np.float32, "init"))
        transpose_node = onnx.helper.make_node("Transpose", ["input"], ["Transpose1_0"], name="Transpose1")
        matmul_node = onnx.helper.make_node("MatMul", ["Transpose1_0", "init"], ["output"], name="MatMul1")
        graph = helper.make_graph(
            [matmul_node, transpose_node],
            "onnx_model_test",
            [test_input],
            [test_ouput],
            initializer=initializers,
        )
        return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])

    @staticmethod
    def construc_matmul_model_without_init():
        #         (input)                 (input)
        #        /       \               /       \
        #   Transpose     |           Transpose   |
        #       |         |             |         |
        #       \        /     ===>   Cast(1) Cast(2)
        #        \      /                \      /
        #         MatMul                  MatMul
        #           |                       |
        #           |                     Cast(3)
        #           |                       |
        #        (output)                (output)

        initializers = []
        test_input = helper.make_tensor_value_info("input", TensorProto.FLOAT, [4, 2])
        test_ouput = helper.make_tensor_value_info("output", TensorProto.FLOAT, [2, 2])
        transpose_node = onnx.helper.make_node("Transpose", ["input"], ["Transpose1_0"], name="Transpose1")
        matmul_node = onnx.helper.make_node("MatMul", ["Transpose1_0", "input"], ["output"], name="MatMul1")
        graph = helper.make_graph(
            [matmul_node, transpose_node],
            "onnx_model_test",
            [test_input],
            [test_ouput],
            initializer=initializers,
        )
        return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])

    def construct_test(self, op: str, with_init: bool = True):
        fp16_nodes = None
        model = None
        test_input = None
        np.random.seed(1)
        model_fp32_path = f"pre_converter_{op}_{'with' if with_init else 'without'}_init_fp32.onnx"
        model_fp16_path = f"post_converter_{op}_{'with' if with_init else 'without'}_init_fp16.onnx"

        if op == "Conv":
            model = self.construct_conv_model()
        elif op == "MatMul":
            if with_init:
                model = self.construc_matmul_model_with_init()
            else:
                model = self.construc_matmul_model_without_init()

        converter = FP16Converter()
        converter.set_model(model)
        converter.export_model_to_path(model_fp32_path)
        op_count = get_op_count_from_model(op, model)
        fp32_nodes = {"Cast": 0, op: op_count}
        check_op_type_count(self, model_fp32_path, **fp32_nodes)
        converter.convert()
        converter.export_model_to_path(model_fp16_path)

        fp16_model = converter.get_model()
        fp16_op_count = get_op_count_from_model(op, fp16_model)
        if op == "Conv":
            fp16_nodes = {"Cast": 7, op: fp16_op_count}
            test_input = {"input": np.random.rand(4, 2, 8, 8).astype(np.float32)}
        elif op == "MatMul":
            if with_init:
                fp16_nodes = {"Cast": 4, op: fp16_op_count}
            else:
                fp16_nodes = {"Cast": 4, op: fp16_op_count}
            test_input = {"input": np.random.rand(4, 2).astype(np.float32)}

        check_op_type_count(self, model_fp16_path, **fp16_nodes)
        check_model_correctness(
            self,
            model_fp32_path,
            model_fp16_path,
            test_input,
        )

    def test_conv_model_converter(self):
        self.construct_test("Conv")

    def test_matmul_model_converter(self):
        self.construct_test("MatMul")
        self.construct_test("MatMul", False)

    def test_model_converter_on_resnet50_v2_allow_list(self):
        filename = "resnet50-v2-7.onnx"
        if not os.path.exists(filename):
            url = f"https://github.com/onnx/models/blob/main/vision/classification/resnet/model/{filename}?raw=true"
            model = download_model_from_url(url)
            onnx.save_model(model, filename)
            print(f"Saved model to {filename}.")
        else:
            model = onnx.load_model(filename)
            print(f"Loaded model from {filename}.")
        converter = FP16Converter()
        converter.set_model(model)
        converter.convert()
        converter.export_model_to_path("resnet50-fp16-v2-7-allow-list.onnx")

    def test_model_converter_on_resnet50_v2_allow_list_simple(self):
        filename = "resnet50-v2-7.onnx"
        if not os.path.exists(filename):
            url = f"https://github.com/onnx/models/blob/main/vision/classification/resnet/model/{filename}?raw=true"
            model = download_model_from_url(url)
            onnx.save_model(model, filename)
            print(f"Saved model to {filename}.")
        else:
            model = onnx.load_model(filename)
            print(f"Loaded model from {filename}.")
        converter = FP16ConverterSimple()
        converter.set_model(model)
        converter.convert()
        converter.export_model_to_path("resnet50-fp16-v2-7-allow-list-simple.onnx")
    # def test_model_converter_on_resnet50_v2_block_list(self):
    #     filename = "resnet50-v2-7.onnx"
    #     if not os.path.exists(filename):
    #         url = f"https://github.com/onnx/models/blob/main/vision/classification/resnet/model/{filename}?raw=true"
    #         model = download_model_from_url(url)
    #         onnx.save_model(model, filename)
    #         print(f"Saved model to {filename}.")
    #     else:
    #         model = onnx.load_model(filename)
    #         print(f"Loaded model from {filename}.")
    #     model = convert_float_to_float16(model)
    #     onnx.save_model(model, "resnet50-fp16--v2-7-block_list.onnx")


def get_op_count_from_model(op, model):
    return len([node for node in list(model.graph.node) if node.op_type == op])


def download_model_from_url(url: str) -> onnx.ModelProto:
    """
    Helper function to download model from url
    """
    r = requests.get(url, allow_redirects=True)
    return onnx.load_model_from_string(r.content)


if __name__ == "__main__":
    unittest.main()
