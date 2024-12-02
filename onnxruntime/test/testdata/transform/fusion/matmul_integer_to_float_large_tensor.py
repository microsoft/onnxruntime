from enum import Enum  # noqa: F401

import onnx
from onnx import TensorProto, helper


def GenerateModel(model_name):  # noqa: N802
    inputs = []
    outputs = []
    initializers = []
    nodes = []

    inputs.append(helper.make_tensor_value_info("inputA", TensorProto.FLOAT, [16, 32, 1280, 1280]))
    inputs.append(helper.make_tensor_value_info("inputB", TensorProto.INT8, [1280, 1280]))
    inputs.append(helper.make_tensor_value_info("inputBZP", TensorProto.INT8, [1]))
    inputs.append(helper.make_tensor_value_info("inputBScale", TensorProto.FLOAT, [1]))

    nodes = [  # construct graph
        helper.make_node(
            "DynamicQuantizeLinear",
            ["inputA"],
            ["a_quantized", "a_scale", "a_zp"],
            "DynamicQuantizeLinear",
        ),
        helper.make_node(
            "MatMulInteger",
            ["a_quantized", "inputB", "a_zp", "inputBZP"],
            ["matmulinteger_output"],
            "MatMulInteger",
        ),
        helper.make_node("Mul", ["a_scale", "inputBScale"], ["mul_1"], "mul_right"),
        helper.make_node("Cast", ["matmulinteger_output"], ["cast_output"], "cast", to=1),
        helper.make_node("Mul", ["mul_1", "cast_output"], ["output"], "mul_bottom"),
    ]

    graph = helper.make_graph(
        nodes,
        "matmul_integer_to_float_large_tensor_fusion",  # name
        inputs,
        outputs,
        initializers,
    )

    model = helper.make_model(graph)
    onnx.save(model, model_name)


if __name__ == "__main__":
    GenerateModel("matmul_integer_to_float_large_tensor.onnx")
