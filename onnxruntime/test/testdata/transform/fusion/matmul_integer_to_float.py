from enum import Enum  # noqa: F401

import onnx
from onnx import TensorProto, helper


def MakeSubGraph(suffix, has_bias):  # noqa: N802
    mul_bottom_output = "mul_output" + suffix if has_bias else "output" + suffix
    nodes = [
        helper.make_node(
            "MatMulInteger",
            ["a_quantized", "b_quantized" + suffix, "a_zp", "b_zp" + suffix],
            ["matmul_output_int32" + suffix],
            "MatMulInteger" + suffix,
        ),
        helper.make_node(
            "Mul",
            ["a_scale", "b_scale" + suffix],
            ["multiplier" + suffix],
            "mul_right" + suffix,
        ),
        helper.make_node(
            "Cast",
            ["matmul_output_int32" + suffix],
            ["matmul_output_float" + suffix],
            "cast" + suffix,
            to=1,
        ),
        helper.make_node(
            "Mul",
            ["matmul_output_float" + suffix, "multiplier" + suffix],
            [mul_bottom_output],
            "mul_bottom" + suffix,
        ),
    ]

    if has_bias:
        nodes.extend(
            [
                helper.make_node(
                    "Add",
                    [mul_bottom_output, "bias" + suffix],
                    ["output" + suffix],
                    "bias_add" + suffix,
                ),
            ]
        )

    return nodes


def MakeInitializer(suffix, output_type_fp16=False):  # noqa: N802
    return [
        helper.make_tensor("b_quantized" + suffix, TensorProto.UINT8, [2, 3], [2, 4, 5, 6, 7, 8]),
        helper.make_tensor("b_zp" + suffix, TensorProto.UINT8, [], [128]),
        helper.make_tensor("b_scale" + suffix, TensorProto.FLOAT16 if output_type_fp16 else TensorProto.FLOAT, [], [1.8]),
    ]


def GenerateModel(model_name, output_type_fp16=False):  # noqa: N802
    nodes = [
        helper.make_node(
            "DynamicQuantizeLinear",
            ["input"],
            ["a_quantized", "a_scale", "a_zp"],
            "DynamicQuantizeLinear",
        ),
    ]
    nodes.extend(MakeSubGraph("_1", True))
    nodes.extend(MakeSubGraph("_2", True))
    nodes.extend(MakeSubGraph("_3", False))

    initializers = []
    initializers.extend(MakeInitializer("_1", output_type_fp16))
    initializers.extend(MakeInitializer("_3", output_type_fp16))

    initializers.extend(
        [
            helper.make_tensor("bias_1", TensorProto.FLOAT16 if output_type_fp16 else TensorProto.FLOAT, [3], [2, 4, 5]),
            helper.make_tensor("bias_2", TensorProto.FLOAT16 if output_type_fp16 else TensorProto.FLOAT, [3, 3], [1, 2, 3, 4, 5, 6, 7, 8, 9]),
        ]
    )

    graph = helper.make_graph(
        nodes,
        "MatMulIntegerToFloat_fusion",  # name
        [  # inputs
            helper.make_tensor_value_info("input", TensorProto.FLOAT16 if output_type_fp16 else TensorProto.FLOAT, [3, 2]),
            # matrix b corresponding inputs for subgraph 2
            helper.make_tensor_value_info("b_quantized_2", TensorProto.UINT8, [2, 3]),
            helper.make_tensor_value_info("b_zp_2", TensorProto.UINT8, [1]),
            helper.make_tensor_value_info("b_scale_2", TensorProto.FLOAT16 if output_type_fp16 else TensorProto.FLOAT, [1]),
        ],
        [  # outputs
            helper.make_tensor_value_info("output_1", TensorProto.FLOAT16 if output_type_fp16 else TensorProto.FLOAT, [3, 3]),
            helper.make_tensor_value_info("output_2", TensorProto.FLOAT16 if output_type_fp16 else TensorProto.FLOAT, [3, 3]),
            helper.make_tensor_value_info("output_3", TensorProto.FLOAT16 if output_type_fp16 else TensorProto.FLOAT, [3, 3]),
        ],
        initializers,
    )

    model = helper.make_model(graph)
    onnx.save(model, model_name)


if __name__ == "__main__":
    GenerateModel("matmul_integer_to_float.onnx")
    GenerateModel("matmul_integer_to_float16.onnx", True)
