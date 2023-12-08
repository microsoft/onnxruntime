from enum import Enum  # noqa: F401

import onnx
from onnx import TensorProto, helper


def GenerateModel(model_name, sign_i, sign_w, output_type_fp16, has_zp=True):  # noqa: N802
    nodes = [  # subgraph
        helper.make_node(
            "QAttention",
            ["input", "weight", "bias", "input_scale", "weight_scale", "input_zero_point", "weight_zero_point", "past"],# if has_zp else ["A", "B"],
            ["present", "output"],
            "QAttention",
            unidirectional=1
        )
    ]

    inputs = [  # inputs
        helper.make_tensor_value_info("input", TensorProto.INT8 if sign_i else TensorProto.UINT8, ["batch_size", "sequence_length", "hidden_size"]),
        helper.make_tensor_value_info("weight", TensorProto.INT8 if sign_w else TensorProto.UINT8, ["hidden_size", "3 * hidden_size"]),
        helper.make_tensor_value_info("bias", TensorProto.FLOAT16 if output_type_fp16 else TensorProto.FLOAT, ["3 * hidden_size"]),
        helper.make_tensor_value_info("input_scale", TensorProto.FLOAT16 if output_type_fp16 else TensorProto.FLOAT, [1]),
        helper.make_tensor_value_info("weight_scale", TensorProto.FLOAT16 if output_type_fp16 else TensorProto.FLOAT, [1]),# per column: ["C"]),
        helper.make_tensor_value_info("input_zero_point", TensorProto.INT8 if sign_i else TensorProto.UINT8, ["1"]),
        helper.make_tensor_value_info("weight_zero_point", TensorProto.INT8 if sign_w else TensorProto.UINT8, ["1"]),
        helper.make_tensor_value_info("past", TensorProto.FLOAT16 if output_type_fp16 else TensorProto.FLOAT, [2, "batch_size", "number_of_heads", "past_sequence_length", "head_size"]),
    ]

    #if has_zp:
    #    inputs.extend(
    #        [
    #            helper.make_tensor_value_info(
    #                "a_zero_point",
    #                TensorProto.INT8 if sign_i else TensorProto.UINT8,
    #                [1],
    #            ),
    #            helper.make_tensor_value_info(
    #                "b_zero_point",
    #                TensorProto.INT8 if sign_w else TensorProto.UINT8,
    #                ["C"],
    #            ),
    #        ]
    #    )

    #if bias:
    #    nodes.extend([helper.make_node("Add", ["mul_bottom_output", "bias"], ["Y"], "add")])
    #
    #    inputs.extend([helper.make_tensor_value_info("bias", TensorProto.FLOAT16 if output_type_fp16 else TensorProto.FLOAT, ["N"])])

    graph = helper.make_graph(
        nodes,
        "QAttention",  # name
        inputs,
        [  # outputs
            helper.make_tensor_value_info("present", TensorProto.FLOAT16 if output_type_fp16 else TensorProto.FLOAT, [2, "batch_size", "number_of_heads", "past_sequence_length + sequence_length", "head_size"]),
            helper.make_tensor_value_info("output", TensorProto.FLOAT16 if output_type_fp16 else TensorProto.FLOAT, ["batch_size", "sequence_length", "hidden_size"]),
        ],
    )

    model = helper.make_model(graph)
    onnx.save(model, model_name)


if __name__ == "__main__":
    GenerateModel("qattentionfp16_int8.onnx", sign_i=False, sign_w=True, output_type_fp16=True)
    GenerateModel("qattentionfp16_uint8.onnx", sign_i=False, sign_w=False, output_type_fp16=True)
    GenerateModel("qattentionfp16_int8_int8.onnx", sign_i=True, sign_w=True, output_type_fp16=True)

    GenerateModel("qattention_int8.onnx", sign_i=False, sign_w=True, output_type_fp16=False)
    GenerateModel("qattention_uint8.onnx", sign_i=False, sign_w=False, output_type_fp16=False)
    GenerateModel("qattention_int8_int8.onnx", sign_i=True, sign_w=True, output_type_fp16=False)
