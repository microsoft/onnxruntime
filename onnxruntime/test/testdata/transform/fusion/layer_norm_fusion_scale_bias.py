# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
import onnx
from onnx import OperatorSetIdProto, TensorProto, helper


def GenerateModel(model_name, has_casts=False, has_identity=False):  # noqa: N802
    nodes = [  # LayerNorm subgraph
        helper.make_node("ReduceMean", ["A"], ["rd_out"], "reduce1", axes=[-1], keepdims=1),
        helper.make_node("Sub", ["A", "rd_out"], ["sub_out"], "sub"),
        helper.make_node("Pow", ["cast_sub_out" if has_casts else "sub_out", "pow_in_2"], ["pow_out"], "pow"),
        helper.make_node("ReduceMean", ["pow_out"], ["rd2_out"], "reduce2", axes=[-1], keepdims=1),
        helper.make_node("Add", ["rd2_out", "const_e12_f32"], ["add1_out"], "add1"),
        helper.make_node("Sqrt", ["add1_out"], ["sqrt_out"], "sqrt"),
        helper.make_node("Div", ["cast_sub_out" if has_casts else "sub_out", "sqrt_out"], ["div_out"], "div"),
        helper.make_node(
            "Mul",
            ["gamma_id_out" if has_identity else "gamma", "cast_div_out" if has_casts else "div_out"],
            ["mul_out"],
            "mul",
        ),
        helper.make_node("Add", ["mul_out", "const_e6_f16_out" if has_identity else "const_e6_f16"], ["C"], "add2"),
    ]

    if has_casts:
        nodes.extend(
            [
                helper.make_node("Cast", ["sub_out"], ["cast_sub_out"], "cast_sub", to=1),
                helper.make_node("Cast", ["div_out"], ["cast_div_out"], "cast_2", to=10),
            ]
        )

    if has_identity:
        nodes.extend(
            [
                helper.make_node("Identity", ["gamma"], ["gamma_id_out"], "gamma_identity"),
                helper.make_node("Identity", ["const_e6_f16"], ["const_e6_f16_out"], "const_e6_f16_identity"),
            ]
        )

    initializers = [  # initializers
        helper.make_tensor("pow_in_2", TensorProto.FLOAT, [], [2]),
        helper.make_tensor("const_e12_f32", TensorProto.FLOAT, [], [1e-12]),
        helper.make_tensor("const_e6_f16", TensorProto.FLOAT16, [4], [1e-6, 1e-6, 1e-6, 1e-6]),
        helper.make_tensor(
            "gamma",
            TensorProto.FLOAT16 if has_casts else TensorProto.FLOAT,
            [4],
            [1, 2, 3, 4],
        ),
    ]

    input_type = TensorProto.FLOAT16 if has_casts else TensorProto.FLOAT
    output_type = TensorProto.FLOAT16 if has_casts else TensorProto.FLOAT

    graph = helper.make_graph(
        nodes,
        "LayerNorm",  # name
        [  # inputs
            helper.make_tensor_value_info("A", input_type, [16, 32, 4]),
        ],
        [  # outputs
            helper.make_tensor_value_info("C", output_type, [16, 32, 4]),
        ],
        initializers,
    )

    onnxdomain = OperatorSetIdProto()
    onnxdomain.version = 12
    # The empty string ("") or absence of this field implies the operator set that is defined as part of the ONNX specification.
    onnxdomain.domain = ""
    msdomain = OperatorSetIdProto()
    msdomain.version = 1
    msdomain.domain = "com.microsoft"
    opsets = [onnxdomain, msdomain]

    model = helper.make_model(graph, opset_imports=opsets)
    onnx.save(model, model_name)


GenerateModel("layer_norm_fusion_scale_bias.onnx", True, True)
