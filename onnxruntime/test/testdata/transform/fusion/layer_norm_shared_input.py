# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
import onnx
from onnx import OperatorSetIdProto, TensorProto, helper


# in gpt_j_residual, there will be 2 LN share the same input
def GenerateModel(model_name):  # noqa: N802
    nodes = [
        # LN1 subgraph
        helper.make_node("ReduceMean", ["A"], ["LN1/rd1_out"], "LN1/reduce", axes=[-1]),
        helper.make_node("Sub", ["A", "LN1/rd1_out"], ["LN1/sub1_out"], "LN1/sub"),
        helper.make_node("Pow", ["LN1/sub1_out", "LN1/pow_in_2"], ["LN1/pow_out"], "LN1/pow"),
        helper.make_node("ReduceMean", ["LN1/pow_out"], ["LN1/rd2_out"], "LN1/reduce2", axes=[-1]),
        helper.make_node("Add", ["LN1/rd2_out", "LN1/const_0"], ["LN1/add1_out"], "LN1/add"),
        helper.make_node("Sqrt", ["LN1/add1_out"], ["LN1/sqrt_out"], "LN1/sqrt"),
        helper.make_node("Div", ["LN1/sub1_out", "LN1/sqrt_out"], ["LN1/div_out"], "LN1/div"),
        helper.make_node("Mul", ["LN1/gamma", "LN1/div_out"], ["LN1/mul_out"], "LN1/mul"),
        helper.make_node("Add", ["LN1/beta", "LN1/mul_out"], ["LN1/C"], "LN1/add2"),
        # LN2 subgraph
        helper.make_node("ReduceMean", ["A"], ["LN2/rd1_out"], "LN2/reduce", axes=[-1]),
        helper.make_node("Sub", ["A", "LN2/rd1_out"], ["LN2/sub1_out"], "LN2/sub"),
        helper.make_node("Pow", ["LN2/sub1_out", "LN2/pow_in_2"], ["LN2/pow_out"], "LN2/pow"),
        helper.make_node("ReduceMean", ["LN2/pow_out"], ["LN2/rd2_out"], "LN2/reduce2", axes=[-1]),
        helper.make_node("Add", ["LN2/rd2_out", "LN2/const_0"], ["LN2/add1_out"], "LN2/add"),
        helper.make_node("Sqrt", ["LN2/add1_out"], ["LN2/sqrt_out"], "LN2/sqrt"),
        helper.make_node("Div", ["LN2/sub1_out", "LN2/sqrt_out"], ["LN2/div_out"], "LN2/div"),
        helper.make_node("Mul", ["LN2/gamma", "LN2/div_out"], ["LN2/mul_out"], "LN2/mul"),
        helper.make_node("Add", ["LN2/beta", "LN2/mul_out"], ["LN2/C"], "LN2/add2"),
    ]

    initializers = [
        # LN1 initializers
        helper.make_tensor("LN1/pow_in_2", TensorProto.FLOAT, [], [2]),
        helper.make_tensor("LN1/const_0", TensorProto.FLOAT, [], [0]),
        helper.make_tensor("LN1/gamma", TensorProto.FLOAT, [4], [1, 2, 3, 4]),
        helper.make_tensor("LN1/beta", TensorProto.FLOAT, [4], [1, 2, 3, 4]),
        # LN2 initializers
        helper.make_tensor("LN2/pow_in_2", TensorProto.FLOAT, [], [2]),
        helper.make_tensor("LN2/const_0", TensorProto.FLOAT, [], [0]),
        helper.make_tensor("LN2/gamma", TensorProto.FLOAT, [4], [1, 2, 3, 4]),
        helper.make_tensor("LN2/beta", TensorProto.FLOAT, [4], [1, 2, 3, 4]),
    ]

    graph = helper.make_graph(
        nodes,
        "2LayerNormShareSameInput",  # name
        [  # inputs
            helper.make_tensor_value_info("A", TensorProto.FLOAT, [16, 32, 4]),
        ],
        [  # outputs
            helper.make_tensor_value_info("LN1/C", TensorProto.FLOAT, [16, 32, 4]),
            helper.make_tensor_value_info("LN2/C", TensorProto.FLOAT, [16, 32, 4]),
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


GenerateModel("layer_norm_shared_input.onnx")
