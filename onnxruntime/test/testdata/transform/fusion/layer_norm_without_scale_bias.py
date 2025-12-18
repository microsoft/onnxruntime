import onnx
from onnx import OperatorSetIdProto, TensorProto, helper


def GenerateModel(model_name):  # noqa: N802
    nodes = [  # LayerNormWithCast2 subgraph
        helper.make_node("ReduceMean", ["X"], ["rd1_out"], "reduce", axes=[-1]),
        helper.make_node("Sub", ["X", "rd1_out"], ["sub1_out"], "sub"),
        helper.make_node("Pow", ["sub1_out", "pow_in_2"], ["pow_out"], "pow"),
        helper.make_node("ReduceMean", ["pow_out"], ["rd2_out"], "reduce2", axes=[-1]),
        helper.make_node("Add", ["rd2_out", "const_0"], ["add1_out"], "add"),
        helper.make_node("Sqrt", ["add1_out"], ["sqrt_out"], "sqrt"),
        helper.make_node("Div", ["sub1_out", "sqrt_out"], ["Y"], "div"),
    ]

    initializers = [  # initializers
        helper.make_tensor("pow_in_2", TensorProto.FLOAT, [], [2]),
        helper.make_tensor("const_0", TensorProto.FLOAT, [], [0]),
    ]

    graph = helper.make_graph(
        nodes,
        "LayerNormWithoutScaleBias",  # name
        [  # inputs
            helper.make_tensor_value_info("X", TensorProto.FLOAT, [16, 32, 4]),
        ],
        [  # outputs
            helper.make_tensor_value_info("Y", TensorProto.FLOAT, [16, 32, 4]),
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


GenerateModel("layer_norm_without_scale_bias.onnx")
