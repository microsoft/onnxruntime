import onnx
from onnx import OperatorSetIdProto, TensorProto, helper


def GenerateModel(model_name):  # noqa: N802
    nodes = [  # LayerNormWithCast3 subgraph
        helper.make_node("ReduceMean", ["A"], ["rd1_out"], "reduce", axes=[-1]),
        helper.make_node("Sub", ["A", "rd1_out"], ["sub1_out"], "sub"),
        helper.make_node("Pow", ["sub1_out", "pow_in_2"], ["pow_out"], "pow"),
        helper.make_node("ReduceMean", ["pow_out"], ["rd2_out"], "reduce2", axes=[-1]),
        helper.make_node("Add", ["rd2_out", "const_0"], ["add1_out"], "add"),
        helper.make_node("Sqrt", ["add1_out"], ["sqrt_out"], "sqrt"),
        helper.make_node("Div", ["sub1_out", "sqrt_out"], ["div_out"], "div"),
        helper.make_node("Cast", ["div_out"], ["cast_out"], "cast", to=10),
        helper.make_node("Mul", ["gamma", "cast_out"], ["mul_out"], "mul"),
        helper.make_node("Add", ["beta", "mul_out"], ["C"], "add2"),
    ]

    initializers = [  # initializers
        helper.make_tensor("pow_in_2", TensorProto.FLOAT, [], [2]),
        helper.make_tensor("const_0", TensorProto.FLOAT, [], [0]),
        helper.make_tensor("gamma", TensorProto.FLOAT16, [4], [1, 2, 3, 4]),
        helper.make_tensor("beta", TensorProto.FLOAT16, [4], [1, 2, 3, 4]),
    ]

    graph = helper.make_graph(
        nodes,
        "LayerNormWithCast3",  # name
        [  # inputs
            helper.make_tensor_value_info("A", TensorProto.FLOAT, [16, 32, 4]),
        ],
        [  # outputs
            helper.make_tensor_value_info("C", TensorProto.FLOAT16, [16, 32, 4]),
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


GenerateModel("layer_norm_with_cast_3.onnx")


def GenerateModel2(model_name):  # noqa: N802
    nodes = [  # LayerNormWithCast4 subgraph
        helper.make_node("Cast", ["A"], ["cast_A"], "cast1", to=1),
        helper.make_node("ReduceMean", ["cast_A"], ["rd1_out"], "reduce", axes=[-1]),
        helper.make_node("Sub", ["cast_A", "rd1_out"], ["sub1_out"], "sub1"),
        helper.make_node("Sub", ["cast_A", "rd1_out"], ["sub2_out"], "sub2"),
        helper.make_node("Pow", ["sub1_out", "pow_in_2"], ["pow_out"], "pow"),
        helper.make_node("ReduceMean", ["pow_out"], ["rd2_out"], "reduce2", axes=[-1]),
        helper.make_node("Add", ["rd2_out", "const_0"], ["add1_out"], "add"),
        helper.make_node("Sqrt", ["add1_out"], ["sqrt_out"], "sqrt"),
        helper.make_node("Div", ["sub2_out", "sqrt_out"], ["div_out"], "div"),
        helper.make_node("Cast", ["div_out"], ["cast_out"], "cast2", to=10),
        helper.make_node("Mul", ["gamma", "cast_out"], ["mul_out"], "mul"),
        helper.make_node("Add", ["beta", "mul_out"], ["C"], "add2"),
    ]

    initializers = [  # initializers
        helper.make_tensor("pow_in_2", TensorProto.FLOAT, [], [2]),
        helper.make_tensor("const_0", TensorProto.FLOAT, [], [0]),
        helper.make_tensor("gamma", TensorProto.FLOAT16, [4], [1, 2, 3, 4]),
        helper.make_tensor("beta", TensorProto.FLOAT16, [4], [1, 2, 3, 4]),
    ]

    graph = helper.make_graph(
        nodes,
        "LayerNormWithCast4",  # name
        [  # inputs
            helper.make_tensor_value_info("A", TensorProto.FLOAT16, [16, 32, 4]),
        ],
        [  # outputs
            helper.make_tensor_value_info("C", TensorProto.FLOAT16, [16, 32, 4]),
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


GenerateModel2("layer_norm_with_cast_4.onnx")
