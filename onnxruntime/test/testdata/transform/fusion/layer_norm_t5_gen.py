import onnx
from onnx import helper
from onnx import TensorProto
from onnx import OperatorSetIdProto
from enum import Enum


def GenerateModel(model_name, has_casts=False):
    nodes = [  # SimplifiedLayerNorm subgraph
        helper.make_node("Pow", ["cast_A" if has_casts else "A", "pow_in_2"], ["pow_out"], "pow"),
        helper.make_node("ReduceMean", ["pow_out"], ["rd2_out"], "reduce", axes=[-1], keepdims=1),
        helper.make_node("Add", ["rd2_out", "const_e12"], ["add1_out"], "add"),
        helper.make_node("Sqrt", ["add1_out"], ["sqrt_out"], "sqrt"),
        helper.make_node("Div", ["cast_A" if has_casts else "A", "sqrt_out"], ["div_out"], "div"),
        helper.make_node("Mul", ["gamma", "cast_div_out" if has_casts else "div_out"], ["C"], "mul"),
    ]

    if has_casts:
        nodes.extend(
            [
                helper.make_node("Cast", ["A"], ["cast_A"], "cast A", to=1),
                helper.make_node("Cast", ["div_out"], ["cast_div_out"], "cast_2", to=10),
            ]
        )

    initializers = [  # initializers
        helper.make_tensor('pow_in_2', TensorProto.FLOAT, [], [2]),
        helper.make_tensor('const_e12', TensorProto.FLOAT, [], [1e-12]),
        helper.make_tensor('gamma', TensorProto.FLOAT16 if has_casts else
                           TensorProto.FLOAT, [4], [1, 2, 3, 4]),
    ]

    input_type = TensorProto.FLOAT16 if has_casts else TensorProto.FLOAT
    output_type = TensorProto.FLOAT16 if has_casts else TensorProto.FLOAT

    graph = helper.make_graph(
        nodes,
        "SimplifiedLayerNorm",  #name
        [  # inputs
            helper.make_tensor_value_info('A', input_type, [16, 32, 4]),
        ],
        [  # outputs
            helper.make_tensor_value_info('C', output_type, [16, 32, 4]),
        ],
        initializers)

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


GenerateModel('layer_norm_t5.onnx')
GenerateModel('simplified_layer_norm_with_casts.onnx', True)
