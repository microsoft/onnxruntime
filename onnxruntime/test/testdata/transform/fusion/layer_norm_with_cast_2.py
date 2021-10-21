import onnx
import numpy as np
from onnx import helper
from onnx import TensorProto
from onnx import OperatorSetIdProto
from enum import Enum


def GenerateModel(model_name):
    nodes = [  # LayerNormWithCast2 subgraph
        helper.make_node("ReduceMean", ["A"], ["rd1_out"], "reduce", axes=[-1]),
        helper.make_node("Sub", ["A", "rd1_out"], ["sub1_out"], "sub"),
        helper.make_node("Cast", ["pow_in_2"], ["cast_out"], "cast", to=10),
        helper.make_node("Pow", ["sub1_out", "cast_out"], ["pow_out"], "pow"),
        helper.make_node("ReduceMean", ["pow_out"], ["rd2_out"], "reduce2", axes=[-1]),
        helper.make_node("Add", ["rd2_out", "const_0"], ["add1_out"], "add"),
        helper.make_node("Sqrt", ["add1_out"], ["sqrt_out"], "sqrt"),
        helper.make_node("Div", ["sub1_out", "sqrt_out"], ["div_out"], "div"),
        helper.make_node("Mul", ["gamma", "div_out"], ["mul_out"], "mul"),
        helper.make_node("Add", ["beta", "mul_out"], ["C"], "add2"),
    ]

    initializers = [  # initializers
        helper.make_tensor('pow_in_2', TensorProto.FLOAT, [], [2]),
        helper.make_tensor('const_0', TensorProto.FLOAT16, [], [0]),
        helper.make_tensor('gamma', TensorProto.FLOAT16, [4], [1, 2, 3, 4]),
        helper.make_tensor('beta', TensorProto.FLOAT16, [4], [1, 2, 3, 4]),
    ]

    graph = helper.make_graph(
        nodes,
        "LayerNormWithCast2",  #name
        [  # inputs
            helper.make_tensor_value_info('A', TensorProto.FLOAT16, [16, 32, 4]),
        ],
        [  # outputs
            helper.make_tensor_value_info('C', TensorProto.FLOAT16, [16, 32, 4]),
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


GenerateModel('layer_norm_with_cast_2.onnx')