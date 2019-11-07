import onnx
from onnx import helper
from onnx import TensorProto

graph = helper.make_graph(
    [ # nodes
        # Add node before Gelu
        helper.make_node("Add", ["A", "B"], ["add0_out"], "add0"),
        
        # Gelu subgraph
        helper.make_node("Div", ["add0_out", "div_const"], ["div_out"], "div"),
        helper.make_node("Mul", ["add0_out", "mul_const"], ["mul_out"], "mul0"),
        helper.make_node("Erf", ["div_out"], ["erf_out"], "erf"),
        helper.make_node("Add", ["erf_out", "add_const"], ["add1_out"], "add1"),
        helper.make_node("Mul", ["mul_out", "add1_out"], ["C"], "mul1"),
    ],
    "Gelu_Add_Fusion",  #name
    [  # inputs
        helper.make_tensor_value_info('A', TensorProto.FLOAT, ['unk_1', 'unk_2', 3072]),
        helper.make_tensor_value_info('B', TensorProto.FLOAT, ['unk_1', 'unk_2', 3072]),
    ],
    [  # outputs
        helper.make_tensor_value_info('C', TensorProto.FLOAT, ['unk_3', 'unk_4', 3072]),
    ],
    [  # initializers
        helper.make_tensor('div_const', TensorProto.FLOAT, [], [1.4142135381698608]),
        helper.make_tensor('mul_const', TensorProto.FLOAT, [], [0.5]),
        helper.make_tensor('add_const', TensorProto.FLOAT, [], [1]),
    ]
)

model = helper.make_model(graph)
onnx.save(model, r'add_gelu_fusion.onnx')