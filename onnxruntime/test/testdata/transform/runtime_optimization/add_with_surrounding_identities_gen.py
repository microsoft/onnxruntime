import onnx
from onnx import helper
from onnx import TensorProto

graph = helper.make_graph(
    [  # nodes
        helper.make_node("Identity", ["A"], ["A_inner"], "id0"),
        helper.make_node("Identity", ["B"], ["B_inner"], "id1"),
        helper.make_node("Add", ["A_inner", "B_inner"], ["C_inner"], "add0"),
        helper.make_node("Identity", ["C_inner"], ["C"], "id2"),
    ],
    "AddWithSurroundingIdentities",  # name
    [  # inputs
        helper.make_tensor_value_info('A', TensorProto.FLOAT, [1]),
        helper.make_tensor_value_info('B', TensorProto.FLOAT, [1]),
    ],
    [  # outputs
        helper.make_tensor_value_info('C', TensorProto.FLOAT, [1]),
    ],
    [  # initializers
    ])

model = helper.make_model(graph)
onnx.save(model, r'add_with_surrounding_identities.onnx')
