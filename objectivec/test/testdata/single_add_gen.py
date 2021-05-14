import onnx
from onnx import helper
from onnx import TensorProto

graph = helper.make_graph(
    [  # nodes
        helper.make_node("Add", ["A", "B"], ["C"], "Add"),
    ],
    "SingleAdd",  # name
    [  # inputs
        helper.make_tensor_value_info('A', TensorProto.FLOAT, [1]),
        helper.make_tensor_value_info('B', TensorProto.FLOAT, [1]),
    ],
    [  # outputs
        helper.make_tensor_value_info('C', TensorProto.FLOAT, [1]),
    ])

model = helper.make_model(graph)
onnx.save(model, r'single_add.onnx')
