from pathlib import Path

import onnx
from onnx import helper
from onnx.onnx_pb import TensorProto

nodes = [
    helper.make_node("Gather", ["X", "indices"], ["Y"], axis=1),
    helper.make_node("Shape", ["Y"], ["Y_shape"]),
]

graph = helper.make_graph(
    nodes,
    "GatherWithScalarIndicesThenShape",
    [  # input
        helper.make_tensor_value_info("X", TensorProto.FLOAT, ["M", "N", "K"]),
        helper.make_tensor_value_info("indices", TensorProto.INT64, []),
    ],
    [  # output
        helper.make_tensor_value_info("Y", TensorProto.FLOAT, None),
        helper.make_tensor_value_info("Y_shape", TensorProto.INT64, None),
    ],
)

model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 19)])
onnx.save(model, str(Path(__file__).with_suffix(".onnx")))
