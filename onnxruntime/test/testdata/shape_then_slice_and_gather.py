from pathlib import Path

import onnx
from onnx import helper
from onnx.onnx_pb import TensorProto

nodes = [
    helper.make_node("Shape", ["X"], ["X_shape"]),
    helper.make_node("Gather", ["X_shape", "gather_indices"], ["gather_out"], axis=0),
    helper.make_node("Slice", ["X_shape", "slice_starts", "slice_ends", "slice_axes", "slice_steps"], ["slice_out"]),
]

graph = helper.make_graph(
    nodes,
    "ShapeThenSliceAndGather",
    [  # input
        helper.make_tensor_value_info("X", TensorProto.FLOAT, ["A", "B", "C", "D", "E"]),
    ],
    [  # output
        helper.make_tensor_value_info("gather_out", TensorProto.INT64, None),
        helper.make_tensor_value_info("slice_out", TensorProto.INT64, None),
    ],
    [  # initializer
        helper.make_tensor("gather_indices", TensorProto.INT64, [2], [1, 3]),
        helper.make_tensor("slice_starts", TensorProto.INT64, [1], [-1]),
        helper.make_tensor("slice_ends", TensorProto.INT64, [1], [-6]),
        helper.make_tensor("slice_axes", TensorProto.INT64, [1], [0]),
        helper.make_tensor("slice_steps", TensorProto.INT64, [1], [-2]),
    ],
)

model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 19)])
onnx.save(model, str(Path(__file__).with_suffix(".onnx")))
