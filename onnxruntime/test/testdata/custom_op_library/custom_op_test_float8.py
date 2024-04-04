"""
This file was used to generate model `custom_op_test_float8.py`.
"""

from onnx import TensorProto
from onnx.checker import check_model
from onnx.helper import make_graph, make_model, make_node, make_opsetid, make_tensor_value_info

X = make_tensor_value_info("X", TensorProto.FLOAT8E4M3FN, [None])
Y = make_tensor_value_info("Y", TensorProto.FLOAT8E4M3FN, [None])
Z = make_tensor_value_info("Z", TensorProto.FLOAT8E4M3FN, [None])
graph = make_graph(
    [make_node("CustomOpOneFloat8", ["X", "Y"], ["Z"], domain="test.customop")],
    "custom_op",
    [X, Y],
    [Z],
)
onnx_model = make_model(
    graph,
    opset_imports=[make_opsetid("", 19), make_opsetid("test.customop", 1)],
    ir_version=9,
)
check_model(onnx_model)
with open("custom_op_test_float8.onnx", "wb") as f:
    f.write(onnx_model.SerializeToString())
