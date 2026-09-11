import onnx
from onnx import TensorProto, helper

# Create a simple ONNX model with DDS output
input = helper.make_tensor_value_info("data", TensorProto.FLOAT, ["d1", "d2"])
output = helper.make_tensor_value_info("output", TensorProto.FLOAT, ["nzr"])

nonzeros_node = helper.make_node("NonZero", ["data"], ["nonzeros"], "nonzeros_node")
transpose_node = helper.make_node("Transpose", ["nonzeros"], ["nonzeros_t"], "transpose_node")
gathernd_node = helper.make_node("GatherND", ["data", "nonzeros_t"], ["output"], "gathernd_node")

value_info = [
    helper.make_tensor_value_info("nonzeros", TensorProto.INT64, [2, "nzr"]),
    helper.make_tensor_value_info("nonzeros_t", TensorProto.INT64, ["nzr", 2]),
]

graph = helper.make_graph(
    [nonzeros_node, transpose_node, gathernd_node],
    "test_graph",
    [input],
    [output],
    value_info=value_info,
)

model = helper.make_model(graph)
onnx.save(model, "ort_github_issue_26272_dds.onnx")
