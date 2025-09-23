import onnx
from onnx import TensorProto, helper

# input tensor
X = helper.make_tensor_value_info("X", TensorProto.FLOAT16, [3, 4])

# output tensors
Values = helper.make_tensor_value_info("Values", TensorProto.FLOAT16, [3, 2])
Indices = helper.make_tensor_value_info("Indices", TensorProto.INT64, [3, 2])

# constant for k
k_const = helper.make_tensor("k", TensorProto.INT64, [1], [2])

node = helper.make_node("TopK", ["X", "k"], ["Values", "Indices"], axis=1)

graph = helper.make_graph([node], "TopKGraph", [X], [Values, Indices], [k_const])
model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])

onnx.save(model, "constant_float16_topk.onnx")
