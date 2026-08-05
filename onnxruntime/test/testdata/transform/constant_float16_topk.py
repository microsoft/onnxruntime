import onnx

# input tensor
X = onnx.helper.make_tensor_value_info("X", onnx.TensorProto.FLOAT16, [3, 4])

# output tensors
Values = onnx.helper.make_tensor_value_info("Values", onnx.TensorProto.FLOAT16, [3, 2])
Indices = onnx.helper.make_tensor_value_info("Indices", onnx.TensorProto.INT64, [3, 2])

# constant for k
k_const = onnx.helper.make_tensor("k", onnx.TensorProto.INT64, [1], [2])

node = onnx.helper.make_node("TopK", ["X", "k"], ["Values", "Indices"], axis=1)

graph = onnx.helper.make_graph([node], "TopKGraph", [X], [Values, Indices], [k_const])
model = onnx.helper.make_model(graph, opset_imports=[onnx.helper.make_opsetid("", 13)])

onnx.save(model, "constant_float16_topk.onnx")
