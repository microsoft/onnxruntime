import onnx
import numpy as np
import onnxruntime as ort

# original input_X and its associated weight
input_X = onnx.helper.make_tensor_value_info("input_X", onnx.TensorProto.FLOAT, [4, 4])

# Original weight
weight_X =  np.array([1, 2, 3, 4, 5, 6, 7, 8,
                      9, 10, 11, 12, 13, 14, 15, 16]).reshape(4, 4).astype(np.float32)

# create a tensor proto for matmul weight
lora_param_a = np.zeros([4, 0], dtype=np.float32)
tensor = onnx.helper.make_tensor("lora_param_a", onnx.TensorProto.FLOAT, [4, 0], lora_param_a.flatten())
# create a tensor value_info for matmul weight input
tensor_input = onnx.helper.make_tensor_value_info("lora_param_a", onnx.TensorProto.FLOAT, [4, "dim"])

# create a matmul node for lora_param_a
node = onnx.helper.make_node("MatMul", ["input1", "lora_param_a"], ["output"])

# create a graph
graph = onnx.helper.make_graph(
    [node], 
    "test", 
    [onnx.helper.make_tensor_value_info("input1", onnx.TensorProto.FLOAT, [1, 4]), tensor_input], 
    [onnx.helper.make_tensor_value_info("output", onnx.TensorProto.FLOAT, [1, "dim"])], 
    [tensor]
)

# create a model
model = onnx.helper.make_model(graph)


session = ort.InferenceSession(model.SerializeToString(), providers=["CUDAExecutionProvider"])

inputs = {
    "input1": np.random.randn(1, 4).astype(np.float32),
    "weight": np.random.randn(4, 3).astype(np.float32)
}

outputs = session.run(None, inputs)
print(outputs)

inputs = {
    "input1": np.random.randn(1, 4).astype(np.float32),
    # "weight": np.random.randn(4, 3).astype(np.float32)
}

outputs = session.run(None, inputs)
print(outputs)
