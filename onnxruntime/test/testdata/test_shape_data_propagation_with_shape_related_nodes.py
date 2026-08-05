import onnx
from onnx import TensorProto, helper

# 1. Define graph input with symbolic shape ['batch', 3, 'width', 'height']
input_tensor = helper.make_tensor_value_info("input", TensorProto.FLOAT, ["batch", 3, "width", "height"])

# 2. Define intermediate and output tensors
shape_out = helper.make_tensor_value_info("shape_out", TensorProto.INT64, [4])  # Shape output
reshape_a_out = helper.make_tensor_value_info("reshape_a_out", TensorProto.FLOAT, None)
output_tensor = helper.make_tensor_value_info("output", TensorProto.FLOAT, None)

# 3. Create the initializer for Reshape A's 'shape' input: [0, 32, -1]
shape_initializer = helper.make_tensor(
    name="reshape_a_shape",
    data_type=TensorProto.INT64,
    dims=[3],
    vals=[0, 32, -1],
)

# 4. Create nodes:
# Shape node
shape_node = helper.make_node("Shape", inputs=["input"], outputs=["shape_out"], name="ShapeNode")

# Reshape A node: takes input + constant shape
reshape_a_node = helper.make_node(
    "Reshape", inputs=["input", "reshape_a_shape"], outputs=["reshape_a_out"], name="ReshapeA"
)

# Reshape B node: takes Shape + ReshapeA outputs, outputs final output
reshape_b_node = helper.make_node("Reshape", inputs=["reshape_a_out", "shape_out"], outputs=["output"], name="ReshapeB")

# 5. Assemble the graph
graph = helper.make_graph(
    nodes=[shape_node, reshape_a_node, reshape_b_node],
    name="Shape_Reshape_Model",
    inputs=[input_tensor],
    outputs=[output_tensor],
    initializer=[shape_initializer],
    value_info=[shape_out, reshape_a_out],
)

# 6. Define the model (set IR and opset)
model = helper.make_model(
    graph,
    opset_imports=[helper.make_operatorsetid("", 18)],
    producer_name="onnx-example-generator",
)
model.ir_version = onnx.IR_VERSION

# 7. Save the model
onnx.save(model, "test_shape_data_propagation_with_shape_related_nodes.onnx")

print("Model saved to test_shape_data_propagation_with_shape_related_nodes.onnx")
