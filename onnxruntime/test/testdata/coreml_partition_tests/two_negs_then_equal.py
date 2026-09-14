import onnx

# Define inputs
input1 = onnx.helper.make_tensor_value_info("x1", onnx.TensorProto.INT64, ["dim"])
input2 = onnx.helper.make_tensor_value_info("x2", onnx.TensorProto.INT64, ["dim"])

# Intermediate outputs
neg1_out = "neg1_out"
neg2_out = "neg2_out"

# Final output
equal_out = onnx.helper.make_tensor_value_info("out", onnx.TensorProto.BOOL, ["dim"])

# Create the nodes
neg1 = onnx.helper.make_node("Neg", inputs=["x1"], outputs=[neg1_out], name="Neg_1")

neg2 = onnx.helper.make_node("Neg", inputs=["x2"], outputs=[neg2_out], name="Neg_2")

equal = onnx.helper.make_node("Equal", inputs=[neg1_out, neg2_out], outputs=["out"], name="Equal")

# Create the graph
graph = onnx.helper.make_graph(
    nodes=[neg1, neg2, equal], name="NegEqualGraph", inputs=[input1, input2], outputs=[equal_out]
)

# Create the model
model = onnx.helper.make_model(graph, opset_imports=[onnx.helper.make_opsetid("", 13)])
model.ir_version = onnx.IR_VERSION

# Save to file
onnx.save_model(model, "two_negs_then_equal.onnx")
print("Model saved as 'two_negs_then_equal.onnx'")

onnx.checker.check_model(model)
