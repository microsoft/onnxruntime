import onnx

axis_count = 16
input_tensor = onnx.helper.make_tensor_value_info("input", onnx.TensorProto.FLOAT, [1] * axis_count)
output_tensor = onnx.helper.make_tensor_value_info("output", onnx.TensorProto.INT64, [1] * axis_count + [axis_count])

shape_node = onnx.helper.make_node("Shape", ["input"], ["shape_out"])
identity_node = onnx.helper.make_node("Identity", ["shape_out"], ["identity_out"])
axes = onnx.helper.make_tensor("unsq_axes", onnx.TensorProto.INT64, [axis_count], list(range(axis_count)))
unsqueeze_node = onnx.helper.make_node("Unsqueeze", ["identity_out", "unsq_axes"], ["output"])

graph = onnx.helper.make_graph(
    [shape_node, identity_node, unsqueeze_node],
    "Unsqueeze_InMemory_INT64_Axes",
    [input_tensor],
    [output_tensor],
    initializer=[axes],
)

model = onnx.helper.make_model(graph, opset_imports=[onnx.helper.make_opsetid("", 18)], producer_name="onnx-example")
onnx.checker.check_model(model)
onnx.save(model, "test_shape_data_propagation_unsqueeze_inmemory_int64.onnx")

print("Model saved to test_shape_data_propagation_unsqueeze_inmemory_int64.onnx")
