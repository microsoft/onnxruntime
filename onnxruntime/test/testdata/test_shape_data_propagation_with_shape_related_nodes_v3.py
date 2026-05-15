import onnx
from onnx import TensorProto, helper

# === Graph input/output ===
input_tensor = helper.make_tensor_value_info("input", TensorProto.FLOAT, ["batch", 3, "width", "height"])
output_tensor = helper.make_tensor_value_info("output", TensorProto.FLOAT, ["batch", 3, "width*height"])

# === Initializers ===
B = helper.make_tensor("B", TensorProto.FLOAT, [], [1.0])

# Gather indices
g0_idx = helper.make_tensor("g0_idx", TensorProto.INT64, [], [0])
g1_idx = helper.make_tensor("g1_idx", TensorProto.INT64, [], [1])
g2_idx = helper.make_tensor("g2_idx", TensorProto.INT64, [], [2])
g3_idx = helper.make_tensor("g3_idx", TensorProto.INT64, [], [3])

# Unsqueeze axes tensors
axes_unsq0 = helper.make_tensor("axes_unsq0", TensorProto.INT64, [1], [0])
axes_unsq1 = helper.make_tensor("axes_unsq1", TensorProto.INT64, [1], [0])
axes_unsq2 = helper.make_tensor("axes_unsq2", TensorProto.INT64, [1], [0])

# === Nodes ===
div = helper.make_node("Div", ["input", "B"], ["div_out"])

# Two Shape nodes from Div
shape_left = helper.make_node("Shape", ["div_out"], ["shape_left_out"])
shape_right = helper.make_node("Shape", ["div_out"], ["shape_right_out"])

# Left Shape path
gather0 = helper.make_node("Gather", ["shape_left_out", "g0_idx"], ["g0_out"])
gather1 = helper.make_node("Gather", ["shape_left_out", "g1_idx"], ["g1_out"])
unsq0 = helper.make_node("Unsqueeze", ["g0_out", "axes_unsq0"], ["u0_out"])
unsq1 = helper.make_node("Unsqueeze", ["g1_out", "axes_unsq1"], ["u1_out"])

# Right Shape path
gather2 = helper.make_node("Gather", ["shape_right_out", "g2_idx"], ["g2_out"])
gather3 = helper.make_node("Gather", ["shape_right_out", "g3_idx"], ["g3_out"])
mul = helper.make_node("Mul", ["g2_out", "g3_out"], ["mul_out"])
unsq2 = helper.make_node("Unsqueeze", ["mul_out", "axes_unsq2"], ["u2_out"])

# Combine
concat = helper.make_node("Concat", ["u0_out", "u1_out", "u2_out"], ["concat_out"], axis=0)

# Axes initializers
axes_u1 = helper.make_tensor("axes_u1", TensorProto.INT64, [1], [1])
axes_u2 = helper.make_tensor("axes_u2", TensorProto.INT64, [1], [1])
axes_s1 = helper.make_tensor("axes_s1", TensorProto.INT64, [1], [1])
axes_s2 = helper.make_tensor("axes_s2", TensorProto.INT64, [1], [1])

# First Unsqueeze
unsqueeze1 = helper.make_node("Unsqueeze", inputs=["concat_out", "axes_u1"], outputs=["u1"], name="Unsqueeze_1")

# Second Unsqueeze
unsqueeze2 = helper.make_node("Unsqueeze", inputs=["u1", "axes_u2"], outputs=["u2"], name="Unsqueeze_2")

# First Squeeze
squeeze1 = helper.make_node("Squeeze", inputs=["u2", "axes_s1"], outputs=["s1"], name="Squeeze_1")

# Second Squeeze
squeeze2 = helper.make_node("Squeeze", inputs=["s1", "axes_s2"], outputs=["squeeze_output"], name="Squeeze_2")

reshape = helper.make_node("Reshape", ["div_out", "squeeze_output"], ["output"])

# === Graph ===
graph = helper.make_graph(
    [
        div,
        shape_left,
        shape_right,
        gather0,
        gather1,
        gather2,
        gather3,
        mul,
        unsq0,
        unsq1,
        unsq2,
        concat,
        unsqueeze1,
        unsqueeze2,
        squeeze1,
        squeeze2,
        reshape,
    ],
    "Div_Shape_Gather_Concat_Reshape",
    [input_tensor],
    [output_tensor],
    initializer=[
        B,
        g0_idx,
        g1_idx,
        g2_idx,
        g3_idx,
        axes_unsq0,
        axes_unsq1,
        axes_unsq2,
        axes_u1,
        axes_u2,
        axes_s1,
        axes_s2,
    ],
)

# === Model ===
model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)], producer_name="onnx-example")
onnx.checker.check_model(model)
onnx.save(model, "test_shape_data_propagation_with_shape_related_nodes_v3.onnx")

print("✅ Model saved as test_shape_data_propagation_with_shape_related_nodes_v3.onnx")
