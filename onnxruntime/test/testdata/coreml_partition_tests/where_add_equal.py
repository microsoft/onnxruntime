import numpy as np
import onnx

# === Inputs ===
cond_input = onnx.helper.make_tensor_value_info("cond", onnx.TensorProto.BOOL, [2, 3])
x_input = onnx.helper.make_tensor_value_info("X", onnx.TensorProto.FLOAT, [2, 3])

# === Outputs ===
neg_output = onnx.helper.make_tensor_value_info("neg_out", onnx.TensorProto.FLOAT, [2, 3])
final_output = onnx.helper.make_tensor_value_info("final_out", onnx.TensorProto.FLOAT, [2, 3])

# === Initializers ===
Y_const = onnx.helper.make_tensor(
    "Y_const", onnx.TensorProto.FLOAT, [2, 3], np.random.randn(2, 3).astype(np.float32).flatten()
)
cmp_const = onnx.helper.make_tensor(
    "cmp_const", onnx.TensorProto.FLOAT, [2, 3], np.random.randn(2, 3).astype(np.float32).flatten()
)
true_val = onnx.helper.make_tensor(
    "true_val", onnx.TensorProto.FLOAT, [2, 3], np.random.randn(2, 3).astype(np.float32).flatten()
)
false_val = onnx.helper.make_tensor(
    "false_val", onnx.TensorProto.FLOAT, [2, 3], np.random.randn(2, 3).astype(np.float32).flatten()
)

# === Nodes ===

# Where(cond, X, Y_const)
where1 = onnx.helper.make_node("Where", inputs=["cond", "X", "Y_const"], outputs=["where1_out"], name="Where1")

# Neg(X)
neg = onnx.helper.make_node("Neg", inputs=["X"], outputs=["neg_out"], name="Neg")

# Add(where1_out, X)
add = onnx.helper.make_node("Add", inputs=["where1_out", "X"], outputs=["add_out"], name="Add")

# Equal(add_out, cmp_const)
equal = onnx.helper.make_node("Equal", inputs=["add_out", "cmp_const"], outputs=["equal_out"], name="Equal")

# Final Where(equal_out, true_val, false_val)
where2 = onnx.helper.make_node(
    "Where", inputs=["equal_out", "true_val", "false_val"], outputs=["final_out"], name="Where2"
)

# === Graph ===
graph = onnx.helper.make_graph(
    nodes=[where1, neg, add, equal, where2],
    name="WhereAddEqualGraph",
    inputs=[cond_input, x_input],
    outputs=[neg_output, final_output],
    initializer=[Y_const, cmp_const, true_val, false_val],
)

# === Model ===
model = onnx.helper.make_model(graph, opset_imports=[onnx.helper.make_opsetid("", 18)])
model.ir_version = onnx.IR_VERSION

# === Check ===
onnx.checker.check_model(model)

# === Save ===
onnx.save_model(model, "where_add_equal.onnx")
print("Model saved as 'where_add_equal.onnx'")
