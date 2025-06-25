import onnx
from onnx import helper, TensorProto, checker, save_model
import numpy as np

# === Inputs ===
cond_input = helper.make_tensor_value_info("cond", TensorProto.BOOL, [2, 3])
x_input = helper.make_tensor_value_info("X", TensorProto.FLOAT, [2, 3])

# === Outputs ===
neg_output = helper.make_tensor_value_info("neg_out", TensorProto.FLOAT, [2, 3])
final_output = helper.make_tensor_value_info("final_out", TensorProto.FLOAT, [2, 3])

# === Initializers ===
Y_const = helper.make_tensor("Y_const", TensorProto.FLOAT, [2, 3], np.random.randn(2, 3).astype(np.float32).flatten())
cmp_const = helper.make_tensor("cmp_const", TensorProto.FLOAT, [2, 3], np.random.randn(2, 3).astype(np.float32).flatten())
true_val = helper.make_tensor("true_val", TensorProto.FLOAT, [2, 3], np.random.randn(2, 3).astype(np.float32).flatten())
false_val = helper.make_tensor("false_val", TensorProto.FLOAT, [2, 3], np.random.randn(2, 3).astype(np.float32).flatten())

# === Nodes ===

# Where(cond, X, Y_const)
where1 = helper.make_node(
    "Where",
    inputs=["cond", "X", "Y_const"],
    outputs=["where1_out"],
    name="Where1"
)

# Neg(X)
neg = helper.make_node(
    "Neg",
    inputs=["X"],
    outputs=["neg_out"],
    name="Neg"
)

# Add(where1_out, X)
add = helper.make_node(
    "Add",
    inputs=["where1_out", "X"],
    outputs=["add_out"],
    name="Add"
)

# Equal(add_out, cmp_const)
equal = helper.make_node(
    "Equal",
    inputs=["add_out", "cmp_const"],
    outputs=["equal_out"],
    name="Equal"
)

# Final Where(equal_out, true_val, false_val)
where2 = helper.make_node(
    "Where",
    inputs=["equal_out", "true_val", "false_val"],
    outputs=["final_out"],
    name="Where2"
)

# === Graph ===
graph = helper.make_graph(
    nodes=[where1, neg, add, equal, where2],
    name="WhereAddEqualGraph",
    inputs=[cond_input, x_input],
    outputs=[neg_output, final_output],
    initializer=[Y_const, cmp_const, true_val, false_val]
)

# === Model ===
model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
model.ir_version = onnx.IR_VERSION

# === Check ===
checker.check_model(model)

# === Save ===
save_model(model, "where_add_equal.onnx")
print("Model saved as 'where_add_equal.onnx'")
