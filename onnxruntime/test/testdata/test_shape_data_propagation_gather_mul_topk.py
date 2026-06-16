# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

# Regression fixture for the Shape -> Gather(1-D index) -> Mul -> TopK chain.
#
# This exercises the elementwise data-propagation consumers (Mul) with rank-1
# single-element operands. Both Gather indices are 1-D constants, so per ONNX
# Gather semantics each Gather output is a rank-1, single-element tensor. The
# Mul of those two rank-1 values must keep propagating (as a rank-1 [1] value)
# so the downstream TopK still receives a valid 1-D K and its output shape is
# inferred concretely.
#
# Before the elementwise ops were made rank-aware, Mul only handled rank-0
# scalar operands; with the Gather producer now emitting rank-1 values, the
# chain's data propagation would otherwise silently stop at Mul (degrading the
# TopK output dim to a symbolic value). This fixture locks in that the chain
# keeps propagating end to end.

from onnx import IR_VERSION, TensorProto, checker, helper, save

# 2-D input whose static dims (50, 40) drive the K computation: 50 * 40 == 2000.
x2d = helper.make_tensor_value_info("X2D", TensorProto.FLOAT, [50, 40])
# 1-D input that TopK selects from; its length matches the computed K (2000).
x1d = helper.make_tensor_value_info("X1D", TensorProto.FLOAT, [2000])

values_out = helper.make_tensor_value_info("V", TensorProto.FLOAT, ["topk"])
indices_out = helper.make_tensor_value_info("I", TensorProto.INT64, ["topk"])

# 1-D single-element Gather indices (rank 1), as emitted by common exporters.
idx_first = helper.make_tensor("idx_first", TensorProto.INT64, [1], [0])
idx_last = helper.make_tensor("idx_last", TensorProto.INT64, [1], [-1])

# Shape(X2D) -> S : 1-D int64 tensor [50, 40].
shape_node = helper.make_node("Shape", inputs=["X2D"], outputs=["S"], name="ShapeNode")
# Gather(S, [0]) -> A : rank-1 size-1 int64 tensor [50].
gather_first = helper.make_node("Gather", inputs=["S", "idx_first"], outputs=["A"], axis=0, name="GatherFirst")
# Gather(S, [-1]) -> B : rank-1 size-1 int64 tensor [40].
gather_last = helper.make_node("Gather", inputs=["S", "idx_last"], outputs=["B"], axis=0, name="GatherLast")
# Mul(A, B) -> K : rank-1 size-1 int64 tensor [2000].
mul_node = helper.make_node("Mul", inputs=["A", "B"], outputs=["K"], name="MulNode")
# TopK(X1D, K) -> V, I : returns all 2000 elements sorted (K == 2000).
topk_node = helper.make_node("TopK", inputs=["X1D", "K"], outputs=["V", "I"], axis=-1, largest=1, name="TopKNode")

graph = helper.make_graph(
    nodes=[shape_node, gather_first, gather_last, mul_node, topk_node],
    name="Shape_Gather_Mul_TopK_Model",
    inputs=[x2d, x1d],
    outputs=[values_out, indices_out],
    initializer=[idx_first, idx_last],
)

model = helper.make_model(
    graph,
    opset_imports=[helper.make_operatorsetid("", 18)],
    producer_name="onnx-example-generator",
)
model.ir_version = IR_VERSION

checker.check_model(model)
save(model, "test_shape_data_propagation_gather_mul_topk.onnx")

print("Model saved to test_shape_data_propagation_gather_mul_topk.onnx")
