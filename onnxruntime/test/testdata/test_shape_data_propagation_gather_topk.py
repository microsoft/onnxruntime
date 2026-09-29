# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

# Regression fixture for the Shape -> Gather(1-D index) -> TopK rank-drop.
#
# The Gather index here is the 1-D constant [-1] (rank 1), so per ONNX Gather
# semantics (output_rank = data_rank - 1 + indices_rank) the Gather output is a
# rank-1, single-element tensor -- exactly what TopK's K input requires.
#
# Before the fix, ORT's Gather custom data propagation scalarized this single
# element (dropping the rank), producing a 0-D K initializer that ONNX TopK
# shape inference rejects with:
#   "K input must be a one-dimensional tensor of size 1."
# The fix preserves the rank-1 shape so the model loads and infers correctly.

from onnx import IR_VERSION, TensorProto, checker, helper, save

# Graph input: a 1-D float tensor of static length 2000.
input_tensor = helper.make_tensor_value_info("X", TensorProto.FLOAT, [2000])

# Outputs: TopK values and indices, each a 1-D tensor whose length is K.
values_out = helper.make_tensor_value_info("V", TensorProto.FLOAT, ["topk"])
indices_out = helper.make_tensor_value_info("I", TensorProto.INT64, ["topk"])

# Gather index is the 1-D constant [-1] (last dimension), as emitted by common
# exporters for the "K = size of the last dimension" pattern.
gather_idx = helper.make_tensor("gather_idx", TensorProto.INT64, [1], [-1])

# Shape(X) -> S : 1-D int64 tensor [2000].
shape_node = helper.make_node("Shape", inputs=["X"], outputs=["S"], name="ShapeNode")
# Gather(S, [-1]) -> K : rank-1 size-1 int64 tensor [2000].
gather_node = helper.make_node("Gather", inputs=["S", "gather_idx"], outputs=["K"], axis=0, name="GatherNode")
# TopK(X, K) -> V, I : returns all 2000 elements sorted (K == 2000).
topk_node = helper.make_node("TopK", inputs=["X", "K"], outputs=["V", "I"], axis=-1, largest=1, name="TopKNode")

graph = helper.make_graph(
    nodes=[shape_node, gather_node, topk_node],
    name="Shape_Gather_TopK_Model",
    inputs=[input_tensor],
    outputs=[values_out, indices_out],
    initializer=[gather_idx],
)

model = helper.make_model(
    graph,
    opset_imports=[helper.make_operatorsetid("", 18)],
    producer_name="onnx-example-generator",
)
model.ir_version = IR_VERSION

checker.check_model(model)
save(model, "test_shape_data_propagation_gather_topk.onnx")

print("Model saved to test_shape_data_propagation_gather_topk.onnx")
