# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

# Lock fixture for Gather's custom data-propagation DECLINE on a rank-2 index.
#
# The Gather index is the constant [[-1]] of shape [1, 1] (rank 2). Per ONNX
# Gather semantics the output rank = data_rank - 1 + index_rank = 1 - 1 + 2 = 2,
# so the Gather output is a rank-2, single-element tensor [[2000]]. The
# single-value data-propagation channel can represent only rank-0 (scalar) and
# rank-1 [1] values, so Gather's custom data propagation must DECLINE here:
# emitting a rank-1 value would fabricate a rank the channel cannot honestly
# carry.
#
# This fixture makes that decline OBSERVABLE through a rank-lowering Squeeze:
#   Shape(X) -> Gather([[ -1 ]]) -> Squeeze -> Range(0, K, 1)
# Because the index is a constant initializer, ONNX's own Gather data
# propagator bails (it has no getInputData for a constant index), so control
# reaches our custom decline branch. With the correct decline, no value is
# propagated, Squeeze has nothing to lower, and Range's limit stays symbolic --
# the Range output length is therefore an unknown dimension.
#
# If the decline were relaxed to emit a rank-1 [1] value, Squeeze would lower it
# to the scalar 2000 and Range(0, 2000, 1) would resolve to a concrete length
# 2000. Asserting the Range length is NOT the concrete 2000 thus locks the
# decline end to end (it discriminates the correct-decline behavior from the
# rank-fabricating bug).

from onnx import IR_VERSION, TensorProto, checker, helper, save

# Graph input: a 3-D float tensor whose last static dimension is 2000.
input_tensor = helper.make_tensor_value_info("X", TensorProto.FLOAT, [3, 4, 2000])

# Output: Range result, a 1-D int64 tensor whose length is the propagated K
# (symbolic when the decline is correct).
range_out = helper.make_tensor_value_info("R", TensorProto.INT64, ["range_len"])

# Gather index is the rank-2 constant [[-1]] of shape [1, 1] (last dimension).
gather_idx = helper.make_tensor("gather_idx", TensorProto.INT64, [1, 1], [-1])
# Range start/delta as 0-D scalar int64 constants.
range_start = helper.make_tensor("range_start", TensorProto.INT64, [], [0])
range_delta = helper.make_tensor("range_delta", TensorProto.INT64, [], [1])

# Shape(X) -> S : 1-D int64 tensor [3, 4, 2000].
shape_node = helper.make_node("Shape", inputs=["X"], outputs=["S"], name="ShapeNode")
# Gather(S, [[-1]]) -> G : rank-2 [1, 1] int64 value [[2000]].
gather_node = helper.make_node("Gather", inputs=["S", "gather_idx"], outputs=["G"], axis=0, name="GatherNode")
# Squeeze(G) -> K : 0-D scalar int64 value (only when a value is propagated).
squeeze_node = helper.make_node("Squeeze", inputs=["G"], outputs=["K"], name="SqueezeNode")
# Range(0, K, 1) -> R : 1-D int64 tensor whose length is K (symbolic on decline).
range_node = helper.make_node("Range", inputs=["range_start", "K", "range_delta"], outputs=["R"], name="RangeNode")

graph = helper.make_graph(
    nodes=[shape_node, gather_node, squeeze_node, range_node],
    name="Shape_Gather_Rank2_Decline_Model",
    inputs=[input_tensor],
    outputs=[range_out],
    initializer=[gather_idx, range_start, range_delta],
)

model = helper.make_model(
    graph,
    opset_imports=[helper.make_operatorsetid("", 18)],
    producer_name="onnx-example-generator",
)
model.ir_version = IR_VERSION

checker.check_model(model)
save(model, "test_shape_data_propagation_gather_rank2_decline.onnx")

print("Model saved to test_shape_data_propagation_gather_rank2_decline.onnx")
