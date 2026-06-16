# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

# Lock fixture for Unsqueeze's custom data-propagation DECLINE on a single-element value.
#
# Shape(X) -> Gather([-1]) produces a rank-1 single-element value (the last dimension of X, 2000).
# Unsqueezing that single-element (scalar-like, rank-1 [1]) value would yield a rank >= 2 result
# ([1, 2000]) that the single-value data-propagation channel cannot faithfully represent, so
# Unsqueeze's custom data propagation must DECLINE rather than fabricate the misleading [1, value].
#
# The decline is made OBSERVABLE through a rank-lowering Squeeze:
#   Shape(X) -> Gather([-1]) -> Unsqueeze([0]) -> Squeeze -> Range(0, K, 1)
# With the correct decline, no value is propagated past Unsqueeze, Squeeze has nothing to lower, and
# Range's limit stays symbolic -- the model loads and the Range output length is an unknown dimension.
# If the decline were relaxed to fabricate [1, 2000], Squeeze would lower a non-scalar value into
# Range, which ONNX Range shape inference rejects ("Input to 'Range' op should be scalars"), so the
# model fails to load. The decline is thus locked end to end: the correct path loads with a symbolic
# Range length, while the rank-fabricating bug fails to load.

from onnx import IR_VERSION, TensorProto, checker, helper, save

# Graph input: a 1-D float tensor of static length 2000.
input_tensor = helper.make_tensor_value_info("X", TensorProto.FLOAT, [2000])

# Output: Range result, a 1-D int64 tensor whose length is the propagated K (symbolic on decline).
range_out = helper.make_tensor_value_info("R", TensorProto.INT64, ["range_len"])

# Gather index is the 1-D constant [-1] (last dimension) -> rank-1 single-element value.
gather_idx = helper.make_tensor("gather_idx", TensorProto.INT64, [1], [-1])
# Unsqueeze axes: insert a leading dimension (axis 0).
unsqueeze_axes = helper.make_tensor("unsqueeze_axes", TensorProto.INT64, [1], [0])
# Range start/delta as 0-D scalar int64 constants.
range_start = helper.make_tensor("range_start", TensorProto.INT64, [], [0])
range_delta = helper.make_tensor("range_delta", TensorProto.INT64, [], [1])

# Shape(X) -> S : 1-D int64 tensor [2000].
shape_node = helper.make_node("Shape", inputs=["X"], outputs=["S"], name="ShapeNode")
# Gather(S, [-1]) -> G : rank-1 size-1 int64 value [2000].
gather_node = helper.make_node("Gather", inputs=["S", "gather_idx"], outputs=["G"], axis=0, name="GatherNode")
# Unsqueeze(G, [0]) -> U : rank-2 [1, 1] tensor; custom data propagation must decline.
unsqueeze_node = helper.make_node("Unsqueeze", inputs=["G", "unsqueeze_axes"], outputs=["U"], name="UnsqueezeNode")
# Squeeze(U) -> K : 0-D scalar int64 value (only when a value is propagated).
squeeze_node = helper.make_node("Squeeze", inputs=["U"], outputs=["K"], name="SqueezeNode")
# Range(0, K, 1) -> R : 1-D int64 tensor whose length is K (symbolic on decline).
range_node = helper.make_node("Range", inputs=["range_start", "K", "range_delta"], outputs=["R"], name="RangeNode")

graph = helper.make_graph(
    nodes=[shape_node, gather_node, unsqueeze_node, squeeze_node, range_node],
    name="Shape_Gather_Unsqueeze_Decline_Model",
    inputs=[input_tensor],
    outputs=[range_out],
    initializer=[gather_idx, unsqueeze_axes, range_start, range_delta],
)

model = helper.make_model(
    graph,
    opset_imports=[helper.make_operatorsetid("", 18)],
    producer_name="onnx-example-generator",
)
model.ir_version = IR_VERSION

checker.check_model(model)
save(model, "test_shape_data_propagation_unsqueeze_decline.onnx")

print("Model saved to test_shape_data_propagation_unsqueeze_decline.onnx")
