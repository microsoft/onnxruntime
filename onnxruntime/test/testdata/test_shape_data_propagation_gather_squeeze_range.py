# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

# Lock fixture for the Shape -> Gather(1-D index) -> Squeeze -> Range chain.
#
# The Gather index is the 1-D constant [-1] (rank 1), so the Gather output is a
# rank-1, single-element int64 value (the last dimension of X, 2000). Squeeze
# then removes the size-1 dimension, producing a 0-D scalar -- this is the
# correct, strict-improvement behavior of Squeeze's custom data propagation
# (rank-1 [1] -> scalar), and this fixture locks it: the scalar must keep its
# value (2000) so the downstream Range produces a concrete 1-D length.
#
# Range(start=0, limit=K, delta=1) with the propagated scalar K=2000 yields a
# 1-D tensor of length 2000. Asserting that output length proves Squeeze
# propagated the single element as a scalar of the correct value (and did not
# drop or corrupt it).

from onnx import IR_VERSION, TensorProto, checker, helper, save

# Graph input: a 1-D float tensor of static length 2000.
input_tensor = helper.make_tensor_value_info("X", TensorProto.FLOAT, [2000])

# Output: Range result, a 1-D int64 tensor whose length is the propagated K.
range_out = helper.make_tensor_value_info("R", TensorProto.INT64, ["range_len"])

# Gather index is the 1-D constant [-1] (last dimension).
gather_idx = helper.make_tensor("gather_idx", TensorProto.INT64, [1], [-1])
# Range start/delta as 0-D scalar int64 constants.
range_start = helper.make_tensor("range_start", TensorProto.INT64, [], [0])
range_delta = helper.make_tensor("range_delta", TensorProto.INT64, [], [1])

# Shape(X) -> S : 1-D int64 tensor [2000].
shape_node = helper.make_node("Shape", inputs=["X"], outputs=["S"], name="ShapeNode")
# Gather(S, [-1]) -> K1 : rank-1 size-1 int64 value [2000].
gather_node = helper.make_node("Gather", inputs=["S", "gather_idx"], outputs=["K1"], axis=0, name="GatherNode")
# Squeeze(K1) -> K : 0-D scalar int64 value 2000 (size-1 dim removed).
squeeze_node = helper.make_node("Squeeze", inputs=["K1"], outputs=["K"], name="SqueezeNode")
# Range(0, K, 1) -> R : 1-D int64 tensor of length K (== 2000).
range_node = helper.make_node("Range", inputs=["range_start", "K", "range_delta"], outputs=["R"], name="RangeNode")

graph = helper.make_graph(
    nodes=[shape_node, gather_node, squeeze_node, range_node],
    name="Shape_Gather_Squeeze_Range_Model",
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
save(model, "test_shape_data_propagation_gather_squeeze_range.onnx")

print("Model saved to test_shape_data_propagation_gather_squeeze_range.onnx")
