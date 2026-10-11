# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

# Combined lock fixture for the two custom data-propagation DECLINE paths guarded by
# microsoft/onnxruntime#29072, exercised through a SINGLE shared graph so the regression
# is covered by one onnxruntime::Model::Load instead of two (microsoft/onnxruntime#29139:
# the single-process onnxruntime_test_all run sits near the AddressSanitizer size-class
# ceiling, so the decline coverage is folded into one model to minimize the live Model/Graph
# footprint).
#
# A single rank-3 input X drives one shared Shape(X) -> S (the 1-D vector [3, 4, 2000]);
# two independent branches gather from S and route into a rank-lowering Squeeze -> Range so
# each decline is observable end to end:
#
#   Branch A (Gather rank-2 index decline):
#     S -> Gather([[ -1 ]])  -> Squeeze -> Range(0, K, 1) -> RA
#   The index [[-1]] has shape [1, 1] (rank 2), so the Gather output rank is
#   data_rank - 1 + index_rank = 1 - 1 + 2 = 2 -- a rank-2 single-element value the
#   single-value channel cannot represent, so Gather's custom propagation must DECLINE.
#   On a correct decline nothing propagates, Squeeze has nothing to lower, and RA's length
#   stays SYMBOLIC. A relaxed decline would emit a rank-1 [1] value that Squeeze lowers to
#   the scalar 2000, concretizing RA to length 2000 -- so asserting RA stays symbolic
#   discriminates the correct decline from the rank-fabricating bug.
#
#   Branch B (Unsqueeze single-element-value decline):
#     S -> Gather([-1]) -> Unsqueeze([0]) -> Squeeze -> Range(0, K, 1) -> RB
#   Gather([-1]) is a rank-1 single-element value (the last dimension of X, 2000).
#   Unsqueezing it would yield a rank >= 2 result ([1, 2000]) the single-value channel
#   cannot faithfully represent, so Unsqueeze's custom propagation must DECLINE. On a
#   correct decline nothing propagates past Unsqueeze, RB's length stays SYMBOLIC, and the
#   model loads. A relaxed decline would fabricate [1, 2000], which Squeeze lowers into a
#   non-scalar Range limit that ONNX Range shape inference rejects ("Input to 'Range' op
#   should be scalars"), so Graph::Resolve (and therefore Model::Load) FAILS -- the decline
#   is observable either as a load failure or, were that to pass, as a concrete RB length.
#
# Because both Gather indices are raw constant initializers, ONNX's own Gather data
# propagator bails (it has no getInputData for a constant index), so control reaches our
# custom decline branches. The two branches are independent: a regression in either one is
# caught (Branch A as a concrete RA dimension, Branch B as a failed load), so the single
# combined model locks both decline paths with the discriminating power of the two
# original per-path fixtures.

from onnx import IR_VERSION, TensorProto, checker, helper, save

# Graph input: a 3-D float tensor whose last static dimension is 2000. Its Shape vector
# [3, 4, 2000] feeds both branches; Gather([-1]) of that vector is the single element 2000.
input_tensor = helper.make_tensor_value_info("X", TensorProto.FLOAT, [3, 4, 2000])

# Outputs: the two Range results, 1-D int64 tensors whose lengths are the propagated K
# (symbolic when each decline is correct).
range_out_a = helper.make_tensor_value_info("RA", TensorProto.INT64, ["range_len_a"])
range_out_b = helper.make_tensor_value_info("RB", TensorProto.INT64, ["range_len_b"])

# Branch A Gather index: the rank-2 constant [[-1]] of shape [1, 1] (last dimension).
gather_idx_rank2 = helper.make_tensor("gather_idx_rank2", TensorProto.INT64, [1, 1], [-1])
# Branch B Gather index: the rank-1 constant [-1] (last dimension).
gather_idx_rank1 = helper.make_tensor("gather_idx_rank1", TensorProto.INT64, [1], [-1])
# Branch B Unsqueeze axes: insert a leading dimension (axis 0).
unsqueeze_axes = helper.make_tensor("unsqueeze_axes", TensorProto.INT64, [1], [0])
# Range start/delta as 0-D scalar int64 constants, shared by both Range nodes.
range_start = helper.make_tensor("range_start", TensorProto.INT64, [], [0])
range_delta = helper.make_tensor("range_delta", TensorProto.INT64, [], [1])

# Shape(X) -> S : 1-D int64 tensor [3, 4, 2000], shared by both branches.
shape_node = helper.make_node("Shape", inputs=["X"], outputs=["S"], name="ShapeNode")

# Branch A: Gather(S, [[-1]]) -> GA : rank-2 [1, 1] value; custom propagation must decline.
gather_a = helper.make_node("Gather", inputs=["S", "gather_idx_rank2"], outputs=["GA"], axis=0, name="GatherNodeA")
squeeze_a = helper.make_node("Squeeze", inputs=["GA"], outputs=["KA"], name="SqueezeNodeA")
range_a = helper.make_node("Range", inputs=["range_start", "KA", "range_delta"], outputs=["RA"], name="RangeNodeA")

# Branch B: Gather(S, [-1]) -> GB : rank-1 single element; Unsqueeze -> rank-2; must decline.
gather_b = helper.make_node("Gather", inputs=["S", "gather_idx_rank1"], outputs=["GB"], axis=0, name="GatherNodeB")
unsqueeze_b = helper.make_node("Unsqueeze", inputs=["GB", "unsqueeze_axes"], outputs=["UB"], name="UnsqueezeNodeB")
squeeze_b = helper.make_node("Squeeze", inputs=["UB"], outputs=["KB"], name="SqueezeNodeB")
range_b = helper.make_node("Range", inputs=["range_start", "KB", "range_delta"], outputs=["RB"], name="RangeNodeB")

graph = helper.make_graph(
    nodes=[shape_node, gather_a, squeeze_a, range_a, gather_b, unsqueeze_b, squeeze_b, range_b],
    name="Shape_Gather_Unsqueeze_Decline_Combined_Model",
    inputs=[input_tensor],
    outputs=[range_out_a, range_out_b],
    initializer=[gather_idx_rank2, gather_idx_rank1, unsqueeze_axes, range_start, range_delta],
)

model = helper.make_model(
    graph,
    opset_imports=[helper.make_operatorsetid("", 18)],
    producer_name="onnx-example-generator",
)
model.ir_version = IR_VERSION

checker.check_model(model)
save(model, "test_shape_data_propagation_decline_combined.onnx")

print("Model saved to test_shape_data_propagation_decline_combined.onnx")
