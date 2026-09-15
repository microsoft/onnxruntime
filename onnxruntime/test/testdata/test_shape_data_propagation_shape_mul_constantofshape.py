# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

# Negative regression fixture for the single-element guard in the elementwise
# data-propagation consumers (Add/Sub/Mul/Div).
#
# Shape(X[3, 4]) produces a rank-1, MULTI-element value [3, 4]. The elementwise
# custom data propagation must only fold a single-element value; for a rank-1
# [N>1] operand it must DECLINE (return no single value) rather than silently
# using element[0]. If it wrongly used element[0], Mul would emit a bogus scalar
# (3 * 3 = 9) that would override the correct, ONNX-propagated multi-element
# value and collapse the downstream ConstantOfShape output from rank-2 [9, 16]
# to a rank-1 [9].
#
# Correct behavior: Mul declines, the multi-element value [9, 16] still flows
# through, and ConstantOfShape(M) is the rank-2 shape [9, 16].

from onnx import IR_VERSION, TensorProto, checker, helper, save

x = helper.make_tensor_value_info("X", TensorProto.FLOAT, [3, 4])
# ConstantOfShape output is always rank 2 here (its shape input has 2 elements);
# declared with symbolic dims so the checker passes while the test asserts the
# concrete inferred dims [9, 16].
y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, ["d0", "d1"])

# Shape(X) -> S : 1-D int64 tensor [3, 4] (two elements).
shape_node = helper.make_node("Shape", inputs=["X"], outputs=["S"], name="ShapeNode")
# Mul(S, S) -> M : elementwise [9, 16] (still two elements; must NOT collapse to a scalar).
mul_node = helper.make_node("Mul", inputs=["S", "S"], outputs=["M"], name="MulNode")
# ConstantOfShape(M) -> Y : output shape equals M's values, i.e. [9, 16].
cos_node = helper.make_node("ConstantOfShape", inputs=["M"], outputs=["Y"], name="ConstantOfShapeNode")

graph = helper.make_graph(
    nodes=[shape_node, mul_node, cos_node],
    name="Shape_Mul_ConstantOfShape_Model",
    inputs=[x],
    outputs=[y],
)

model = helper.make_model(
    graph,
    opset_imports=[helper.make_operatorsetid("", 18)],
    producer_name="onnx-example-generator",
)
model.ir_version = IR_VERSION

checker.check_model(model)
save(model, "test_shape_data_propagation_shape_mul_constantofshape.onnx")

print("Model saved to test_shape_data_propagation_shape_mul_constantofshape.onnx")
