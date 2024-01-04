import numpy as np
import onnx
from onnx import TensorProto, helper


# Create a model with shared initializers that can be updated in-place by the transpose optimizer,
# including ones behind a DQ node. The transpose optimizer updates the first usage and inserts
# Transpose/Unsqueeze ops on the others (see UnsqueezeInput and TransposeInput).
# When we push the Transpose past other usages we should be able to cancel out those Transpose/Unsqueeze ops.
# We need 3 DQ nodes to ensure the Transpose or Unsqueeze added by the transpose optimizer is not
# removed prematurely.
def create_model(broadcast_weights: bool):
    if broadcast_weights:
        bias_shape = [2, 2]
        bias_values = np.random.randn(2, 2)
    else:
        bias_shape = [1, 3, 2, 2]
        bias_values = np.random.randn(1, 3, 2, 2)

    graph = helper.make_graph(
        name="graph",
        inputs=[
            helper.make_tensor_value_info("input0", TensorProto.FLOAT, [1, 2, 2, 3]),
        ],
        initializer=[
            helper.make_tensor("bias_quant", TensorProto.UINT8, bias_shape, bias_values.astype(np.uint8)),
            helper.make_tensor("bias_fp32", TensorProto.FLOAT, bias_shape, bias_values.astype(np.float32)),
            helper.make_tensor("dq_scale0", TensorProto.FLOAT, [], [1.5]),
            helper.make_tensor("dq_zp0", TensorProto.UINT8, [], [5]),
            helper.make_tensor("dq_scale1", TensorProto.FLOAT, [], [0.5]),
        ],
        nodes=[
            # Transpose input from channels last to channels first
            helper.make_node("Transpose", ["input0"], ["input_T"], perm=[0, 3, 1, 2]),
            helper.make_node("DequantizeLinear", ["bias_quant", "dq_scale0", "dq_zp0"], ["DQ0"], "DQ0"),
            helper.make_node("Add", ["input_T", "DQ0"], ["A0"], "A0"),
            helper.make_node("DequantizeLinear", ["bias_quant", "dq_scale1"], ["DQ1"], "DQ1"),
            helper.make_node("Add", ["A0", "DQ1"], ["A1"], "A1"),
            helper.make_node("DequantizeLinear", ["bias_quant", "dq_scale0"], ["DQ2"], "DQ2"),
            helper.make_node("Add", ["A1", "DQ2"], ["A2"], "A2"),
            helper.make_node("Add", ["A2", "bias_fp32"], ["A3"], "A3"),
            helper.make_node("Add", ["A3", "bias_fp32"], ["A4"], "A4"),
            # NCHW to NHWC
            helper.make_node("Transpose", ["A4"], ["output0"], perm=[0, 2, 3, 1]),
        ],
        outputs=[
            helper.make_tensor_value_info("output0", TensorProto.FLOAT, [1, 2, 2, 3]),
        ],
    )

    model = helper.make_model(graph)
    onnx.checker.check_model(model, full_check=True)
    return model


def create_model_with_Where():  # noqa 'Where' is the operator name
    """
    Create a model to validate the logic to cancel out the Transpose -> Squeeze -> DQ between an updated shared
    initializer and other usage. We need to use Where as we require more than 2 inputs.
    The `condition` input will be having a Transpose pushed through it will have a negative cost.
    The `X` input will have a positive cost which cancels out the negative value.
    The `Y` input will be a shared initializer that is braodcast. If we don't find the Transpose to make the cost of it
    negative we will not push the Transpose though.

    If we only have 2 inputs, the broadcast initializer will always cost less due to its smaller rank, meaning we don't
    actually need to look for the Squeeze in that case.
    """
    cond_0_shape = [3, 2]  # transpose to 2, 3
    cond_1_shape = [2, 3]
    x_0_shape = [3]  # broadcast so Transpose goes through Where0
    x_1_shape = [3]  # also broadcast
    y_shape = [3]  # should be transposed and broadcast to [3, 1] if we push the transpose through the Where
    y_values = np.random.randn(3)

    graph = helper.make_graph(
        name="graph",
        inputs=[
            helper.make_tensor_value_info("cond_in_0", TensorProto.BOOL, cond_0_shape),
            helper.make_tensor_value_info("cond_in_1", TensorProto.BOOL, cond_1_shape),
            helper.make_tensor_value_info("x_in_0", TensorProto.FLOAT, x_0_shape),
            helper.make_tensor_value_info("x_in_1", TensorProto.FLOAT, x_1_shape),
        ],
        initializer=[
            helper.make_tensor("y_quant", TensorProto.UINT8, y_shape, y_values.astype(np.uint8)),
            helper.make_tensor("dq_scale0", TensorProto.FLOAT, [], [1.5]),
            helper.make_tensor("dq_scale1", TensorProto.FLOAT, [], [0.5]),
        ],
        nodes=[
            # Transpose the cond input
            helper.make_node("Transpose", ["cond_in_0"], ["cond_in_T"], perm=[1, 0]),
            helper.make_node("DequantizeLinear", ["y_quant", "dq_scale0"], ["DQ0"], "DQ0"),
            # first usage of shared initializer. simple so we know the Transpose can push through it
            helper.make_node("Where", ["cond_in_T", "x_in_0", "DQ0"], ["Where0"], "Where0"),
            helper.make_node("DequantizeLinear", ["y_quant", "dq_scale1"], ["DQ1"], "DQ1"),
            helper.make_node("Add", ["x_in_1", "Where0"], ["Add0"], "Add0"),
            # second usage of shared initializer. requires looking past the Squeeze to push the transpose through
            helper.make_node("Where", ["cond_in_1", "Add0", "DQ1"], ["Where1"], "Where1"),
            helper.make_node("Transpose", ["Where1"], ["output0"], perm=[1, 0]),
        ],
        outputs=[
            helper.make_tensor_value_info("output0", TensorProto.FLOAT, [3, 2]),
        ],
    )

    model = helper.make_model(graph)
    onnx.checker.check_model(model, full_check=True)
    return model


if __name__ == "__main__":
    model = create_model(broadcast_weights=False)
    onnx.save(model, "transpose_optimizer_shared_initializers.onnx")
    model = create_model(broadcast_weights=True)
    onnx.save(model, "transpose_optimizer_shared_initializers_broadcast.onnx")
    model = create_model_with_Where()
    onnx.save(model, "transpose_optimizer_shared_initializers_broadcast2.onnx")
