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


if __name__ == "__main__":
    model = create_model(broadcast_weights=False)
    onnx.save(model, "transpose_optimizer_shared_initializers.onnx")
    model = create_model(broadcast_weights=True)
    onnx.save(model, "transpose_optimizer_shared_initializers_broadcast.onnx")
