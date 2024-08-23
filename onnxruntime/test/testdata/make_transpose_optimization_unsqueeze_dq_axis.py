# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import numpy as np
import onnx

if __name__ == "__main__":
    """
    Creates a QDQ model with a per-channel DQ weight that is Unsqueezed and Transposed by the Transpose optimizer.
    """
    input0_shape = (1, 3, 4, 4)

    input0 = onnx.helper.make_tensor_value_info("input0", onnx.TensorProto.FLOAT, input0_shape)
    output0 = onnx.helper.make_tensor_value_info("output0", onnx.TensorProto.FLOAT, None)

    scale_1 = onnx.numpy_helper.from_array(np.array(1.0, dtype=np.float32), "scale_1")
    zp_128 = onnx.numpy_helper.from_array(np.array(128, dtype=np.uint8), "zp_128")
    scale_inv_255 = onnx.numpy_helper.from_array(np.array(1.0 / 255.0, dtype=np.float32), "scale_inv_255")
    zp_0 = onnx.numpy_helper.from_array(np.array(0, dtype=np.uint8), "zp_0")

    mul_weight_i8_data = np.array([1, 2, 3], dtype=np.int8)
    mul_weight_scales_data = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    mul_weight_zps_data = np.array([0, 0, 0], dtype=np.int8)
    mul_weight = onnx.numpy_helper.from_array(mul_weight_i8_data, "mul_weight")
    mul_weight_scales = onnx.numpy_helper.from_array(mul_weight_scales_data, "mul_weight_scales")
    mul_weight_zps = onnx.numpy_helper.from_array(mul_weight_zps_data, "mul_weight_zps")

    # Transpose to channel-last
    tp0_node = onnx.helper.make_node("Transpose", ["input0"], ["tp0_out"], name="tp0_node", perm=(0, 2, 3, 1))

    # Q_0
    q0_node = onnx.helper.make_node("QuantizeLinear", ["tp0_out", "scale_1", "zp_128"], ["q0_out"], name="q0_node")

    # DQ_0
    dq0_node = onnx.helper.make_node("DequantizeLinear", ["q0_out", "scale_1", "zp_128"], ["dq0_out"], name="dq0_node")

    # Sigmoid
    sigmoid_node = onnx.helper.make_node("Sigmoid", ["dq0_out"], ["sigmoid_out"], name="sigmoid_node")

    # Q_1
    q1_node = onnx.helper.make_node(
        "QuantizeLinear", ["sigmoid_out", "scale_inv_255", "zp_0"], ["q1_out"], name="q1_node"
    )

    # DQ_1
    dq1_node = onnx.helper.make_node(
        "DequantizeLinear", ["q1_out", "scale_inv_255", "zp_0"], ["dq1_out"], name="dq1_node"
    )

    # DQ_weight
    dq_weight_node = onnx.helper.make_node(
        "DequantizeLinear",
        ["mul_weight", "mul_weight_scales", "mul_weight_zps"],
        ["dq_weight_out"],
        name="dq_weight_node",
        axis=0,
    )

    # Mul
    mul_node = onnx.helper.make_node("Mul", ["dq1_out", "dq_weight_out"], ["mul_out"], name="mul_node")

    # Q_2
    q2_node = onnx.helper.make_node("QuantizeLinear", ["mul_out", "scale_inv_255", "zp_0"], ["q2_out"], name="q2_node")

    # DQ_2
    dq2_node = onnx.helper.make_node(
        "DequantizeLinear", ["q2_out", "scale_inv_255", "zp_0"], ["dq2_out"], name="dq2_node"
    )

    # Transpose to channel-first
    tp1_node = onnx.helper.make_node("Transpose", ["dq2_out"], ["output0"], name="tp1_node", perm=(0, 3, 1, 2))

    graph = onnx.helper.make_graph(
        [
            tp0_node,
            q0_node,
            dq0_node,
            sigmoid_node,
            q1_node,
            dq1_node,
            dq_weight_node,
            mul_node,
            q2_node,
            dq2_node,
            tp1_node,
        ],
        "transpose_opt_unsqueeze_dq_axis",
        [input0],
        [output0],
        initializer=[scale_1, zp_128, scale_inv_255, zp_0, mul_weight, mul_weight_scales, mul_weight_zps],
    )
    opset_imports = [
        onnx.helper.make_opsetid("", 19),
    ]
    qdq_model = onnx.helper.make_model(graph, opset_imports=opset_imports)

    print("[INFO]: Running onnx.checker on qdq model")
    qdq_model = onnx.shape_inference.infer_shapes(qdq_model)
    onnx.checker.check_model(qdq_model, True)
    qdq_model_path = "transpose_optimization_unsqueeze_dq_axis.qdq.onnx"

    print(f"[INFO]: Saving {qdq_model_path}")
    onnx.save_model(qdq_model, qdq_model_path)
