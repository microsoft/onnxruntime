# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import numpy as np
import onnx


def make_model(model_path: str):
    """
    Creates a QDQ model with a (DQ -> Transpose -> Q -> GRAPH OUTPUT) sequence. The Transpose is optimized out
    and the TransposeOptimizer should also remove the empty (DQ -> Q) sequence.
    """
    input0_shape = (1, 3, 4, 4)

    inputs = [onnx.helper.make_tensor_value_info("input0", onnx.TensorProto.FLOAT, input0_shape)]
    outputs = [onnx.helper.make_tensor_value_info("output0", onnx.TensorProto.UINT8, None)]

    mul_weight_scale_data = np.array(1.0, dtype=np.float32)
    mul_weight_zp_data = np.array(0, dtype=np.int8)

    initializers = [
        onnx.numpy_helper.from_array(np.array(1.0, dtype=np.float32), "scale_1"),
        onnx.numpy_helper.from_array(np.array(128, dtype=np.uint8), "zp_128"),
        onnx.numpy_helper.from_array(np.array(1.0 / 255.0, dtype=np.float32), "scale_inv_255"),
        onnx.numpy_helper.from_array(np.array(0, dtype=np.uint8), "zp_0"),
        onnx.numpy_helper.from_array(mul_weight_scale_data, "mul_weight_scale"),
        onnx.numpy_helper.from_array(mul_weight_zp_data, "mul_weight_zp"),
    ]
    nodes = []

    # Transpose to channel-last
    tp0_node = onnx.helper.make_node("Transpose", ["input0"], ["tp0_out"], name="tp0_node", perm=(0, 2, 3, 1))
    nodes.append(tp0_node)

    # Q_0
    q0_node = onnx.helper.make_node("QuantizeLinear", ["tp0_out", "scale_1", "zp_128"], ["q0_out"], name="q0_node")
    nodes.append(q0_node)

    # DQ_0
    dq0_node = onnx.helper.make_node("DequantizeLinear", ["q0_out", "scale_1", "zp_128"], ["dq0_out"], name="dq0_node")
    nodes.append(dq0_node)

    # Sigmoid
    sigmoid_node = onnx.helper.make_node("Sigmoid", ["dq0_out"], ["sigmoid_out"], name="sigmoid_node")
    nodes.append(sigmoid_node)

    # Q_1
    q1_node = onnx.helper.make_node(
        "QuantizeLinear", ["sigmoid_out", "scale_inv_255", "zp_0"], ["q1_out"], name="q1_node"
    )
    nodes.append(q1_node)

    # DQ_1
    dq1_node = onnx.helper.make_node(
        "DequantizeLinear", ["q1_out", "scale_inv_255", "zp_0"], ["dq1_out"], name="dq1_node"
    )
    nodes.append(dq1_node)

    # DQ for mul input[1]
    mul_weight_i8_data = np.array([1, 2, 3], dtype=np.int8)
    mul_weight = onnx.numpy_helper.from_array(mul_weight_i8_data, "mul_weight")
    initializers.append(mul_weight)

    nodes.append(
        onnx.helper.make_node(
            "DequantizeLinear",
            ["mul_weight", "mul_weight_scale", "mul_weight_zp"],
            ["mul_input_1"],
            name="dq_mul_input_1",
        )
    )

    # Mul
    mul_node = onnx.helper.make_node("Mul", ["dq1_out", "mul_input_1"], ["mul_out"], name="mul_node")
    nodes.append(mul_node)

    # Q_2
    q2_node = onnx.helper.make_node("QuantizeLinear", ["mul_out", "scale_inv_255", "zp_0"], ["q2_out"], name="q2_node")
    nodes.append(q2_node)

    # DQ_2
    dq2_node = onnx.helper.make_node(
        "DequantizeLinear", ["q2_out", "scale_inv_255", "zp_0"], ["dq2_out"], name="dq2_node"
    )
    nodes.append(dq2_node)

    # Transpose to channel-first
    tp1_node = onnx.helper.make_node("Transpose", ["dq2_out"], ["tp1_out"], name="tp1_node", perm=(0, 3, 1, 2))
    nodes.append(tp1_node)

    # Q_3 to graph output
    nodes.append(
        onnx.helper.make_node("QuantizeLinear", ["tp1_out", "scale_inv_255", "zp_0"], ["output0"], name="q3_node")
    )

    graph = onnx.helper.make_graph(
        nodes,
        "transpose_opt_empty_dqq_graph_output",
        inputs,
        outputs,
        initializer=initializers,
    )
    opset_imports = [
        onnx.helper.make_opsetid("", 19),
    ]
    qdq_model = onnx.helper.make_model(graph, opset_imports=opset_imports)

    print("[INFO]: Running onnx.checker on qdq model")
    qdq_model = onnx.shape_inference.infer_shapes(qdq_model)
    onnx.checker.check_model(qdq_model, True)

    print(f"[INFO]: Saving {model_path}")
    onnx.save_model(qdq_model, model_path)


if __name__ == "__main__":
    make_model("transpose_optimizer_empty_dq_q_at_graph_output.onnx")
