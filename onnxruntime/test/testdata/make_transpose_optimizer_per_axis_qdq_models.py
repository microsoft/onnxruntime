# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import numpy as np
import onnx


def subgraph_1d_const_input_dq(inputs, initializers, nodes) -> str:
    """
    Creates mul_weight -> DQ. mul_weight is a constant of rank 1.
    """
    mul_weight_i8_data = np.array([1, 2, 3], dtype=np.int8)
    mul_weight = onnx.numpy_helper.from_array(mul_weight_i8_data, "mul_weight")
    initializers.append(mul_weight)

    dq_output_name = "mul_input_1"
    nodes.append(
        onnx.helper.make_node(
            "DequantizeLinear",
            ["mul_weight", "mul_weight_scales", "mul_weight_zps"],
            [dq_output_name],
            name="dq_mul_input_1",
            axis=0,
        )
    )

    return dq_output_name


def subgraph_1d_input_dq(inputs, initializers, nodes) -> str:
    """
    Creates input1 -> DQ. input1 is a graph input of rank 1.
    """
    input1_shape = (3,)
    inputs.append(onnx.helper.make_tensor_value_info("input1", onnx.TensorProto.INT8, input1_shape))

    dq_output_name = "mul_input_1"
    nodes.append(
        onnx.helper.make_node(
            "DequantizeLinear",
            ["input1", "mul_weight_scales", "mul_weight_zps"],
            [dq_output_name],
            name="dq_mul_input_1",
            axis=0,
        )
    )

    return dq_output_name


def subgraph_4d_input_squeeze_dq(inputs, initializers, nodes) -> str:
    """
    Creates input1 -> Squeeze -> DQ. input1 is a graph input of rank 4.
    """
    input1_shape = (1, 1, 1, 3)
    inputs.append(onnx.helper.make_tensor_value_info("input1", onnx.TensorProto.INT8, input1_shape))

    axes_data = np.array([0, 1, 2], dtype=np.int64)
    initializers.append(onnx.numpy_helper.from_array(axes_data, "axes_const"))
    nodes.append(onnx.helper.make_node("Squeeze", ["input1", "axes_const"], ["squeeze_out"], name="squeeze_node"))

    dq_output_name = "mul_input_1"
    nodes.append(
        onnx.helper.make_node(
            "DequantizeLinear",
            ["squeeze_out", "mul_weight_scales", "mul_weight_zps"],
            [dq_output_name],
            name="dq_mul_input_1",
            axis=0,
        )
    )

    return dq_output_name


def subgraph_4d_input_transpose_dq(inputs, initializers, nodes) -> str:
    """
    Creates input1 -> Transpose -> DQ. input1 is a graph input of rank 4.
    """
    input1_shape = (1, 3, 1, 1)
    inputs.append(onnx.helper.make_tensor_value_info("input1", onnx.TensorProto.INT8, input1_shape))

    perm = [0, 2, 3, 1]  # To channel-last
    nodes.append(onnx.helper.make_node("Transpose", ["input1"], ["tp_out_"], perm=perm, name="transpose_"))

    dq_output_name = "mul_input_1"
    nodes.append(
        onnx.helper.make_node(
            "DequantizeLinear",
            ["tp_out_", "mul_weight_scales", "mul_weight_zps"],
            [dq_output_name],
            name="dq_mul_input_1",
            axis=-1,
        )
    )

    return dq_output_name


def make_model(model_path: str, build_mul_input_1_subgraph):
    """
    Creates a QDQ model with a per-axis DQ input that is Unsqueezed and Transposed by the Transpose optimizer.
    """
    input0_shape = (1, 3, 4, 4)

    inputs = [onnx.helper.make_tensor_value_info("input0", onnx.TensorProto.FLOAT, input0_shape)]
    outputs = [onnx.helper.make_tensor_value_info("output0", onnx.TensorProto.FLOAT, None)]

    mul_weight_scales_data = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    mul_weight_zps_data = np.array([0, 0, 0], dtype=np.int8)

    initializers = [
        onnx.numpy_helper.from_array(np.array(1.0, dtype=np.float32), "scale_1"),
        onnx.numpy_helper.from_array(np.array(128, dtype=np.uint8), "zp_128"),
        onnx.numpy_helper.from_array(np.array(1.0 / 255.0, dtype=np.float32), "scale_inv_255"),
        onnx.numpy_helper.from_array(np.array(0, dtype=np.uint8), "zp_0"),
        onnx.numpy_helper.from_array(mul_weight_scales_data, "mul_weight_scales"),
        onnx.numpy_helper.from_array(mul_weight_zps_data, "mul_weight_zps"),
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
    mul_input_1_name = build_mul_input_1_subgraph(inputs, initializers, nodes)

    # Mul
    mul_node = onnx.helper.make_node("Mul", ["dq1_out", mul_input_1_name], ["mul_out"], name="mul_node")
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
    tp1_node = onnx.helper.make_node("Transpose", ["dq2_out"], ["output0"], name="tp1_node", perm=(0, 3, 1, 2))
    nodes.append(tp1_node)

    graph = onnx.helper.make_graph(
        nodes,
        "transpose_opt_unsqueeze_dq_axis",
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
    make_model(
        "transpose_optimizer_qdq_fixup_unsqueeze_per_axis_dq.onnx",
        subgraph_1d_input_dq,
    )
    make_model(
        "transpose_optimizer_in_place_transpose_unsqueeze_per_axis_dq.onnx",
        subgraph_1d_const_input_dq,
    )
    make_model(
        "transpose_optimizer_cancel_squeeze_per_axis_dq.onnx",
        subgraph_4d_input_squeeze_dq,
    )
    make_model(
        "transpose_optimizer_cancel_transpose_per_axis_dq.onnx",
        subgraph_4d_input_transpose_dq,
    )
