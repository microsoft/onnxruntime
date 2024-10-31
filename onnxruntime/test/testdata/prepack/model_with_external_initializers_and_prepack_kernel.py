# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os

import numpy as np
import onnx
from onnx import TensorProto, helper
from onnx.external_data_helper import set_external_data
from onnx.numpy_helper import from_array

M = 1
K = 1
N = 1
q_cols = 1
q_rows = 1
q_scale_size = 1


def create_external_data_tensor(value, tensor_name, data_type):
    tensor = from_array(np.array(value))
    tensor.name = tensor_name
    tensor_filename = f"{tensor_name}.bin"
    set_external_data(tensor, location=tensor_filename)

    with open(os.path.join(tensor_filename), "wb") as data_file:
        data_file.write(tensor.raw_data)
    tensor.ClearField("raw_data")
    tensor.data_location = onnx.TensorProto.EXTERNAL
    tensor.data_type = data_type
    return tensor


def create_internal_data_tensor(value, tensor_name, data_type):
    tensor = helper.make_tensor(name=tensor_name, data_type=data_type, dims=value.shape, vals=value.flatten().tolist())
    print(tensor)
    tensor.data_location = onnx.TensorProto.DEFAULT
    return tensor


def GenerateMatmulNBitsModel(model_name, external_data_name):  # noqa: N802
    A = helper.make_tensor_value_info("A", TensorProto.FLOAT, [M, K])  # noqa: N806
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [M, N])  # noqa: N806

    # Create a node (NodeProto)
    node_def = helper.make_node(
        op_type="MatMulNBits",  # op type
        inputs=["A", external_data_name, "scales"],  # inputs
        outputs=["Y"],  # outputs
        name="MatMul_0",  # node name
        domain="com.microsoft",  # Custom domain for this operator
        accuracy_level=4,  # Attributes
        bits=4,  # Attributes
        block_size=32,  # Attributes
        K=K,  # Attributes
        N=N,  # Attributes
    )

    # Create the graph (GraphProto)
    graph_def = helper.make_graph(
        [node_def],
        "test-model-matmul4bits",
        [A],
        [Y],
        [
            create_external_data_tensor([[171]], external_data_name, TensorProto.UINT8),
            create_internal_data_tensor(np.array([1.5], dtype=np.float32), "scales", TensorProto.FLOAT),
        ],
    )

    # Create the model
    model_def = helper.make_model(
        graph_def,
        producer_name="onnx-example",
        opset_imports=[helper.make_operatorsetid("", 14), helper.make_operatorsetid("com.microsoft", 1)],
    )

    print(f"The ir_version in model: {model_def.ir_version}\n")
    print(f"The producer_name in model: {model_def.producer_name}\n")
    print(f"The graph in model:\n{model_def.graph}")
    onnx.checker.check_model(model_def)
    print("The model is checked!")
    with open(model_name, "wb") as model_file:
        model_file.write(model_def.SerializeToString())


def GenerateGemmModel(model_name, external_data_name):  # noqa: N802
    A = helper.make_tensor_value_info("A", TensorProto.FLOAT, [M, K])  # noqa: N806
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [M, N])  # noqa: N806

    # Create a node (NodeProto)
    node_def = helper.make_node(
        op_type="Gemm",  # op type
        inputs=["A", external_data_name],  # inputs
        outputs=["Y"],  # outputs
        name="Gemm_0",  # node name
        alpha=3.5,  # Attributes
        beta=6.25,  # Attributes
        transA=0,  # Attributes
        transB=1,  # Attributes
    )

    # Create the graph (GraphProto)
    graph_def = helper.make_graph(
        [node_def],
        "test-model-gemm",
        [A],
        [Y],
        [create_external_data_tensor(np.random.rand(N, K).astype(np.float32), external_data_name, TensorProto.FLOAT)],
    )

    # Create the model
    model_def = helper.make_model(
        graph_def,
        producer_name="onnx-example",
        opset_imports=[helper.make_operatorsetid("", 14)],
    )

    print(f"The ir_version in model: {model_def.ir_version}\n")
    print(f"The producer_name in model: {model_def.producer_name}\n")
    print(f"The graph in model:\n{model_def.graph}")
    onnx.checker.check_model(model_def)
    print("The model is checked!")
    with open(model_name, "wb") as model_file:
        model_file.write(model_def.SerializeToString())


def GenerateMatMulModel(model_name, external_data_name):  # noqa: N802
    A = helper.make_tensor_value_info("A", TensorProto.FLOAT, [N, N])  # noqa: N806
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [N, N])  # noqa: N806

    # Create a node (NodeProto)
    node_def = helper.make_node(
        op_type="MatMul",  # op type
        inputs=["A", external_data_name],  # inputs
        outputs=["Y"],  # outputs
        name="MatMul_0",  # node name
    )

    # Create the graph (GraphProto)
    graph_def = helper.make_graph(
        [node_def],
        "test-model-matmul",
        [A],
        [Y],
        [create_external_data_tensor(np.random.rand(N, N).astype(np.float32), external_data_name, TensorProto.FLOAT)],
    )

    # Create the model
    model_def = helper.make_model(
        graph_def,
        producer_name="onnx-example",
        opset_imports=[helper.make_operatorsetid("", 14)],
    )

    print(f"The ir_version in model: {model_def.ir_version}\n")
    print(f"The producer_name in model: {model_def.producer_name}\n")
    print(f"The graph in model:\n{model_def.graph}")
    onnx.checker.check_model(model_def)
    print("The model is checked!")
    with open(model_name, "wb") as model_file:
        model_file.write(model_def.SerializeToString())


def GenerateConvTransposeModel(model_name, external_data_name):  # noqa: N802
    input = helper.make_tensor_value_info("A", TensorProto.FLOAT, [1, 3, 64, 64])
    output = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 6, 127, 127])

    # Create a node (NodeProto)
    node_def = helper.make_node(
        op_type="ConvTranspose",  # op type
        inputs=["A", external_data_name, "bias"],  # inputs
        outputs=["Y"],  # outputs
        name="ConvTranspose_0",  # node name
        output_padding=[0, 0],
        pads=[1, 1, 1, 1],
        strides=[2, 2],
        dilations=[1, 1],
        group=1,
    )

    # Create the graph (GraphProto)
    graph_def = helper.make_graph(
        [node_def],
        "test-model-conv_transpose",
        [input],
        [output],
        [
            create_external_data_tensor(
                np.random.rand(3, 6, 3, 3).astype(np.float32), external_data_name, TensorProto.FLOAT
            ),
            create_internal_data_tensor(np.random.rand(3).astype(np.float32), "bias", TensorProto.FLOAT),
        ],
    )

    # Create the model
    model_def = helper.make_model(
        graph_def,
        producer_name="onnx-example",
        opset_imports=[helper.make_operatorsetid("", 14)],
    )

    print(f"The ir_version in model: {model_def.ir_version}\n")
    print(f"The producer_name in model: {model_def.producer_name}\n")
    print(f"The graph in model:\n{model_def.graph}")
    onnx.checker.check_model(model_def)
    print("The model is checked!")
    with open(model_name, "wb") as model_file:
        model_file.write(model_def.SerializeToString())


def GenerateDeepCpuLSTMModel(model_name, external_data_name_w, external_data_name_r):  # noqa: N802
    input = helper.make_tensor_value_info("A", TensorProto.FLOAT, [16, 2, 32])
    output = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [16, 1, 2, 4])

    # Create a node (NodeProto)
    node_def = helper.make_node(
        op_type="LSTM",  # op type
        inputs=["A", external_data_name_w, external_data_name_r],  # inputs
        outputs=["Y"],  # outputs
        name="DeepCpuLstmOp_0",  # node name
        direction="forward",
        hidden_size=4,
    )

    # Create the graph (GraphProto)
    graph_def = helper.make_graph(
        [node_def],
        "test-model-deep_cpu_lstm",
        [input],
        [output],
        [
            create_external_data_tensor(
                np.random.rand(1, 16, 32).astype(np.float32), external_data_name_w, TensorProto.FLOAT
            ),
            create_external_data_tensor(
                np.random.rand(1, 16, 32).astype(np.float32), external_data_name_r, TensorProto.FLOAT
            ),
        ],
    )

    # Create the model
    model_def = helper.make_model(
        graph_def,
        producer_name="onnx-example",
        opset_imports=[helper.make_operatorsetid("", 14)],
    )

    print(f"The ir_version in model: {model_def.ir_version}\n")
    print(f"The producer_name in model: {model_def.producer_name}\n")
    print(f"The graph in model:\n{model_def.graph}")
    onnx.checker.check_model(model_def)
    print("The model is checked!")
    with open(model_name, "wb") as model_file:
        model_file.write(model_def.SerializeToString())


def GenerateAttentionModel(model_name, external_data_name):  # noqa: N802
    input = helper.make_tensor_value_info("A", TensorProto.FLOAT, [4, 16, 8])
    output = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [4, 16, 2])

    # Create a node (NodeProto)
    node_def = helper.make_node(
        op_type="Attention",  # op type
        inputs=["A", external_data_name, "bias"],  # inputs
        outputs=["Y"],  # outputs
        name="Attention_0",  # node name
        domain="com.microsoft",  # Custom domain for this operator
        num_heads=2,
        past_present_share_buffer=0,
        mask_filter_value=-10000.0,
    )

    # Create the graph (GraphProto)
    graph_def = helper.make_graph(
        [node_def],
        "test-model-attention",
        [input],
        [output],
        [
            create_external_data_tensor(np.random.rand(8, 6).astype(np.float32), external_data_name, TensorProto.FLOAT),
            create_internal_data_tensor(np.random.rand(6).astype(np.float32), "bias", TensorProto.FLOAT),
        ],
    )

    # Create the model
    model_def = helper.make_model(
        graph_def,
        producer_name="onnx-example",
        opset_imports=[helper.make_operatorsetid("", 12), helper.make_operatorsetid("com.microsoft", 1)],
    )

    print(f"The ir_version in model: {model_def.ir_version}\n")
    print(f"The producer_name in model: {model_def.producer_name}\n")
    print(f"The graph in model:\n{model_def.graph}")
    onnx.checker.check_model(model_def)
    print("The model is checked!")
    with open(model_name, "wb") as model_file:
        model_file.write(model_def.SerializeToString())


def GenerateQAttentionModel(model_name, external_data_name):  # noqa: N802
    input = helper.make_tensor_value_info("A", TensorProto.UINT8, [4, 16, 8])
    output = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [4, 16, 2])

    # Create a node (NodeProto)
    node_def = helper.make_node(
        op_type="QAttention",  # op type
        inputs=["A", external_data_name, "bias", "input_scale", "weight_scale"],  # inputs
        outputs=["Y"],  # outputs
        name="QAttention_0",  # node name
        domain="com.microsoft",  # Custom domain for this operator
        num_heads=2,
        unidirectional=0,
    )

    # Create the graph (GraphProto)
    graph_def = helper.make_graph(
        [node_def],
        "test-model-qattention",
        [input],
        [output],
        [
            create_external_data_tensor(np.random.rand(8, 6).astype(np.uint8), external_data_name, TensorProto.UINT8),
            create_internal_data_tensor(np.random.rand(6).astype(np.float32), "bias", TensorProto.FLOAT),
            create_internal_data_tensor(np.random.rand(1).astype(np.float32), "input_scale", TensorProto.FLOAT),
            create_internal_data_tensor(np.random.rand(1).astype(np.float32), "weight_scale", TensorProto.FLOAT),
        ],
    )

    # Create the model
    model_def = helper.make_model(
        graph_def,
        producer_name="onnx-example",
        opset_imports=[helper.make_operatorsetid("", 12), helper.make_operatorsetid("com.microsoft", 1)],
    )

    print(f"The ir_version in model: {model_def.ir_version}\n")
    print(f"The producer_name in model: {model_def.producer_name}\n")
    print(f"The graph in model:\n{model_def.graph}")
    onnx.checker.check_model(model_def)
    print("The model is checked!")
    with open(model_name, "wb") as model_file:
        model_file.write(model_def.SerializeToString())


def GenerateDynamicQuantizeLSTMModel(model_name, external_data_name_w, external_data_name_r):  # noqa: N802
    input = helper.make_tensor_value_info("A", TensorProto.FLOAT, [4, 16, 8])
    output = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [4, 1, 16, 2])

    # Create a node (NodeProto)
    node_def = helper.make_node(
        op_type="DynamicQuantizeLSTM",  # op type
        inputs=[
            "A",
            external_data_name_w,
            external_data_name_r,
            "B",
            "sequence_lens",
            "initial_h",
            "initial_c",
            "P",
            "W_scale",
            "W_zero_point",
            "R_scale",
            "R_zero_point",
        ],  # inputs
        outputs=["Y"],  # outputs
        name="DynamicQuantizeLSTM_0",  # node name
        domain="com.microsoft",  # Custom domain for this operator
        direction="forward",
        hidden_size=2,
    )

    # Create the graph (GraphProto)
    graph_def = helper.make_graph(
        [node_def],
        "test-model-dynamic-quantize-LSTM",
        [input],
        [output],
        [
            create_external_data_tensor(
                np.random.rand(1, 8, 8).astype(np.uint8), external_data_name_w, TensorProto.UINT8
            ),
            create_external_data_tensor(
                np.random.rand(1, 2, 8).astype(np.uint8), external_data_name_r, TensorProto.UINT8
            ),
            create_internal_data_tensor(np.random.rand(1, 16).astype(np.float32), "B", TensorProto.FLOAT),
            create_internal_data_tensor(np.random.rand(16).astype(np.int32), "sequence_lens", TensorProto.INT32),
            create_internal_data_tensor(np.random.rand(1, 16, 2).astype(np.float32), "initial_h", TensorProto.FLOAT),
            create_internal_data_tensor(np.random.rand(1, 16, 2).astype(np.float32), "initial_c", TensorProto.FLOAT),
            create_internal_data_tensor(np.random.rand(1, 6).astype(np.float32), "P", TensorProto.FLOAT),
            create_internal_data_tensor(np.random.rand(1, 8).astype(np.float32), "W_scale", TensorProto.FLOAT),
            create_internal_data_tensor(np.random.rand(1, 8).astype(np.uint8), "W_zero_point", TensorProto.UINT8),
            create_internal_data_tensor(np.random.rand(1, 8).astype(np.float32), "R_scale", TensorProto.FLOAT),
            create_internal_data_tensor(np.random.rand(1, 16).astype(np.uint8), "R_zero_point", TensorProto.UINT8),
        ],
    )

    # Create the model
    model_def = helper.make_model(
        graph_def,
        producer_name="onnx-example",
        opset_imports=[helper.make_operatorsetid("", 12), helper.make_operatorsetid("com.microsoft", 1)],
    )

    print(f"The ir_version in model: {model_def.ir_version}\n")
    print(f"The producer_name in model: {model_def.producer_name}\n")
    print(f"The graph in model:\n{model_def.graph}")
    onnx.checker.check_model(model_def)
    print("The model is checked!")
    with open(model_name, "wb") as model_file:
        model_file.write(model_def.SerializeToString())


def GenerateMatMulIntegerModel(model_name, external_data_name):  # noqa: N802
    A = helper.make_tensor_value_info("A", TensorProto.UINT8, [8, 16])  # noqa: N806
    Y = helper.make_tensor_value_info("Y", TensorProto.INT32, [8, 8])  # noqa: N806

    # Create a node (NodeProto)
    node_def = helper.make_node(
        op_type="MatMulInteger",  # op type
        inputs=["A", external_data_name],  # inputs
        outputs=["Y"],  # outputs
        name="MatMulInteger_0",  # node name
    )

    # Create the graph (GraphProto)
    graph_def = helper.make_graph(
        [node_def],
        "test-model-matmul-integer",
        [A],
        [Y],
        [create_external_data_tensor(np.random.rand(16, 8).astype(np.uint8), external_data_name, TensorProto.UINT8)],
    )

    # Create the model
    model_def = helper.make_model(
        graph_def,
        producer_name="onnx-example",
        opset_imports=[helper.make_operatorsetid("", 14)],
    )

    print(f"The ir_version in model: {model_def.ir_version}\n")
    print(f"The producer_name in model: {model_def.producer_name}\n")
    print(f"The graph in model:\n{model_def.graph}")
    onnx.checker.check_model(model_def)
    print("The model is checked!")
    with open(model_name, "wb") as model_file:
        model_file.write(model_def.SerializeToString())


def GenerateConvFP16Model(model_name, external_data_name):  # noqa: N802
    input = helper.make_tensor_value_info("A", TensorProto.FLOAT, [1, 1, 1, 5])
    output = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 1, 1, 5])

    nodes = [
        # Cast node to cast from float to fp16
        helper.make_node("Cast", ["A"], ["A_fp16"], name="Cast_fp_fp16", to=10),
        # Conv node
        helper.make_node(
            op_type="FusedConv",  # op type
            inputs=["A_fp16", external_data_name],  # inputs
            outputs=["Y_fp16"],  # outputs
            name="ConvFP16_0",  # node name
            domain="com.microsoft",  # Custom domain for this operator
            auto_pad="SAME_UPPER",
            # pads=[1, 1, 1, 1],
            strides=[1, 1],
            dilations=[1, 1],
            group=1,
            kernel_shape=[3, 3],
        ),
        # Cast node to cast from fp16 back to float
        helper.make_node("Cast", ["Y_fp16"], ["Y"], name="Cast_fp16_fp", to=1),
    ]

    # Create the graph (GraphProto)
    graph_def = helper.make_graph(
        nodes,
        "test-model-conv_fp16",
        [input],
        [output],
        [
            create_external_data_tensor(
                np.random.rand(1, 1, 3, 3).astype(np.float16), external_data_name, TensorProto.FLOAT16
            ),
        ],
    )

    # Create the model
    model_def = helper.make_model(
        graph_def,
        producer_name="onnx-example",
        opset_imports=[helper.make_operatorsetid("", 14), helper.make_operatorsetid("com.microsoft", 1)],
    )

    print(f"The ir_version in model: {model_def.ir_version}\n")
    print(f"The producer_name in model: {model_def.producer_name}\n")
    print(f"The graph in model:\n{model_def.graph}")
    onnx.checker.check_model(model_def)
    print("The model is checked!")
    with open(model_name, "wb") as model_file:
        model_file.write(model_def.SerializeToString())


def GenerateGRUModel(model_name, external_data_name_w, external_data_name_r):  # noqa: N802
    A = helper.make_tensor_value_info("A", TensorProto.FLOAT, [32, 2, 8])  # noqa: N806
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [32, 1, 2, 4])  # noqa: N806

    # Create a node (NodeProto)
    node_def = helper.make_node(
        op_type="GRU",  # op type
        inputs=["A", external_data_name_w, external_data_name_r],  # inputs
        outputs=["Y"],  # outputs
        name="GRU_0",  # node name
        direction="forward",
        hidden_size=4,
        clip=100.0,
        layout=0,
        linear_before_reset=0,
    )

    # Create the graph (GraphProto)
    graph_def = helper.make_graph(
        [node_def],
        "test-model-gru",
        [A],
        [Y],
        [
            create_external_data_tensor(
                np.random.rand(1, 12, 8).astype(np.float32), external_data_name_w, TensorProto.FLOAT
            ),
            create_external_data_tensor(
                np.random.rand(1, 12, 4).astype(np.float32), external_data_name_r, TensorProto.FLOAT
            ),
        ],
    )

    # Create the model
    model_def = helper.make_model(
        graph_def,
        producer_name="onnx-example",
        opset_imports=[helper.make_operatorsetid("", 14)],
    )

    print(f"The ir_version in model: {model_def.ir_version}\n")
    print(f"The producer_name in model: {model_def.producer_name}\n")
    print(f"The graph in model:\n{model_def.graph}")
    onnx.checker.check_model(model_def)
    print("The model is checked!")
    with open(model_name, "wb") as model_file:
        model_file.write(model_def.SerializeToString())


def GenerateQLinearConvModel(model_name, external_data_name):  # noqa: N802
    input = helper.make_tensor_value_info("A", TensorProto.UINT8, [2, 4, 8, 16])
    output = helper.make_tensor_value_info("Y", TensorProto.UINT8, [2, 10, 8, 16])

    # Create a node (NodeProto)
    node_def = helper.make_node(
        op_type="QLinearConv",  # op type
        inputs=[
            "A",
            "x_scale",
            "x_zero_point",
            external_data_name,
            "w_scale",
            "w_zero_point",
            "y_scale",
            "y_zero_point",
        ],  # inputs
        outputs=["Y"],  # outputs
        name="QLinearConv_0",  # node name
        auto_pad="SAME_LOWER",
        dilations=[1, 1],
        group=1,
        strides=[1, 1],
        # kernel_shape=[2, 4],
    )

    # Create the graph (GraphProto)
    graph_def = helper.make_graph(
        [node_def],
        "test-model-qlinearconv",
        [input],
        [output],
        [
            create_external_data_tensor(
                np.random.rand(10, 4, 4, 8).astype(np.uint8), external_data_name, TensorProto.UINT8
            ),
            create_internal_data_tensor(np.random.rand(1).astype(np.float32), "x_scale", TensorProto.FLOAT),
            create_internal_data_tensor(np.random.rand(1).astype(np.uint8), "x_zero_point", TensorProto.UINT8),
            create_internal_data_tensor(np.random.rand(1).astype(np.float32), "w_scale", TensorProto.FLOAT),
            create_internal_data_tensor(np.random.rand(1).astype(np.uint8), "w_zero_point", TensorProto.UINT8),
            create_internal_data_tensor(np.random.rand(1).astype(np.float32), "y_scale", TensorProto.FLOAT),
            create_internal_data_tensor(np.random.rand(1).astype(np.uint8), "y_zero_point", TensorProto.UINT8),
        ],
    )

    # Create the model
    model_def = helper.make_model(
        graph_def,
        producer_name="onnx-example",
        opset_imports=[helper.make_operatorsetid("", 14), helper.make_operatorsetid("com.microsoft", 1)],
    )

    print(f"The ir_version in model: {model_def.ir_version}\n")
    print(f"The producer_name in model: {model_def.producer_name}\n")
    print(f"The graph in model:\n{model_def.graph}")
    onnx.checker.check_model(model_def)
    print("The model is checked!")
    with open(model_name, "wb") as model_file:
        model_file.write(model_def.SerializeToString())


if __name__ == "__main__":
    GenerateMatmulNBitsModel("model_with_matmul_nbits.onnx", "MatMul.Weight")
    GenerateGemmModel("model_with_gemm.onnx", "Gemm_B")
    GenerateMatMulModel("model_with_matmul.onnx", "MatMul_B")
    GenerateConvTransposeModel("model_with_conv_transpose.onnx", "Conv_Transpose_weights")
    GenerateDeepCpuLSTMModel("model_with_deep_cpu_lstm.onnx", "deep_cpu_lstm_w", "deep_cpu_lstm_r")
    GenerateAttentionModel("model_with_attention.onnx", "Attention.Weight")
    GenerateQAttentionModel("model_with_quant_attention.onnx", "QAttention.Weight")
    GenerateDynamicQuantizeLSTMModel(
        "model_with_dynamic_quan_lstm.onnx", "DynamicQuantizeLSTM.Weight", "DynamicQuantizeLSTM.RecurrenceWeight"
    )
    GenerateMatMulIntegerModel("model_with_matmul_integer_quant.onnx", "MatMulInteger.B")
    GenerateConvFP16Model("model_with_fp16_conv.onnx", "ConvFP16.Weight")
    GenerateGRUModel("model_with_deep_cpu_gru.onnx", "GRU.Weight", "GRU.RecurrentceWeight")
    GenerateQLinearConvModel("model_with_quant_linearconv.onnx", "QLinearConv.Weight")
