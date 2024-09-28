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


if __name__ == "__main__":
    GenerateMatmulNBitsModel("model_with_matmul_nbits.onnx", "MatMul.Weight")
