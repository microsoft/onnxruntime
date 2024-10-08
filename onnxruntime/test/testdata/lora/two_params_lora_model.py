# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os

import numpy as np
import onnx

import onnxruntime as ort

model_path = "two_params_lora_model.onnx"
adapter_path = "two_params_lora_model.onnx_adapter"


def create_model(model_path: os.PathLike):
    #### Inputs
    # original input_x and its associated weight
    input_x = onnx.helper.make_tensor_value_info("input_x", onnx.TensorProto.FLOAT, [4, 4])

    # Inputs overriding default Lora initializers
    lora_param_a_input = onnx.helper.make_tensor_value_info("lora_param_a", onnx.TensorProto.FLOAT, [4, "dim"])
    lora_param_b_input = onnx.helper.make_tensor_value_info("lora_param_b", onnx.TensorProto.FLOAT, ["dim", 4])

    ### Outputs
    output = onnx.helper.make_tensor_value_info("output", onnx.TensorProto.FLOAT, [4, 4])

    #### Initializers
    # Base weight tensor proto
    weight_x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]).reshape(4, 4).astype(np.float32)
    weight_x_tensor = onnx.helper.make_tensor("weight_x", onnx.TensorProto.FLOAT, [4, 4], weight_x.flatten())

    # tensor proto for default lora parameter A
    lora_weight_a = np.zeros([4, 0], dtype=np.float32)
    lora_weight_a_tensor = onnx.helper.make_tensor(
        "lora_param_a", onnx.TensorProto.FLOAT, [4, 0], lora_weight_a.flatten()
    )

    # tensor proto for default lora parameter B
    lora_weight_b = np.zeros([0, 4], dtype=np.float32)
    lora_weight_b_tensor = onnx.helper.make_tensor(
        "lora_param_b", onnx.TensorProto.FLOAT, [0, 4], lora_weight_b.flatten()
    )

    ##### Linear nodes
    # Create matmul for base case
    matmul_x = onnx.helper.make_node("MatMul", ["input_x", "weight_x"], ["mm_output_x"])
    # create matmul node for lora_param_a
    matmul_a = onnx.helper.make_node("MatMul", ["input_x", "lora_param_a"], ["mm_output_a"])
    # Create matmul for lora_param_b
    matmul_b = onnx.helper.make_node("MatMul", ["mm_output_a", "lora_param_b"], ["mm_output_b"])

    # Create Add
    add_node = onnx.helper.make_node("Add", ["mm_output_x", "mm_output_b"], ["output"])

    graph = onnx.helper.make_graph(
        name="two_params_lora_model",
        nodes=[matmul_x, matmul_a, matmul_b, add_node],
        inputs=[input_x, lora_param_a_input, lora_param_b_input],
        outputs=[output],
        initializer=[weight_x_tensor, lora_weight_a_tensor, lora_weight_b_tensor],
    )

    # create a model
    model = onnx.helper.make_model(graph)

    # onnx.checker.check_model(model, full_check=True)

    onnx.save_model(model, model_path)


def create_adapter(adapter_path: os.PathLike):
    """
    Creates an test adapter for the model above
    """
    param_a = np.array([3, 4, 5, 6]).astype(np.float32).reshape(4, 1)
    param_b = np.array([7, 8, 9, 10]).astype(np.float32).reshape(1, 4)
    ort_value_a = ort.OrtValue.ortvalue_from_numpy(param_a)
    ort_value_b = ort.OrtValue.ortvalue_from_numpy(param_b)

    numpy_a = ort_value_a.numpy()
    numpy_b = ort_value_b.numpy()
    np.allclose(param_a, numpy_a)
    np.allclose(param_b, numpy_b)

    print(param_a)
    print(param_b)

    name_to_value = {"lora_param_a": ort_value_a, "lora_param_b": ort_value_b}

    adapter_format = ort.AdapterFormat()
    adapter_format.set_adapter_version(1)
    adapter_format.set_model_version(1)
    adapter_format.set_parameters(name_to_value)
    adapter_format.export_adapter(adapter_path)


def read_adapter(adapter_path: os.PathLike):
    adapter = ort.AdapterFormat.read_adapter(adapter_path)
    params = adapter.get_parameters()

    assert "lora_param_a" in params
    assert "lora_param_b" in params

    numpy_a = params["lora_param_a"].numpy()
    print(numpy_a)

    numpy_b = params["lora_param_b"].numpy()
    print(numpy_b)


def run_base_model(model_path: os.PathLike):
    session = ort.InferenceSession(model_path)

    # Run the base case
    inputs = {"input_x": np.ones((4, 4), dtype=np.float32)}

    outputs = session.run(None, inputs)
    print(outputs)


def run_with_override(model_path: os.PathLike):
    session = ort.InferenceSession(model_path)

    inputs = {
        "input_x": np.ones((4, 4), dtype=np.float32),
        "lora_param_a": np.array([3, 4, 5, 6]).astype(np.float32).reshape(4, 1),
        "lora_param_b": np.array([7, 8, 9, 10]).astype(np.float32).reshape(1, 4),
    }

    outputs = session.run(None, inputs)
    print(outputs)


def run_with_adapter(model_path: os.PathLike, adapter_path: os.PathLike):
    adapter = ort.LoraAdapter()
    adapter.Load(adapter_path)

    run_options = ort.RunOptions()
    run_options.set_adapter_active(adapter)

    session = ort.InferenceSession(model_path)

    inputs = {"input_x": np.ones((4, 4), dtype=np.float32)}

    outputs = session.run(None, inputs, run_options)

    print(outputs)


if __name__ == "__main__":
    # create_model(model_path)
    # run_base_model(model_path)
    run_with_override(model_path)
    # create_adapter(adapter_path)
    # read_adapter(adapter_path)
    run_with_adapter(model_path, adapter_path)
