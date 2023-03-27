# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import numpy as np
import onnx
from onnx import TensorProto, helper
from onnx.external_data_helper import set_external_data
from onnx.numpy_helper import from_array


def create_external_data_tensor(value, tensor_name):  # type: (List[Any], Text) -> TensorProto
    tensor = from_array(value)
    tensor.name = tensor_name
    tensor_filename = f"{tensor_name}.bin"
    set_external_data(tensor, location=tensor_filename)
    tensor.ClearField("raw_data")
    tensor.data_location = onnx.TensorProto.EXTERNAL
    return tensor


def GenerateModel(model_name):  # noqa: N802
    # Create one input (ValueInfoProto)
    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 2])  # noqa: N806

    # Create second input (ValueInfoProto)
    Pads = helper.make_tensor_value_info("Pads_not_on_disk", TensorProto.INT64, [4])  # noqa: N806

    # Create one output (ValueInfoProto)
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 4])  # noqa: N806

    # Create a node (NodeProto)
    node_def = helper.make_node(
        "Pad",  # node name
        ["X", "Pads_not_on_disk"],  # inputs
        ["Y"],  # outputs
        mode="constant",  # Attributes
    )

    # Create the graph (GraphProto)
    initializer_data = np.array([0, 0, 1, 1]).astype(np.int64)
    graph_def = helper.make_graph(
        [node_def],
        "test-model",
        [X, Pads],
        [Y],
        [create_external_data_tensor(initializer_data, "Pads_not_on_disk")],
    )

    # Create the model (ModelProto)
    model_def = helper.make_model(graph_def, producer_name="onnx-example")

    print(f"The ir_version in model: {model_def.ir_version}\n")
    print(f"The producer_name in model: {model_def.producer_name}\n")
    print(f"The graph in model:\n{model_def.graph}")
    with open(model_name, "wb") as model_file:
        model_file.write(model_def.SerializeToString())


if __name__ == "__main__":
    GenerateModel("model_with_external_initializer_come_from_user.onnx")
