# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Generate a simple ONNX model that computes C = A + B.

Inputs:
  A : float tensor of shape [1, 3]
  B : float tensor of shape [1, 3]

Output:
  C : float tensor of shape [1, 3]

Usage:
  pip install onnx
  python generate_model.py
"""

from onnx import TensorProto, helper, save_model
from onnx.checker import check_model


def main():
    # Define inputs and output
    a = helper.make_tensor_value_info("A", TensorProto.FLOAT, [1, 3])
    b = helper.make_tensor_value_info("B", TensorProto.FLOAT, [1, 3])
    c = helper.make_tensor_value_info("C", TensorProto.FLOAT, [1, 3])

    # Create the Add node
    add_node = helper.make_node("Add", inputs=["A", "B"], outputs=["C"])

    # Build the graph and model
    graph = helper.make_graph([add_node], "add_graph", [a, b], [c])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])

    # Validate and save
    check_model(model)
    save_model(model, "add_model.onnx")
    print("Saved add_model.onnx")


if __name__ == "__main__":
    main()
