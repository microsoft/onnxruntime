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

import onnx
from onnx import TensorProto, helper

def main():
    # Define inputs and output
    A = helper.make_tensor_value_info("A", TensorProto.FLOAT, [1, 3])
    B = helper.make_tensor_value_info("B", TensorProto.FLOAT, [1, 3])
    C = helper.make_tensor_value_info("C", TensorProto.FLOAT, [1, 3])

    # Create the Add node
    add_node = helper.make_node("Add", inputs=["A", "B"], outputs=["C"])

    # Build the graph and model
    graph = helper.make_graph([add_node], "add_graph", [A, B], [C])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])

    # Validate and save
    onnx.checker.check_model(model)
    onnx.save(model, "add_model.onnx")
    print("Saved add_model.onnx")


if __name__ == "__main__":
    main()
