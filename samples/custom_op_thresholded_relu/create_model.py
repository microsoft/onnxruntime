#!/usr/bin/env python3
"""
Creates an ONNX model with a ThresholdedRelu custom operator.

This demonstrates the correct way to define a custom operator in ONNX
that matches the C++ implementation.
"""

import onnx
from onnx import helper, checker, shape_inference, onnx_pb as onnx_proto


def create_thresholded_relu_model():
    """Create ONNX model with ThresholdedRelu custom operator."""
    
    # Define the custom operator node
    # Note: operator name MUST match the C++ registration name
    nodes = [
        helper.make_node(
            "ThresholdedRelu",  # This must match C++ registration
            ["X"],              # Single input
            ["Y"],              # Single output  
            domain='custom.ops',  # Custom domain name
            alpha=1.0           # Optional attribute for threshold value
        )
    ]
    
    # Define inputs and outputs
    # Note: shapes and types must match what C++ implementation expects
    inputs = [
        helper.make_tensor_value_info(
            "X", 
            onnx_proto.TensorProto.FLOAT, 
            [10]  # 1D tensor with 10 elements
        )
    ]
    
    outputs = [
        helper.make_tensor_value_info(
            "Y", 
            onnx_proto.TensorProto.FLOAT, 
            [10]  # Same shape as input
        )
    ]
    
    # Create the graph
    graph = helper.make_graph(
        nodes,
        "ThresholdedReluGraph",
        inputs,
        outputs
    )
    
    # Create the model with custom operator set
    model = helper.make_model(
        graph,
        opset_imports=[
            helper.make_operatorsetid('custom.ops', 1)  # Custom domain version 1
        ]
    )
    
    # Add metadata
    model.doc_string = "ThresholdedRelu Custom Operator Example"
    
    # Validate the model
    checker.check_model(model)
    
    return model


def main():
    """Create and save the model."""
    model = create_thresholded_relu_model()
    
    # Save the model
    output_path = "thresholded_relu.onnx"
    onnx.save(model, output_path)
    print(f"Model saved to {output_path}")
    
    # Print model info
    print(f"Model inputs: {[input.name for input in model.graph.input]}")
    print(f"Model outputs: {[output.name for output in model.graph.output]}")
    print(f"Custom operators: {[node.op_type for node in model.graph.node]}")


if __name__ == "__main__":
    main()