#!/usr/bin/env python3
"""
Fixed version of the user's original Python script.

This shows what the user's original script should have looked like
to work correctly with their C++ implementation.
"""

import onnx
from onnx import helper, checker, shape_inference, onnx_pb as onnx_proto


def create_fixed_model():
    """Create ONNX model matching the user's intent but with fixes."""
    
    # FIXED: Use consistent operator name and domain
    nodes = [
        helper.make_node(
            "ThresholdedRelu",  # FIXED: Consistent with C++ registration
            ["X"],              # FIXED: Single input as defined
            ["Y"],              # FIXED: Single output as defined  
            domain='custom.ops'  # FIXED: Use a standard custom domain name
        )
    ]
    
    # Input/output definitions remain the same (these were correct)
    inputs = [
        helper.make_tensor_value_info("X", onnx_proto.TensorProto.FLOAT, [10])
    ]
    outputs = [
        helper.make_tensor_value_info("Y", onnx_proto.TensorProto.FLOAT, [10])
    ]
    
    graph = helper.make_graph(nodes, "graph_name", inputs, outputs)
    
    # FIXED: Use consistent domain name
    model = helper.make_model(
        graph, 
        opset_imports=[helper.make_operatorsetid('custom.ops', 1)]  # FIXED: Match domain name
    )
    
    model = shape_inference.infer_shapes(model)
    checker.check_model(model)
    onnx.save(model, "fixed_thresholded_relu.onnx")
    
    print("Fixed model created successfully!")
    print("Key fixes:")
    print("1. Consistent operator name: 'ThresholdedRelu'")
    print("2. Consistent domain name: 'custom.ops'")
    print("3. Proper input/output mapping: 1 input -> 1 output")


if __name__ == "__main__":
    create_fixed_model()