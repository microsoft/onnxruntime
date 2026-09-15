# ThresholdedRelu Custom Operator Example

This sample demonstrates how to create a custom operator for ONNX Runtime using the modern lite custom op API.

## Problem Analysis

The original issue had several problems:

1. **Operator Name Mismatch**: The ONNX model defined "ThresholdedRelu" but the C++ implementation registered "AvggKernel"
2. **Input/Output Mismatch**: The ONNX model expected 1 input and 1 output, but the C++ implementation expected 2 inputs and 1 output
3. **Domain Inconsistency**: Mixed usage of domain names and registration methods
4. **API Misuse**: Incorrect mixing of registration approaches

## Solution

This example provides:

1. `create_model.py` - Creates a correct ONNX model with ThresholdedRelu operator
2. `thresholded_relu_op.h/.cpp` - Proper C++ implementation using lite custom op API
3. `test_thresholded_relu.cpp` - Correct usage example
4. `CMakeLists.txt` - Build configuration

## Key Patterns

### 1. Operator Name Consistency
The operator name must match everywhere:
- ONNX model node type
- C++ registration name  
- Domain registration

### 2. Input/Output Matching
The C++ signature must match the ONNX model:
- Number of inputs/outputs
- Data types
- Tensor shapes

### 3. Proper Domain Usage
Use consistent domain names and registration methods.

### 4. Modern API Usage
Use the lite custom op API for simplified development.

## Files

- `create_model.py` - Python script to create the ONNX model
- `thresholded_relu_op.h` - Header file for the custom operator
- `thresholded_relu_op.cpp` - Implementation of the custom operator
- `test_thresholded_relu.cpp` - Test application
- `CMakeLists.txt` - Build configuration