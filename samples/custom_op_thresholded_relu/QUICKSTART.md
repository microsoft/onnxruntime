# Quick Start Guide

## Overview

This directory contains a complete working solution for implementing a ThresholdedRelu custom operator in ONNX Runtime, addressing the issues found in GitHub issue #25644.

## Files

### Working Implementation
- **`create_model.py`** - Creates the ONNX model
- **`thresholded_relu_op.h/.cpp`** - Modern implementation using lite custom op API
- **`test_thresholded_relu.cpp`** - Test application
- **`CMakeLists.txt`** - Build configuration

### Issue Analysis & Learning
- **`SOLUTION_SUMMARY.md`** - Complete problem analysis and solution
- **`ISSUE_ANALYSIS.md`** - Detailed breakdown of what went wrong
- **`README.md`** - Overview and patterns

### Fixed Original Code
- **`create_fixed_original.py`** - Corrected version of user's Python script
- **`fixed_original.h/.cpp`** - Corrected version of user's C++ implementation  
- **`fixed_original_test.cpp`** - Corrected version of user's test code

## Quick Test

1. **Create model:**
   ```bash
   python3 create_model.py
   ```

2. **Verify compilation:**
   ```bash
   g++ -I../../include -I../../include/onnxruntime/core/session -std=c++17 -c thresholded_relu_op.cpp
   ```

3. **Expected model output:**
   - Input: `[0.5, 1.5, -0.5, 2.0, 0.0, 1.0, -1.0, 3.0, 0.8, 1.2]`
   - Output: `[0.0, 1.5, 0.0, 2.0, 0.0, 0.0, 0.0, 3.0, 0.0, 1.2]`

## Key Learnings

1. **Operator names must be consistent** across ONNX model, C++ registration, and usage
2. **Input/output signatures must match** between ONNX model and C++ implementation
3. **Use modern lite custom op API** for simplified development
4. **Test incrementally** - model creation, compilation, registration, execution
5. **Manage domain lifetimes** properly to avoid crashes

This solution demonstrates the correct patterns for custom operator development in ONNX Runtime.