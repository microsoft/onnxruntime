# iOS End-to-End Test App for ORT-Mobile

This End-to-End test app for iOS will test ORT Mobile C/C++ API framework using XCode and CocoaPods

## Requirements

- [Prerequisites for building ORT-Mobile for iOS](https://onnxruntime.ai/docs/build/ios.html#prerequisites)
- [CocoaPods](https://cocoapods.org/)

## iOS End-to-End Test App Overview

The iOS End-to-End Test App will use CocoaPods to install the Onnx Runtime C/C++ framework, and run basic End-to-End tests of Onnx Runtime C and C++ API.

### Model used
- [sigmoid ONNX model](https://github.com/onnx/onnx/blob/f9b0cc99344869c246b8f4011b8586a39841284c/onnx/backend/test/data/node/test_sigmoid/model.onnx) converted to ORT format

    Here's [documentation](https://onnxruntime.ai/docs/reference/ort-format-models.html#convert-onnx-models-to-ort-format) about how you can convert an ONNX model into ORT format.

    Run `python -m onnxruntime.tools.convert_onnx_models_to_ort --optimization_style=Fixed /path/to/model.onnx` and rename the resulting .ort file accordingly.

### Tests
- [Tests for C++ API ](./ios_package_testUITests/ios_package_uitest_cpp_api.mm)

## Build and Test iOS Framework using [build.py](../../../../../tools/ci_build/build.py)

Use the [build for iOS simulator](https://onnxruntime.ai/docs/build/ios.html#cross-compile-for-ios-simulator) with `--build_apple_framework`
