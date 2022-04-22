# iOS End-to-End Test App for ORT-Mobile

This End-to-End test app for iOS will test ORT Mobile C/C++ API framework using XCode and CocoaPods

## Requirements

- [Prerequisites for building ORT-Mobile for iOS](https://onnxruntime.ai/docs/build/android-ios.html#prerequisites-1)
- [CocoaPods](https://cocoapods.org/)

## iOS End-to-End Test App Overview

The iOS End-to-End Test App will use CocoaPods to install the Onnx Runtime C/C++ framework, and run basic End-to-End tests of Onnx Runtime C and C++ API.

### Model used
- [sigmoid ONNX model](https://github.com/onnx/onnx/blob/f9b0cc99344869c246b8f4011b8586a39841284c/onnx/backend/test/data/node/test_sigmoid/model.onnx) converted to ORT format

    Here's the [document](https://onnxruntime.ai/docs/tutorials/mobile/model-conversion.html) about how you can convert an ONNX model into ORT format.

### Tests
- [Tests for C++ API ](./ios_package_testUITests/ios_package_uitest_cpp_api.mm)

## Build and Test iOS Framework using [build.py](../../../../../tools/ci_build/build.py)

Use the [build for iOS simulator](https://onnxruntime.ai/docs/build/android-ios.html#cross-build-for-ios-simulator) with `--build_apple_framework`
