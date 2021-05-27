# iOS End-to-End Test App for ORT-Mobile

This End-to-End test app for iOS will test ORT Mobile C/C++ API framework using XCode and CocoaPods

## Requirements

- [Prerequisites for building ORT-Mobile for iOS](http://www.onnxruntime.ai/docs/how-to/build/android-ios.html#prerequisites-1)
- [CocoaPods](https://cocoapods.org/)

## iOS End-to-End Test App Overview

The iOS End-to-End Test App will use CocoaPods to install the Onnx Runtime C/C++ framework, and run basic End-to-End tests of Onnx Runtime C and C++ API.

### Model used
- [sigmoid ONNX model](https://github.com/onnx/onnx/blob/f9b0cc99344869c246b8f4011b8586a39841284c/onnx/backend/test/data/node/test_sigmoid/model.onnx) converted to ORT format

    Here's the [document](http://www.onnxruntime.ai/docs/how-to/deploy-on-mobile.html#1-create-ort-format-model-and-configuration-file-with-required-operators) about how you can convert an ONNX model into ORT format.

### Tests
- [Tests for C API ](./ios_package_testTests/ios_package_test_c_api.m)
- [Tests for C++ API ](./ios_package_testTests/ios_package_test_cpp_api.mm)

## Build and Test iOS Framework using [build.py](../../../../../tools/ci_build/build.py)

Use the [build for iOS simulator](http://www.onnxruntime.ai/docs/how-to/build/android-ios.html#cross-build-for-ios-simulator) with `--build_apple_framework`

## Run the iOS End-to-End Test App standalone

### Requirements

- A pre-built ORT Mobile iOS framework, which can be built using the [instruction](#build-and-test-ios-framework-using-buildpy) above. The framework can be found as `<build_dir>/iOS/<build-config>/<build-config>-iphonesimulator/onnxruntime.framework`

### Steps

1. Go to this folder
2. Copy the [OnnxRuntimeBase.podspec.template](./OnnxRuntimeBase.podspec.template) to `OnnxRuntimeBase.podspec`
3. Update the `OnnxRuntimeBase.podspec`, replace `${ORT_BASE_FRAMEWORK_ARCHIVE}` with the path of a zip archive contains the pre-built ORT Mobile iOS framework
4. Run `pod install` to install the pre-built ORT Mobile iOS framework
5. Run the following command to perform the test

```
    xcrun xcodebuild \
        -workspace ./ios_package_test.xcworkspace \
        -destination '<Your choice of test target device' \
        -scheme ios_package_test \
        test
```