# iOS End-to-End Test App for ORT-Mobile

This End-to-End test app for iOS will test ORT Mobile C/C++ API framework using CoocaPods

## Requirements

- [CocoaPods](https://cocoapods.org/)

## Build and Test iOS Framework using build.py

Use the [build for iOS simulator](http://www.onnxruntime.ai/docs/how-to/build/android-ios.html#cross-build-for-ios-simulator) with `--build_apple_framework`

## Run the iOS End-to-End Test App standalone

### Requirements

- A pre-built ORT Mobile iOS framework

### Steps

1. Update the [OnnxRuntimeBase.podspec](./OnnxRuntimeBase.podspec), replace `${ORT_BASE_FRAMEWORK_FILE}` with the path of the pre-built ORT Mobile iOS framework
2. Go to this folder
3. Run `pod install` to install the pre-built ORT Mobile iOS framework
4. Run the following command to perform the test

    ```
    xcodebuild test -workspace ios_package_test.xcworkspace -destination '<Your choice of test target device' -scheme ios_package_test
    ```