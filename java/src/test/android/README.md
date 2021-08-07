# Android Test Application for ORT-Mobile 

This directory contains a simple android application for testing [ONNX Runtime AAR package](https://www.onnxruntime.ai/docs/how-to/build.html#build-android-archive-aar).

## Background

For general usage and build purpose of ORT-Mobile Android, please see the [documentation](https://www.onnxruntime.ai/docs/how-to/build.html#android) here.

### Test Android Application Overview

This android application is mainly aimed for testing:

- Model used: A simple [sigmoid ONNX model](https://github.com/onnx/onnx/blob/f9b0cc99344869c246b8f4011b8586a39841284c/onnx/backend/test/data/node/test_sigmoid/model.onnx) (converted to ORT format under `app\src\androidTest\assets` folder).
    - Here's a [documentation](https://github.com/microsoft/onnxruntime/blob/master/docs/ONNX_Runtime_for_Mobile_Platforms.md#1-create-ort-format-model-and-configuration-file-with-required-operators) about how you can convert an ONNX model into ORT format.
- Main test file: An android instrumentation test under `app\src\androidtest\java\ai.onnxruntime.example.javavalidator\SimpleTest.kt`
- The main dependency of this application is `onnxruntime` aar package under `app\libs`.
- The MainActivity of this application is set to be empty.

### Requirements

- JDK version 8 or later is required.
- The [Gradle](https://gradle.org/) build system is required for building the APKs used to run [android instrumentation tests](https://source.android.com/compatibility/tests/development/instrumentation). Version 6 or newer is required.

### Building

Use the android's [build instructions](https://www.onnxruntime.ai/docs/how-to/build.html#android-build-instructions) with `--build_java` and `--android_run_emulator` option.

Please note that you may need to set the `--android_abi=x86_64` (the default option is `arm64-v8a`). This is because android instrumentation test is run on an android emulator which requires an abi of `x86_64`.

#### Build Output

The build will generate two apks which is required to run the test application in `$YOUR_BUILD_DIR/java/androidtest/android/app/build/outputs/apk`:

* `androidtest/debug/app-debug-androidtest.apk` 
* `debug/app-debug.apk`

After running the build script, the two apks will be installed on `ort_android` emulator and it will automatically run the test application in an adb shell.
