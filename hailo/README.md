# Hailo Execution Provider
Hailo ONNX Runtime integrates ONNX Runtime with HailoRT to enable Hailo-EP, providing hardware accelerated inference on the Hailo-8 device.

### Supported versions
* ONNX Runtime version 1.11.1 with Python 3.7 and above

# Prerequisites
* HailoRT v4.8.1

# Build Instructions
To build ONNXRuntime with HailoRT please follow the following steps:
* Clone ONNXRuntime-Hailo from github.
* Compile ONNXRuntime with Hailo using the following command:
    ```
        ./build.sh --use_hailo --parallel --skip_tests --enable_pybind --build_wheel --config Release
    ```

# Run ONNX Runtime with HailoRT
To run your ONNX model on ONNXRuntime with Hailo execution provider, follow the following steps:
1. Convert your ONNX model with DFC tool - see [Model Compilation](https://hailo.ai/developer-zone/documentation/dataflow-compiler/latest/?sp_referrer=compilation.html#for-inference-using-onnx-runtime).
2. Create the ONNXRuntime session with `"HailoExecutionProvider"` in the execution providers list, and run the ONNX model.

## Examples:
* C++

    The file [hailo_basic_test](./../onnxruntime/test/providers/hailo/hailo_basic_test.cc) contains basic tests that run with Hailo-EP.
    
    The ONNX models used in these tests are located in [testdata/hailo directory](./../onnxruntime/test/testdata/hailo/).
    To run the tests, do the following:
    1. Compile onnxruntime with Hailo.
    2. Go to `build/Linux/Release/`.
    3. Run a test with the name `Test_Name`: `./onnxruntime_test_all --gtest_filter=HailoCustomOpTest.Test_Name`.
* Python

    The example `hailo/examples/hailo_example.py` contains a basic inference example using onnxruntime with Hailo-EP.
