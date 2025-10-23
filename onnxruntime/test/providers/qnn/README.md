# ONNX Runtime QNN Execution Provider Tests
## Overview
1. The `onnxruntime/test/providers/qnn` directory contains integration tests for the Qualcomm Neural Network (QNN) execution provider.
2. Most testcases run an ONNX model through the QNN-EP, then verifies the inference result against the one on CPU-EP

## Building the Tests
The tests are built as part of the regular ONNX Runtime build. After a successful build you will have an executable named
- onnxruntime_provider_test.exe   (Windows)
- onnxruntime_provider_test      (Linux/macOS)

## Running the Tests
1. QNN supports several backends. You can use the standard Google‑Test syntax for filtering:
    - `onnxruntime_provider_test.exe --gtest_filter=QnnCPUBackendTests.*`
    - `onnxruntime_provider_test.exe --gtest_filter=QnnHTPBackendTests.*`
    - `onnxruntime_provider_test.exe --gtest_filter=QnnGPUBackendTests.*`
    - `onnxruntime_provider_test.exe --gtest_filter=QnnIRBackendTests.*`
2. Saving Test Artifacts
    - For debugging it is often helpful to keep the intermediate files that the tests generate. The following custom flags are
    recognized by the test binary:
        - `--dump_onnx`: Saves the input ONNX model used for the test
        - `--dump_json`: Save json qnn graph with provider_option `dump_json_qnn_graph`
        - `--dump_dlc`: Saves the compiled QNN DLC file by specifying the provider_option `backend_path` to `QnnIr.dll`
    - The artifacts will be saved to a directory named with `<TestSuite>_<TestName>`
        ```
        .
        ├── QnnCPUBackendTests_BatchNorm2D_fp32         # RunQnnModelTest
        │   ├── dumped_f32_model.onnx                   # float32 ONNX model
        │   ├── QNNExecutionProvider_QNN_XXXX_X_X.dlc
        │   └── QNNExecutionProvider_QNN_XXXX_X_X.json
        ├── QnnHTPBackendTests_BatchNorm_FP16           # TestFp16ModelAccuracy
        │   ├── dumped_f16_model.onnx                   # float16 ONNX model
        │   ├── dumped_f32_model.onnx                   # float32 ONNX model
        │   ├── QNNExecutionProvider_QNN_XXXX_X_X.dlc
        │   └── QNNExecutionProvider_QNN_XXXX_X_X.json
        └── QnnHTPBackendTests_BatchNorm2D_U8U8S32      # TestQDQModelAccuracy
            ├── dumped_f32_model.onnx                   # float32 ONNX model
            ├── dumped_qdq_model.onnx                   # QDQ ONNX model
            ├── QNNExecutionProvider_QNN_XXXX_X_X.dlc
            └── QNNExecutionProvider_QNN_XXXX_X_X.json

        # All artifact files are placed under the current working directory from which the test binary is invoked.
        ```
3. Verbose
    - `--verbose`: Sets the ONNX Runtime log level to `ORT_LOGGING_LEVEL_VERBOSE`

4. You can enable any combination of these flags, for example:
    - `onnxruntime_provider_test.exe --gtest_filter=QnnHTPBackendTests.* --dump_onnx --dump_json --dump_dlc --verbose`

# Note
- An issue on QNN backends can prevent the test artifacts from being successfully saved.
- The `onnxruntime_provider_test.exe` does not automatically delete the artifact directories, so you may want to prune them after a debugging session.
