<!-- Copyright (c) Microsoft Corporation. All rights reserved.
Licensed under the MIT License. -->

# Onnxruntime Qualcomm Neural Network Executaion Provider Tool (QNN EP Tool)
- The tool runs onnxruntime session inference with QNN Executaion Provider on inputs and save the outputs.
- The inputs/outputs can be .pb or .raw format.

## Model and Inputs Data Directory Structure
The tool expects the onnx model and inputs data to be arranged in the following directory structure:
```bash
resnet18-v1-7
├── resnet18-v1-7.onnx
├── test_data_set_0   
│   └── input_0.pb (or input_0.raw)
└── test_data_set_1   
    └── input_0.pb (or input_0.raw)
```
In each test_data_set_X/
1. If only .pb input data provided, the tool will use .pb as input
2. If only .raw input data provided, the tool will use .raw as input
3. If both are provided (not recommended), the tool will prioritize .pb

## Build
The tool can be built with the flag "--build_qnn_ep_tool" set.
```cmd
.\build.bat --config RelWithDebInfo --build_shared_lib --parallel --compile_no_warning_as_error --cmake_generator "Visual Studio 17 2022" --use_qnn --qnn_home <path-to-qnn-sdk> --build_qnn_ep_tool
```

## Command Line Usage
1. The following command serves as an example to run the tool
    ```ps1
    # qnn_ep_tools.exe <model_dir> <backend_path>
    .\qnn_ep_tools.exe resnet18-v1-7 QnnCpu.dll
    ```

2. The tool will produce .pb / .raw under the corresponding directory
    ```bash
    resnet18-v1-7
    ├── resnet18-v1-7.onnx
    ├── test_data_set_0   
    │   ├── input_0.pb (input_0.raw)
    │   └── out_0.pb (out_0.raw)
    └── test_data_set_1   
        ├── input_0.pb (input_0.raw)
        └── out_0.pb (out_0.raw)
    ```