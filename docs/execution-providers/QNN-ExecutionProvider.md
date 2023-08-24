---
title: Qualcomm - QNN
description: Execute ONNX models with QNN Execution Provider 
parent: Execution Providers
nav_order: 6
redirect_from: /docs/reference/execution-providers/QNN-ExecutionProvider
---

# QNN Execution Provider
{: .no_toc }

The QNN Execution Provider for ONNX Runtime enables hardware accelerated execution on Qualcomm chipsets. 
It uses the Qualcomm AI Engine Direct SDK (QNN SDK) to construct a QNN graph from an ONNX model which can 
be executed by a supported accelerator backend library.


## Contents
{: .no_toc }

* TOC placeholder
{:toc}

## Install Pre-requisites

Download the Qualcomm AI Engine Direct SDK (QNN SDK) from [https://qpm.qualcomm.com/main/tools/details/qualcomm_ai_engine_direct](https://qpm.qualcomm.com/main/tools/details/qualcomm_ai_engine_direct)

### QNN Version Requirements

ONNX Runtime QNN Execution Provider has been built and tested with QNN 2.10.x and Qualcomm SC8280, SM8350 SOC's

## Build
For build instructions, please see the [BUILD page](../build/eps.md#qnn).
[prebuilt NuGet package](https://www.nuget.org/packages/Microsoft.ML.OnnxRuntime.QNN)

## Configuration Options
The QNN Execution Provider supports a number of configuration options. The `provider_option_keys`, `provider_options_values` enable different options for the application. Each `provider_options_keys` accepts values as shown below:

|`provider_options_values` for `provider_options_keys = "backend_path"`|Description|
|---|-----|
|'libQnnCpu.so' or 'QnnCpu.dll'|Enable CPU backend. Useful for integration testing. CPU backend is a reference implementation of QNN operators|
|'libQnnHtp.so' or 'QnnHtp.dll'|Enable Htp backend. Offloads compute to NPU.|

|`provider_options_values` for `provider_options_keys = "profiling_level"`|Description|
|---|---|
|'off'||
|'basic'||
|'detailed'||

|`provider_options_values` for `provider_options_keys = "rpc_control_latency"`|Description|
|---|---|
|microseconds (string)|allows client to set up RPC control latency in microseconds|

|`provider_options_values` for `provider_options_keys = "htp_performance_mode"`|Description|
|---|---|
|'burst'||
|'balanced'||
|'default'||
|'high_performance'||
|'high_power_saver'||
|'low_balanced'||
|'low_power_saver'||
|'power_saver'||
|'sustained_high_performance'||


|`provider_options_values` for `provider_options_keys = "qnn_context_cache_enable"`|Description|
|---|---|
|'0'|disabled (default)|
|'1'|enable qnn context cache. write out prepared Htp Context Binary to disk to save initialization costs.|


|`provider_options_values` for `provider_options_keys = "qnn_context_cache_path"`|Description|
|---|---|
|'/path/to/context/cache'|string path to context cache binary|


## Usage
### C++
C API details are [here](../get-started/with-c.md).
```
Ort::Env env = Ort::Env{ORT_LOGGING_LEVEL_ERROR, "Default"};
std::unordered_map<std::string, std::string> qnn_options;
qnn_options["backend_path"] = "QnnHtp.dll";
Ort::SessionOptions session_options;
session_options.AppendExecutionProvider("QNN", qnn_options);
Ort::Session session(env, model_path, session_options);
```
### Python
```
import onnxruntime as ort
# Create a session with QNN EP using HTP (NPU) backend.
sess = ort.InferenceSession(model_path, providers=['QNNExecutionProvider'], provider_options=[{'backend_path':'QnnHtp.dll'}])`
```

### Inference example

[Image classification with Mobilenetv2 in CPP using QNN Execution Provider with QNN CPU & HTP Backend](https://github.com/microsoft/onnxruntime-inference-examples/tree/main/c_cxx/QNN_EP)
