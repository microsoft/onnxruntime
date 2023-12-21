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


|`provider_options_values` for `provider_options_keys = "qnn_context_priority"`|[Description](https://docs.qualcomm.com/bundle/publicresource/topics/80-63442-50/htp_yielding.html)|
|---|---|
|'low'||
|'normal'|default.|
|'normal_high'||
|'high'||


|`provider_options_values` for `provider_options_keys = "htp_graph_finalization_optimization_mode"`|Description|
|---|---|
|'0'|default.|
|'1'|faster preparation time, less optimal graph.|
|'2'|longer preparation time, more optimal graph.|
|'3'|longest preparation time, most likely even more optimal graph.|


## QNN context binary cache feature
There's a QNN context which contains QNN graphs after converting, compiling, filnalizing the model. QNN can serialize the context into binary file, so that user can use it for futher inference direclty (without the QDQ model) to improve the model loading cost.
The QNN Execution Provider supports a number of session options to configure this.

### Dump QNN context binary
1. Create session option, set "ep.context_enable" to "1" to enable QNN context dump. The key "ep.context_enable" is defined as kOrtSessionOptionEpContextEnable in [onnxruntime_session_options_config_keys.h](https://github.com/microsoft/onnxruntime/blob/8931854528b1b2a3f320d012c78d37186fbbdab8/include/onnxruntime/core/session/onnxruntime_session_options_config_keys.h#L239-L252).
2. Create the session with the QDQ model using session options created in step 1, and use HTP backend
A Onnx model with QNN context binary will be created once the session is created/initialized. No need to run the session.
The QNN context binary generation can be done on the QualComm device which has HTP using Arm64 build. It can also be done on x64 machine using x64 build (not able to run it since there's no HTP device).

The generated Onnx model which has QNN context binary can be deployed to production/real device to run inference. This Onnx model is treated as a normal model by QNN Execution Provider. Inference code keeps same as inference with QDQ model on HTP backend.

[Code example](https://github.com/microsoft/onnxruntime-inference-examples/blob/733ce6f3e8dd2ede8b67a8465684bca2f62a4a33/c_cxx/QNN_EP/mobilenetv2_classification/main.cpp#L90-L97)
```
#include "onnxruntime_session_options_config_keys.h"

// C++
Ort::SessionOptions so;
so.AddConfigEntry(kOrtSessionOptionEpContextEnable, "1");

// C
const OrtApi* g_ort = OrtGetApiBase()->GetApi(ORT_API_VERSION);
OrtSessionOptions* session_options;
CheckStatus(g_ort, g_ort->CreateSessionOptions(&session_options));
g_ort->AddSessionConfigEntry(session_options, kOrtSessionOptionEpContextEnable, "1");
```

### Configure the context binary file path
The generated Onnx model with QNN context binary is default to [input_QDQ_model_path]_ctx.onnx in case user does not specify the path. User can to set the path in the session option with the key "ep.context_file_path". Example code below:

```
// C++
so.AddConfigEntry(kOrtSessionOptionEpContextFilePath, "./model_a_ctx.onnx");

// C
g_ort->AddSessionConfigEntry(session_options, kOrtSessionOptionEpContextFilePath, "./model_a_ctx.onnx");
```

### Disable the embed mode
The QNN context binary content is embeded in the generated Onnx model by default. User can to disable it by setting "ep.context_embed_mode" to "0". In that case, a bin file will be generated separately. The file name looks like [ctx.onnx]_QNNExecutionProvider_QNN_[hash_id]_x_x.bin. The name is provided by Ort and tracked in the generated Onnx model. It will cause problems if any changes to the bin file. This bin file needs to sit together with the generated Onnx file.

```
// C++
so.AddConfigEntry(kOrtSessionOptionEpContextEmbedMode, "0");

// C
g_ort->AddSessionConfigEntry(session_options, kOrtSessionOptionEpContextEmbedMode, "0");
```

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

[Image classification with Mobilenetv2 in CPP using QNN Execution Provider with QNN CPU & HTP Backend](https://github.com/microsoft/onnxruntime-inference-examples/tree/main/c_cxx/QNN_EP/mobilenetv2_classification)
