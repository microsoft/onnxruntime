---
title: Qualcomm - QNN
description: Execute ONNX models with QNN Execution Provider 
parent: Execution Providers
nav_order: 6
redirect_from: /docs/reference/execution-providers/QNN-ExecutionProvider
---

# QNN Execution Provider
{: .no_toc }

The QNN Execution Provider for ONNX Runtime enables hardware accelerated execution on Qualcomm chipsets. It uses the Qualcomm Neural Network SDK to
construct a QNN graph from an ONNX model which can be executed by a supported accelerator backend library.


## Contents
{: .no_toc }

* TOC placeholder
{:toc}

## Install Pre-requisites

Download the Qualcomm Neural Network (QNN) SDK from the Qualcomm Developer Network for [Android/Linux](https://developer.qualcomm.com/TBD)
or [Windows](https://developer.qualcomm.com/TBD)

### QNN Version Requirements

ONNX Runtime QNN Execution Provider has been built and tested with QNN 2.10.x and Qualcomm SC8280, SM8350 SOC's

## Build
For build instructions, please see the [BUILD page](../build/eps.md#qnn).

## Configuration Options
The QNN Execution Provider supports a number of configuration options. The `provider_option_keys`, `provider_options_values` enable different options for the application. Each `provider_options_keys` accepts values as shown below:

|`provider_options_values` for `provider_options_keys = "backend_path"`|Description|
|---|-----|
|'libQnnCpu.so' or 'QnnCpu.dll'|Enable CPU backend|
|'libQnnHtp.do' or 'QnnHtp.dll'|Enable Htp backend|

|`provider_options_values` for `provider_options_keys = "profiling_level"`|Description|
|---|---|

|`provider_options_values` for `provider_options_keys = "rpc_control_latency"`|Description|
|---|---|

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
sess = ort.InferenceSession(model_path, providers=['QNNExecutionProvider'], provider_options=[{'backend_path':'QnnHtp.dll'}])`
```

### Inference example

[Image classification with SqueezeNet in CPP using SNPE Execution Provider](https://github.com/microsoft/onnxruntime-inference-examples/tree/main/c_cxx/QNN_EP)
