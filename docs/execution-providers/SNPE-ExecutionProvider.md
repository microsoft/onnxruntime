---
title: SNPE (Qualcomm)
description: Execute ONNX models with SNPE Execution Provider 
parent: Execution Providers
nav_order: 13
redirect_from: /docs/reference/execution-providers/SNPE-ExecutionProvider
---

# SNPE Execution Provider
{: .no_toc }

The SNPE Execution Provider for ONNX Runtime enables hardware accelerated execution on Qualcomm Snapdragon CPU, the Qualcomm Adreno<sup>TM</sup> GPU, or the Hexagon DSP. This execution provider makes use of the Qualcomm Snapdragon Neural Processing Engine SDK.

This execution provider uses the AOT converted DLC code as an embedded node in the ONNX model file.

## Contents
{: .no_toc }

* TOC placeholder
{:toc}

## Install Pre-requisites

Download the SNPE toolkit from the Qualcomm Developer Network for [Android/Linux](https://developer.qualcomm.com/software/qualcomm-neural-processing-sdk)
or [Windows](https://developer.qualcomm.com/software/qualcomm-neural-processing-sdk/windows-on-snapdragon)

### SNPE Version Requirements

The SNPE version used with the ONNX Runtime SNPE Execution Provider must match the version used to generate the quantized SNPE-DLC file.

## Build
For build instructions, please see the [BUILD page](../build/eps.md#snpe).

## Configuration Options
The SNPE Execution Provider supports a number of options to set the SNPE Runtime configuration for executing the model. The `provider_option_keys`, `provider_options_values` and `num_keys` enable different options for the application. Each `provider_options_keys` accepts values as shown below:

|`provider_options_values` for `provider_options_keys = "runtime"`|Description|
|---|-----|
|CPU or CPU_FLOAT32|Using SnapDragon CPU with 32 bit data storage and math|
|DSP or DSP_FIXED8_TF|Using Hexagon DSP with 8bit fixed point Tensorflow style format data storage and 8bit fixed point Tensorflow style format math|
|GPU or GPU_FLOAT32_16_HYBRID|Using Adreno GPU with 16 bit data storage and 32 bit math|
|GPU_FLOAT16|GPU with 16 bit data storage and 16 bit math|
|AIP_FIXED_TF or AIP_FIXED8_TF|Using Snapdragon AIX+HVX with 8bit fixed point Tensorflow style format data storage and 8bit fixed point Tensorflow style format math|

|`provider_options_values` for `provider_options_keys = "buffer_type"`|Description|
|---|---|
|ITensor|Represents a tensor with n-dimensional data|
|TF8|User defined buffer with 8-bit quantized value|
|TF16|User defined buffer with 16-bit quantized value|
|UINT8|User defined buffer with unsigned int value|
|FLOAT|User defined buffer with float value|

## Usage
### C++
```
Ort::Env env = Ort::Env{ORT_LOGGING_LEVEL_ERROR, "Default"};
std::unordered_map<std::string, std::string> snpe_options;
snpe_options["runtime"] = "DSP";
snpe_options["buffer_type"] = "FLOAT";
Ort::SessionOptions session_options;
session_options.AppendExecutionProvider("SNPE", snpe_options);
Ort::Session session(env, model_path, session_options);
```

The C API details are [here](../get-started/with-c.md).
