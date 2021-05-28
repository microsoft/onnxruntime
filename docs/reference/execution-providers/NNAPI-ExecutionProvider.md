---
title: NNAPI
parent: Execution Providers
grand_parent: Reference
---
{::options toc_levels="2" /}

# NNAPI Execution Provider

[Android Neural Networks API (NNAPI)](https://developer.android.com/ndk/guides/neuralnetworks) is a unified interface to CPU, GPU, and NN accelerators on Android.

## Contents
{: .no_toc }

* TOC placeholder
{:toc}

## Requirements

The NNAPI Execution Provider (EP) requires Android devices with Android 8.1 or higher. It is recommended to use Android devices with Android 9 or higher to achieve optimal performance.

## Build

Please see the [ONNX Runtime Mobile](../../how-to/mobile) deployment information for instructions on building or using a pre-built package that includes the NNAPI EP.

## Usage

The ONNX Runtime API details are [here](../api). 

The NNAPI EP can be used via the C, C++ or Java APIs

The NNAPI EP must be explicitly registered when creating the inference session. For example:

```C++
Ort::Env env = Ort::Env{ORT_LOGGING_LEVEL_ERROR, "Default"};
Ort::SessionOptions so;
uint32_t nnapi_flags = 0;
Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_Nnapi(so, nnapi_flags));
Ort::Session session(env, model_path, so);
```

## Configuration Options

There are several run time options available for the NNAPI EP.

To use the NNAPI EP run time options, create an unsigned integer representing the options, and set each individual options by using the bitwise OR operator.

```
uint32_t nnapi_flags = 0;
nnapi_flags |= NNAPI_FLAG_USE_FP16;
```

### Available Options
##### NNAPI_FLAG_USE_FP16

Use fp16 relaxation in NNAPI EP. 

This may improve performance but can also reduce accuracy due to the lower precision.

##### NNAPI_FLAG_USE_NCHW

Use the NCHW layout in NNAPI EP. 

This is only available for Android API level 29 and later. Please note that for now, NNAPI might have worse performance using NCHW compared to using NHWC.

##### NNAPI_FLAG_CPU_DISABLED

Prevent NNAPI from using CPU devices.

NNAPI is more efficient using GPU or NPU for execution, however NNAPI might fall back to its CPU implementation for operations that are not supported by GPU/NPU. The CPU implementation of NNAPI (which is called nnapi-reference) is often less efficient than the optimized versions of the operation of ORT. Due to this, it may be advantageous to disable the NNAPI CPU fallback and handle execution using ORT kernels.

For some models, if NNAPI would use CPU to execute an operation, and this flag is set, the execution of the model may fall back to ORT kernels.

This option is only available after Android API level 29, and will be ignored for Android API level 28 and lower.

For NNAPI device assignments, see [https://developer.android.com/ndk/guides/neuralnetworks#device-assignment](https://developer.android.com/ndk/guides/neuralnetworks#device-assignment)

For NNAPI CPU fallback, see [https://developer.android.com/ndk/guides/neuralnetworks#cpu-fallback](https://developer.android.com/ndk/guides/neuralnetworks#cpu-fallback)



