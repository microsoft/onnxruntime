---
title: NNAPI
parent: Execution Providers
grand_parent: Reference
nav_order: 6
---


# NNAPI Execution Provider
{: .no_toc }

[Android Neural Networks API (NNAPI)](https://developer.android.com/ndk/guides/neuralnetworks) is a unified interface to CPU, GPU, and NN accelerators on Android.

## Contents
{: .no_toc }

* TOC placeholder
{:toc}

## Minimum requirements

The NNAPI EP requires Android devices with Android 8.1 or higher, it is recommended to use Android devices with Android 9 or higher to achieve optimal performance.

## Build NNAPI EP

For build instructions, please see the [BUILD page](../../how-to/build.md#android-nnapi-execution-provider).

## Using NNAPI EP in C/C++

```c++
Ort::Env env = Ort::Env{ORT_LOGGING_LEVEL_ERROR, "Default"};
Ort::SessionOptions sf;
uint32_t nnapi_flags = 0;
Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_Nnapi(sf, nnapi_flags));
Ort::Session session(env, model_path, sf);
```

The C API details are [here](../api/c-api.md).

## Configuring NNAPI Execution Provider run time options

There are several run time options for NNAPI Execution Provider.

* NNAPI_FLAG_USE_FP16

   Using fp16 relaxation in NNAPI EP, this may improve perf but may also reduce precision.

* NNAPI_FLAG_USE_NCHW

   Use NCHW layout in NNAPI EP, this is only available after Android API level 29. Please note for now, NNAPI might perform worse using NCHW compare to using NHWC.

* NNAPI_FLAG_CPU_DISABLED

   Prevent NNAPI from using CPU devices.

   NNAPI is more efficient using GPU or NPU for execution, and NNAPI might fall back to its own CPU implementation for operations not supported by GPU/NPU. The CPU implementation of NNAPI (which is called nnapi-reference) might be less efficient than the optimized versions of the operation of ORT. It might be advantageous to disable the NNAPI CPU fallback and handle execution using ORT kernels.

   For some models, if NNAPI would use CPU to execute an operation, and this flag is set, the execution of the model may fall back to ORT kernels.

   This option is only available after Android API level 29, and will be ignored for Android API level 28 and lower.

   For NNAPI device assignments, see https://developer.android.com/ndk/guides/neuralnetworks#device-assignment

   For NNAPI CPU fallback, see https://developer.android.com/ndk/guides/neuralnetworks#cpu-fallback


To use NNAPI execution provider run time options, create an unsigned integer representing the options, and set each individual options by using the bitwise OR operator,

```
uint32_t nnapi_flags = 0;
nnapi_flags |= NNAPI_FLAG_USE_FP16;
```