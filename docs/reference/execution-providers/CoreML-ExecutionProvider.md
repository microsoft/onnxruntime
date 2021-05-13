---
title: CoreML
parent: Execution Providers
grand_parent: Reference
nav_order: 13
---


# CoreML Execution Provider
{: .no_toc }

[Core ML](https://developer.apple.com/machine-learning/core-ml/) is a machine learning framework introduced by Apple. It is designed to seamlessly take advantage of powerful hardware technology including CPU, GPU, and Neural Engine, in the most efficient way in order to maximize performance while minimizing memory and power consumption.

## Contents
{: .no_toc }

* TOC placeholder
{:toc}

## Minimum requirements

The CoreML Execution Provider requires iOS devices with iOS 13 or higher, or Mac computers with macOS 10.15 or higher. It is recommended to use Apple devices equipped with Apple Neural Engine to achieve optimal performance.

## Build CoreML Execution Provider

For build instructions for iOS devices, please see the [How to: Build for Android/iOS](../../how-to/build/android-ios.md#ios-coreml-execution-provider).

## Using CoreML Execution Provider in C/C++

```c++
Ort::Env env = Ort::Env{ORT_LOGGING_LEVEL_ERROR, "Default"};
Ort::SessionOptions so;
uint32_t coreml_flags = 0;
Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_CoreML(so, coreml_flags));
Ort::Session session(env, model_path, so);
```

The C API details are [here](../api/c-api.md).

## Configuring CoreML Execution Provider run time options

There are several run time options for CoreML Execution Provider.

* COREML_FLAG_USE_CPU_ONLY

   Limit CoreML Execution Provider to CPU based execution. Using this option may decrease the performance but will provide reference output value without precision loss, which is useful for validation.

* COREML_FLAG_ENABLE_ON_SUBGRAPH

   Enable CoreML Execution Provider on subgraph within control flow operators, such as the body graph of [Loop](https://github.com/onnx/onnx/blob/master/docs/Operators.md#loop) operator.

* COREML_FLAG_ONLY_ENABLE_DEVICE_WITH_ANE

   By default CoreML Execution Provider will be enabled for all compatible Apple devices.

   Enable this option will only enable CoreML Execution Provider for Apple devices with compatible ANE (Apple Neural Engine).

   Please note, enable this option does not guarantee the entire model to be executed using ANE only.

   For more information, see [Which devices have an ANE?](https://github.com/hollance/neural-engine/blob/master/docs/supported-devices.md)

To use CoreML Execution Provider run time options, create an unsigned integer representing the options, and set each individual options by using the bitwise OR operator,

```
uint32_t coreml_flags = 0;
coreml_flags |= COREML_FLAG_ONLY_ENABLE_DEVICE_WITH_ANE;
```