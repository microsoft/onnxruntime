---
title: CoreML
parent: Execution Providers
grand_parent: Reference
---
{::options toc_levels="2" /}

# CoreML Execution Provider

[Core ML](https://developer.apple.com/machine-learning/core-ml/) is a machine learning framework introduced by Apple. It is designed to seamlessly take advantage of powerful hardware technology including CPU, GPU, and Neural Engine, in the most efficient way in order to maximize performance while minimizing memory and power consumption.

## Contents
{: .no_toc }

* TOC placeholder
{:toc}

## Requirements

The CoreML Execution Provider (EP) requires iOS devices with iOS 13 or higher, or Mac computers with macOS 10.15 or higher. 

It is recommended to use Apple devices equipped with Apple Neural Engine to achieve optimal performance.

## Build

For build instructions for iOS devices, please see the [How to: Build for Android/iOS](../../how-to/build/android-ios.md#coreml-execution-provider).

## Usage

The ONNX Runtime API details are [here](../api). 

The CoreML EP can be used via the C or C++ APIs currently. Additional support via the Objective-C API is in progress.

The CoreML EP must be explicitly registered when creating the inference session. For example:

```C++
Ort::Env env = Ort::Env{ORT_LOGGING_LEVEL_ERROR, "Default"};
Ort::SessionOptions so;
uint32_t coreml_flags = 0;
Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_CoreML(so, coreml_flags));
Ort::Session session(env, model_path, so);
```

## Configuration Options

There are several run time options available for the CoreML EP.

To use the CoreML EP run time options, create an unsigned integer representing the options, and set each individual option by using the bitwise OR operator.

```
uint32_t coreml_flags = 0;
coreml_flags |= COREML_FLAG_ONLY_ENABLE_DEVICE_WITH_ANE;
```

### Available Options

##### COREML_FLAG_USE_CPU_ONLY

Limit CoreML to running on CPU only.

This may decrease the performance but will provide reference output value without precision loss, which is useful for validation.

#####  COREML_FLAG_ENABLE_ON_SUBGRAPH

Enable CoreML EP to run on a subgraph in the body of a control flow operator (i.e. a [Loop](https://github.com/onnx/onnx/blob/master/docs/Operators.md#loop), [Scan](https://github.com/onnx/onnx/blob/master/docs/Operators.md#scan) or [If](https://github.com/onnx/onnx/blob/master/docs/Operators.md#if) operator).


##### COREML_FLAG_ONLY_ENABLE_DEVICE_WITH_ANE

By default the CoreML EP will be enabled for all compatible Apple devices.

Setting this option will only enable CoreML EP for Apple devices with a compatible Apple Neural Engine (ANE). 
Note, enabling this option does not guarantee the entire model to be executed using ANE only. 

For more information, see [Which devices have an ANE?](https://github.com/hollance/neural-engine/blob/master/docs/supported-devices.md)

