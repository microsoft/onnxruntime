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

The CoreML EP requires iOS devices with iOS 13 or higher, or mac computers with macOS 10.15 or higher. It is recommended to use Apple devices equipped with Apple Neural Engine to achieve optimal performance.

## Build CoreML EP

For build instructions, please see the [BUILD page](../../how-to/build.md#coreml-execution-provider).

## Using CoreML EP in C/C++

```c++
Ort::Env env = Ort::Env{ORT_LOGGING_LEVEL_ERROR, "Default"};
Ort::SessionOptions sf;
uint32_t coreml_flags = 0;
Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_CoreML(sf, coreml_flags));
Ort::Session session(env, model_path, sf);
```

The C API details are [here](../api/c-api.md).

## Configuring CoreML Execution Provider run time options

There are several run time options for CoreML Execution Provider.

* COREML_FLAG_USE_CPU_ONLY

   Using CPU only in CoreML EP, using this option may decrease the perf but will provide reference output value without precision loss, which is useful for validation.

* COREML_FLAG_ENABLE_ON_SUBGRAPH

   Enable CoreML EP on subgraph.

* COREML_FLAG_ONLY_ENABLE_DEVICE_WITH_ANE

   By default CoreML Execution provider will be enabled for all compatible Apple devices.

   Enable this option will only enable CoreML EP for Apple devices with compatible ANE (Apple Neural Engine).

   Please note, enable this option does not guarantee the entire model to be executed using ANE only.

   For more information, see [Which devices have an ANE?](https://github.com/hollance/neural-engine/blob/master/docs/supported-devices.md)

To use CoreML execution provider run time options, create an unsigned integer representing the options, and set each individual options by using the bitwise OR operator,

```
uint32_t coreml_flags = 0;
coreml_flags |= COREML_FLAG_ONLY_ENABLE_DEVICE_WITH_ANE;
```