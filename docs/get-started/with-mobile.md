---
title: Mobile
parent: Get Started
toc: true
nav_order: 7
---
# Get Started with ONNX Runtime Mobile

The execution environment on mobile devices has fixed memory and disk storage. Therefore, it is essential that any AI execution library is optimized to consume minimum resources in terms of disk footprint, memory and network usage (both model size and binary size).

ONNX Runtime Mobile uses the ORT formatted model which enables us to create a [custom ORT build](../build/custom.md) that minimizes the binary size and reduces memory usage for client side inference. The ORT formatted model file is generated from the regular ONNX model using the `onnxruntime` python package. The custom build does this primarily by only including specified operators and types in the build, as well as trimming down dependencies per custom needs.

An ONNX model must be converted to an ORT format model to be used with minimal build in ONNX Runtime Mobile.

![Steps to build for mobile platforms](../../images/mobile.png){:width="60%"}

There are two options for deploying ONNX Runtime with ORT format model.

## APIs by platform

| Platform | Available APIs |
|----------|----------------|
| Android | C, C++, Java |
| iOS | C, C++, Objective-C (Swift via bridge) |

## ORT format model loading

If you provide a filename for the ORT format model, a file extension of '.ort' will be inferred to be an ORT format model.

If you provide in-memory bytes for the ORT format model, a marker in those bytes will be checked to infer if it's an ORT format model.

If you wish to explicitly say that the InferenceSession input is an ORT format model you can do so via SessionOptions, although this generally should not be necessary.

C++ API

```c++
Ort::SessionOptions session_options;
session_options.AddConfigEntry('session.load_model_format', 'ORT');

Ort::Env env;
Ort::Session session(env, <path to model>, session_options);
```

Java API

```java
SessionOptions session_options = new SessionOptions();
session_options.addConfigEntry("session.load_model_format", "ORT");

OrtEnvironment env = OrtEnvironment.getEnvironment();
OrtSession session = env.createSession(<path to model>, opsession_optionstions);
```

## Learn More
- [Deploy on mobile](../tutorials/mobile/)