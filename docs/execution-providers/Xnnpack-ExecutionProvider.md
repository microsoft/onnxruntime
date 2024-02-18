---
title: XNNPACK
description: Instructions to execute ONNX Runtime with the XNNPACK execution provider
parent: Execution Providers
nav_order: 9
---
{::options toc_levels="2" /}

# XNNPACK Execution Provider

Accelerate ONNX models on Android/iOS devices and WebAssembly with ONNX Runtime and the XNNPACK execution provider. [XNNPACK](https://github.com/google/XNNPACK) is a highly optimized library of floating-point neural network inference operators for ARM, WebAssembly, and x86 platforms.

## Contents
{: .no_toc }

* TOC placeholder
{:toc}


## Install
Pre-built packages of ONNX Runtime ([`onnxruntime-android`](https://mvnrepository.com/artifact/com.microsoft.onnxruntime/onnxruntime-android)) with XNNPACK EP for Android are published on Maven.
See [here](../install/index.md#install-on-android) for installation instructions.

Pre-built binaries(`onnxruntime-objc` and `onnxruntime-c`) of ONNX Runtime with XNNPACK EP for iOS are published to CocoaPods.
See [here](../install/index.md#install-on-ios) for installation instructions.

## Build

Please see the [Build page](../build/eps.md#xnnpack) for instructions on building a package that includes the XNNPACK EP.

You can build ONNX Runtime with the XNNPACK EP for Android, iOS, Windows, and Linux.

## Usage

The ONNX Runtime API details are [here](../api).

The XNNPACK EP can be used via the C, C++ or Java APIs

The XNNPACK EP must be explicitly registered when creating the inference session. For example:

```C++
Ort::Env env = Ort::Env{ORT_LOGGING_LEVEL_ERROR, "Default"};
Ort::SessionOptions so;
so.AppendExecutionProvider("XNNPACK", {"intra_op_num_threads", std::to_string(intra_op_num_threads)});
Ort::Session session(env, model_path, so);
```

## Configuration Options

### Recommended configuration
XNNPACK has a separate internal threadpool which can lead to contention with the ORT intra-op threadpool.
To minimize this, we recommend setting the following options:
1. Disable the ORT intra-op thread-pool spinning by adding the following to the session options:
```C++
    so.AddConfigEntry(kOrtSessionOptionsConfigAllowIntraOpSpinning, "0");
```
2. Set the XNNPACK intra-op thread-pool size when registering the XNNPACK EP. The suggested value would be the number of physical cores on the device.
```C++
    so.AppendExecutionProvider("XNNPACK", {"intra_op_num_threads", std::to_string(intra_op_num_threads)});
```
3. Set the ORT intra-op thread-pool size to 1:
```C++
    so.SetIntraOpNumThreads(1);
```

This configuration will work well if your model is using XNNPACK for the nodes performing the compute-intensive work, as these operators are likely to use the intra-op threadpool. e.g. Conv, Gemm, MatMul operators.

If your model contains compute-intensive nodes using operators that are not currently supported by the XNNPACK EP these will be handled by the CPU EP. In that case better performance may be achieved by increasing the size of the ORT intra-op threadpool and potentially re-enabling spinning. Performance testing is the best way to determine the optimal configuration for your model.
### Available Options
##### intra_op_num_threads

The number of threads to use for the XNNPACK EP's internal intra-op thread-pool. This is the number of threads used to parallelize the execution within a node. The default value is 1. The value should be >= 1.


## Supported ops
Following ops are supported by the XNNPACK Execution Provider,

|Operator|Note||
|--------|------|-----|
|ai.onnx:AveragePool|Only 2D Pool is supported.|
|ai.onnx:Conv|Only 2D Conv is supported.<br/>Weights and bias should be constant.|
|ai.onnx:ConvTranspose|Only 2D ConvTranspose is supported.<br/>Weights and bias should be constant.|since 1.14|
|ai.onnx:MaxPool|Only 2D Pool is supported.|
|ai.onnx:Softmax|all opset below 13 is supported, only support opset 13 when AXIS is the last dimension|
|ai.onnx:QLinearConv|Only 2D Conv is supported.<br/>Weights and bias should be constant.<br/>All quantization scales and zero points should be constant.|
|ai.onnx:Resize|2D/4D Resize in `Bilinear mode` are supported|since 1.14|
|ai.onnx:Gemm|Only 2D Op is supported|since 1.14|
|ai.onnx:Matmul|Only 2D Op is supported|since 1.14|
|com.microsoft:QLinearAveragePool|Only 2D Pool is supported.<br/>All quantization scales and zero points should be constant.|
|com.microsoft:QLinearSoftmax|All quantization scales and zero points should be constant.|
|com.microsoft:QLinearConvTranspose|All quantization scales and zero points should be constant.|since 1.14|
