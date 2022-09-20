---
title: Xnnpack (Android)
description: Instructions to execute ONNX Runtime with the Xnnpack execution provider
parent: Execution Providers
nav_order: 7
redirect_from: /docs/reference/execution-providers/Xnnpack-ExecutionProvider
---
{::options toc_levels="2" /}

# Xnnpack Execution Provider

Accelerate ONNX models on Android devices and WebAssembly with ONNX Runtime and the Xnnpack execution provider. [(Xnnpack)](https://github.com/google/XNNPACK) is a highly optimized library of floating-point neural network inference operators for ARM, WebAssembly, and x86 platforms.

## Contents
{: .no_toc }

* TOC placeholder
{:toc}

## Requirements

No

## Install
Pre-built packages of ONNX Runtime with Xnnpack EP for Android are published on Maven.

See [here](../install/index.md#install-on-android) for installation instructions.

## Build

Please see the [Build Android EP](../build/eps.md#Xnnpack) for instructions on building a package that includes the Xnnpack EP.

## Usage

The ONNX Runtime API details are [here](../api).

The Xnnpack EP can be used via the C, C++ or Java APIs

The Xnnpack EP must be explicitly registered when creating the inference session. For example:

```C++
Ort::Env env = Ort::Env{ORT_LOGGING_LEVEL_ERROR, "Default"};
Ort::SessionOptions so;
uint32_t Xnnpack_flags = 0;
so.AppendExecutionProvider("XNNPACK", {{"intra_op_num_threads", 4/*how many threads setup for xnnpack*/}});
Ort::Session session(env, model_path, so);
```

## Configuration Options

To achieve the best performance, the Xnnpack EP requires the following configuration options:
1. Disable thread-pool Spinning by adding the following to the session options:
```C++
    so.AddConfigEntry(kOrtSessionOptionsConfigAllowIntraOpSpinning, "0");
```
2. Xnnpack EP takes the intra-threadpool size from provider-options. The default value is 1. 
the intra-threadpool size should be set to the half of number of cores on the device. For example:
```C++
    so.AppendExecutionProvider("XNNPACK", {{"intra_op_num_threads", intra_op_num_threads});
    // intra_op_num_threads is the key of the provider option, and 4 is the value of the provider option.
```
3. Try to set ort thread-pool intra_op_num_threads as 1 or equal to Xnnpack thread-pool size, and pick the best value for your model. Generally, 1 would be the best fit if your model run most of calculation heavy ops on Xnnpack EP or same as Xnnpack thread-pool size in contrast.:
```C++
    so.SetIntraOpNumThreads(intra_op_num_threads/*1 or same size as Xnnpack thread-pool*/);

```

### Available Options
##### intra_op_num_threads

The thread-pool size for Xnnpack EP. The default value is 1. Xnnpack Ep use [pthreadpool](https://github.com/Maratyszcza/pthreadpool) for parallelization. However, CPU use ORT thread-pool for parallelization. ORT thread-pool will spinning threads by default for which can improve the performance. But in that case, threads will not release CPU resources when they are idle and it will lead to serious contention between the two thread-pool, which will hurt performance dramatically and even produce more power consumption. So, it is recommended to disable thread-pool Spinning by adding the following to the session options.

To alleviate the contention further, it is recommended to set ort thread-pool intra_op_num_threads as 1 so ORT thread-pool wouldn't be created. The trade-off is that all ops are assigned to CPU EP will running on single thread.


## Supported ops
Following ops are supported by the Xnnpack Execution Provider,

|Operator|Note|
|--------|------|
|ai.onnx:AveragePool|Only 2D Pool is supported.|
|ai.onnx:Conv|Only 2D Conv is supported.<br/>Weights and bias should be constant.|
|ai.onnx:MaxPool|Only 2D Pool is supported.|
|ai.onnx:QLinearConv|Only 2D Conv is supported.<br/>Weights and bias should be constant.<br/>All quantization scales and zero points should be constant.|
|ai.onnx:Softmax||
|com.microsoft:QLinearAveragePool|Only 2D Pool is supported.<br/>All quantization scales and zero points should be constant.|
|com.microsoft:QLinearSoftmax|All quantization scales and zero points should be constant.|
