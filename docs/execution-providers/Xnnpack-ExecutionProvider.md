---
title: XNNPACK
description: Instructions to execute ONNX Runtime with the XNNPACK execution provider
parent: Execution Providers
nav_order: 16
redirect_from: /docs/reference/execution-providers/Xnnpack-ExecutionProvider
---
{::options toc_levels="2" /}

# XNNPACK Execution Provider

Accelerate ONNX models on Android devices and WebAssembly with ONNX Runtime and the XNNPACK execution provider. [(XNNPACK)](https://github.com/google/XNNPACK) is a highly optimized library of floating-point neural network inference operators for ARM, WebAssembly, and x86 platforms.

## Contents
{: .no_toc }

* TOC placeholder
{:toc}

## Requirements


## Install
No Pre-built packages of ONNX Runtime with XNNPACK EP for Android are available.


## Build

Please see the [Build page](../build/eps.md#xnnpack) for instructions on building a package that includes the XNNPACK EP.

Support build for Android/Windows/Linux

## Usage

The ONNX Runtime API details are [here](../api).

The XNNPACK EP can be used via the C, C++ or Java APIs

The XNNPACK EP must be explicitly registered when creating the inference session. For example:

```C++
Ort::Env env = Ort::Env{ORT_LOGGING_LEVEL_ERROR, "Default"};
Ort::SessionOptions so;
so.AppendExecutionProvider("XNNPACK", {{"intra_op_num_threads", std::to_string(intra_op_num_threads)}});
Ort::Session session(env, model_path, so);
```

## Configuration Options

To achieve the best performance, the XNNPACK EP requires the following configuration options:
1. Disable thread-pool Spinning by adding the following to the session options:
```C++
    so.AddConfigEntry(kOrtSessionOptionsConfigAllowIntraOpSpinning, "0");
```
2. XNNPACK EP takes the intra-threadpool size from provider-options. The default value is 1. For example:
```C++
    so.AppendExecutionProvider("XNNPACK", {{"intra_op_num_threads", std::to_string(intra_op_num_threads)}});
```
3. Try to set ORT thread-pool intra_op_num_threads as 1 or equal to XNNPACK thread-pool size, and pick the best value for your model. Generally, 1 would be the best fit if your model run most of computation heavy ops on XNNPACK EP or same as XNNPACK thread-pool size in contrast.:
```C++
    so.SetIntraOpNumThreads(intra_op_num_threads/*1 or same size as XNNPACK thread-pool*/);
```

### Available Options
##### intra_op_num_threads

The thread-pool size (default 1) for XNNPACK EP. XNNPACK Ep use [pthreadpool](https://github.com/Maratyszcza/pthreadpool) for parallelization implementation. Thus, there would be two threadpools inside. However, ORT thread-pool will spinning threads by default as which can improve the performance. But in that case, threads will not release CPU resources when ops finished, switched to XNNPACK EP and it will lead to serious contention between the two thread-pools, which will hurt performance dramatically and even produce more power consumption. So, it is recommended to disable thread-pool Spinning.

To alleviate the contention further, it is recommended to set ort thread-pool intra_op_num_threads as 1 so ORT thread-pool wouldn't be created. The trade-off is that all ops are assigned to CPU EP will be running on single thread.


## Supported ops
Following ops are supported by the XNNPACK Execution Provider,

|Operator|Note|
|--------|------|
|ai.onnx:AveragePool|Only 2D Pool is supported.|
|ai.onnx:Conv|Only 2D Conv is supported.<br/>Weights and bias should be constant.|
|ai.onnx:MaxPool|Only 2D Pool is supported.|
|ai.onnx:Softmax|all opset below 13 is supported, only support opset 13 when AXIS is the last dimension|
|ai.onnx:QLinearConv|Only 2D Conv is supported.<br/>Weights and bias should be constant.<br/>All quantization scales and zero points should be constant.|
|com.microsoft:QLinearAveragePool|Only 2D Pool is supported.<br/>All quantization scales and zero points should be constant.|
|com.microsoft:QLinearSoftmax|All quantization scales and zero points should be constant.|
