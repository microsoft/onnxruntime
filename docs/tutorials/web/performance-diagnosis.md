---
title: Performance Diagnosis
description: Performance Diagnosis
parent: Web
grand_parent: Tutorials
has_children: false
nav_order: 4
---
{::options toc_levels="2..4" /}

# Performance Diagnosis
{: .no_toc }

ONNX Runtime Web is designed to be fast and efficient, but there are a number of factors that can affect the performance of your application. This document provides some guidance on how to diagnose performance issues in ONNX Runtime Web.

Before you start, make sure that ONNX Runtime Web successfully loads and runs your model. If you encounter any issues, see the [troubleshooting guide](./trouble-shooting.md) for help first.


## Contents
{: .no_toc}

* TOC
{:toc}


## General performance tips

Here are some general tips to improve the performance of your application:

### Use the right model

Choose a model that is appropriate for web scenario. A model that is too large or too complex may not run efficiently on less powerful hardware. Usually, the "tiny" or "small" versions of models are more commonly used in web applications. This does not mean that you cannot use larger models, but you should be aware of the potential hit on user experience due to longer load times and slower inference.

### Use the right execution provider

Choose the right execution provider for your scenario.

  - **WebAssembly (wasm)**: This is the default CPU execution provider for ONNX Runtime Web. Use it for very small models or environments where GPU is not available.
    
  - **WebGPU (webgpu)**: This is the default GPU execution provider. Use it when the device has a decent GPU which supports WebGPU.

  - **WebNN (webnn)**: This is the option which offers potential near-native performance on the web. It is currently not supported by default in browsers, but you can enable WebNN feature manually in browser's settings.
    
  - **WebGL (webgl)**: This execution provider is designed to run models using GPU on older devices that do not support WebGPU.

### Use the diagnostic features

Use the [diagnostic features](#diagnostic-features) to get detailed information about the execution of the model. This can be helpful to understand the performance characteristics of the model and to identify potential problems or bottlenecks.

## CPU tips

If you are using the WebAssembly (wasm) execution provider, you can use the following tips to improve the performance of your application:

### Enable multi-threading

Always enable multi-threading if the environment supports it. Multi-threading can significantly improve the performance of your application by utilizing multiple CPU cores.

This feature is enabled by default in ONNX Runtime Web, however it only works when `crossOriginIsolated` mode is enabled. See [https://web.dev/cross-origin-isolation-guide/](https://web.dev/cross-origin-isolation-guide/) for more info.

You can also use flag `ort.env.wasm.numThreads` to set the number of threads to be used.

```js
// Set the number of threads to 4
ort.env.wasm.numThreads = 4;

// Disable multi-threading
ort.env.wasm.numThreads = 1;

// Let ONNX Runtime Web decide the number of threads to use
ort.env.wasm.numThreads = 0;
```

See [API reference: env.wasm.numThreads](https://onnxruntime.ai/docs/api/js/interfaces/Env.WebAssemblyFlags.html#numThreads) for more details.

### Enable SIMD

Always enable SIMD if it's supported. SIMD (Single Instruction, Multiple Data) is a set of instructions that perform the same operation on multiple data points simultaneously. This can significantly improve the performance of your application.

This feature is enabled by default in ONNX Runtime Web, unless you explicitly disable it by setting `ort.env.wasm.simd = false`.

See [API reference: env.wasm.simd](https://onnxruntime.ai/docs/api/js/interfaces/Env.WebAssemblyFlags.html#simd) for more details.

### Prefer uint8 quantized models

If you are using a quantized model, prefer uint8 quantized models. Avoid float16 models if possible, as float16 is not natively supported by CPU and it is going to be slow.

### Enable Proxy Worker

Proxy worker is a feature that allows ONNX Runtime Web to offload the heavy computation to a separate Web Worker. Using the proxy worker cannot improve the performance of the model, but it can improve the responsiveness of the UI to improve the user experience.

If you didn't import ONNX Runtime Web in a Web Worker, and the model takes a while to inference, it is recommended to enable the proxy worker.

```js
// Enable proxy worker
ort.env.wasm.proxy = true;
```

See [API reference: env.wasm.proxy](https://onnxruntime.ai/docs/api/js/interfaces/Env.WebAssemblyFlags.html#proxy) for more details.

## WebGPU tips

If you are using the WebGPU execution provider, you can use the following tips to improve the performance of your application:

### Use capture and replay if possible

See [Capture and replay](ep-webgpu.md#capture-and-replay) for feature introduction.

It is always recommended to enable capture and replay feature, unless you need to feed input data with dynamic shape (eg. transformer based decoder model). Even with static shape input, this feature does not always work for all models. You can try it out and see if it works for your model. If it doesn't work, the model initialization will fail, and you can disable this feature for this model.

### Try using free dimension override

See [Free dimension override](ep-webgpu.md#free-dimension-override) for feature introduction.

Using free dimension override does not necessarily improve the performance. It's quite model by model. You can try it out and see if it works for your model. If you see performance degradation or larger memory usage, you may disable this feature.

### Try keep tensor data on GPU

See [Keep tensor data on GPU (IO binding)](ep-webgpu.md#keep-tensor-data-on-gpu-io-binding) for feature introduction.

Keeping tensor data on GPU can avoid unnecessary data transfer between CPU and GPU, which can improve the performance. Try to find out the best way to use this feature for your model.

Please be careful of the [GPU tensor life cycle management](ep-webgpu.md#gpu-tensor-life-cycle-management) when using this feature.

## Diagnostic features

### Profiling

You can enable profiling to get detailed information about the execution of the model. This can be helpful to understand the performance characteristics of the model and to identify potential bottlenecks.

To enable CPU profiling:
- step.1: Specify the `enableProfiling` option in the session options:
  ```js
  const mySessionOptions = {
    ...,
    enableProfiling: true
  };
  ```

  By specifying this option, ONNX Runtime Web will collect CPU profiling data for each run.

- step.2: Get the profiling data after the inference:
  ```js
  mySession.endProfiling();
  ```

  After calling `endProfiling()`, the profiling data will be outputted to the console.

  See [In Code Performance Profiling](../../performance/tune-performance/profiling-tools.md#in-code-performance-profiling) for how to use the profiling data.

To enable WebGPU profiling:

 - set `ort.env.webgpu.profiling = { mode: 'default' }` to enable WebGPU profiling. GPU Profiling data will be outputted to the console with prefix `[profiling]`.
 - alternatively, you can set `ort.env.webgpu.profiling` with a function to handle the profiling data:
   ```js
    ort.env.webgpu.profiling = {
        mode: 'default',
        ondata: (data) => {
            // handle the profiling data
        }
    };
   ```
   See [API reference: env.webgpu.profiling](https://onnxruntime.ai/docs/api/js/interfaces/Env.WebGpuFlags.html#profiling) for more details.

### Trace

You can enable trace by specifying the following flag:
```js
ort.env.trace = true;
```

This feature uses `console.timeStamp` to log the trace data. You can use the browser's performance tool to analyze the trace data.

See [API reference: env.trace](https://onnxruntime.ai/docs/api/js/interfaces/Env-1.html#trace) for more details.

### Log level 'verbose'

You can set the log level to 'verbose' to get more detailed logs:
```js
ort.env.logLevel = 'verbose';
```

See [API reference: env.logLevel](https://onnxruntime.ai/docs/api/js/interfaces/Env-1.html#logLevel) for more details.

### Enable debug mode

You can enable the debug mode by specifying the following flag:
```js
ort.env.debug = true;
```

In debug mode, ONNX Runtime Web will log detailed information about the execution of the model, and also apply some additional checks. Usually you need to use `verbose` log level to see the debug logs.

See [API reference: env.debug](https://onnxruntime.ai/docs/api/js/interfaces/Env-1.html#debug) for more details.

## Analyze the profiling data

This part is under construction.