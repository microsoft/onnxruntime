---
title: The 'env' Flags and Session Options
description: The 'env' Flags and Session Options
parent: Web
grand_parent: Tutorials
has_children: false
nav_order: 2
---
{::options toc_levels="2..4" /}

# The 'env' Flags and Session Options
{: .no_toc }

This document explains how to configure ONNX Runtime Web, using the following methods:

- [The 'env' flags](#the-environment-flags-env)
- [Session options](#session-options)

The biggest difference between the two is that the 'env' flags are global settings that affect the entire ONNX Runtime Web environment, while session options are settings that are specific to a single inference session.

## Contents
{: .no_toc}

* TOC
{:toc}


## The environment flags ('env')

### Summary

The environment flags are a set of global flags that can be used to configure the behavior of ONNX Runtime Web. They are accessible via the `ort.env` object:

```js
import * as ort from 'onnxruntime-web';

// get the 'env' object
const env = ort.env;
```

These flags are usually required to be set before any inference session is created.


For more information, see [API reference: Interface Env](https://onnxruntime.ai/docs/api/js/interfaces/Env-1.html).

### `env.debug`

The `env.debug` flag is used to enable/disable the debug mode. When enabled, ONNX Runtime Web will do extra checks and logging to help diagnose issues. It is disabled by default.

```js
// enable the debug mode
ort.env.debug = true;
```

For more information, see [API reference: env.debug](https://onnxruntime.ai/docs/api/js/interfaces/Env-1.html#debug).

### `env.logLevel`

The `env.logLevel` flag is used to set the log level. It can be set to one of `"error" | "verbose" | "info" | "warning" | "fatal"`. The default value is `"warning"`.

```js
// set the log level to 'verbose'
ort.env.logLevel = 'verbose';
```

For more information, see [API reference: env.logLevel](https://onnxruntime.ai/docs/api/js/interfaces/Env-1.html#logLevel).

### `env.wasm`

The `env.wasm` object contains flags that are used to configure the behavior of the WebAssembly instance.

For more information, see [API reference: Interface WebAssemblyFlags](https://onnxruntime.ai/docs/api/js/interfaces/Env.WebAssemblyFlags.html).

#### `env.wasm.numThreads`

The `env.wasm.numThreads` flag is used to set the number of threads that ONNX Runtime Web will use for model inference. This value includes the main thread.

The default value is `0`, which means it will be determined by ONNX Runtime Web based on the environment. In browsers, it will be set to half of `navigator.hardwareConcurrency` or `4`, whichever is smaller.

Setting it to `1` will force disable multi-threading. Otherwize, ONNX Runtime Web will perform a check for whether the environment supports multi-threading. Only when the browser supports WebAssembly multi-threading and `crossOriginIsolated` mode is enabled, multi-threading will be enabled. See [Cross Origin Isolation Guide](https://web.dev/cross-origin-isolation-guide/) for more info.

When multi-threading is enabled, ONNX Runtime Web will load the multi-threaded WebAssembly binary file. The corresponding file name will include `-threaded`.

```js
// Disable multi-threading
ort.env.wasm.numThreads = 1;
```

For more information, see [API reference: env.wasm.numThreads](https://onnxruntime.ai/docs/api/js/interfaces/Env.WebAssemblyFlags.html#numThreads).

#### `env.wasm.simd`

The `env.wasm.simd` flag is used to enable/disable the SIMD (Single Instruction, Multiple Data) feature. It is enabled by default.

When SIMD is enabled, ONNX Runtime Web will perform a check for whether the environment supports SIMD. If the environment supports SIMD, ONNX Runtime Web will load the SIMD WebAssembly binary file. The corresponding file name will include `-simd`.

It is not recommended to set this flag to `false` unless you are sure that the environment does not support SIMD.

For more information, see [API reference: env.wasm.simd](https://onnxruntime.ai/docs/api/js/interfaces/Env.WebAssemblyFlags.html#simd).

#### `env.wasm.proxy`

The `env.wasm.proxy` flag is used to enable/disable the proxy worker feature. It is disabled by default.

When the proxy worker is enabled, ONNX Runtime Web will offload the heavy computation to a separate Web Worker. Using the proxy worker can improve the responsiveness of the UI to improve the user experience, because the computation will not block the main thread.

```js
// Enable proxy worker
ort.env.wasm.proxy = true;
```

However, there are some limitations when using the proxy worker:
- The proxy worker cannot work with WebGPU EP. This is because a GPU buffer is not transferable. If you want to use WebGPU EP in a Web Worker, you can use `importScripts()` to import the ONNX Runtime Web library in the Web Worker.
- The proxy worker cannot work in a Content Security Policy (CSP) restricted environment. This is because the proxy worker uses `Blob` to create a Web Worker, and the CSP may block the creation of the Web Worker.

For more information, see [API reference: env.wasm.proxy](https://onnxruntime.ai/docs/api/js/interfaces/Env.WebAssemblyFlags.html#proxy).

#### `env.wasm.wasmPaths`

The `env.wasm.wasmPaths` flag is used to override the WebAssembly binary file path. It can be used in 2 ways:

- Set `env.wasm.wasmPaths` to a string as a path prefix.
  ```js
  // Set the WebAssembly binary file path to jsdelivr CDN for a specific release version
  ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.17.3/dist/';
  ```

- Set `env.wasm.wasmPaths` to an object with keys as the WebAssembly binary file name and values as the path to the WebAssembly binary file.
  ```js
  // Set separate WebAssembly binary file paths
  ort.env.wasm.wasmPaths = {
    'ort-wasm-simd.jsep.wasm': 'https://example.com/path/to/ort-wasm-simd.jsep.wasm'
    'ort-wasm-simd-threaded.jsep.wasm': 'https://example.com/path/to/ort-wasm-simd-threaded.jsep.wasm',
  };
  ```

This flag is useful when the WebAssembly binary file(s) are not located in the same directory as the JavaScript code bundle. It is also useful when you want to use a public CDN to serve the WebAssembly binary file(s).

NOTE: Please make sure the the JavaScript code bundle and the WebAssembly binary file(s) are from the same build. Otherwise, ONNX Runtime Web will fail to initialize due to a mismatch of the minimized function names between the JavaScript code bundle and the WebAssembly binary file(s). This means you cannot use this feature to load the WebAssembly binary file(s) from a different version.

For more information, see [API reference: env.wasm.wasmPaths](https://onnxruntime.ai/docs/api/js/interfaces/Env.WebAssemblyFlags.html#wasmPaths).

### `env.webgpu`

The `env.webgpu` object contains flags that are used to configure the behavior of the WebGPU EP.

For more information, see [API reference: Interface WebGpuFlags](https://onnxruntime.ai/docs/api/js/interfaces/Env.WebGpuFlags.html).

#### `env.webgpu.device` and `env.webgpu.adapter`

These 2 flags are used to get the WebGPU device and adapter after a WebGPU inference session is created.

The `env.webgpu.adapter` flag can also be used to set the adapter that will be used by the WebGPU EP before the first WebGPU inference session is created. It is useful when you want to use a specific adapter.

For more information, see [API reference: env.webgpu.device](https://onnxruntime.ai/docs/api/js/interfaces/Env.WebGpuFlags.html#device) and [API reference: env.webgpu.adapter](https://onnxruntime.ai/docs/api/js/interfaces/Env.WebGpuFlags.html#adapter).

#### `env.webgpu.powerPreference` and `env.webgpu.forceFallbackAdapter`

These 2 flags are used to set the power preference and force fallback adapter for the WebGPU EP. They will be used when the WebGPU EP is initialized without any pre-configured adapter is set via `env.webgpu.adapter`.

For more information, see [API reference: env.webgpu.powerPreference](https://onnxruntime.ai/docs/api/js/interfaces/Env.WebGpuFlags.html#powerPreference) and [API reference: env.webgpu.forceFallbackAdapter](https://onnxruntime.ai/docs/api/js/interfaces/Env.WebGpuFlags.html#forceFallbackAdapter).

#### `env.webgpu.profiling`

The `env.webgpu.profiling` flag is used to enable WebGPU profiling.

Please see [WebGPU Profiling](./performance-diagnosis.md#webgpu-profiling) for more details.

For more information, see [API reference: env.webgpu.profiling](https://onnxruntime.ai/docs/api/js/interfaces/Env.WebGpuFlags.html#profiling).

## Session options

### Summary

Session options are used to configure the behavior of a single inference session. They are passed to the `InferenceSession.create()` method.

For more information, see [API reference: Interface InferenceSession.SessionOptions](https://onnxruntime.ai/docs/api/js/interfaces/InferenceSession.SessionOptions.html).

### `executionProviders`

The `executionProviders` option is used to specify a list of execution providers that will be used by the inference session.

The following execution providers are available in ONNX Runtime Web:

- `'wasm'`: The default CPU execution provider.
- `'webgpu'`: The WebGPU execution provider. See [WebGPU EP](./ep-webgpu.md) for more details.
- `'webnn'`: The WebNN execution provider.
- `'webgl'`: The WebGL execution provider.

```js
const mySession = await ort.InferenceSession.create(modelUrl, {
    ...,
    // specify the EP list
    executionProviders: ['webgpu', 'wasm']
});
```

For more information, see [API reference: executionProviders](https://onnxruntime.ai/docs/api/js/interfaces/InferenceSession.SessionOptions.html#executionProviders).

### `externalData`

The `externalData` option is used to pass the external data information to ONNX Runtime Web. When a model's weights are stored in external data files, you need to pass the external data information to ONNX Runtime Web. See [External Data](./large-models.md#external-data) for more details.

For more information, see [API reference: externalData](https://onnxruntime.ai/docs/api/js/interfaces/InferenceSession.SessionOptions.html#externalData).

### `freeDimensionOverrides`

The `freeDimensionOverrides` option is used to override the free dimensions of the model.

ONNX models may have some dimensions as free dimensions, which means that the model can accept inputs of any size in that dimension. For example, an image model may define its input shape as `[batch, 3, height, width]`, which means that the model can accept any numbers of images of any size, as long as the number of channels is 3. However, if your application always uses images of a specific size, you can override the free dimensions to a specific size, which can be helpful to optimize the performance of the model. For example, if your web app always use a single image of 224x224, you can override the free dimensions to `[1, 3, 224, 224]` by specifying the following config in your session options:

```js
const mySessionOptions = {
  ...,
  freeDimensionOverrides: {
    batch: 1,
    height: 224,
    width: 224
  }
};
```

For more information, see [API reference: freeDimensionOverrides](https://onnxruntime.ai/docs/api/js/interfaces/InferenceSession.SessionOptions.html#freeDimensionOverrides).

### `enableGraphCapture`

The `enableGraphCapture` option is used to enable graph capture feature. Currently, this feature is only available for WebGPU EP.

If ONNX Runtime determines that a model has static shapes, and all its computing kernels are running on the registered EP, it can capture the kernel executions in the first run and replay them in the following runs. This can lead to better performance when CPU sometimes is the bottleneck to prepare for the commands.

```js
const mySessionOptions = {
  ...,
  enableGraphCapture: true
};
```

Not all models are suitable for graph capture. Some models with dynamic input shapes can use this feature together with [free dimension override](#freedimensionoverrides). Some models just don't work with this feature. You can try it out and see if it works for your model. If it doesn't work, the model initialization will fail, and you can disable this feature for this model.

See [API reference: enableGraphCapture](https://onnxruntime.ai/docs/api/js/interfaces/InferenceSession.SessionOptions.html#enablegraphcapture) for more details.


### `optimizedModelFilePath`

The `optimizedModelFilePath` option is used to specify the file path of the optimized model. In browsers, the value of this option is ignored. Instead, the a new tab is opened with the content of the optimized model as a blob, allowing the user to download and save the optimized model.

```js
const mySessionOptions = {
  ...,
  // specify this option to allow downloading the optimized model
  optimizedModelFilePath: 'optimized_model.onnx'
};
```

NOTE: This feature is not available by default. It requires to rebuild ONNX Runtime Web with the `--cmake_extra_defines onnxruntime_ENABLE_WEBASSEMBLY_OUTPUT_OPTIMIZED_MODEL=1` command line option.

For more information, see [API reference: optimizedModelFilePath](https://onnxruntime.ai/docs/api/js/interfaces/InferenceSession.SessionOptions.html#optimizedModelFilePath).

### `preferredOutputLocation`

The `preferredOutputLocation` option is used to specify the preferred location of the output data. It can be used to keep the output data on GPU for further processing. See [Keep tensor data on GPU (IO binding)](./ep-webgpu.md#keep-tensor-data-on-gpu-io-binding) for more details.

For more information, see [API reference: preferredOutputLocation](https://onnxruntime.ai/docs/api/js/interfaces/InferenceSession.SessionOptions.html#preferredOutputLocation).
