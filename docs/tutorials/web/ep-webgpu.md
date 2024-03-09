---
title: Using WebGPU
description: Using WebGPU
parent: Web
grand_parent: Tutorials
has_children: false
nav_order: 2
---

# Using WebGPU Execution Provider
{: .no_toc }

This document explains how to use the WebGPU execution provider in ONNX Runtime.

## Contents
{: .no_toc}

* TOC
{:toc}


## Basics

### What is WebGPU? Should I use it?

WebGPU is a new web standard for general purpose GPU compute and graphics. It is designed to be a low-level API, similar to Vulkan and Metal, and is designed to be used in the browser. It is designed to be more efficient and performant than WebGL, and is designed to be used for machine learning, graphics, and other compute tasks.

WebGPU is available out-of-box in latest versions of Chrome and Edge on Windows, macOS and Android. It is also available in Firefox under a flag and Safari Technology Preview. Check [WebGPU status](https://webgpu.io/status/) for the latest information.

If you are using ONNX Runtime Web for inferencing very lightweight models in you web application, and you want to have a small binary size, you can keep using the default WebAssembly (WASM) execution provider. If you want to run more complex models, or you want to take advantage of the GPU in the client's device, you can use the WebGPU execution provider.

### How to use WebGPU EP in ONNX Runtime Web

This section assumes you have already set up your web application with ONNX Runtime Web. If you haven't, you can follow the [Get Started](../../get-started/with-javascript/web.md) for some basic info.

To use WebGPU EP, you just need to make 2 small changes:
  1. Update your import statement:

     - For HTML script tag, change `ort.min.js` to `ort.webgpu.min.js`:
       ```html
       <script src="https://example.com/path/ort.webgpu.min.js"></script>
       ```
     - For JavaScript import statement, change `onnxruntime-web` to `onnxruntime-web/webgpu`:
       ```js
       import * as ort from 'onnxruntime-web/webgpu';
       ```

     See [Conditional Importing](https://github.com/microsoft/onnxruntime-inference-examples/tree/main/js/importing_onnxruntime-web#conditional-importing) for details.

  2. Specify 'webgpu' EP explicitly in session options:
     ```js
     const session = await ort.InferenceSession.create(modelPath, { ..., executionProviders: ['webgpu'] });
     ```

It is also recommended to install the latest nightly build version of ONNX Runtime Web (onnxruntime-web@dev) to get the latest features and bug fixes.

## WebGPU EP features

ONNX Runtime Web offers the following features which may be helpful to use with WebGPU EP:

### Free dimension override

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

Because WebGPU is shader based, if the engine knows the input shape in advance, it can do extra optimizations, which can lead to better performance.

See [API reference: freeDimensionOverrides](https://onnxruntime.ai/docs/api/js/interfaces/InferenceSession.SessionOptions.html#freeDimensionOverrides) for more details.

### Capture and replay

If ONNX Runtime determines that a model have static shapes, and all its computing kernels are running on WebGPU EP, it can capture the kernel execution and replay it in the next run. This can lead to better performance, especially for relatively lightweighted models.

```js
const mySessionOptions = {
  ...,
  enableGraphCapture: true
};
```

Not all models are suitable for graph capture and replay. Some models with dynamic input shapes can use this feature together with free dimension override. Some models just don't work with this feature. You can try it out and see if it works for your model. If it doesn't work, the model initialization will fail, and you can disable this feature for this model.

See [API reference: enableGraphCapture](https://onnxruntime.ai/docs/api/js/interfaces/InferenceSession.SessionOptions.html#enableGraphCapture) for more details.

### Keep tensor data on GPU (IO binding)

By default, a model's inputs and outputs are tensors that hold data in CPU memory. When you run a session with WebGPU EP, the data is copied to GPU memory, and the results are copied back to CPU memory. If you get your input data from a GPU-based source, or you want to keep the output data on GPU for further processing, you can use IO binding to keep the data on GPU. This will be specially helpful when running transformer based models, which usually run a single model multiple times with previous output as the next input.

There are 2 ways to use the IO binding feature:
- Use pre-allocated GPU tensors
- Specify the output data location

#### Use pre-allocated GPU tensors

#### Specify the output data location