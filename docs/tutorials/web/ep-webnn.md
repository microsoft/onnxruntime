---
title: Using WebNN
description: Using WebNN
parent: Web
grand_parent: Tutorials
has_children: false
nav_order: 4
---
{::options toc_levels="2..4" /}

# Using the WebNN Execution Provider
{: .no_toc }

This document explains how to use the WebNN execution provider in ONNX Runtime.

## Contents
{: .no_toc}

* TOC
{:toc}


## Basics

### What is WebNN? Should I use it?

[Web Neural Network (WebNN)](https://webnn.dev/) API is a new web standard that allows web apps and frameworks to accelerate deep neural networks with on-device hardware such as GPUs, CPUs, or purpose-built AI accelerators(NPUs).

WebNN is available in latest versions of Chrome and Edge on Windows, Linux, macOS, Android and ChromeOS behind a "*Enables WebNN API*" flag. Check [WebNN status](https://webmachinelearning.github.io/webnn-status/) for the latest implementation status.

Refer to the [WebNN operators](https://github.com/microsoft/onnxruntime/blob/main/js/web/docs/webnn-operators.md) for the most recent status of operator support in the WebNN execution provider. If the WebNN execution provider supports most of the operators in your model (with unsupported operators falling back to the WASM EP), and you wish to achieve power-efficient, faster processing and smoother performance by utilizing on-device accelerators, consider using the WebNN execution provider.

### How to use WebNN EP in ONNX Runtime Web

This section assumes you have already set up your web application with ONNX Runtime Web. If you haven't, you can follow the [Get Started](../../get-started/with-javascript/web.md) for some basic info.

To use WebNN EP, you just need to make 3 small changes:
  1. Update your import statement:

     - For HTML script tag, change `ort.min.js` to `ort.all.min.js`:
       ```html
       <script src="https://example.com/path/ort.all.min.js"></script>
       ```
     - For JavaScript import statement, change `onnxruntime-web` to `onnxruntime-web/all`:
       ```js
       import * as ort from 'onnxruntime-web/all';
       ```

     See [Conditional Importing](https://github.com/microsoft/onnxruntime-inference-examples/tree/main/js/importing_onnxruntime-web#conditional-importing) for details.

  2. Specify 'webnn' EP explicitly in session options:
     ```js
     const session = await ort.InferenceSession.create(modelPath, { ..., executionProviders: ['webnn'] });
     ```
     WebNN EP also offers a set of options for creating diverse types of WebNN MLContext.
     - `deviceType`: `'cpu'|'gpu'|'npu'`(default value is `'cpu'`), specifies the preferred type of device to be used for the MLContext.
     - `powerPreference`: `'default'|'low-power'|'high-performance'`(default value is `'default'`), specifies the preferred type of power consumption to be used for the MLContext.
     - `context`: type of `MLContext`, allows users to pass a pre-created `MLContext` to WebNN EP, it is required in IO binding feature. If this option is provided, the other options will be ignored.

     Example of using WebNN EP options:
     ```js
     const options = {
        executionProviders: [
          {
            name: 'webnn',
            deviceType: 'gpu',
            powerPreference: "default",
          },
        ],
     }
     ```
  3. If it is dynamic shape model, ONNX Runtime Web offers `freeDimensionOverrides` session option to override the free dimensions of the model. See [freeDimensionOverrides introduction](https://onnxruntime.ai/docs/tutorials/web/env-flags-and-session-options.html#freedimensionoverrides) for more details.

WebNN API and WebNN EP are in actively development, you might consider installing the latest nightly build version of ONNX Runtime Web (onnxruntime-web@dev) to benefit from the latest features and improvements.

## Keep tensor data on WebNN MLTensor (IO binding)

By default, a model's inputs and outputs are tensors that hold data in CPU memory. When you run a session with WebNN EP with 'gpu' or 'npu' device type, the data is copied to GPU or NPU memory, and the results are copied back to CPU memory. Memory copy between different devices as well as different sessions will bring much overhead to the inference time, WebNN provides a new opaque device-specific storage type [MLTensor](https://webmachinelearning.github.io/webnn/#api-mltensor) to address this issue.
If you get your input data from a MLTensor, or you want to keep the output data on MLTensor for further processing, you can use IO binding to keep the data on MLTensor. This will be especially helpful when running transformer based models, which usually runs a single model multiple times with previous output as the next input.

For model input, if your input data is a WebNN storage MLTensor, you can [create a MLTensor tensor and use it as input tensor](#create-input-tensor-from-a-mltensor).

For model output, there are 2 ways to use the IO binding feature:
- [Use pre-allocated MLTensor tensors](#use-pre-allocated-mltensor-tensors)
- [Specify the output data location](#specify-the-output-data-location)

Please also check the following topic:
- [MLTensor tensor life cycle management](#mltensor-tensor-life-cycle-management)

**Note:** The MLTensor necessitates a shared MLContext for IO binding. This implies that the MLContext should be pre-created as a WebNN EP option and utilized across all sessions.

### Create input tensor from a MLTensor

If your input data is a WebNN storage MLTensor, you can create a MLTensor tensor and use it as input tensor:

```js
// Create WebNN MLContext
const mlContext = await navigator.ml.createContext({deviceType, ...});
// Create a WebNN MLTensor
const inputMLTensor = await mlContext.createTensor({
  dataType: 'float32',
  shape: [1, 3, 224, 224],
  writable: true,
});
// Write data to the MLTensor
const inputArrayBuffer = new Float32Array(1 * 3 * 224 * 224).fill(1.0);
mlContext.writeTensor(inputMLTensor, inputArrayBuffer);

// Create an ORT tensor from the MLTensor
const inputTensor = ort.Tensor.fromMLTensor(inputMLTensor, {
  dataType: 'float32',
  dims: [1, 3, 224, 224],
});

```

Use this tensor as model inputs(feeds) so that the input data will be kept on MLTensor.

### Use pre-allocated MLTensor tensors

If you know the output shape in advance, you can create a MLTensor tensor and use it as output tensor:

```js

// Create a pre-allocated MLTensor and the corresponding ORT tensor. Assuming that the output shape is [10, 1000].
const mlContext = await navigator.ml.createContext({deviceType, ...});
const preallocatedMLTensor = await mlContext.createTensor({
  dataType: 'float32',
  shape: [10, 1000],
  readable: true,
});

const preallocatedOutputTensor = ort.Tensor.fromMLTensor(preallocatedMLTensor, {
  dataType: 'float32',
  dims: [10, 1000],
});

// ...

// Run the session with fetches
const feeds = { 'input_0': inputTensor };
const fetches = { 'output_0': preallocatedOutputTensor };
await session.run(feeds, fetches);

// Read output_0 data from preallocatedMLTensor if need
const output_0 = await mlContext.readTensor(preallocatedMLTensor);
console.log('output_0 value:', new Float32Array(output_0));
```

By specifying the output tensor in the fetches, ONNX Runtime Web will use the pre-allocated MLTensor as the output tensor. If there is a shape mismatch, the `run()` call will fail.

### Specify the output data location

If you don't want to use pre-allocated MLTensor tensors for outputs, you can also specify the output data location in the session options:

```js
const sessionOptions1 = {
  ...,
  // keep all output data on MLTensor
  preferredOutputLocation: 'ml-tensor'
};

const sessionOptions2 = {
  ...,
  // alternatively, you can specify the output location for each output tensor
  preferredOutputLocation: {
    'output_0': 'cpu',         // keep output_0 on CPU. This is the default behavior.
    'output_1': 'ml-tensor'   // keep output_1 on MLTensor tensor
  }
};

// ...

// Run the session
const feeds = { 'input_0': inputTensor };
const results = await session.run(feeds);

// Read output_1 data
const output_1 = await results['output_1'].getData();
console.log('output_1 value:', new Float32Array(output_1));
```

By specifying the config `preferredOutputLocation`, ONNX Runtime Web will keep the output data on the specified device.

See [API reference: preferredOutputLocation](https://onnxruntime.ai/docs/api/js/interfaces/InferenceSession.SessionOptions.html#preferredOutputLocation) for more details.

## Notes

### MLTensor tensor life cycle management

It is important to understand how the underlying MLTensor is managed so that you can avoid memory leaks and improve tensor usage efficiency.

A MLTensor tensor is created either by user code or by ONNX Runtime Web as model's output.
- When it is created by user code, it is always created with an existing MLTensor using `Tensor.fromMLTensor()`. In this case, the tensor does not "own" the MLTensor.

  - It is user's responsibility to make sure the underlying MLTensor is valid during the inference, and call `mlTensor.destroy()` to dispose the MLTensor when it is no longer needed.
  - Avoid calling `tensor.getData()` and `tensor.dispose()`. Use the MLTensor tensor directly.
  - Using a MLTensor tensor with a destroyed MLTensor will cause the session run to fail.
- When it is created by ONNX Runtime Web as model's output (not a pre-allocated MLTensor tensor), the tensor "owns" the MLTensor.

  - You don't need to worry about the case that the MLTensor is destroyed before the tensor is used.
  - Call `tensor.getData()` to download the data from the MLTensor to CPU and get the data as a typed array.
  - Call `tensor.dispose()` explicitly to destroy the underlying MLTensor when it is no longer needed.
