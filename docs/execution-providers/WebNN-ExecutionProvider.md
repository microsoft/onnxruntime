---
title: WebNN
description: Execute ONNX models with the WebNN execution provider in ONNX Runtime Web
parent: Execution Providers
nav_order: 14
redirect_from: /docs/reference/execution-providers/WebNN-ExecutionProvider
---

# WebNN Execution Provider
{: .no_toc }

The WebNN Execution Provider enables ONNX Runtime Web to run model operators through the [Web Neural Network API](https://www.w3.org/TR/webnn/). WebNN lets the browser select on-device CPUs, GPUs, or neural processing units (NPUs), providing a portable path to hardware-accelerated inference without a vendor-specific JavaScript API.

Unsupported model operators fall back to the WebAssembly execution provider. For best performance, check the [current WebNN operator support](https://github.com/microsoft/onnxruntime/blob/main/js/web/docs/webnn-operators.md) before choosing this execution provider.

{: .note }
> For a complete browser tutorial, including MLTensor I/O binding and tensor lifetime management, see [Using the WebNN Execution Provider](../tutorials/web/ep-webnn.md).

## Contents
{: .no_toc }

* TOC placeholder
{:toc}

## Requirements

WebNN support depends on the browser, operating system, and available hardware. Consult the [WebNN implementation status](https://webmachinelearning.github.io/webnn-status/) for the current browser and platform matrix.

WebNN is an experimental ONNX Runtime Web feature. Use a recent Chrome or Edge build and enable the browser's WebNN feature if the API is not enabled by default. The `navigator.ml` API must be available to the page.

## Install

Install ONNX Runtime Web from npm:

```bash
npm install onnxruntime-web
```

Import the bundle that includes all web execution providers:

```js
import * as ort from 'onnxruntime-web/all';
```

When loading ONNX Runtime Web with a script tag, use `ort.all.min.js` instead of `ort.min.js`.

## Usage

List `webnn` in the session's execution providers:

```js
import * as ort from 'onnxruntime-web/all';

const session = await ort.InferenceSession.create('model.onnx', {
  executionProviders: ['webnn'],
});
```

To request a specific device type or power profile, provide an execution provider options object:

```js
const session = await ort.InferenceSession.create('model.onnx', {
  executionProviders: [
    {
      name: 'webnn',
      deviceType: 'gpu',
      powerPreference: 'high-performance',
    },
  ],
});
```

## Configuration options

| Option | Allowed values | Default | Description |
|--------|----------------|---------|-------------|
| `deviceType` | `cpu`, `gpu`, `npu` | `cpu` | Preferred device type for the WebNN `MLContext`. Availability depends on the browser and platform. |
| `powerPreference` | `default`, `low-power`, `high-performance` | `default` | Preferred power and performance profile for the `MLContext`. |
| `context` | `MLContext` | none | Use a caller-created `MLContext`. When set, `deviceType` and `powerPreference` are ignored. A shared context is required for MLTensor I/O binding. |

Models with dynamic dimensions can use the ONNX Runtime Web [`freeDimensionOverrides`](../tutorials/web/env-flags-and-session-options.md#freedimensionoverrides) session option to provide concrete dimensions.

## Keep data on the device

WebNN exposes `MLTensor` for device-resident data. ONNX Runtime Web can create an input tensor from an `MLTensor`, write results into a preallocated `MLTensor`, or keep selected outputs on the device with `preferredOutputLocation`.

These workflows require a shared, caller-created `MLContext`. See the [WebNN I/O binding guide](../tutorials/web/ep-webnn.md#keep-tensor-data-on-webnn-mltensor-io-binding) for examples and resource-lifetime requirements.

## Build from source

ONNX Runtime Web uses the JavaScript Execution Provider (JSEP) infrastructure for WebNN. Enable JSEP with `--use_jsep` and add `--use_webnn` when building the WebAssembly package. See [Build ONNX Runtime Web](../build/web.md) for the complete build workflow and generated artifact names.

## Limitations

- Operator coverage depends on the browser's WebNN implementation. Unsupported operators are assigned to the WebAssembly execution provider.
- Device availability and performance characteristics differ across browsers, operating systems, and hardware.
- WebNN and its ONNX Runtime integration are under active development, so browser flags and build requirements may change.
- MLTensor I/O binding requires all participating sessions and tensors to use the same `MLContext`.

## Additional resources

- [Using the WebNN Execution Provider](../tutorials/web/ep-webnn.md)
- [ONNX Runtime Web browser support](../get-started/with-javascript/web.md#supported-versions)
- [WebNN operator support](https://github.com/microsoft/onnxruntime/blob/main/js/web/docs/webnn-operators.md)
- [WebNN specification](https://www.w3.org/TR/webnn/)
- [WebNN implementation status](https://webmachinelearning.github.io/webnn-status/)
