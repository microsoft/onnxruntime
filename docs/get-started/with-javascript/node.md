---
title: Node.js binding
parent: JavaScript
grand_parent: Get Started
has_children: false
nav_order: 2
---

# Get started with ONNX Runtime Node.js binding

## Contents
{: .no_toc }

* TOC placeholder
{:toc}

## Install

```bash
# install latest release version
npm install onnxruntime-node
```

## Import

```js
// use ES6 style import syntax (recommended)
import * as ort from 'onnxruntime-node';
```
```js
// or use CommonJS style import syntax
const ort = require('onnxruntime-node');
```

## Examples

- Follow the [Quick Start](https://github.com/microsoft/onnxruntime-inference-examples/tree/main/js/quick-start_onnxruntime-node) instructions for ONNX Runtime Node.js binding.

The following table lists the supported versions of ONNX Runtime Node.js binding provided with pre-built binaries.

| EPs/Platforms | Windows x64        | Windows arm64      | Linux x64          | Linux arm64        | MacOS x64          | MacOS arm64        |
| ------------- | ------------------ | ------------------ | ------------------ | ------------------ | ------------------ | ------------------ |
| CPU           | ✔️                 | ✔️                 | ✔️                 | ✔️                 | ✔️                 | ✔️                 |
| WebGPU        | ✔️ <sup>\[1]</sup> | ✔️ <sup>\[1]</sup> | ✔️ <sup>\[1]</sup> | ❌ <sup>\[2]</sup> | ✔️ <sup>\[1]</sup> | ✔️ <sup>\[1]</sup> |
| DirectML      | ✔️                 | ✔️                 | ❌                 | ❌                 | ❌                 | ❌                 |
| CUDA          | ❌                 | ❌                 | ✔️<sup>\[3]</sup>  | ❌                 | ❌                 | ❌                 |
| CoreML        | ❌                 | ❌                 | ❌                 | ❌                 | ✔️                 | ✔️                 |

- \[1]: WebGPU support is currently experimental.
- \[2]: WebGPU support is not available on Linux arm64 yet in the pre-built binaries.
- \[3]: CUDA v12. See [CUDA EP Installation](#cuda-ep-installation) for details.

To use on platforms without pre-built binaries, you can build Node.js binding from source and consume it by `npm install <onnxruntime_repo_root>/js/node/`. See also [instructions](https://onnxruntime.ai/docs/build/inferencing.html#apis-and-language-bindings) for building ONNX Runtime Node.js binding locally.

# GPU Support

Right now, the Windows version supports WebGPU execution provider and DML execution provider. Linux x64 can use CUDA and TensorRT.

## CUDA EP Installation

To use CUDA EP, you need to install the CUDA EP binaries. By default, the CUDA EP binaries are installed automatically when you install the package. If you want to skip the installation, you can pass the `--onnxruntime-node-install=skip` flag to the installation command.

```
npm install onnxruntime-node --onnxruntime-node-install=skip
```

~~You can also use this flag to specify the version of the CUDA: (v11 or v12)~~ CUDA v11 is no longer supported since v1.22.
