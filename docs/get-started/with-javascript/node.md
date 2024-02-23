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

## Supported Versions

The following table lists the supported versions of ONNX Runtime Node.js binding provided with pre-built binaries.


| EPs/Platforms | Windows x64 | Windows arm64 | Linux x64 | Linux arm64 | MacOS x64 | MacOS arm64 |
|--------------|--------|---------|--------|------|---|----|
| CPU  |   ✔️    |    ✔️    |   ✔️   |  ✔️  |  ✔️  |  ✔️  |
| DirectML  |   ✔️    |    ✔️    |  ❌  |  ❌  |  ❌  |  ❌  |
| CUDA     |  ❌  |  ❌  |  ✔️<sup>\[1]</sup>  | ❌ | ❌ |  ❌  |


- \[1]: CUDA v11.8.


For platforms not on the list or want a custom build, you can [build Node.js binding from source](../../build/inferencing.md#apis-and-language-bindings) and consume using `npm install <onnxruntime_repo_root>/js/node/`.
