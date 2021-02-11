---
title: Node.js API
parent: API docs
grand_parent: Reference
nav_order: 4
---

# ONNX Runtime Node.js API
{: .no_toc }

ONNX Runtime Node.js binding enables Node.js applications to run ONNX model inference.

## Contents
{: .no_toc }

* TOC placeholder
{:toc}

## Usage

Install the latest stable version:

```
npm install onnxruntime
```

Install the latest dev version:

```
npm install onnxruntime@dev
```

Refer to [Node.js samples](https://github.com/microsoft/onnxruntime/tree/master/samples/nodejs) for samples and tutorials.

## Requirements

ONNXRuntime works on Node.js v12.x+ or Electron v5.x+.

Following platforms are supported with pre-built binaries:

- Windows x64 CPU NAPI_v3
- Linux x64 CPU NAPI_v3
- MacOS x64 CPU NAPI_v3

To use on platforms without pre-built binaries, you can build Node.js binding from source and consume it by `npm install <onnxruntime_repo_root>/nodejs/`. 

See also [build instructions](../../how-to/build.md#apis-and-language-bindings) for building ONNX Runtime Node.js binding locally.
