---
title: JavaScript API
parent: API docs
grand_parent: Reference
---

# ONNX Runtime JavaScript API
{: .no_toc }

ONNX Runtime JavaScript API is the unified interface used by [ONNX Runtime Node.js binding](https://github.com/microsoft/onnxruntime/tree/master/js/node), [ONNX Runtime Web](https://github.com/microsoft/onnxruntime/tree/master/js/web) and [ONNX Runtime for React Native](https://github.com/microsoft/onnxruntime/tree/master/js/react_native).

## Contents
{: .no_toc }

* TOC placeholder
{:toc}


## Supported Versions

- ONNX Runtime Node.js binding: Node.js v12.x+ or Electron v5.x+
- ONNX Runtime Web: mainstream modern browsers on Windows, macOS, Android and iOS.
- ONNX Runtime for React Native: TBD

## Builds

Builds are published to **npm** and can be installed using `npm install`

| Package | Artifact  | Description | Supported Platforms |
|---------|-----------|-------------|---------------------|
|Node.js binding|[onnxruntime-node](https://www.npmjs.com/package/onnxruntime-node)|CPU (Release)| Windows x64 CPU NAPI_v3, Linux x64 CPU NAPI_v3, MacOS x64 CPU NAPI_v3|
|Web|[onnxruntime-web](https://www.npmjs.com/package/onnxruntime-web)||Browsers (wasm, webgl), Node.js (wasm)|
|React Native|[onnxruntime-react-native](https://www.npmjs.com/package/onnxruntime-react-native)|||

For Node.js binding, to use on platforms without pre-built binaries, you can [build Node.js binding from source](../../how-to/build/inferencing.md#apis-and-language-bindings) and consume using `npm install <onnxruntime_repo_root>/js/node/`.

## API Reference
See Typescript declarations for [Inference Session](https://github.com/microsoft/onnxruntime/blob/master/js/common/lib/inference-session.ts), [Tensor](https://github.com/microsoft/onnxruntime/blob/master/js/common/lib/tensor.ts) and [Environment Flags](https://github.com/microsoft/onnxruntime/blob/master/js/common/lib/env.ts) for reference.

See also [ONNX Runtime JavaScript examples](https://github.com/microsoft/onnxruntime-inference-examples/tree/main/js).
