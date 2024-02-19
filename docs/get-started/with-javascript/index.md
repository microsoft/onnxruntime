---
title: JavaScript
parent: Get Started
has_children: true
toc: true
nav_order: 6
---

# Get started with ORT for JavaScript
{: .no_toc }

ONNX Runtime JavaScript API is the unified interface used by [ONNX Runtime Node.js binding](https://github.com/microsoft/onnxruntime/tree/main/js/node), [ONNX Runtime Web](https://github.com/microsoft/onnxruntime/tree/main/js/web), and [ONNX Runtime for React Native](https://github.com/microsoft/onnxruntime/tree/main/js/react_native).

See [how to choose the right package](../../tutorials/web/build-web-app#options-for-deployment-target) for your JavaScript application.

## Contents
{: .no_toc }

* Get Started with [ONNX Runtime Web](web.md)
* Get Started with [ONNX Runtime Node.js binding](node.md)
* Get Started with [ONNX Runtime for React Native](react-native.md)
* [Builds](#builds)
* [API Reference](#api-reference)

## Builds

[Builds](https://onnxruntime.ai/docs/build/web.html) are published to **npm** and can be installed using `npm install`

| Package | Artifact  | Description | Supported Platforms |
|---------|-----------|-------------|---------------------|
|Node.js binding|[onnxruntime-node](https://www.npmjs.com/package/onnxruntime-node)|CPU and GPU (Release/NAPI_v3)| Windows x64: cpu, dml<br/> Windows arm64: cpu, dml<br/> Linux x64: cpu, cuda<br/> Linux arm64: cpu<br/> MacOS x64: cpu<br/> MacOS arm64: cpu|
|Web|[onnxruntime-web](https://www.npmjs.com/package/onnxruntime-web)|CPU and GPU|Chromium Browsers (Chrome, Edge): wasm, webgl, webgpu, webnn<br/>Safari: wasm, webgl<br/>Other Browsers: wasm<br/> Node.js: wasm|
|React Native|[onnxruntime-react-native](https://www.npmjs.com/package/onnxruntime-react-native)|CPU|Android, iOS|

- For Web, pre-built binaries are published in NPM package as well as served in CDNs. If you want to use a custom build, you can [build ONNX Runtime Web from source](../../build/web.md).
- For Node.js binding, to use on platforms without pre-built binaries, you can [build Node.js binding from source](../../build/inferencing.md#apis-and-language-bindings) and consume using `npm install <onnxruntime_repo_root>/js/node/`.
- Explore a simple web application to [classify images with ONNX Runtime Web](https://onnxruntime.ai/docs/tutorials/web/classify-images-nextjs-github-template.html).

## API Reference

See [ONNX Runtime JavaScript API](../../api/js/index.html){:target="_blank"} for API reference.

See also:

- [ONNX Runtime JavaScript examples and API Usage](https://github.com/microsoft/onnxruntime-inference-examples/tree/main/js).

- [ONNX Runtime Web demos](https://microsoft.github.io/onnxruntime-web-demo/#/) for image recognition, handwriting analysis, real-time emotion detection, object detection, and so on.

- Typescript declarations for [Inference Session](https://github.com/microsoft/onnxruntime/blob/main/js/common/lib/inference-session.ts), [Tensor](https://github.com/microsoft/onnxruntime/blob/main/js/common/lib/tensor.ts), and [Environment Flags](https://github.com/microsoft/onnxruntime/blob/main/js/common/lib/env.ts) for reference.
