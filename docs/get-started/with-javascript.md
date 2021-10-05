---
title: JavaScript
parent: Get Started
toc: true
nav_order: 4
---

# Get started with ORT for JavaScript
{: .no_toc }

ONNX Runtime JavaScript API is the unified interface used by [ONNX Runtime Node.js binding](https://github.com/microsoft/onnxruntime/tree/master/js/node), [ONNX Runtime Web](https://github.com/microsoft/onnxruntime/tree/master/js/web) and [ONNX Runtime for React Native](https://github.com/microsoft/onnxruntime/tree/master/js/react_native).

## Contents
{: .no_toc }

* TOC placeholder
{:toc}

## JavaScript Examples (Install and import)

### Web ORT (client)

```bash
npm install onnxruntime-web
```
```javascript
const ort = require('onnxruntime-web');
```

### Node ORT (server)

```bash
npm install onnxruntime-node
```
```javascript
const ort = require('onnxruntime-node');
```

### React Native ORT

```bash
npm install onnxruntime-react-native
```
```javascript
const ort = require('onnxruntime-react-native');
```

### JavaScript Usage Example
```javascript
// use an async context to call onnxruntime functions.
async function main() {
    try {
        // create a new session and load the specific model.
        //
        // the model in this example contains a single MatMul node
        // it has 2 inputs: 'a'(float32, 3x4) and 'b'(float32, 4x3)
        // it has 1 output: 'c'(float32, 3x3)
        const session = await ort.InferenceSession.create('./model.onnx');

        // prepare inputs. a tensor need its corresponding TypedArray as data
        const dataA = Float32Array.from([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]);
        const dataB = Float32Array.from([10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120]);
        const tensorA = new ort.Tensor('float32', dataA, [3, 4]);
        const tensorB = new ort.Tensor('float32', dataB, [4, 3]);

        // prepare feeds. use model input names as keys.
        const feeds = { a: tensorA, b: tensorB };

        // feed inputs and run
        const results = await session.run(feeds);

        // read from results
        const dataC = results.c.data;
        document.write(`data of result tensor 'c': ${dataC}`);

    } catch (e) {
        document.write(`failed to inference ONNX model: ${e}.`);
    }
}

```

## Supported Versions

- ONNX Runtime Node.js binding: Node.js v12.x+ or Electron v5.x+
- ONNX Runtime Web: mainstream modern browsers on Windows, macOS, Android and iOS.
- ONNX Runtime for React Native: same as [ORT Mobile](./with-mobile) (Android/iOS)

## Builds

Builds are published to **npm** and can be installed using `npm install`

| Package | Artifact  | Description | Supported Platforms |
|---------|-----------|-------------|---------------------|
|Node.js binding|[onnxruntime-node](https://www.npmjs.com/package/onnxruntime-node)|CPU (Release)| Windows x64 CPU NAPI_v3, Linux x64 CPU NAPI_v3, MacOS x64 CPU NAPI_v3|
|Web|[onnxruntime-web](https://www.npmjs.com/package/onnxruntime-web)|CPU and GPU|Browsers (wasm, webgl), Node.js (wasm)|
|React Native|[onnxruntime-react-native](https://www.npmjs.com/package/onnxruntime-react-native)|CPU|Android, iOS|

For Node.js binding, to use on platforms without pre-built binaries, you can [build Node.js binding from source](../build/inferencing.md#apis-and-language-bindings) and consume using `npm install <onnxruntime_repo_root>/js/node/`.

## API Reference
See Typescript declarations for [Inference Session](https://github.com/microsoft/onnxruntime/blob/master/js/common/lib/inference-session.ts), [Tensor](https://github.com/microsoft/onnxruntime/blob/master/js/common/lib/tensor.ts) and [Environment Flags](https://github.com/microsoft/onnxruntime/blob/master/js/common/lib/env.ts) for reference.

See also [ONNX Runtime JavaScript examples](https://github.com/microsoft/onnxruntime-inference-examples/tree/main/js).
