---
title: JavaScript
parent: Get Started
toc: true
nav_order: 3
---

# Get started with ORT for JavaScript
{: .no_toc }

ONNX Runtime JavaScript API is the unified interface used by [ONNX Runtime Node.js binding](https://github.com/microsoft/onnxruntime/tree/master/js/node), [ONNX Runtime Web](https://github.com/microsoft/onnxruntime/tree/master/js/web) and [ONNX Runtime for React Native](https://github.com/microsoft/onnxruntime/tree/master/js/react_native).

## Contents
{: .no_toc }

* TOC placeholder
{:toc}

## ONNX Runtime Node.js binding

### Install
{: .no_toc }

```bash
# install latest release version
npm install onnxruntime-node
```

### Import
{: .no_toc }

```js
// use ES6 style import syntax (recommended)
import * as ort from 'onnxruntime-node';
```
```js
// or use CommonJS style import syntax
const ort = require('onnxruntime-node');
```

### Examples
{: .no_toc }

- [Quick Start](https://github.com/microsoft/onnxruntime-inference-examples/tree/main/js/quick-start_onnxruntime-node)

### Supported Versions
{: .no_toc }

Node.js v12.x+ or Electron v5.x+

## ONNX Runtime Web

### Install
{: .no_toc }

```bash
# install latest release version
npm install onnxruntime-web

# install nightly build dev version
npm install onnxruntime-web@dev
```

### Import
{: .no_toc }

```js
// use ES6 style import syntax (recommended)
import * as ort from 'onnxruntime-web';
```
```js
// or use CommonJS style import syntax
const ort = require('onnxruntime-web');
```
ONNX Runtime Web can also be imported via a script tag in a HTML file, from a CDN server. See examples below for detail.

### Examples
{: .no_toc }

- [Quick Start (using bundler)](https://github.com/microsoft/onnxruntime-inference-examples/tree/main/js/quick-start_onnxruntime-web-bundler)
- [Quick Start (using script tag)](https://github.com/microsoft/onnxruntime-inference-examples/tree/main/js/quick-start_onnxruntime-web-script-tag)

### Supported Versions
{: .no_toc }

mainstream modern browsers on Windows, macOS, Android and iOS.

## ONNX Runtime for React Native

### Install
{: .no_toc }

```bash
# install latest release version
npm install onnxruntime-react-native
```

### Import
{: .no_toc }

```js
// use ES6 style import syntax (recommended)
import * as ort from 'onnxruntime-react-native';
```
```js
// or use CommonJS style import syntax
const ort = require('onnxruntime-react-native');
```

### Supported Versions
{: .no_toc }

same as [ORT Mobile](./with-mobile) (Android/iOS)

## Builds

Builds are published to **npm** and can be installed using `npm install`

| Package | Artifact  | Description | Supported Platforms |
|---------|-----------|-------------|---------------------|
|Node.js binding|[onnxruntime-node](https://www.npmjs.com/package/onnxruntime-node)|CPU (Release)| Windows x64 CPU NAPI_v3, Linux x64 CPU NAPI_v3, MacOS x64 CPU NAPI_v3|
|Web|[onnxruntime-web](https://www.npmjs.com/package/onnxruntime-web)|CPU and GPU|Browsers (wasm, webgl), Node.js (wasm)|
|React Native|[onnxruntime-react-native](https://www.npmjs.com/package/onnxruntime-react-native)|CPU|Android, iOS|

For Node.js binding, to use on platforms without pre-built binaries, you can [build Node.js binding from source](../build/inferencing.md#apis-and-language-bindings) and consume using `npm install <onnxruntime_repo_root>/js/node/`.

## API Reference

See [ONNX Runtime JavaScript API](../api/js/index.html){:target="_blank"} for API reference.

See also:

- [ONNX Runtime JavaScript examples](https://github.com/microsoft/onnxruntime-inference-examples/tree/main/js).

- Typescript declarations for [Inference Session](https://github.com/microsoft/onnxruntime/blob/master/js/common/lib/inference-session.ts), [Tensor](https://github.com/microsoft/onnxruntime/blob/master/js/common/lib/tensor.ts) and [Environment Flags](https://github.com/microsoft/onnxruntime/blob/master/js/common/lib/env.ts) for reference.
