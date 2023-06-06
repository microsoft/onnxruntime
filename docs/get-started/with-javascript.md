---
title: JavaScript
parent: Get Started
toc: true
nav_order: 6
---

# Get started with ORT for JavaScript
{: .no_toc }

ONNX Runtime JavaScript API is the unified interface used by [ONNX Runtime Node.js binding](https://github.com/microsoft/onnxruntime/tree/main/js/node), [ONNX Runtime Web](https://github.com/microsoft/onnxruntime/tree/main/js/web), and [ONNX Runtime for React Native](https://github.com/microsoft/onnxruntime/tree/main/js/react_native).

## Contents
{: .no_toc }

* TOC placeholder
{:toc}

## ONNX Runtime Node.js binding
ONNX Runtime Node.js binding can be achieved by installing and importing.
### Install

```bash
# install latest release version
npm install onnxruntime-node
```

### Import

```js
// use ES6 style import syntax (recommended)
import * as ort from 'onnxruntime-node';
```
```js
// or use CommonJS style import syntax
const ort = require('onnxruntime-node');
```

### Examples

- Follow the [Quick Start](https://github.com/microsoft/onnxruntime-inference-examples/tree/main/js/quick-start_onnxruntime-node) instructions for ONNX Runtime Node.js binding.

### Supported Versions

ONNX Runtime Node.js binding supports Node.js v12.x+ or Electron v5.x+

## ONNX Runtime Web
You can install and import ONNX Runtime Web.
### Install

```bash
# install latest release version
npm install onnxruntime-web

# install nightly build dev version
npm install onnxruntime-web@dev
```

### Import


```js
// use ES6 style import syntax (recommended)
import * as ort from 'onnxruntime-web';
```
```js
// or use CommonJS style import syntax
const ort = require('onnxruntime-web');
```

If you want to use ONNX Runtime Web with WebGPU support (experimental feature), you need to import as below:

```js
// use ES6 style import syntax (recommended)
import * as ort from 'onnxruntime-web/webgpu';
```
```js
// or use CommonJS style import syntax
const ort = require('onnxruntime-web/webgpu');
```

### Examples

ONNX Runtime Web can also be imported via a script tag in a HTML file, from a CDN server. Here are some examples:
- [Quick Start (using bundler)](https://github.com/microsoft/onnxruntime-inference-examples/tree/main/js/quick-start_onnxruntime-web-bundler)
- [Quick Start (using script tag)](https://github.com/microsoft/onnxruntime-inference-examples/tree/main/js/quick-start_onnxruntime-web-script-tag)
- [ONNX Runtime Web for In Browser Inference](https://youtu.be/0dskvE4IvGM)
- [Inference in Javascript with ONNX Runtime Web](https://youtu.be/vYzWrT3A7wQ)


### Supported Versions


ONNX Runtime supports mainstream modern browsers/OS on Windows, Ubuntu, macOS, Android, and iOS. You can check the [compatibility](https://github.com/Microsoft/onnxjs#Compatibility) of ONNX Runtime with modern browsers and operating systems for your desktop and mobile platforms. In-browser inference is possible with [ONNX Runtime Web JavaScript](https://cloudblogs.microsoft.com/opensource/2021/09/02/onnx-runtime-web-running-your-machine-learning-model-in-browser/) that can enable cross-platform portability for web-applications. 



## ONNX Runtime for React Native
You can install and import ONNX Runtime Web for React Native.
### Install


```bash
# install latest release version
npm install onnxruntime-react-native
```

### Import


```js
// use ES6 style import syntax (recommended)
import * as ort from 'onnxruntime-react-native';
```
```js
// or use CommonJS style import syntax
const ort = require('onnxruntime-react-native');
```


#### Enable ONNX Runtime Extensions for React Native
To enable support for [ONNX Runtime Extensions](https://github.com/microsoft/onnxruntime-extensions) in your React Native app,
you need to specify the following configuration as a top-level entry (note: usually where the package `name`and `version`fields are) in your project's root directory `package.json` file. 

```js
"onnxruntimeExtensionsEnabled": "true"
```


## Builds

[Builds](https://onnxruntime.ai/docs/build/web.html) are published to **npm** and can be installed using `npm install`

| Package | Artifact  | Description | Supported Platforms |
|---------|-----------|-------------|---------------------|
|Node.js binding|[onnxruntime-node](https://www.npmjs.com/package/onnxruntime-node)|CPU (Release)| Windows x64 CPU NAPI_v3, Linux x64 CPU NAPI_v3, MacOS x64 CPU NAPI_v3|
|Web|[onnxruntime-web](https://www.npmjs.com/package/onnxruntime-web)|CPU and GPU|Browsers (wasm, webgl), Node.js (wasm)|
|React Native|[onnxruntime-react-native](https://www.npmjs.com/package/onnxruntime-react-native)|CPU|Android, iOS|

- For Node.js binding, to use on platforms without pre-built binaries, you can [build Node.js binding from source](../build/inferencing.md#apis-and-language-bindings) and consume using `npm install <onnxruntime_repo_root>/js/node/`.
- Consider the [options and considerations](https://onnxruntime.ai/docs/reference/build-web-app.html) for building a Web app with ONNX Runtime Web using JavaScript. 
- Explore a simple web application to [classify images with ONNX Runtime Web](https://onnxruntime.ai/docs/tutorials/web/classify-images-nextjs-github-template.html). 

## API Reference

See [ONNX Runtime JavaScript API](../api/js/index.html){:target="_blank"} for API reference. Check out the [ONNX Runtime Web demos!](https://microsoft.github.io/onnxruntime-web-demo/#/) for image recognition, handwriting analysis, real-time emotion detection, object detection, and so on.

See also:

- [ONNX Runtime JavaScript examples and API Usage](https://github.com/microsoft/onnxruntime-inference-examples/tree/main/js).

- Typescript declarations for [Inference Session](https://github.com/microsoft/onnxruntime/blob/main/js/common/lib/inference-session.ts), [Tensor](https://github.com/microsoft/onnxruntime/blob/main/js/common/lib/tensor.ts), and [Environment Flags](https://github.com/microsoft/onnxruntime/blob/main/js/common/lib/env.ts) for reference.
