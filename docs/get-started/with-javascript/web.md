---
title: Web
parent: JavaScript
grand_parent: Get Started
has_children: false
nav_order: 1
---

# Get started with ONNX Runtime Web

## Contents
{: .no_toc }

* TOC placeholder
{:toc}

## Install

Use the following command in shell to install ONNX Runtime Web:

```bash
# install latest release version
npm install onnxruntime-web

# install nightly build dev version
npm install onnxruntime-web@dev
```

## Import

Use the following JavaScript code to import ONNX Runtime Web:

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

For a complete table for importing, see [Conditional Importing](https://github.com/microsoft/onnxruntime-inference-examples/tree/main/js/importing_onnxruntime-web#conditional-importing).

## Documentation

See [Tutorial: Web](../../tutorials/web/index.md) for more details. Please also check the following links:
- [Tensor](https://github.com/microsoft/onnxruntime-inference-examples/blob/main/js/api-usage_tensor) - a demonstration of basic usage of Tensor.
- [Tensor <--> Image conversion](https://github.com/microsoft/onnxruntime-inference-examples/blob/main/js/api-usage-tensor-image) - a demonstration of conversions from Image elements to and from Tensor.
- [InferenceSession](https://github.com/microsoft/onnxruntime-inference-examples/blob/main/js/api-usage_inference-session) - a demonstration of basic usage of InferenceSession.
- [SessionOptions](https://github.com/microsoft/onnxruntime-inference-examples/blob/main/js/api-usage_session-options) - a demonstration of how to configure creation of an InferenceSession instance.
- [ort.env flags](https://github.com/microsoft/onnxruntime-inference-examples/blob/main/js/api-usage_ort-env-flags) - a demonstration of how to configure a set of global flags.

See [Training on web demo](https://github.com/microsoft/onnxruntime-training-examples/tree/master/on_device_training/web) for training using onnxruntime-web.

## Examples

The following examples describe how to use ONNX Runtime Web in your web applications for model inferencing:
- [Quick Start (using bundler)](https://github.com/microsoft/onnxruntime-inference-examples/tree/main/js/quick-start_onnxruntime-web-bundler)
- [Quick Start (using script tag)](https://github.com/microsoft/onnxruntime-inference-examples/tree/main/js/quick-start_onnxruntime-web-script-tag)

The following are E2E examples that uses ONNX Runtime Web in web applications:
- [OpenAI Whisper](https://github.com/microsoft/onnxruntime-inference-examples/blob/main/js/ort-whisper) - demonstrates how to run [whisper tiny.en](https://github.com/openai/whisper) in your browser using onnxruntime-web and the browser's audio interfaces.
- [Facebook Segment-Anything](https://github.com/microsoft/onnxruntime-inference-examples/blob/main/js/segment-anything) - demonstrates how to run [segment-anything](https://github.com/facebookresearch/segment-anything) in your browser using onnxruntime-web with webgpu.

The following are video tutorials that use ONNX Runtime Web in web applications:
- [ONNX Runtime Web for In Browser Inference](https://youtu.be/0dskvE4IvGM)
- [Inference in Javascript with ONNX Runtime Web](https://youtu.be/vYzWrT3A7wQ)


## Supported Versions


ONNX Runtime supports mainstream modern browsers/OS on Windows, Ubuntu, macOS, Android, and iOS. Specifically, for Chromium-based browsers, ONNX Runtime Web supports wasm, webgl, webgpu, and webnn EPs. For Safari, ONNX Runtime Web supports wasm and webgl EPs. For other browsers or Node.js, ONNX Runtime Web supports wasm EP.
