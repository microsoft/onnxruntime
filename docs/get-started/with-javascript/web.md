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

See [ONNX Runtime JavaScript API](../../api/js/index.html){:target="_blank"} for API reference. Please also check the following links for API usage examples:
- [Tensor](https://github.com/microsoft/onnxruntime-inference-examples/blob/main/js/api-usage_tensor) - a demonstration of basic usage of Tensor.
- [Tensor <--> Image conversion](https://github.com/microsoft/onnxruntime-inference-examples/blob/main/js/api-usage-tensor-image) - a demonstration of conversions from Image elements to and from Tensor.
- [InferenceSession](https://github.com/microsoft/onnxruntime-inference-examples/blob/main/js/api-usage_inference-session) - a demonstration of basic usage of InferenceSession.
- [SessionOptions](https://github.com/microsoft/onnxruntime-inference-examples/blob/main/js/api-usage_session-options) - a demonstration of how to configure creation of an InferenceSession instance.
- [ort.env flags](https://github.com/microsoft/onnxruntime-inference-examples/blob/main/js/api-usage_ort-env-flags) - a demonstration of how to configure a set of global flags.

- See also: Typescript declarations for [Inference Session](https://github.com/microsoft/onnxruntime/blob/main/js/common/lib/inference-session.ts), [Tensor](https://github.com/microsoft/onnxruntime/blob/main/js/common/lib/tensor.ts), and [Environment Flags](https://github.com/microsoft/onnxruntime/blob/main/js/common/lib/env.ts) for reference.

See [Tutorial: Web](../../tutorials/web/index.md) for tutorials.

See [Training on web demo](https://github.com/microsoft/onnxruntime-training-examples/tree/master/on_device_training/web) for training using onnxruntime-web.

## Examples

The following examples describe how to use ONNX Runtime Web in your web applications for model inferencing:
- [Quick Start (using bundler)](https://github.com/microsoft/onnxruntime-inference-examples/tree/main/js/quick-start_onnxruntime-web-bundler)
- [Quick Start (using script tag)](https://github.com/microsoft/onnxruntime-inference-examples/tree/main/js/quick-start_onnxruntime-web-script-tag)

The following are E2E examples that uses ONNX Runtime Web in web applications:
- [Classify images with ONNX Runtime Web](https://onnxruntime.ai/docs/tutorials/web/classify-images-nextjs-github-template.html) - a simple web application using Next.js for image classifying.
- [ONNX Runtime Web demos](https://microsoft.github.io/onnxruntime-web-demo/#/) for image recognition, handwriting analysis, real-time emotion detection, object detection, and so on.
- [OpenAI Whisper](https://github.com/microsoft/onnxruntime-inference-examples/blob/main/js/ort-whisper) - demonstrates how to run [whisper tiny.en](https://github.com/openai/whisper) in your browser using onnxruntime-web and the browser's audio interfaces.
- [Facebook Segment-Anything](https://github.com/microsoft/onnxruntime-inference-examples/blob/main/js/segment-anything) - demonstrates how to run [segment-anything](https://github.com/facebookresearch/segment-anything) in your browser using onnxruntime-web with webgpu.


The following are video tutorials that use ONNX Runtime Web in web applications:
- [ONNX Runtime Web for In Browser Inference](https://youtu.be/0dskvE4IvGM)
- [Inference in Javascript with ONNX Runtime Web](https://youtu.be/vYzWrT3A7wQ)


## Supported Versions

| EPs/Browsers | Chrome/Edge (Windows) | Chrome/Edge (Android) | Chrome/Edge (MacOS) | Chrome/Edge (iOS) | Safari (MacOS) | Safari (iOS) | Firefox (Windows) | Node.js |
|--------------|--------|---------|--------|------|---|----|------|-----|
| WebAssembly (CPU)  |   ✔️    |    ✔️    |   ✔️   |  ✔️  |  ✔️  |  ✔️  |  ✔️  |  ✔️<sup>\[1]</sup>  |
| WebGPU         |   ✔️<sup>\[2]</sup>    |    ✔️<sup>\[3]</sup>    |   ✔️   |  ❌  |  ❌  |  ❌  |  ❌  |  ❌  |
| WebGL          |   ✔️<sup>\[4]</sup>    |    ✔️<sup>\[4]</sup>    |   ✔️<sup>\[4]</sup>   |  ✔️<sup>\[4]</sup>  |  ✔️<sup>\[4]</sup>  | ✔️<sup>\[4]</sup>  | ✔️<sup>\[4]</sup>  |  ❌  |
| WebNN          |   ✔️<sup>\[5]</sup>    |    ❌    |   ❌   |  ❌  |  ❌  |  ❌  |  ❌  |  ❌  |

- \[1]: Node.js only support single-threaded `wasm` EP.
- \[2]: WebGPU requires Chromium v113 or later on Windows. Float16 support requires Chrome v121 or later, and Edge v122 or later.
- \[3]: WebGPU requires Chromium v121 or later on Windows.
- \[4]: WebGL support is in maintenance mode. It is recommended to use WebGPU for better performance.
- \[5]: Requires to launch browser with commandline flag `--enable-experimental-web-platform-features`.