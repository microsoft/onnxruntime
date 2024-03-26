---
title: Build a web app with ONNX Runtime
description: Considerations and options for building a web application with ONNX Runtime
parent: Web
grand_parent: Tutorials
nav_order: 1
redirect_from: /reference/build-web-app

---
# Build a web application with ONNX Runtime
{: .no_toc}
This document explains the options and considerations for building a web application with ONNX Runtime.

## Contents
{: .no_toc}

* TOC
{:toc}


## Options for deployment target

1. Inference in browser

   Runtime and model are downloaded to client and inferencing happens inside browser. Use onnxruntime-web in this scenario.

2. Inference on server

   Browser sends user's input to server, server inferences and gets the result and sends back to client.

   Use native ONNX Runtime to get best performance.

   To use Node.js as the server application, use onnxruntime-node (ONNX Runtime node.js binding) on the server.

3. Electron

   Electron uses a frontend (based on chromium, technically a browser core) and a backend (based on Node.js).

   If possible, use onnxruntime-node for inference in the backend, which is faster. Using onnxruntime-web in frontend is also an option (for security and compatibility concerns).

4. React Native

   React-native is a framework that uses the same API to reactjs, but builds native applications instead of web app on mobile. Should use onnxruntime-react-native.

## Options to obtain a model

You need to understand your web app's scenario and get an ONNX model that is appropriate for that scenario.

ONNX models can be obtained from the [ONNX model zoo](https://github.com/onnx/models), converted from PyTorch or TensorFlow, and many other places.

You can [convert the ONNX format model to ORT format model](../../performance/model-optimizations/ort-format-models.md), for optimized binary size, faster initialization and peak memory usage.

You can [perform a model-specific custom build](../../build/custom.md) to further optimize binary size.

## Bootstrap your application

Bootstrap your web application according in your web framework of choice e.g. vuejs, reactjs, angularjs.

You can skip this step if you already have a web application and you are adding machine learning to it with ONNX Runtime.

## Add ONNX Runtime Web as dependency

Install onnxruntime-web. These command line will update the application's `package.json` file.

### With yarn

```bash
yarn add onnxruntime-web 
```

### With npm

```bash
npm install onnxruntime-web
```

Add "@dev" to the package name to use the nightly build (eg. npm install onnxruntime-web@dev).

## Consume onnxruntime-web in your code

1. Import onnxruntime-web
   See  [import onnxruntime-web](../../get-started/with-javascript/web.md#import)

2. Initialize the inference session
   See [InferenceSession.create](https://github.com/microsoft/onnxruntime-inference-examples/blob/main/js/quick-start_onnxruntime-web-bundler/main.js#L14)

   Session initialization should only happen once.

3. Run the session
   See [session.run](https://github.com/microsoft/onnxruntime-inference-examples/blob/main/js/quick-start_onnxruntime-web-bundler/main.js#L26)

   Session run happens each time their is new user input.

Refer to [ONNX Runtime Web API docs](../../api/js) for more detail.
  
## Pre and post processing

Raw input is usually a string (for NLP model) or an image (for image model). They come from multiple forms and formats.

### String inputs

1. Use a tokenizer in JS/wasm to pre-process it to number data, create tensors from the data and feed to ORT for model inferencing.

2. Use one or more custom ops to deal with strings. Build with the custom ops. The model can directly process string tensor inputs. Refer to the[onnxruntime-extensions](https://github.com/microsoft/onnxruntime-extensions) library, which contain a set of possible custom operators.

### Image input

1. Use a JS/wasm library to pre-process the data, and create tensor as input to fulfill the requirement of the model. See the [image classification using ONNX Runtime Web](./classify-images-nextjs-github-template.md) tutorial.

2. Modify the model to include the pre-processing inside the model as operators. The model will expect a certain web image format (eg. A bitmap or texture from canvas).

### Outputs

The output of a model vary, and most need their own post-processing code. Refer to the above tutorial as an example of Javascript post processing.

## Bundlers

_[This section is coming soon]_