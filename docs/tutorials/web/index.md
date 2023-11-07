---
title: Deploy on web
parent: Tutorials
has_children: true
nav_order: 7
---

# How to add machine learning to your web application with ONNX Runtime

ONNX Runtime Web enables you to run and deploy machine learning models in your web application using JavaScript APIs and libraries. This page outlines the general flow through the development process.

You can also integrate machine learning into the server side of your web application with ONNX Runtime using other language libraries, depending on your application development environment.

To see an example of the web development flow in practice, you can follow the steps in the following tutorial to [build a web application to classify images using Next.js](classify-images-nextjs-github-template.md).

For more detail on the steps below, see the [build a web application](./build-web-app.md) with ONNX Runtime reference guide.

## ONNX Runtime web application development flow

1. Choose deployment target and ONNX Runtime package

   ONNX Runtime can be integrated into your web application in a number of different ways depending on the requirements of your application.

   * Inference in browser. Use the `onnxruntime-web` package.

     There are benefits to doing on-device and in-browser inference.

     * **It's faster.** That's right, you can cut inferencing time way down which inferencing is done right on the client for models that are optimized to work on less powerful hardware.
     * **It's safer** and helps with privacy. Since the data never leaves the device for inferencing, it is a safer method of doing inferencing.
     * **It works offline.** If you lose internet connection, the model will still be able to inference.
     * **It's cheaper.** You can reduce cloud serving costs by offloading inference to the browser.

     You can also use the onnxruntime-web package in the frontend of an electron app.

     With onnxruntime-web, you have the option to use `webgl` or `webgpu` for GPU processing, and WebAssembly (`wasm`, alias to `cpu`) for CPU processing. All ONNX operators are supported by WASM but only a subset are currently supported by WebGL and WebGPU.

   * Inference on server in JavaScript. Use the `onnxruntime-node` package.

     Your application may have constraints that means it is better to perform inference server side.

     * **The model is too large** and requires higher hardware specs. In order to do inference on the client you need to have a model that is small enough to run efficiently on less powerful hardware.
     * You don't want the model to be downloaded onto the device.

     You can also use the onnxruntime-node package in the backend of an electron app.

   * Inference on server using other language APIs. Use the ONNX Runtime packages for C/C++ and other languages.

     * **If you are not developing your web backend in node.js** If the backend of your web application is developed in another language, you can use ONNX Runtime APIs in the language of your choice.

   * Inference in a React Native application. Use the `onnxruntime-react-native` package.

2. Which machine learning model does my application use?

   You need to understand your web app's scenario and get an ONNX model that is appropriate for that scenario. For example, does the app classify images, do object detection in a video stream, summarize or predict text, or do numerical prediction.

   ONNX Runtime web applications process models in ONNX format. ONNX models can be obtained from the [ONNX model zoo](https://github.com/onnx/models), converted from PyTorch or TensorFlow, and many other places.

   You can also create a custom model that is specific to the task you are trying to solve. Use code to build your model or use low code/no code tools to create the model. Check out the resources below to learn about some different ways to create a customized model. All of these resources have an export to ONNX format functionality so that you can leverage this template and source code.

   * [Use AutoML to create a custom model](https://docs.microsoft.com/azure/machine-learning/concept-automated-ml)
   * [Use Custom Vision Cognitive Services to create a custom model](https://docs.microsoft.com/azure/cognitive-services/custom-vision-service/overview)
   * [Use Azure Machine Learning Designer to create a custom model](https://docs.microsoft.com/en-us/azure/machine-learning/concept-designer)
   * [Build your own model with PyTorch.](https://docs.microsoft.com/learn/paths/pytorch-fundamentals/)

3. How do I bootstrap my app development?

   Bootstrap your web application according in your web framework of choice e.g. vuejs, reactjs, angularjs.

   1. [Add the ONNX Runtime dependency](./build-web-app.md#add-onnx-runtime-web-as-dependency)

   1. [Consume the onnxruntime-web API in your application](./build-web-app.md#consume-onnxruntime-web-in-your-code)

   1. [Add pre and post processing](./build-web-app.md#pre-and-post-processing) appropriate to your application and model

4. How do I optimize my application?

   The libraries and models mentioned in the previous steps can be optimized to meet memory and processing demands.

   a. Models in ONNX format can be [converted to ORT format](../../performance/model-optimizations/ort-format-models.md), for optimized model binary size, faster initialization and peak memory usage.

   b. The size of the ONNX Runtime itself can reduced by [building a custom package](../../build/custom.md) that only includes support for your specific model/s.

   c. Tune ONNX Runtime inference session options, including trying different Execution Providers
