---
title: Deploy on web
parent: Tutorials
has_children: true
nav_order: 6
---

# How to develop a web application with ONNX Runtime

ONNX Runtime gives you a variety of options to add machine learning to your web application. This page outlines the general flow through the development process. You can also build a web application to classify images using Next.js with this [tutorial](classify-images-nextjs-github-template.md). For more detail on the steps below, see the [build a web application](../../reference-build-web-app.md) with ONNX Runtime reference guide.

## ONNX Runtime web application development flow

1. Which ONNX Runtime web package should I use?

   We publish 3 ONNX Runtime web packages:
   * onnxruntime-web: for inference (running a machine learning model) in the web browser. Can also be used in the frontend of electron applications.
   * onnxruntime-node: for inference on the web server. Can also be used in the backend of electron applications. If the application is performance critical, one of the native ONNX Runtime libraries is also an option here.
   * onnxruntime-react-native: for react native web applications.

2. Which machine learning model does my application use?

   You need to understand your web app's scenario and get an ONNX model that is appropriate for that scenario. For example, does the app classify images, do object detection in a video stream, summarize or predict text, or do numerical prediction.

   ONNX Runtime web applications process models in ONNX format. ONNX models can be obtained from the [ONNX model zoo](https://github.com/onnx/models), converted from PyTorch or TensorFlow, and many other places.

3. How do I bootstrap my app development?

   Bootstrap your web application according in your web framework of choice e.g. vuejs, reactjs, angularjs.

   a. Add the ONNX Runtime dependency
   b. Consume the onnxruntime-web API in your application
   c. Add pre and post processing appropriate to your application and model

4. How do I optimize my application?

   The libraries and models mentioned in the previous steps can be optimized to meet memory and processing demands.

   a. Models in ONNX format can be [converted to ORT format](../../reference/ort-format-model-conversion.md), for optimized model binary size, faster initialization and peak memory usage.

   b. The size of the ONNX Runtime itself can reduced by [building a custom package](../../build/custom.md) that only includes support for your specific model/s.
