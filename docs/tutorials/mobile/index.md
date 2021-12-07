---
title: Deploy on mobile
description: Learn how to deploy an ONNX model on a mobile device or as a web application with ONNX Runtime
parent: Tutorials
has_children: true
nav_order: 5
redirect_from: /get-started/with-mobile
---

# How to develop a mobile application with ONNX Runtime

ONNX Runtime gives you a variety of options to add machine learning to your mobile application. This page outlines the general flow through the development process. You can also check out the tutorials in this section:

* [Build an objection detection application on iOS](./deploy-ios.md)
* [Build an image classification application on Android](./deploy-android.md)

## ONNX Runtime mobile application development flow

![Steps to build for mobile platforms](../../images/mobile.png){:width="60%"}

1. Which ONNX Runtime mobile library should I use?

   We publish the following ONNX Runtime mobile libraries:
   * Android C/C++
   * Android Java
   * iOS C/C++
   * iOS Objective C

2. Which machine learning model does my application use?

   You need to understand your mobile app's scenario and get an ONNX model that is appropriate for that scenario. For example, does the app classify images, do object detection in a video stream, summarize or predict text, or do numerical prediction.

   ONNX models can be obtained from the [ONNX model zoo](https://github.com/onnx/models), converted from PyTorch or TensorFlow, and many other places.

   Once you have sourced or converted the model into ONNX format, there is a further step required to optimize the model for mobile deployments. [Convert the model to ORT format](../../reference/ort-format-model-conversion.md) for optimized model binary size, faster initialization and peak memory usage.

3. How do I bootstrap my app development?

   If you are starting from scratch, bootstrap your mobile application according in your mobile framework XCode or Android Development Kit. TODO check this.

   a. Add the ONNX Runtime dependency
   b. Consume the onnxruntime-web API in your application
   c. Add pre and post processing appropriate to your application and model

4. How do I optimize my application?

   The libraries in step 1 can be optimized to meet memory and processing demands.

   The size of the ONNX Runtime itself can reduced by [building a custom package](../../build/custom.md) that only includes support for your specific model/s.

## Helpful resources


TODO

Can this be included anywhere:

The execution environment on mobile devices has fixed memory and disk storage. Therefore, it is essential that any AI execution library is optimized to consume minimum resources in terms of disk footprint, memory and network usage (both model size and binary size).

ONNX Runtime Mobile uses the ORT formatted model which enables us to create a [custom ORT build](../build/custom.md) that minimizes the binary size and reduces memory usage for client side inference. The ORT formatted model file is generated from the regular ONNX model using the `onnxruntime` python package. The custom build does this primarily by only including specified operators and types in the build, as well as trimming down dependencies per custom needs.

An ONNX model must be converted to an ORT format model to be used with minimal build in ONNX Runtime Mobile.