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

![Steps to build for mobile platforms](../../../images/mobile.png){:width="60%"}

1. Which ONNX Runtime package should I use?

   We publish the following ONNX Runtime packages that can be used in mobile applications:
   * Android Java/C/C++
     * Mobile (onnxruntime-mobile) and full (onnxruntime-android) packages
   * iOS C/C++
     * Mobile (onnxruntime-mobile-c) and full (onnxruntime-c) packages
   * iOS Objective-C
     * Mobile (onnxruntime-mobile-objc) and full (onnxruntime-objc) packages

   The full package has the full ONNX Runtime feature set.
   The mobile package has a smaller binary size but limited feature support, like a reduced set of operator implementations and no support for running ONNX models.

   A [custom build](../../build/custom.md) is tailored to your model(s) and can be even smaller than the mobile package. However, using a custom build is more involved than using one of the published packages.

   If the binary size of the full package is acceptable, using the full package is recommended because it is easier to use.
   Otherwise, consider using the mobile package or a custom build.

   To give an idea of the binary size difference between mobile and full packages:

   ONNX Runtime 1.11.0 Android package `jni/arm64-v8a/libonnxruntime.so` dynamic library file size:

   |Package|Size|
   |-|-|
   |onnxruntime-mobile|3.3 MB|
   |onnxruntime-android|12 MB|

   ONNX Runtime 1.11.0 iOS package `onnxruntime.xcframework/ios-arm64/onnxruntime.framework/onnxruntime` static library file size:

   |Package|Size|
   |-|-|
   |onnxruntime-mobile-c|22 MB|
   |onnxruntime-c|48 MB|

   Note: The iOS package is a static framework that will have a reduced binary size impact when compiled into your app.

   See [here](../../install/index.md#install-on-web-and-mobile) for installation instructions.

2. Which machine learning model does my application use?

   You need to understand your mobile app's scenario and get an ONNX model that is appropriate for that scenario. For example, does the app classify images, do object detection in a video stream, summarize or predict text, or do numerical prediction.

   ONNX models can be obtained from the [ONNX model zoo](https://github.com/onnx/models), converted from PyTorch or TensorFlow, and many other places.

   Once you have sourced or converted the model into ONNX format, it must be [converted to an ORT format model](../../reference/ort-format-models.md#convert-onnx-models-to-ort-format) in order to be used with the ONNX Runtime mobile package. This conversion is not necessary if you are using the full package.

3. How do I bootstrap my app development?

   If you are starting from scratch, bootstrap your mobile application according in your mobile framework XCode or Android Development Kit. TODO check this.

   a. Add the ONNX Runtime dependency
  
   b. Consume the onnxruntime API in your application
   
   c. Add pre and post processing appropriate to your application and model

4. How do I optimize my application?

   **To reduce binary size:**  Use the ONNX Runtime mobile package or a custom build to reduce the binary size. The mobile package requires use of an ORT format model.

   **To reduce memory usage:** Use an ORT format model as that uses less memory.
