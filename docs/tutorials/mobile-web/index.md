---
title: Deploy on mobile and web
description: Learn how to deploy an ONNX model on a mobile device or as a web application with ONNX Runtime
parent: Tutorials
has_children: true
nav_order: 5
redirect_from: /docs/tutorials/mobile/,/docs/tutorials/mobile/limitations
---

# Deploy on mobile devices and to the web

ORT format model is supported by version 1.5.2 of ONNX Runtime or later.

## Overview

The execution environment on Mobile devices, and for web browsers, have fixed memory and disk storage. The execution scenario on web browsers have strict memory consumption and network bandwidths requirement. Therefore, it is essential that any AI execution library is optimized to consume minimum resources in terms of disk footprint, memory and network usage (both model size and binary size). ONNX Runtime was enhanced to target these size constrained environments. These enhancements are packaged in the ONNX Runtime Mobile and web offering.

ONNX Runtime Mobile and web uses the ORT formatted model which enables us to create a [custom ORT build](../build/custom.md) that minimizes the binary size and reduces memory usage for client side inference. The ORT formatted model file is generated from the regular ONNX model using the `onnxruntime` python package. The custom build does this primarily by only including specified operators and types in the build, as well as trimming down dependencies per custom needs.

An ONNX model must be converted to an ORT format model to be used with minimal build in ONNX Runtime Mobile or ONNX Runtime Web.

![Steps to build for mobile platforms](../../../images/mobile.png){:width="60%"}

There are two options for deploying ONNX Runtime with ORT format model.

* TOC
{:toc}

## Pre-built package

The pre-built package includes support for selected operators and ONNX opset versions based on the requirements for popular models. If you choose to use the pre-built package you will not need a development environment to perform a custom build of ONNX Runtime, however the binary size will be larger than if you do a custom build with just the operators required by your model/s. Your model can only use the opsets and operators supported by the pre-built package.

### Available pre-built packages

| Platform | Package location | Included Execution Providers |
|----------|------------------|----------|
| Android | onnxruntime-mobile package in Maven  | CPU Execution Provider <br>NNAPI Execution Provider |
| iOS | onnxruntime-mobile CocoaPod (C/C++ APIs) <br>onnxruntime-mobile-objc CocoaPod (Objective-C API) | CPU Execution Provider <br>CoreML Execution Provider |
| Web (preview) | onnxruntime-web package in NPM | WebAssembly Execution Provider <br>WebGL Execution Provider |

### Operators and Types supported

ONNX operator and types supported by the pre-built package for each ONNX Runtime release:
- [mobile](../../reference/mobile/prebuilt-package)

## Custom build

Performing a custom build will ensure the smallest possibly binary size and that the build will work with your model/s. You can also choose whether to enable features such as exceptions or traditional ML operators. You will however need a development environment to build for all platforms you're targeting.

## Limitations

A minimal build has the following limitations:

* No support for ONNX format models, that is model must be converted to ORT format
* No support for runtime optimizations. Optimizations are performed during conversion to ORT format
* Limited support for runtime partitioning (assigning nodes in a model to specific execution providers). Execution providers that statically register kernels (e.g. ONNX Runtime CPU Execution Provider) are supported by default. All execution providers that will be used at runtime MUST be registered when creating the ORT format model
    - Execution providers that compile nodes are optionally supported
      - currently this is limited to the NNAPI and CoreML Execution Providers
        - see [here](./using-platform-specific-ep.html#using-nnapi-and-coreml-with-onnx-runtime-mobile) for details on using the NNAPI or CoreML Execution Providers with ONNX Runtime Mobile.

We do not currently offer backwards compatibility guarantees for ORT format models, as we will be expanding the capabilities in the short term and may need to update the internal format in an incompatible manner to accommodate these changes. You may need to regenerate the ORT format models to use with a future version of ONNX Runtime. Once the feature set stabilizes we will provide backwards compatibility guarantees.
