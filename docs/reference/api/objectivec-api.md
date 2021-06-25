---
title: Objective-C API
parent: API docs
grand_parent: Reference
---

# ONNX Runtime Objective-C API
{: .no_toc }

ONNX Runtime provides an Objective-C API for running ONNX models on iOS devices.

## Contents
{: .no_toc }

* TOC placeholder
{:toc}

## Supported Versions

iOS 11+.

## Builds

The artifacts are published to CocoaPods.

| Artifact | Description | Supported Platforms | Notes |
|-|-|-|-|
| onnxruntime-mobile-objc | CPU | iOS | Currently a pre-release version (1.8.0-preview) |

Refer to the installation instructions [here](../../how-to/mobile/initial-setup.md#iOS).

## Swift Usage

The Objective-C API can be called from Swift code.
To enable this, use a bridging header (more info [here](https://developer.apple.com/documentation/swift/imported_c_and_objective-c_apis/importing_objective-c_into_swift)) that imports the ORT Objective-C API header.

```objectivec
// In the bridging header, import the ORT Objective-C API header.
#import <onnxruntime.h>
```

## API Reference

[Objective-C API Reference](../../../objectivec/index.html)

## Samples

[Basic Usage](https://github.com/microsoft/onnxruntime-inference-examples/tree/main/mobile/examples/basic_usage/ios)
