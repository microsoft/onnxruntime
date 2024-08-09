---
title: Objective-C
parent: Get Started
nav_order: 7
---
# Get started with ORT for Objective-C
{: .no_toc }

ONNX Runtime provides an Objective-C API for running ONNX models on iOS devices.

## Contents
{: .no_toc }

* TOC placeholder
{:toc}

## Supported Versions

See iOS [compatibility info](../reference/compatibility.md).

## Builds

The artifacts are published to CocoaPods.

| Artifact | Description | Supported Platforms |
|-|-|-|
| onnxruntime-objc | CPU and CoreML | iOS |

Refer to the [installation instructions](../install/index.md#install-on-ios).

## Swift Usage

The Objective-C API can be called from Swift code.
To enable this, use a bridging header (more info [here](https://developer.apple.com/documentation/swift/importing-objective-c-into-swift)) that imports the ORT Objective-C API header.

```objectivec
// In the bridging header, import the ORT Objective-C API header.
#import <onnxruntime.h>
```

## API Reference

[Objective-C API Reference](../api/objectivec/index.html)

## Samples

See the iOS examples [here](https://github.com/microsoft/onnxruntime-inference-examples/tree/main/mobile).