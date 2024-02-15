---
title: Build for iOS
parent: Build ONNX Runtime
nav_order: 5
---

# Build ONNX Runtime for iOS
{: .no_toc }

Follow the instructions below to build ONNX Runtime for iOS. 


## Contents
{: .no_toc }

* TOC placeholder
{:toc}

## General Info

* iOS Platforms

  The following two platforms are supported
  * iOS device (iPhone, iPad) with arm64 architecture
  * iOS simulator with x86_64 architecture

  The following platforms are *not* supported
  * armv7
  * armv7s
  * i386 architectures
  * tvOS
  * watchOS platforms are not currently supported.

* apple_deploy_target

  Specify the minimum version of the target platform (iOS) on which the target binaries are to be deployed.


## Prerequisites

* A Mac computer with latest macOS
* Xcode, https://developer.apple.com/xcode/
* CMake, https://cmake.org/download/
* Python 3, https://www.python.org/downloads/mac-osx/

## Build Instructions

Run one of the following build scripts from the ONNX Runtime repository root:

### Cross compile for iOS simulator

```bash
./build.sh --config <Release|Debug|RelWithDebInfo|MinSizeRel> --use_xcode \
           --ios --apple_sysroot iphonesimulator --osx_arch x86_64 --apple_deploy_target <minimal iOS version>
```

### Cross compile for iOS device

```bash
./build.sh --config <Release|Debug|RelWithDebInfo|MinSizeRel> --use_xcode \
           --ios --apple_sysroot iphoneos --osx_arch arm64 --apple_deploy_target <minimal iOS version>
```

### CoreML Execution Provider

If you want to use CoreML Execution Provider on iOS or macOS, see [CoreML Execution Provider](../execution-providers/CoreML-ExecutionProvider).

#### Build Instructions

CoreML Execution Provider can be built using building commands in [iOS Build instructions](#build-instructions-1) with `--use_coreml`

## Building a Custom iOS Package

Refer to the documentation for [custom builds](./custom.md). In particular, see the section about the [iOS Package](./custom.md#ios).
