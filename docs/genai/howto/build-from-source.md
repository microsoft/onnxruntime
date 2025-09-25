---
title: Build from source
description: How to build the ONNX Runtime generate() API from source
has_children: false
parent: How to
grand_parent: Generate API (Preview)
nav_order: 2
---

# Build the generate() API from source
{: .no_toc }

* TOC placeholder
{:toc}

## Pre-requisites

- `cmake`
- `.NET6` (if building C#)

## Clone the onnxruntime-genai repo

```bash
git clone https://github.com/microsoft/onnxruntime-genai
cd onnxruntime-genai
```

## Build the generate() API

This step assumes that you are in the root of the onnxruntime-genai repo.

All of the build commands below have a `--config` argument, which takes the following options:
- `Release` builds release binaries
- `Debug` build binaries with debug symbols
- `RelWithDebInfo` builds release binaries with debug info

### Build Python API

#### Windows CPU build

```bash
python build.py --config Release
```

#### Windows DirectML build

```bash
python build.py --use_dml --config Release
```

#### Windows NvTensorRtRtx build

```bash
python build.py --use_trt_rtx --config Release --cuda_home <cuda_path>
```

#### Linux build

```bash
python build.py --config Release
```

#### Linux CUDA build

```bash
python build.py --use_cuda --config Release
```

#### Mac build

```bash
python build.py --config Release
```

### Build Java API

```bash
python build.py --build_java --config Release
```

### Build for Android

If building on Windows, install `ninja`.

```bash
pip install ninja
```

Run the build script.

```bash
python build.py --build_java --android --android_home <path to your Android SDK> --android_ndk_path <path to your NDK installation> --android_abi  [armeabi-v7a|arm64-v8a|x86|x86_64] --config Release
```

## Install the library into your application

### Install Python wheel

```bash
# Change dir to the folder containing the onnxruntime-genai wheel
# Example for Linux: cd build/Linux/Release/wheel/
pip install *.whl
```

### Install NuGet

_Coming soon_

### Install JAR

Copy `build/Windows/Release/src/java/build/libs/*.jar` into your application.

### Install AAR

Copy `build/Android/Release/src/java/build/android/outputs/aar/onnxruntime-genai-release.aar` into your application.


### Install C/C++ header file and library

#### Windows

Use the header in `src\ort_genai.h` and the libraries in `build\Windows\Release`

#### Linux

Use the header in `src/ort_genai.h` and the libraries in `build/Linux/Release`
