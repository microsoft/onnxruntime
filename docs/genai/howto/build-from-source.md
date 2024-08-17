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

## Download ONNX Runtime binaries

By default, the onnxruntime-genai build expects to find the ONNX Runtime include and binaries in a folder called `ort` in the root directory of onnxruntime-genai. You can put the ONNX Runtime files in a different location and specify this location to the onnxruntime-genai build via the --ort_home command line argument.


These instructions assume you are in the `onnxruntime-genai` folder.

#### Windows

These instruction use `win-x64`. Replace this if you are using a different architecture.

```bash
curl -L https://github.com/microsoft/onnxruntime/releases/download/v1.18.1/onnxruntime-win-x64-1.18.1.zip -o onnxruntime-win-x64-1.18.1.zip
tar xvf onnxruntime-win-x64-1.18.1.zip
move onnxruntime-win-x64-1.18.1 ort 
```

#### Linux and Mac

These instruction use `linux-x64-gpu`. Replace this if you are using a different architecture.

```bash
curl -L https://github.com/microsoft/onnxruntime/releases/download/v1.18.1/onnxruntime-linux-x64-gpu-1.18.1.tgz -o onnxruntime-linux-x64-gpu-1.18.1.tgz
tar xvzf onnxruntime-linux-x64-gpu-1.18.1.tgz
mv onnxruntime-linux-x64-gpu-1.18.1 ort 
```

#### Android

If you do not already have an `ort` folder, create one.

```bash
mkdir ort
```

```bash
curl -L https://repo1.maven.org/maven2/com/microsoft/onnxruntime/onnxruntime-android/1.18.0/onnxruntime-android-1.18.0.aar -o ort/onnxruntime-android-1.18.0.aar
cd ort
tar xvf onnxruntime-android-1.18.0.aar
cd ..
```

## Build the generate() API

This step assumes that you are in the root of the onnxruntime-genai repo, and you have followed the previous steps to copy the onnxruntime headers and binaries into the folder specified by <ORT_HOME>, which defaults to `onnxruntime-genai/ort`.

All of the build commands below have a `--config` argument, which takes the following options:
- `Release` builds release binaries
- `Debug` build binaries with debug symbols
- `RelWithDebInfo` builds release binaries with debug info

### Build Python API

#### Windows CPU build

```bash
python build.py --config `Release`
```

#### Windows DirectML build

```bash
python build.py --use_dml --config `Release`
```

#### Linux build

```bash
python build.py --config `Release`
```

#### Linux CUDA build

```bash
python build.py --use_cuda --config `Release`
```

#### Mac build

```bash
python build.py --config `Release`
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
python build.py --build_java --android --android_home <path to your Android SDK> --android_ndk_path <path to your NDK installation>` --android_abi  [armeabi-v7a|arm64-v8a|x86|x86_64] --config Release
```

## Install the library into your application

### Install Python wheel

```bash
cd build/wheel
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



