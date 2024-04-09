---
title: Build from source
description: How to build ONNX Runtime GenAI from source
has_children: false
parent: How to
grand_parent: Generative AI (Preview)
nav_order: 2
---

# Build onnxruntime-genai from source
{: .no_toc }

* TOC placeholder
{:toc}

## Pre-requisites

`cmake`

## Clone the onnxruntime-genai repo

```bash
git clone https://github.com/microsoft/onnxruntime-genai
cd onnxruntime-genai
```

## Install ONNX Runtime

By default, the onnxruntime-genai build expects to find the ONNX Runtime include and binaries in a folder called `ort` in the root directory of onnxruntime-genai. You can put the ONNX Runtime files in a different location and specify this location to the onnxruntime-genai build. These instructions use `ORT_HOME` as the location.

### Option 1: Install from release

These instructions are for the Linux GPU build of ONNX Runtime. Replace `linux-gpu` with your target of choice.

```bash
cd <ORT_HOME>
curl -L https://github.com/microsoft/onnxruntime/releases/download/v1.17.1/onnxruntime-linux-x64-gpu-1.17.1.tgz
tar xvzf onnxruntime-linux-x64-gpu-1.17.1.tgz 
mv onnxruntime-linux-x64-gpu-1.17.1/include .
mv onnxruntime-linux-x64-gpu-1.17.1/lib .
```

### Option 2: Install from nightly

Download the nightly nuget package `Microsoft.ML.OnnxRuntime` from: https://aiinfra.visualstudio.com/PublicPackages/_artifacts/feed/ORT-Nightly.
  
Extract the nuget package.
  
```bash
tar xvf Microsoft.ML.OnnxRuntime.1.18.0-dev-20240322-0323-ca825cb6e6.nupkg
```
  
Copy the include and lib files into `ORT_HOME`.
  
On Windows
  
Example is given for `win-x64`. Change this to your architecture if different.

```cmd
copy build\native\include\onnxruntime_c_api.h <ORT_HOME>\include
copy runtimes\win-x64\native\*.dll <ORT_HOME>\lib
```

On Linux

```cmd
cp build/native/include/onnxruntime_c_api.h <ORT_HOME>/include
cp build/linux-x64/native/libonnxruntime*.so* <ORT_HOME>/lib
```      
      
### Option 3: Build from source

```
git clone https://github.com/microsoft/onnxruntime.git
cd onnxruntime
```

Create include and lib folders in the `ORT_HOME` directory

```bash
mkdir <ORT HOME>/include
mkdir <ORT_HOME>/lib
```

Build from source and copy the include and libraries into `ORT_HOME`

On Windows

```cmd
build.bat --build_shared_lib --skip_tests --parallel [--use_cuda]
copy include\onnxruntime\core\session\onnxruntime_c_api.h <ORT_HOME>\include
copy build\Windows\Debug\Debug\*.dll <ORT_HOME>\lib
copy build\Windows\Debug\Debug\onnxruntime.lib <ORTHOME>\lib
```

On Linux

```cmd
./build.sh --build_shared_lib --skip_tests --parallel [--use_cuda]
cp include/onnxruntime/core/session/onnxruntime_c_api.h <ORT_HOME>/include
cp build/Linux/RelWithDebInfo/libonnxruntime*.so* <ORT_HOME>/lib
```

## Build onnxruntime-genai

### Build for CPU

```bash
cd ..
python build.py [--ort_home <ORT_HOME>]
```

### Build for CUDA

These instructions assume you already have CUDA installed.

```bash
cd ..
python build.py --cuda_home <path to cuda home> [--ort_home <ORT_HOME>]
```
   
## Install the library into your application

### Install Python wheel

```bash
cd build/wheel
pip install *.whl
```

### Install Nuget package

_Coming soon_

### Install C/C++ header file and library

_Coming soon_
