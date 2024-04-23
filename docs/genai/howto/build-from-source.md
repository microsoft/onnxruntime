---
title: Build from source
description: How to build the ONNX Runtime generate() API from source
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

#### Clone the repo 

```bash
git clone https://github.com/microsoft/onnxruntime.git
cd onnxruntime
```

#### Build ONNX Runtime for DirectML on Windows

```bash
build.bat --build_shared_lib --skip_tests --parallel --use_dml --config Release
```

#### Build ONNX Runtime for CPU on Windows

```bash
build.bat --build_shared_lib --skip_tests --parallel --config Release
```

#### Build ONNX Runtime for CUDA on Windows

```bash
build.bat --build_shared_lib --skip_tests --parallel --use_cuda --config Release
```

#### Build ONNX Runtine on Linux

```bash
./build.sh --build_shared_lib --skip_tests --parallel [--use_cuda] --config Release
```

You may need to provide extra command line options for building with CUDA on Linux. An example full command is as follows.

```bash
./build.sh --parallel --build_shared_lib --use_cuda --cuda_version 11.8 --cuda_home /usr/local/cuda-11.8 --cudnn_home /usr/lib/x86_64-linux-gnu/ --config Release --build_wheel --skip_tests --cmake_extra_defines CMAKE_CUDA_ARCHITECTURES="80" --cmake_extra_defines CMAKE_CUDA_COMPILER=/usr/local/cuda-11.8/bin/nvcc
```

Replace the values given above for different versions and locations of CUDA.

#### Build ONNX Runtime on Mac

```bash
./build.sh --build_shared_lib --skip_tests --parallel --config Release
```

## Build the generate() API

### Build on Windows

If building for DirectML

```bash
copy ..\onnxruntime\include\onnxruntime\core\providers\dml\dml_provider_factory.h ort\include
```

```bash
copy ..\onnxruntime\include\onnxruntime\core\session\onnxruntime_c_api.h ort\include
copy ..\onnxruntime\build\Windows\Release\Release\*.dll ort\lib
copy ..\onnxruntime\build\Windows\Release\Release\onnxruntime.lib ort\lib
python build.py [--use_dml | --use_cuda]
```

### Build on Linux

```bash
cp ../onnxruntime/include/onnxruntime/core/session/onnxruntime_c_api.h ort/include
cp ../onnxruntime/build/Linux/Release/libonnxruntime*.so* ort/lib
python build.py [--use_cuda]
```

### Build on Mac

```bash
cp ../onnxruntime/include/onnxruntime/core/session/onnxruntime_c_api.h ort/include
cp ../onnxruntime/build/MacOS/Release/libonnxruntime*.dylib* ort/lib
python build.py
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
