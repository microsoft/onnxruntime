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
- `.Net v6` (if building C#)

## Clone the onnxruntime-genai repo

```bash
git clone https://github.com/microsoft/onnxruntime-genai
cd onnxruntime-genai
```

## Install ONNX Runtime

By default, the onnxruntime-genai build expects to find the ONNX Runtime include and binaries in a folder called `ort` in the root directory of onnxruntime-genai. You can put the ONNX Runtime files in a different location and specify this location to the onnxruntime-genai build via the --ort_home command line argument.

### Option 1: Install from release

These instructions assume you are in the `onnxruntime-genai` folder.

#### Windows

These instruction use `win-x64`. Replace this if you are using a different architecture.

```bash
curl -L https://github.com/microsoft/onnxruntime/releases/download/v1.18.0/onnxruntime-win-x64-1.18.0.zip -o onnxruntime-win-x64-1.18.0.zip
tar xvf onnxruntime-win-x64-1.18.0.zip
move onnxruntime-win-x64-1.18.0 ort 
```

#### Linux and Mac

These instruction use `linux-x64-gpu`. Replace this if you are using a different architecture.

```bash
curl -L https://github.com/microsoft/onnxruntime/releases/download/v1.18.0/onnxruntime-linux-x64-gpu-1.18.0.tgz -o onnxruntime-linux-x64-gpu-1.18.0.tgz
tar xvzf onnxruntime-linux-x64-gpu-1.18.0.tgz
mv onnxruntime-linux-x64-gpu-1.18.0 ort 
```

### Option 2: Install from nightly

Download the nightly nuget package `Microsoft.ML.OnnxRuntime` from: https://aiinfra.visualstudio.com/PublicPackages/_artifacts/feed/ORT-Nightly.
  
Extract the nuget package.
  
```bash
tar xvf Microsoft.ML.OnnxRuntime.1.18.0-dev-20240322-0323-ca825cb6e6.nupkg
```
  
Copy the include and lib files into `ort`.
  
On Windows
  
Example is given for `win-x64`. Change this to your architecture if different.

```cmd
copy build\native\include\onnxruntime_c_api.h ort\include
copy runtimes\win-x64\native\*.dll ort\lib
```

On Linux

Example is given for `linux-x64`. Change this to your architecture if different.

```cmd
cp build/native/include/onnxruntime_c_api.h ort/include
cp build/linux-x64/native/libonnxruntime*.so* ort/lib
```      
      
### Option 3: Build from source

#### Clone the onnxruntime repo 

```bash
cd ..
git clone https://github.com/microsoft/onnxruntime.git
cd onnxruntime
```

#### Build ONNX Runtime for CPU on Windows

```bash
build.bat --build_shared_lib --skip_tests --parallel --config Release
copy include\onnxruntime\core\session\onnxruntime_c_api.h ..\onnxruntime-genai\ort\include
copy build\Windows\Release\Release\*.dll ..\onnxruntime-genai\ort\lib
copy build\Windows\Release\Release\onnxruntime.lib ..\onnxruntime-genai\ort\lib
```

#### Build ONNX Runtime for DirectML on Windows

```bash
build.bat --build_shared_lib --skip_tests --parallel --use_dml --config Release
copy include\onnxruntime\core\session\onnxruntime_c_api.h ..\onnxruntime-genai\ort\include
copy include\onnxruntime\core\providers\dml\dml_provider_factory.h ..\onnxruntime-genai\ort\include
copy build\Windows\Release\Release\*.dll ..\onnxruntime-genai\ort\lib
copy build\Windows\Release\Release\onnxruntime.lib ..\onnxruntime-genai\ort\lib
```


#### Build ONNX Runtime for CUDA on Windows

```bash
build.bat --build_shared_lib --skip_tests --parallel --use_cuda --config Release
copy include\onnxruntime\core\session\onnxruntime_c_api.h ..\onnxruntime-genai\ort\include
copy include\onnxruntime\core\providers\cuda\*.h ..\onnxruntime-genai\ort\include
copy build\Windows\Release\Release\*.dll ..\onnxruntime-genai\ort\lib
copy build\Windows\Release\Release\onnxruntime.lib ..\onnxruntime-genai\ort\lib
```

#### Build ONNX Runtime on Linux

```bash
./build.sh --build_shared_lib --skip_tests --parallel [--use_cuda] --config Release
cp include/onnxruntime/core/session/onnxruntime_c_api.h ../onnxruntime-genai/ort/include
cp build/Linux/Release/libonnxruntime*.so* ../onnxruntime-genai/ort/lib
```

You may need to provide extra command line options for building with CUDA on Linux. An example full command is as follows.

```bash
./build.sh --parallel --build_shared_lib --use_cuda --cuda_version 11.8 --cuda_home /usr/local/cuda-11.8 --cudnn_home /usr/lib/x86_64-linux-gnu/ --config Release --build_wheel --skip_tests --cmake_extra_defines CMAKE_CUDA_ARCHITECTURES="80" --cmake_extra_defines CMAKE_CUDA_COMPILER=/usr/local/cuda-11.8/bin/nvcc
```

Replace the values given above for different versions and locations of CUDA.

#### Build ONNX Runtime on Mac

```bash
./build.sh --build_shared_lib --skip_tests --parallel --config Release
cp include/onnxruntime/core/session/onnxruntime_c_api.h ../onnxruntime-genai/ort/include
cp build/MacOS/Release/libonnxruntime*.dylib* ../onnxruntime-genai/ort/lib
```

## Build the generate() API

This step assumes that you are in the root of the onnxruntime-genai repo, and you have followed the previos steps to copy the onnxruntime headers and binaries into the folder specified by <ORT_HOME>, which defaults to `onnxruntime-genai/ort`.

```bash
cd ../onnxruntime-genai
```

### Build Python API

#### Build for Windows CPU

```bash
python build.py
```

#### Build for Windows DirectML

```bash
python build.py --use_dml
```

#### Build on Linux

```bash
python build.py
```

#### Build on Linux with CUDA

```bash
python build.py --use_cuda
```

#### Build on Mac

```bash
python build.py
```

### Build Java API

```bash
python build.py --build_java --config Release
```
Change config to Debug for debug builds.

## Install the library into your application

### Install Python wheel

```bash
cd build/wheel
pip install *.whl
```

### Install .jar

Copy `build/Windows/Release/src/java/build/libs/*.jar` into your application.

### Install Nuget package

### Install C/C++ header file and library

_Coming soon_
