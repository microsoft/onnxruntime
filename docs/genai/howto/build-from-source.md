---
title: Build from source
description: How to build ONNX Runtime GenAI from source
has_children: false
parent: How to
grand_parent: Generative AI
nav_order: 2
---

# Build onnxruntime-genai from source
{: .no_toc }

* TOC placeholder
{:toc}

## Pre-requisites

`cmake`

## Build steps

1. Clone this repo

   ```bash
   git clone https://github.com/microsoft/onnxruntime-genai
   cd onnxruntime-genai
   ```

2. Install ONNX Runtime

    By default, the onnxruntime-genai build expects to find the ONNX Runtime include and binaries in a folder called `ort` in the root directory of onnxruntime-genai. You can put the ONNX Runtime files in a different location and specify this location to the onnxruntime-genai build. These instructions use ORT_HOME as the location.

    * Install from release

      These instructions are for the Linux GPU build of ONNX Runtime. Replace the location with the operating system and target of choice. 

      ```bash
      cd $ORT_HOME
      wget https://github.com/microsoft/onnxruntime/releases/download/v1.17.0/onnxruntime-linux-x64-gpu-1.17.0.tgz
      tar xvzf onnxruntime-linux-x64-gpu-1.17.0.tgz 
      mv onnxruntime-linux-x64-gpu-1.17.0/include .
      mv onnxruntime-linux-x64-gpu-1.17.0/lib .
      ```

    * Or build from source

      ```
      git clone https://github.com/microsoft/onnxruntime.git
      cd onnxruntime
      ```

      Create include and lib folders in the ORT_HOME directory

      ```bash
      mkdir $ORT_HOME/include
      mkdir $ORT_HOME/lib
      ```

      Build from source and copy the include and libraries into ORT_HOME

      On Windows

      ```cmd
      build.bat --build_shared_lib --skip_tests --parallel [--use_cuda]
      copy include\onnxruntime\core\session\onnxruntime_c_api.h $ORT_HOME\include
      copy build\Windows\Debug\Debug\*.dll $ORT_HOME\lib
      ```

      On Linux

      ```cmd
      ./build.sh --build_shared_lib --skip_tests --parallel [--use_cuda]
      cp include/onnxruntime/core/session/onnxruntime_c_api.h $ORT_HOME/include
      cp build/Linux/RelWithDebInfo/libonnxruntime*.so* $ORT_HOME/lib
      ```

3. Build onnxruntime-genai

   If you are building for CUDA, add the cuda_home argument.

   ```bash
   cd ..
   python build.py [--cuda_home <path_to_cuda_home>]
   ```


   
4. Install Python wheel

   ```bash
   cd build/wheel
   pip install *.whl
   ```