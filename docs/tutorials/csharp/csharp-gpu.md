---
title: Configure CUDA for C#
description: Configure CUDA and cuDNN for ONNX Runtime with C# on Windows 11
parent: Inference with C#
grand_parent: Tutorials
has_children: false
nav_order: 1
---


# Configure CUDA and cuDNN for ONNX Runtime with C# on Windows 11

{: .no_toc }

## Contents
{: .no_toc }

* TOC placeholder
{:toc}

## Prerequisites
- Windows 11
- Visual Studio 2019 or 2022
 
## Steps to Configure CUDA and cuDNN for ONNX Runtime with C# on Windows 11

1. Install CUDA toolkit based on the supported version for the ONNX Runtime Version. See this link for compatible versions:
[NVIDIA - CUDA - ONNX Runtime](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html)

2. Install Driver from NVIDIA based on your GPU. Find the driver download here: [Official Drivers - NVIDIA](https://www.nvidia.com/download/index.aspx?lang=en-us)

3. Install the cuDNN version based on the supported version for the ONNX Runtime Version. See this link for compatible versions: [NVIDIA - CUDA - ONNX Runtime](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html) then download the cuDNN version from here: [cuDNN Archive - NVIDIA Developer](https://developer.nvidia.com/rdp/cudnn-archive)

4. Follow the steps here from NVIDIA to install cuDNN. Skip step 5 on updating visual studio settings, this is only for C++ projects.
[Installation Guide - NVIDIA Deep Learning cuDNN Documentation](https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html#install-windows)

5. Restart your computer as prompted by the NVIDIA installer.

6. You will need to move the `zlib` DLL and rename it to `zlibwapi` in the CUDA bin directory.
    - Copy the `zlib` DLL from `C:\Program Files\NVIDIA Corporation\Nsight Systems 2022.4.2\host-windows-x64\zlib.dll`
    - Paste and rename `zlibwapi` to `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\bin\zlibwapi.dll`

7. Now you can enable GPU in the C# ONNX Runtime API with the following code:
```cs
var session = new InferenceSession(modelPath, SessionOptions.MakeSessionOptionWithCudaProvider(0));
```

## Checkout more C# ONNX Runtime resources
- [C# API Doc](https://onnxruntime.ai/docs/api/csharp/api)
- [Get Started with C# in ONNX Runtime](https://onnxruntime.ai/docs/get-started/with-csharp.html)

