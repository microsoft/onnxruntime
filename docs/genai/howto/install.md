---
title: Install
description: Instructions to install ONNX Runtime generate() API on your target platform in your environment
has_children: false
parent: How to
grand_parent: Generate API (Preview)
nav_order: 1
---

# Install ONNX Runtime generate() API
{: .no_toc }

* TOC placeholder
{:toc}

## Pre-requisites

### CUDA

If you are installing the CUDA variant of onnxruntime-genai, the CUDA toolkit must be installed.

The CUDA toolkit can be downloaded from the [CUDA Toolkit Archive](https://developer.nvidia.com/cuda-toolkit-archive).

Ensure that the `CUDA_PATH` environment variable is set to the location of your CUDA installation.

## Python packages

Note: only one of these packages should be installed in your application.

### CPU

```bash
pip install numpy
pip install onnxruntime-genai --pre
```

### DirectML

Append `-directml` for the library that is optimized for DirectML on Windows

```bash
pip install numpy
pip install onnxruntime-genai-directml --pre
```

### CUDA

Append `-cuda` for the library that is optimized for CUDA environments

#### CUDA 11

```bash
pip install numpy
pip install onnxruntime-genai-cuda --pre --index-url=https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-genai/pypi/simple/
```

#### CUDA 12

```bash
pip install numpy
pip install onnxruntime-genai-cuda --pre --index-url=https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-12/pypi/simple/
```

## Nuget packages

Note: only one of these packages should be installed in your application.

```bash
dotnet add package Microsoft.ML.OnnxRuntimeGenAI --prerelease
```

For the package that has been optimized for CUDA:

```bash
dotnet add package Microsoft.ML.OnnxRuntimeGenAI.Cuda --prerelease
```

For the package that has been optimized for DirectML:

```bash
dotnet add package Microsoft.ML.OnnxRuntimeGenAI.DirectML --prerelease
```





   

