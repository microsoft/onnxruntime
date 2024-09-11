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


## Python package installation

Note: only one of these sets of packages (CPU, DirectML, CUDA) should be installed in your environment.

### CPU

```bash
pip install onnxruntime-genai
```

### DirectML

```bash
pip install onnxruntime-genai-directml
```

### CUDA

If you are installing the CUDA variant of onnxruntime-genai, the CUDA toolkit must be installed.

The CUDA toolkit can be downloaded from the [CUDA Toolkit Archive](https://developer.nvidia.com/cuda-toolkit-archive).

Ensure that the `CUDA_PATH` environment variable is set to the location of your CUDA installation.

#### CUDA 11

```bash
pip install onnxruntime-genai-cuda --index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-11/pypi/simple/
```

#### CUDA 12

```bash
pip install onnxruntime-genai-cuda
```


## Nuget package installation

Note: install only one of these packages (CPU, DirectML, CUDA) in your project.

### Pre-requisites

#### ONNX Runtime dependency

ONNX Runtime generate() versions 0.3.0 and earlier came bundled with the core ONNX Runtime binaries. From version 0.4.0 onwards, the packages are separated to allow a more flexible developer experience.

### CPU

```bash
dotnet add package Microsoft.ML.OnnxRuntimeGenAI
```

### CUDA 

Note: only CUDA 11 is supported for versions 0.3.0 and earlier, and only CUDA 12 is supported for versions 0.4.0 and later.

```bash
dotnet add package Microsoft.ML.OnnxRuntimeGenAI.Cuda
```

### DirectML

```bash
dotnet add package Microsoft.ML.OnnxRuntimeGenAI.DirectML
```





   

