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

### ONNX Runtime

Previous versions of ONNX Runtime generate() came bundled with the core ONNX Runtime binaries. From version 0.4.0 onwards, the packages are separated to allow a better developer experience. Specific platform instructions are included in each section below.

### CUDA

If you are installing the CUDA variant of onnxruntime-genai, the CUDA toolkit must be installed.

The CUDA toolkit can be downloaded from the [CUDA Toolkit Archive](https://developer.nvidia.com/cuda-toolkit-archive).

Ensure that the `CUDA_PATH` environment variable is set to the location of your CUDA installation.

Versions later than ONNX Runtime release bundle CUDA 12 by default. CUDA 11 is also supported in a separate package. Instructions are given below.

## Python packages

Note: only one of these set of packages (CPU, DirectML, CUDA) should be installed in your environment.

### CPU

#### Intel CPU

```bash
pip install onnxruntime
pip install onnxruntime-genai --pre
```

#### Arm CPU

```bash
pip install onnxruntime-qnn
pip install onnxruntime-genai --pre
```


### DirectML

```bash
pip install numpy
pip install onnxruntime-directml
pip install onnxruntime-genai-directml --pre
```

### CUDA

#### CUDA 11

```bash
# Install ORT nightly CUDA 11, change this to released version when it is released
pip install ort-nightly-gpu --index-url
https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/ort-cuda-11-nightly/pypi/simple


# Install onnxruntime-genai built for CUDA 11
pip install onnxruntime-genai-cuda-11 --pre --index-url=https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-genai/pypi/simple/
```

#### CUDA 12

```bash
# Install ORT nightly CUDA 12, change this to released version when it is released
pip install onnxruntime-gpu --extra-index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-12/pypi/simple/

# Install onnxruntime-genai build for CUDA 12
pip install onnxruntime-genai-cuda --pre --index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-genai/pypi/simple/
```

## Nuget packages

Note: only one of these set of packages (CPU, DirectML, CUDA) should be installed in your application.

### CPU

```bash
dotnet add package Microsoft.ML.OnnxRuntime
dotnet add package Microsoft.ML.OnnxRuntimeGenAI --prerelease
```

### CUDA 

#### CUDA 11

Add the following lines to a `nuget.config` file in the same folder as your `.csproj` file.

```xml
<?xml version="1.0" encoding="utf-8"?>
<configuration>
    <packageSources>
        <clear/>
        <add key="onnxruntime-cuda-11"
             value="https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-11/nuget/v3/index.json"/>
    </packageSources>
</configuration>
```

```bash
dotnet add package Microsoft.ML.OnnxRuntime.Gpu
dotnet add package Microsoft.ML.OnnxRuntimeGenAI.Cuda --prerelease
```

#### CUDA 12

Add the following lines to a `nuget.config` file in the same folder as your `.csproj` file.

```xml
<?xml version="1.0" encoding="utf-8"?>
<configuration>
    <packageSources>
        <clear/>
        <add key="onnxruntime-cuda-12"
             value="https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-12/nuget/v3/index.json"/>
    </packageSources>
</configuration>
```


```bash
dotnet add package Microsoft.ML.OnnxRuntime.Gpu
dotnet add package Microsoft.ML.OnnxRuntimeGenAI.Cuda --prerelease
```

### DirectML

For the package that has been optimized for DirectML:

```bash
dotnet add package Microsoft.ML.OnnxRuntime.DirectML
dotnet add package Microsoft.ML.OnnxRuntimeGenAI.DirectML --prerelease
```





   

