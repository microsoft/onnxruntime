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

ONNX Runtime generate() versions 0.3.0 and earlier came bundled with the core ONNX Runtime binaries. From version 0.4.0 onwards, the packages are separated to allow a better developer experience. 

Platform specific instructions to install ONNX Runtime are included in each section below.

### CUDA

If you are installing the CUDA variant of onnxruntime-genai, the CUDA toolkit must be installed.

The CUDA toolkit can be downloaded from the [CUDA Toolkit Archive](https://developer.nvidia.com/cuda-toolkit-archive).

Ensure that the `CUDA_PATH` environment variable is set to the location of your CUDA installation.


## Python packages

Note: only one of these sets of packages (CPU, DirectML, CUDA) should be installed in your environment.

### CPU

#### x64

```bash
# Install ONNX Runtime nightly. Update this to the released version when it is available.
pip install ort-nightly --extra-index-url
https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/ort-nightly/pypi/simple


pip install onnxruntime-genai --pre
```

#### Arm64

```bash
# Install ONNX Runtime nightly QNN version (which supports Arm CPU). Update this to the released version when it is available.
pip install ort-nightly-qnn --extra-index-url
https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/ort-nightly/pypi/simple

pip install onnxruntime-genai --pre
```


### DirectML

```bash
# Install the nightly version of ONNX Runtime. Update this to the released version when it is available.
pip install ort-nightly-directml --extra-index-url
https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/ort-nightly/pypi/simple

pip install onnxruntime-genai-directml --pre
```

### CUDA

#### CUDA 11

```bash
# Install the nightly version of ONNX Runtime. Update this to the released version when it is available.
pip install ort-nightly-gpu --extra-index-url
https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/ort-cuda-11-nightly/pypi/simple


# Install onnxruntime-genai built for CUDA 11
pip install onnxruntime-genai-cuda-11 --pre --extra-index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-genai/pypi/simple/
```

#### CUDA 12

```bash
# Install ORT nightly CUDA 12, change this to released version when it is released
pip install onnxruntime-gpu --extra-index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-12/pypi/simple/

# Install onnxruntime-genai build for CUDA 12
pip install onnxruntime-genai-cuda --pre --extra-index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-genai/pypi/simple/
```

## Nuget packages

Note: only one of these set of packages (CPU, DirectML, CUDA) should be installed in your project.

### CPU

Add the following lines to a `nuget.config` file in the same folder as your `.csproj` file.

```xml
<?xml version="1.0" encoding="utf-8"?>
<configuration>
  <packageSources>
    <clear />
    <add key="ORT-Nightly" value="https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/ORT-Nightly/nuget/v3/index.json" />
  </packageSources>
</configuration>
```

```bash
dotnet add package Microsoft.ML.OnnxRuntime --source ORT-Nightly --prerelease
dotnet add package Microsoft.ML.OnnxRuntimeGenAI --prerelease
```

### CUDA 

#### CUDA 11

Add the following lines to a `nuget.config` file in the same folder as your `.csproj` file.

```xml
<?xml version="1.0" encoding="utf-8"?>
<configuration>
  <packageSources>
    <clear />
    <add key="ORT-Nightly" value="https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/ORT-Nightly/nuget/v3/index.json" />
  </packageSources>
</configuration>
```

```bash
dotnet add package Microsoft.ML.OnnxRuntime.Gpu --source ORT-Nightly --prerelease
dotnet add package Microsoft.ML.OnnxRuntimeGenAI.Cuda --source --prerelease
```

#### CUDA 12

Add the following lines to a `nuget.config` file in the same folder as your `.csproj` file.

```xml
<?xml version="1.0" encoding="utf-8"?>
<configuration>
  <packageSources>
    <clear />
    <add key="ORT-Nightly" value="https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/ORT-Nightly/nuget/v3/index.json" />
  </packageSources>
</configuration>
```


```bash
dotnet add package Microsoft.ML.OnnxRuntime.Gpu --source ORT-Nightly --prerelease
dotnet add package Microsoft.ML.OnnxRuntimeGenAI.Cuda --prerelease
```

### DirectML

Add the following lines to a `nuget.config` file in the same folder as your `.csproj` file.

```xml
<?xml version="1.0" encoding="utf-8"?>
<configuration>
  <packageSources>
    <clear />
    <add key="ORT-Nightly" value="https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/ORT-Nightly/nuget/v3/index.json" />
  </packageSources>
</configuration>
```

```bash
dotnet add package Microsoft.ML.OnnxRuntime.DirectML --source ORT-Nightly --prerelease
dotnet add package Microsoft.ML.OnnxRuntimeGenAI.DirectML --prerelease
```





   

