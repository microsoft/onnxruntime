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

## Python packages

```bash
pip install numpy
pip install onnxruntime-genai --pre
```
Append `-directml` for the library that is optimized for DirectML on Windows

```bash
pip install numpy
pip install onnxruntime-genai-directml --pre
```

Append `-cuda` for the library that is optimized for CUDA environments

```bash
pip install numpy
pip install onnxruntime-genai-cuda --pre --index-url=https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-genai/pypi/simple/
```

## Nuget packages

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





   

