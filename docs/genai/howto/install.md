---
title: Install
description: Instructions to install ONNX Runtime GenAI on your target platform in your environment
has_children: false
parent: How to
grand_parent: Generate API (Preview)
nav_order: 1
---

# Install ONNX Runtime GenAI
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
pip install onnxruntime-genai-cuda --pre --index-url=https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-genai/pypi/simple/
```

## Nuget packages




To run with CUDA, use the following packages instead:

- `Microsoft.ML.OnnxRuntimeGenAI.Cuda`
- `Microsoft.ML.OnnxRuntime.Gpu`


   

