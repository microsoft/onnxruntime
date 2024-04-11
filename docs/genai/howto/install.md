---
title: Install
description: Instructions to install ONNX Runtime generate() API on your target platform in your environment
has_children: false
parent: How to
grand_parent: Generative AI (Preview)
nav_order: 1
---

# Install ONNX Runtime GenAI
{: .no_toc }

* TOC placeholder
{:toc}

## Python package

```bash
pip install numpy
pip install onnxruntime-genai --pre --index-url=https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-genai/pypi/simple/
```

Append `-cuda` for the library that is optimized for CUDA environments

```bash
pip install onnxruntime-genai-cuda --pre --index-url=https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-genai/pypi/simple/
```

## Nuget package

Add the `Microsoft.ML.OnnxRuntimeGenAI` package

Add the `Microsoft.ML.OnnxRuntime` package

To run with CUDA, use the following packages instead:

- `Microsoft.ML.OnnxRuntimeGenAI.Cuda`
- `Microsoft.ML.OnnxRuntime.Gpu`

## C/C++ binaries

_Coming soon_
   

