---
title: Install
description: Instructions to install ONNX Runtime GenAI on your target platform in your environment
has_children: false
parent: How to
grand_parent: Generative AI (Preview)
nav_order: 1
---

# Install ONNX Runtime GenAI
{: .no_toc }

* TOC placeholder
{:toc}

## Python package release candidates

```bash
pip install numpy
pip install onnxruntime-genai --pre --index-url=
https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-genai/pypi/simple/`
```

Append `-cuda` for the library that is optimized for CUDA environments

```bash
pip install onnxruntime-genai-cuda --pre --index-url=
https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-genai/pypi/simple/`

```

## Nuget package release candidates

To install the NuGet release candidates, add a new package source in Visual Studio, go to `Project` -> `Manage NuGet Packages`.

1. Click on the `Settings` cog icon

2. Click the `+` button to add a new package source

   - Change the Name to `onnxruntime-genai`
   - Change the Source to `https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-genai/nuget/v3/index.json`

3. Check the `Include prerelease` button

4. Add the `Microsoft.ML.OnnxRuntimeGenAI` package

5. Add the `Microsoft.ML.OnnxRuntime` package

To run with CUDA, use the following packages instead:

- `Microsoft.ML.OnnxRuntimeGenAI.Cuda`
- `Microsoft.ML.OnnxRuntime.Gpu`


   

