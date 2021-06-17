---
title: C++ API
parent: API docs
grand_parent: Reference
---

# ONNX Runtime C++ API
{: .no_toc }

## Contents
{: .no_toc }

* TOC placeholder
{:toc}

## Builds

| Artifact  | Description | Supported Platforms |
|-----------|-------------|---------------------|
| [Microsoft.ML.OnnxRuntime](https://www.nuget.org/packages/Microsoft.ML.OnnxRuntime) | CPU (Release) |Windows, Linux,  Mac, X64, X86 (Windows-only), ARM64 (Windows-only)...more details: [compatibility](../../resources/compatibility.md) |
| [Microsoft.ML.OnnxRuntime.Gpu](https://www.nuget.org/packages/Microsoft.ML.OnnxRuntime.gpu) | GPU - CUDA (Release) | Windows, Linux, Mac, X64...more details: [compatibility](../../resources/compatibility.md) |
| [Microsoft.ML.OnnxRuntime.DirectML](https://www.nuget.org/packages/Microsoft.ML.OnnxRuntime.directml) | GPU - DirectML (Release) | Windows 10 1709+ |
| [ort-nightly](https://aiinfra.visualstudio.com/PublicPackages/_packaging?_a=feed&feed=ORT-Nightly) | CPU, GPU (Dev) | Same as Release versions |

.zip and .tgz files are also included as assets in each [Github release](https://github.com/microsoft/onnxruntime/releases).

## API Reference
The C++ API is a thin wrapper of the C API. Please refer to [C API](./c-api.html) for more details.

## Samples
See [Tutorials: API Basics - C++](../../tutorials/inferencing/api-basics.md#c)