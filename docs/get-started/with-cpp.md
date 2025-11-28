Update with-python.md---
title: C++
parent: Get Started
nav_order: 2
---

# Get started with ORT for C++
{: .no_toc }

## Contents
{: .no_toc }

* TOC placeholder
{:toc}

## Builds

| Artifact  | Description | Supported Platforms |
|-----------|-------------|---------------------|
| [Microsoft.ML.OnnxRuntime](https://www.nuget.org/packages/Microsoft.ML.OnnxRuntime) | CPU (Release) |Windows, Linux,  Mac, X64, X86 (Windows-only), ARM64 (Windows-only)...more details: [compatibility](../reference/compatibility.md) |
| [Microsoft.ML.OnnxRuntime.Gpu](https://www.nuget.org/packages/Microsoft.ML.OnnxRuntime.gpu) | GPU - CUDA (Release) | Windows, Linux, Mac, X64...more details: [compatibility](../reference/compatibility.md) |
| [Microsoft.ML.OnnxRuntime.DirectML](https://www.nuget.org/packages/Microsoft.ML.OnnxRuntime.directml) | GPU - DirectML (Release) | Windows 10 1709+ |
| [onnxruntime](https://aiinfra.visualstudio.com/PublicPackages/_packaging?_a=feed&feed=ORT-Nightly) | CPU, GPU (Dev), CPU (On-Device Training) | Same as Release versions |
| [Microsoft.ML.OnnxRuntime.Training](https://www.nuget.org/packages/Microsoft.ML.OnnxRuntime) | CPU On-Device Training (Release) |Windows, Linux,  Mac, X64, X86 (Windows-only), ARM64 (Windows-only)...more details: [compatibility](../reference/compatibility.md) |
| [Microsoft.ML.OnnxRuntime.ROCm](https://github.com/Looong01/onnxruntime-rocm-build/releases) | GPU - ROCm (External) | Linux, X64...more details: [compatibility](../reference/compatibility.md) |

.zip and .tgz files are also included as assets in each [Github release](https://github.com/microsoft/onnxruntime/releases).

## API Reference
The C++ API is a thin wrapper of the C API. Please refer to [C API](./with-c.md) for more details.

## Samples
See [Tutorials: API Basics - C++](../tutorials/api-basics)
