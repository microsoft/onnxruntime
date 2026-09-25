---
title: Windows
parent: Get Started
toc: true
description: Get started with ONNX Runtime and Windows ML on Windows.
nav_order: 9
---

# Get started with ONNX Runtime for Windows
{: .no_toc }

**Windows ML is the recommended Windows development path for ONNX Runtime.** It combines ONNX Runtime APIs with Windows APIs that discover, install, and register execution providers for the device.

## Contents
{: .no_toc }

* TOC placeholder
{:toc}

## Requirements

Windows ML apps can target any Windows version supported by the Windows App SDK on x64 and Arm64. Installing hardware-optimized execution providers through the Windows ML execution provider catalog requires Windows 11, version 24H2 (build 26100) or later.

Language-specific requirements are listed in the [Windows ML getting-started guide](https://learn.microsoft.com/windows/ai/new-windows-ml/get-started).

## Install Windows ML

Windows ML supports framework-dependent and self-contained deployment. For a framework-dependent C# or C++/WinRT app, reference the Windows App SDK:

```bash
dotnet add package Microsoft.WindowsAppSDK
```

For self-contained deployment, C/C++, or Python, follow the [Windows ML installation and deployment guide](https://learn.microsoft.com/windows/ai/new-windows-ml/distributing-your-app). It lists the required `Microsoft.WindowsAppSDK.ML`, `Microsoft.Windows.AI.MachineLearning`, runtime, and Python packages for each deployment mode.

## Use ONNX Runtime APIs

- C# applications use the `Microsoft.ML.OnnxRuntime` namespace supplied by Windows ML.
- C++ applications use the ONNX Runtime C API after Windows ML registers the selected execution providers.
- Python applications use the `onnxruntime-windowsml` wheel.

See the [Windows ML API reference](https://learn.microsoft.com/windows/ai/new-windows-ml/api-reference) and [run an ONNX model](https://learn.microsoft.com/windows/ai/new-windows-ml/run-onnx-models) guide for current examples.

## Other ONNX Runtime options on Windows

Use the standalone [ONNX Runtime packages](../install/#cccwinml-installs) when you need a cross-platform API or want to manage execution providers directly. [DirectML](../execution-providers/DirectML-ExecutionProvider.md) remains supported in sustained engineering mode, but Windows ML is recommended for new Windows applications.
