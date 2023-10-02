---
title: Windows - DirectML
description: Instructions to execute ONNX Runtime with the DirectML execution provider
parent: Execution Providers
nav_order: 5
redirect_from: /docs/reference/execution-providers/DirectML-ExecutionProvider
---

# DirectML Execution Provider
{: .no_toc }

The DirectML Execution Provider is a component of ONNX Runtime that uses [DirectML](https://docs.microsoft.com/en-us/windows/ai/directml/dml-intro) to accelerate inference of ONNX models. The DirectML execution provider is capable of greatly improving evaluation time of models using commodity GPU hardware, without sacrificing broad hardware support or requiring vendor-specific extensions to be installed.


DirectML is a high-performance, hardware-accelerated DirectX 12 library for machine learning on Windows.  DirectML provides GPU acceleration for common machine learning tasks across a broad range of supported hardware and drivers.

When used standalone, the DirectML API is a low-level DirectX 12 library and is suitable for high-performance, low-latency applications such as frameworks, games, and other real-time applications. The seamless interoperability of DirectML with Direct3D 12 as well as its low overhead and conformance across hardware makes DirectML ideal for accelerating machine learning when both high performance is desired, and the reliability and predictability of results across hardware is critical.

The DirectML Execution Provider currently uses DirectML version [1.12.1](https://www.nuget.org/packages/Microsoft.AI.DirectML/1.12.1) and supports up to ONNX opset 17 ([ONNX v1.12](https://github.com/onnx/onnx/releases/tag/v1.12.0)). Evaluating models which require a higher opset version is unsupported and will yield poor performance.

## Contents
{: .no_toc }

* TOC placeholder
{:toc}

## Install

Pre-built packages of ORT with the DirectML EP is published on Nuget.org. See: [Install ONNX Runtime](../install#cccwinml-installs).

## Requirements

The DirectML execution provider requires a DirectX 12 capable device. Almost all commercially-available graphics cards released in the last several years support DirectX 12. Here are some examples of compatible hardware:

* NVIDIA Kepler (GTX 600 series) and above
* AMD GCN 1st Gen (Radeon HD 7000 series) and above
* Intel Haswell (4th-gen core) HD Integrated Graphics and above
* Qualcomm Adreno 600 and above

DirectML was introduced in Windows 10, version 1903, and in the corresponding version of the [Windows SDK](https://docs.microsoft.com/en-us/windows/ai/directml/dml).

## Build

Requirements for building the DirectML execution provider:

1. Visual Studio 2017 toolchain
2. [The Windows 10 SDK (10.0.18362.0) for Windows 10, version 1903](https://developer.microsoft.com/en-us/windows/downloads/windows-10-sdk) (or newer)

To build onnxruntime with the DML EP included, supply the `--use_dml` flag to `build.bat`. 
For example:

```powershell
    build.bat --config RelWithDebInfo --build_shared_lib --parallel --use_dml
```

The DirectML execution provider supports building for both x64 (default) and x86 architectures.

Note that, you can build [ONNX Runtime with DirectML](https://docs.microsoft.com/en-us/windows/ai/windows-ml/onnx-versions). This allows DirectML re-distributable package download automatically as part of the build. Find additional license information on the NuGet docs.


## Usage

When using the [C API](../get-started/with-c.md) with a DML-enabled build of onnxruntime, the DirectML execution provider can be enabled using one of the two factory functions included in `include/onnxruntime/core/providers/dml/dml_provider_factory.h`.

### `OrtSessionOptionsAppendExecutionProvider_DML` function
{: .no_toc }

 Creates a DirectML Execution Provider which executes on the hardware adapter with the given `device_id`, also known as the adapter index. The device ID corresponds to the enumeration order of hardware adapters as given by [IDXGIFactory::EnumAdapters](https://docs.microsoft.com/windows/win32/api/dxgi/nf-dxgi-idxgifactory-enumadapters). A `device_id` of 0 always corresponds to the default adapter, which is typically the primary display GPU installed on the system. Beware that in systems with multiple GPU's, the primary display (GPU 0) is often not the most performant one, particularly on laptops with dual adapters where battery lifetime is preferred over performance. So you can double check in Task Manager's performance tab to see which GPU is which. A negative `device_id` is invalid.

Example for C API:
```c
OrtStatus* OrtSessionOptionsAppendExecutionProvider_DML(
    _In_ OrtSessionOptions* options,
    int device_id
    );
```

Example for C# API:

Install the Nuget Package [Microsoft.ML.OnnxRuntime.DirectML](https://www.nuget.org/packages/Microsoft.ML.OnnxRuntime.DirectML/1.14.1) and use the following code to enable the DirectML EP:

```csharp
SessionOptions sessionOptions = newSessionOptions();
sessionOptions.GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL;
sessionOptions.AppendExecutionProvider_DML(0);
```

### `OrtSessionOptionsAppendExecutionProviderEx_DML` function
{: .no_toc }

Creates a DirectML Execution Provider using the given DirectML device, and which executes work on the supplied D3D12 command queue. The DirectML device and D3D12 command queue must have the same parent [ID3D12Device](https://docs.microsoft.com/windows/win32/api/d3d12/nn-d3d12-id3d12device), or an error will be returned. The D3D12 command queue must be of type `DIRECT` or `COMPUTE` (see [D3D12_COMMAND_LIST_TYPE](https://docs.microsoft.com/windows/win32/api/d3d12/ne-d3d12-d3d12_command_list_type)). If this function succeeds, the inference session once created will maintain a strong reference on both the `dml_device` and `command_queue` objects.

```c
OrtStatus* OrtSessionOptionsAppendExecutionProviderEx_DML(
    _In_ OrtSessionOptions* options,
    _In_ IDMLDevice* dml_device,
    _In_ ID3D12CommandQueue* cmd_queue
    );
```


## Configuration Options

The DirectML execution provider does not support the use of memory pattern optimizations or parallel execution in onnxruntime. When supplying session options during InferenceSession creation, these options must be disabled or an error will be returned.

If using the onnxruntime C API, you must call `DisableMemPattern` and `SetSessionExecutionMode` functions to set the options required by the DirectML execution provider.

See [onnxruntime\include\onnxruntime\core\session\onnxruntime_c_api.h](https://github.com/microsoft/onnxruntime/tree/main/include//onnxruntime/core/session/onnxruntime_c_api.h).

    OrtStatus*(ORT_API_CALL* DisableMemPattern)(_Inout_ OrtSessionOptions* options)NO_EXCEPTION;

    OrtStatus*(ORT_API_CALL* SetSessionExecutionMode)(_Inout_ OrtSessionOptions* options, ExecutionMode execution_mode)NO_EXCEPTION;

If creating the onnxruntime InferenceSession object directly, you must set the appropriate fields on the `onnxruntime::SessionOptions` struct. Specifically, `execution_mode` must be set to `ExecutionMode::ORT_SEQUENTIAL`, and `enable_mem_pattern` must be `false`.

Additionally, as the DirectML execution provider does not support parallel execution, it does not support multi-threaded calls to `Run` on the same inference session. That is, if an inference session using the DirectML execution provider, only one thread may call `Run` at a time. Multiple threads are permitted to call `Run` simultaneously if they operate on different inference session objects.

## Performance Tuning

The DirectML execution provider works most efficiently when tensor shapes are known at the time a session is created.  Here are a few performance benefits:

1. Constant folding can occur more often, reducing the CPU / GPU copies and stalls during evaluations.
2. More initialization work occurs when sessions are created rather than during the first evaluation.
3. Weights may be pre-processed within DirectML, enabling more efficient algorithms.
4. Graph optimization occurs within DirectML. For example, Concat operators may be removed and more optimal tensor layouts may be used for the input and output of operators.

Normally, when the shapes of model inputs are known during a session creation, the shapes for the rest of the model are inferred by OnnxRuntime when that session is created.  

However, if a model input contains a free dimension (such as for batch size), additional steps must be taken to retain the above performance benefits. These include:

1. Edit the model to replace an input's free dimension (specified through ONNX using "dim_param") with a fixed size (specified through ONNX using "dim_value").
2. Specify values of named dimensions within model inputs when creating the session using the OnnxRuntime *AddFreeDimensionOverrideByName* ABI.
3. Edit the model to ensure that an input's free dimension has a [denotation](https://github.com/onnx/onnx/blob/master/docs/DimensionDenotation.md) (such as "DATA_BATCH," or a custom denotation).  Then, when creating the session, specify the dimension size for each denotation. This can be done using the OnnxRuntime *AddFreeDimensionOverride* ABI.

## Samples

A complete sample of onnxruntime using the DirectML execution provider can be found under [samples/c_cxx/fns_candy_style_transfer](https://github.com/microsoft/onnxruntime-inference-examples/tree/main/c_cxx/fns_candy_style_transfer)


## Additional Resources

* [DirectML documentation \(docs.microsoft.com\)](https://docs.microsoft.com/en-us/windows/win32/direct3d12/dml)
* [DMLCreateDevice function](https://docs.microsoft.com/windows/win32/api/directml/nf-directml-dmlcreatedevice)
* [ID3D12Device::CreateCommandQueue method](https://docs.microsoft.com/windows/win32/api/d3d12/nf-d3d12-id3d12device-createcommandqueue)
* [Direct3D 12 programming guide](https://docs.microsoft.com/windows/win32/direct3d12/directx-12-programming-guide)
* [ONNX versions and Windows builds](https://docs.microsoft.com/en-us/windows/ai/windows-ml/onnx-versions)
* [Windows Machine Learning](https://docs.microsoft.com/en-us/windows/ai/windows-ml/)
* [ONNX models](https://docs.microsoft.com/en-us/windows/ai/windows-ml/get-onnx-model)
* [DirectML Github](https://github.com/microsoft/DirectML)

<p><a href="#">Back to top</a></p>
