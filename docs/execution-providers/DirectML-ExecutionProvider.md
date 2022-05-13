---
title: DirectML
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

The DirectML Execution Provider currently uses DirectML version [1.8.0](https://www.nuget.org/packages/Microsoft.AI.DirectML/1.8.0) and supports up to ONNX opset 12 ([ONNX v1.7](https://github.com/onnx/onnx/releases/tag/v1.7.0)). Evaluating models which require a higher opset version is unsupported and will yield poor performance.

## Contents
{: .no_toc }

* TOC placeholder
{:toc}


## Windows OS integration

ONNX Runtime is available in Windows 10 versions >= 1809 and all versions of Windows 11. It is embedded inside Windows.AI.MachineLearning.dll and exposed via the WinRT API (WinML for short). It includes the CPU execution provider and the [DirectML execution provider](../execution-providers/DirectML-ExecutionProvider) for GPU support.

The high level design looks like this:

![ONNX + WinML layered architecture](../../images/layered-architecture.png)

### API choice
{: .no_toc }

You can choose to use either the WinRT API or the C API.

||WinRT|C API|
|--|--|--|
|Type system| Integration with Windows RT types| Platform neutral types|
|Language support| Language support via WinRT Projections| Language support via per language projections|
|Tensorization| Accepts VideoFrames and converts to tensors (support for CPU and GPU)| Accepts tensors|

### Using the NuGet WinRT API with other C-API distributions
{: .no_toc }

The WinRT API NuGet package is distributed with a specific version of ONNX Runtime, but apps can include their own version of ONNX Runtime (either a [released version](../install/#cccwinml-installs) or [a custom build](../build/)). You may wish to do this to use non-default execution providers.
To use your own version of ONNX Runtime, replace onnxruntime.dll with your desired version.

<p><a href="#">Back to top</a></p>


## Install
Pre-built packages of ORT with the DirectML EP is published on Nuget.org. See: [Install ORT](../install#cccwinml-installs).

## Requirements

The [DirectML](https://github.com/microsoft/DirectML) execution provider requires a [DirectX 12](https://answers.microsoft.com/en-us/windows/forum/all/direct-x-120-download/98d1137a-653d-40eb-9863-3b474665add7) capable device. Almost all commercially-available graphics cards released in the last several years support DirectX 12. Here are some examples of compatible hardware:

* NVIDIA Kepler (GTX 600 series) and above
* AMD GCN 1st Gen (Radeon HD 7000 series) and above
* Intel Haswell (4th-gen core) HD Integrated Graphics and above
* Qualcomm Adreno 600 and above

[DirectML](https://docs.microsoft.com/en-us/windows/ai/directml/dml) is introduced in Windows 10, version 1903, and in the corresponding version of the Windows SDK.

<p><a href="#">Back to top</a></p>

## Build

Requirements for building the DirectML execution provider:
1. Visual Studio 2017 toolchain
2. [The Windows 10 SDK (10.0.18362.0) for Windows 10, version 1903](https://developer.microsoft.com/en-us/windows/downloads/windows-10-sdk) (or newer)

To build onnxruntime with the DML EP included, supply the `--use_dml` parameter to `build.bat`. 
For example:

```powershell
    build.bat --config RelWithDebInfo --build_shared_lib --parallel --use_dml
```

The DirectML execution provider supports building for both x64 (default) and x86 architectures.

Note that, you can build [ONNX Runtime with DirectML](https://docs.microsoft.com/en-us/windows/ai/windows-ml/onnx-versions). This allows DirectML re-distributable package download automatically as part of the build. Find additional license information on the NuGet docs.


<p><a href="#">Back to top</a></p>

## Usage

When using the [C API](../get-started/with-c.md) with a DML-enabled build of onnxruntime, the DirectML execution provider can be enabled using one of the two factory functions included in `include/onnxruntime/core/providers/dml/dml_provider_factory.h`.

### `OrtSessionOptionsAppendExecutionProvider_DML` function
{: .no_toc }

 Creates a DirectML Execution Provider which executes on the hardware adapter with the given `device_id`, also known as the adapter index. The device ID corresponds to the enumeration order of hardware adapters as given by [IDXGIFactory::EnumAdapters](https://docs.microsoft.com/windows/win32/api/dxgi/nf-dxgi-idxgifactory-enumadapters). A `device_id` of 0 always corresponds to the default adapter, which is typically the primary display GPU installed on the system. Beware that in systems with multiple GPU's, the primary display (GPU 0) is often not the most performant one, particularly on laptops with dual adapters where battery lifetime is preferred over performance. So you can double check in Task Manager's performance tab to see which GPU is which. A negative `device_id` is invalid.

```c
OrtStatus* OrtSessionOptionsAppendExecutionProvider_DML(
    _In_ OrtSessionOptions* options,
    int device_id
    );
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

<p><a href="#">Back to top</a></p>

## Configuration Options

The DirectML execution provider does not support the use of memory pattern optimizations or parallel execution in onnxruntime. When supplying session options during InferenceSession creation, these options must be disabled or an error will be returned.

If using the onnxruntime C API, you must call `DisableMemPattern` and `SetSessionExecutionMode` functions to set the options required by the DirectML execution provider.

See [onnxruntime\include\onnxruntime\core\session\onnxruntime_c_api.h](https://github.com/microsoft/onnxruntime/tree/master/include//onnxruntime/core/session/onnxruntime_c_api.h).

    OrtStatus*(ORT_API_CALL* DisableMemPattern)(_Inout_ OrtSessionOptions* options)NO_EXCEPTION;

    OrtStatus*(ORT_API_CALL* SetSessionExecutionMode)(_Inout_ OrtSessionOptions* options, ExecutionMode execution_mode)NO_EXCEPTION;

If creating the onnxruntime InferenceSession object directly, you must set the appropriate fields on the `onnxruntime::SessionOptions` struct. Specifically, `execution_mode` must be set to `ExecutionMode::ORT_SEQUENTIAL`, and `enable_mem_pattern` must be `false`.

Additionally, as the DirectML execution provider does not support parallel execution, it does not support multi-threaded calls to `Run` on the same inference session. That is, if an inference session using the DirectML execution provider, only one thread may call `Run` at a time. Multiple threads are permitted to call `Run` simultaneously if they operate on different inference session objects.

<p><a href="#">Back to top</a></p>

## Performance Tuning

The DirectML execution provider works most efficiently when tensor shapes are known at the time a session is created.  Here are a few performance benefits:

1. Constant folding can occur more often, there by reducing the CPU / GPU copies and stalls during evaluations.
2. More initialization work occurs when sessions are created rather than during the first evaluation.
3. Weights may be pre-processed within DirectML, enabling more efficient algorithms.
4. Graph optimization occurs within DirectML. For example, Concat operators may be removed and more optimal tensor layouts may be used for the input and output of operators.

Normally, when the shapes of model inputs are known during a session creation, the shapes for the rest of the model are inferred by OnnxRuntime when that session is created.  

However, if a model input contains a free dimension (such as for batch size), additional steps must be taken to retain the above performance benefits. These include:

1. Edit the model to replace an input's free dimension (specified through ONNX using "dim_param") with a fixed size (specified through ONNX using "dim_value").
2. Specify values of named dimensions within model inputs when creating the session using the OnnxRuntime *AddFreeDimensionOverrideByName* ABI.
3. Edit the model to ensure that an input's free dimension has a [denotation](https://github.com/onnx/onnx/blob/master/docs/DimensionDenotation.md) (such as "DATA_BATCH," or a custom denotation).  Then, when creating the session, specify the dimension size for each denotation. This can be done using the OnnxRuntime *AddFreeDimensionOverride* ABI.

<p><a href="#">Back to top</a></p>

## Samples

DirectML and Onnx Runtime can be used to achieve many models. Here are some examples given on [Github](https://github.com/microsoft/DirectML#directml-samples):

- HelloDirectML: A minimal "hello world" application that executes a single DirectML operator.
- DirectMLSuperResolution: A sample that uses DirectML to execute a basic super-resolution model to upscale video from 540p to 1080p in real time.
- yolov4: YOLOv4 is an object detection model capable of recognizing up to 80 different classes of objects in an image. This sample contains a complete end-to-end implementation of the model using DirectML, and is able to run in real time on a user-provided video stream.
- MobileNet: Adapted from the ONNX MobileNet model. MobileNet classifies an image into 1000 different classes. It is highly efficient in speed and size, ideal for mobile applications.
- MNIST: Adapted from the ONNX MNIST model. MNIST predicts handwritten digits using a convolution neural network.
- SqueezeNet: Based on the ONNX SqueezeNet model. SqueezeNet performs image classification trained on the ImageNet dataset. It is highly efficient and provides results with good accuracy.
- FNS-Candy: Adapted from the Windows ML Style Transfer model sample, FNS-Candy re-applies specific artistic styles on regular images.
- Super Resolution: Adapted from the ONNX Super Resolution model, Super-Res upscales and sharpens the input images to refine the details and improve image quality.

A complete sample of onnxruntime using the DirectML execution provider can be found under [samples/c_cxx/fns_candy_style_transfer](https://github.com/microsoft/onnxruntime/tree/master/samples//c_cxx/fns_candy_style_transfer).

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