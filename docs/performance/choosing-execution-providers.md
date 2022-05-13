---
title: Choosing Execution Providers for Performance
parent: Performance
nav_order: 3
description: Choosing the right Execution Provider for optimizing the ONNX Runtime performance.
redirect_from: /docs/how-to/tune-performance
---
<div class="container">

## Choosing the Execution Provider for best performance

Performance is dependent on the specific model you're trying to run, the session, the run options, and your specific hardware target. Here is some additional information for selecting the right Execution Provider for optimizing the ONNX Runtime performance.

## Contents
{: .no_toc }

* TOC placeholder
{:toc}

## CUDA (Default GPU) or CPU?

The CPU version of ONNX Runtime provides a complete implementation of all operators in the ONNX spec. This ensures that your ONNX-compliant model can execute successfully. In order to keep the binary size small, common data types are supported for the ops. If you are using an uncommon data type that is not supported, you can file a Github issue and/or contribute a Github Pull Request. (see examples - [PR #2112](https://github.com/microsoft/onnxruntime/pull/2112), [PR #2034](https://github.com/microsoft/onnxruntime/pull/2034), [PR #1565](https://github.com/microsoft/onnxruntime/pull/1565)). Please make sure you provide details on usage justification.

Additionally, not all CUDA kernels are implemented, as these have been prioritized on an as-needed basis. This means that if your model contains operators that do not have a CUDA implementation, it will fall back to CPU. Switching between CPU and GPU can cause significant performance impact. If you require a specific operator that is not currently supported, please consider [contributing](https://github.com/microsoft/onnxruntime/tree/master/CONTRIBUTING.md) and/or [file an issue](https://github.com/microsoft/onnxruntime/issues) clearly describing your use case and share your model if possible.

## TensorRT or CUDA?

TensorRT and CUDA are separate execution providers for ONNX Runtime. On the same hardware, TensorRT will generally provide better performance; however, this depends on the specific model and whether the operators in the model can be supported by TensorRT. In cases where TensorRT cannot handle the subgraph(s), it will fall back to CUDA. Note that the TensorRT EP may depend on a different version of CUDA than the CUDA EP.

## TensorRT/CUDA or DirectML?

DirectML is the hardware-accelerated DirectX 12 library for machine learning on Windows and supports all DirectX 12 capable devices (Nvidia, Intel, AMD). This means that if you are targeting Windows GPUs, using the DirectML Execution Provider is likely your best bet. This can be used with both the ONNX Runtime as well as [WinML APIs](https://docs.microsoft.com/en-us/windows/ai/windows-ml/api-reference).


<p><a href="#" id="back-to-top">Back to top</a></p>

</div>