---
title: Add a new execution provider
parent: How to
nav_order: 7
---

# Add a new Execution Provider to ONNX Runtime
{: .no_toc }

## Contents
{: .no_toc }

* TOC placeholder
{:toc}

## Create the Execution Provider

1. Create a folder under onnxruntime/core/providers
2. Create a folder under include/onnxruntime/core/providers, it should has the same name as the first step.
3. Create a new class, which must inherit from [IExecutionProvider](https://github.com/microsoft/onnxruntime/tree/master/include//onnxruntime/core/framework/execution_provider.h). The source code should be put in 'onnxruntime/core/providers/[your_provider_name]'
4. Create a new header file under include/onnxruntime/core/providers/[your_provider_name]. The file should provide one function for creating an OrtProviderFactoryInterface. You may use 'include/onnxruntime/core/providers/cpu/cpu_provider_factory.h' as a template. You don't need to provide a function for creating MemoryInfo.
5. Put a symbols.txt under 'onnxruntime/core/providers/[your_provider_name]'. The file should contain all the function names that would be exported from you provider. Usually, just a single function for creating provider factory is enough.
6. Add your provider in onnxruntime_providers.cmake. Build it as a static lib.
7. Add one line in cmake/onnxruntime.cmake, to the 'target_link_libraries' function call. Put your provider there.

Examples:

* [CPU Execution Provider](https://github.com/microsoft/onnxruntime/blob/master/onnxruntime/core/providers/cpu/cpu_execution_provider.h)
* [CUDA Execution Provider](https://github.com/microsoft/onnxruntime/blob/master/onnxruntime/core/providers/cuda/cuda_execution_provider.h)               
* [DNNL Execution Provider](https://github.com/microsoft/onnxruntime/blob/master/onnxruntime/core/providers/dnnl/dnnl_execution_provider.h)               

## Use the Execution Provider

1. Create a factory for that provider, by using the c function you exported in 'symbols.txt'
2. Put the provider factory into session options
3. Create session from that session option

Example:

```c
  OrtEnv* env;
  OrtInitialize(ORT_LOGGING_LEVEL_WARNING, "test", &env)
  OrtSessionOptions* session_option = OrtCreateSessionOptions();
  OrtProviderFactoryInterface** factory;
  OrtCreateCUDAExecutionProviderFactory(0, &factory);
  OrtSessionOptionsAppendExecutionProvider(session_option, factory);
  OrtReleaseObject(factory);
  OrtCreateSession(env, model_path, session_option, &session);
```
