# Introduction

ONNX Runtime is capable of working with different HW acceleration libraries to execute the ONNX models on the hardware platform. ONNX Runtime supports an extensible framework, called **Execution Providers** (EP), to integrate with the HW specific libraries. This interface enables flexibility for the AP application developer to deploy their ONNX models in different environments in the cloud and the edge and optimize the execution by taking advantage of the compute capabilities of the platform.

<p align="center"><img width="50%" src="images/ONNX_Runtime_EP1.png" alt="Executing ONNX models across different HW environments"/></p>

ONNX Runtime works with the execution provider(s) using the `GetCapability()` interface to allocate specific nodes or sub-graphs for execution by the EP library in supported hardware. The EP libraries that are preinstalled in the execution environment processes and executes the ONNX sub-graph on the hardware. This architecture abstracts out the details of the hardware specific libraries that are essential to optimizing the execution of deep neural networks across hardware platforms like CPU, GPU, FPGA or specialized NPUs.

<p align="center"><img width="50%" src="images/ONNX_Runtime_EP3.png" alt="ONNX Runtime GetCapability()"/></p>

ONNX Runtime supports many different execution providers today. Some of the EPs are in GA and used in live service. Many are in released in preview to enable developers to develop and customize their application using the different options.

## Adding an Execution Provider

Developers of specialized HW acceleration solutions can integrate with ONNX Runtime to execute ONNX models on their stack. To create an EP to interface with ONNX Runtime you must first identify a unique name for the EP. Follow the steps outlined [here](../AddingExecutionProvider.md) to integrat your code in the repo.



## Building ONNX Runtime with EPs

## Using Execution Providers

### Python APIs

### C/C++ APIs

1. Create a factory for that provider, by using the c function you exported in 'symbols.txt'
2. Put the provider factory into session options
3. Create session from that session option
e.g.

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

===========================================================================
* Create a folder under onnxruntime/core/providers
* Create a folder under include/onnxruntime/core/providers, it should has the same name as the first step.
* Create a new class, which must inherit from [IExecutionProvider](../include/onnxruntime/core/framework/execution_provider.h). The source code should be put in 'onnxruntime/core/providers/[your_provider_name]'
* Create a new header file under include/onnxruntime/core/providers/[your_provider_name]. The file should provide one function for creating an OrtProviderFactoryInterface. You may use 'include/onnxruntime/core/providers/cpu/cpu_provider_factory.h' as a template. You don't need to provide a function for creating MemoryInfo.
* Put a symbols.txt under 'onnxruntime/core/providers/[your_provider_name]'. The file should contain all the function names that would be exported from you provider. Usually, just a single function for creating provider factory is enough.
* Add your provider in onnxruntime_providers.cmake. Build it as a static lib.
* Add one line in cmake/onnxruntime.cmake, to the 'target_link_libraries' function call. Put your provider there.


Examples:     

 * [CPU Execution
       Provider](../onnxruntime/core/providers/cpu/cpu_execution_provider.h)               
 * [CUDA Execution
       Provider](../onnxruntime/core/providers/cuda/cuda_execution_provider.h)               
 * [DNNL Execution
       Provider](../onnxruntime/core/providers/dnnl/dnnl_execution_provider.h)               


