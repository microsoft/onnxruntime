# DNNL Execution Provider

Intel® Math Kernel Library for Deep Neural Networks (Intel® DNNL) is an open-source performance library for deep-learning applications. The library accelerates deep-learning applications and frameworks on Intel® architecture and Intel® Processor Graphics Architecture. Intel DNNL contains vectorized and threaded building blocks that you can use to implement deep neural networks (DNN) with C and C++ interfaces. For more, please see the DNNL documentation on (https://intel.github.io/mkl-dnn/).

Intel and Microsoft have developed DNNL Execution Provider (EP) for ONNX Runtime to accelerate performance of ONNX Runtime using Intel® Math Kernel Library for Deep Neural Networks (Intel® DNNL) optimized primitives.

For information on how DNNL optimizes subgraphs, see [Subgraph Optimization](./MKL-DNN-Subgraphs.md)

## Build
For build instructions, please see the [BUILD page](../../BUILD.md#dnnl-and-mklml).

## Supported OS
* Ubuntu 16.04
* Windows 10 
* Mac OS X

## Supported backend
*	CPU

## Using the DNNL Execution Provider
### C/C++
The DNNLExecutionProvider execution provider needs to be registered with ONNX Runtime to enable in the inference session.
```
InferenceSession session_object{so};
session_object.RegisterExecutionProvider(std::make_unique<::onnxruntime:: DNNLExecutionProvider >());
status = session_object.Load(model_file_name);
```
The C API details are [here](../C_API.md#c-api).

### Python
When using the python wheel from the ONNX Runtime built with DNNL execution provider, it will be automatically prioritized over the CPU execution provider. Python APIs details are [here](https://aka.ms/onnxruntime-python).

## Performance Tuning
For performance tuning, please see guidance on this page: [ONNX Runtime Perf Tuning](../ONNX_Runtime_Perf_Tuning.md)
