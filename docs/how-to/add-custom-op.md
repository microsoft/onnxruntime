---
title: Add a new operator
parent: Recipes
nav_order: 2
---

# Add a new operator to ONNX Runtime
{: .no_toc }

## Contents
{: .no_toc }

* TOC placeholder
{:toc}

## A new op can be written and registered with ONNXRuntime in the following 3 ways

### 1. Using the experimental custom op API in the C API (onnxruntime_c_api.h)

Note: These APIs are experimental and will change in the next release. They're released now for feedback and experimentation.
* Create an OrtCustomOpDomain with the domain name used by the custom ops
* Create an OrtCustomOp structure for each op and add them to the OrtCustomOpDomain with OrtCustomOpDomain_Add
* Call OrtAddCustomOpDomain to add the custom domain of ops to the session options
See [this](https://github.com/microsoft/onnxruntime/blob/master/onnxruntime/test/shared_lib/test_inference.cc) for an example called MyCustomOp that uses the C++ helper API (onnxruntime_cxx_api.h).
Currently, the only supported Execution Providers (EPs) for custom ops registered via this approach are the `CUDA` and the `CPU` EPs. 

### 2. Using RegisterCustomRegistry API
* Implement your kernel and schema (if required) using the OpKernel and OpSchema APIs (headers are in the include folder).
* Create a CustomRegistry object and register your kernel and schema with this registry.
* Register the custom registry with ONNXRuntime using RegisterCustomRegistry API.

See
[this](https://github.com/microsoft/onnxruntime/blob/master/onnxruntime/test/framework/local_kernel_registry_test.cc) for an example.

### 3. Contributing the op to ONNXRuntime
This is mostly meant for ops that are in the process of being proposed to ONNX. This way you don't have to wait for an approval from the ONNX team
if the op is required in production today.
See [this](https://github.com/microsoft/onnxruntime/blob/master/onnxruntime/contrib_ops) for an example.
