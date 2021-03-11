---
title: Add a new operator
parent: How to
nav_order: 2
---

# Add a new operator to ONNX Runtime
{: .no_toc }

## Contents
{: .no_toc }

* TOC placeholder
{:toc}

A new op can be written and registered with ONNXRuntime in the following 3 ways

## Custom Operator API

Use the custom operator C/C++ API (onnxruntime_c_api.h)

* Create an OrtCustomOpDomain with the domain name used by the custom ops
* Create an OrtCustomOp structure for each op and add them to the OrtCustomOpDomain with OrtCustomOpDomain_Add
* Call OrtAddCustomOpDomain to add the custom domain of ops to the session options
  
See [this](https://github.com/microsoft/onnxruntime/blob/master/onnxruntime/test/shared_lib/test_inference.cc) for examples called `MyCustomOp` and `SliceCustomOp` that use the C++ helper API (onnxruntime_cxx_api.h).

You can also compile the custom ops into a shared library and use that to run a model via the C++ API. The same test file contains an example.

The source code for a sample custom op shared library containing two custom kernels is [here](https://github.com/microsoft/onnxruntime/blob/master/onnxruntime/test/testdata/custom_op_library/custom_op_library.cc).

See [this](https://github.com/microsoft/onnxruntime/blob/master/onnxruntime/test/python/onnxruntime_test_python.py) for an example called `testRegisterCustomOpsLibrary` that uses the Python API to register a shared library that contains custom op kernels. Currently, the only supported Execution Providers (EPs) for custom ops registered via this approach are the CUDA and the CPU EPs.

Note that when a model being inferred on gpu, onnxruntime will insert MemcpyToHost op before a cpu custom op and append MemcpyFromHost after to make sure tensor(s) are accessible throughout calling, meaning there are no extra efforts required from custom op developer for the case.

When using CUDA custom ops, to ensure synchronization between ORT's CUDA kernels and the custom CUDA kernels, they must all use the same CUDA compute stream. To ensure this, you may first create a CUDA stream and pass it to the underlying Session via SessionOptions (use `OrtCudaProviderOptions` struct). This will ensure ORT's CUDA kernels use that stream and if the custom CUDA kernels are launched using the same stream, synchronization is now taken care of implicitly. For a sample, please see how the afore-mentioned `MyCustomOp` is being launched and how the Session using this custom op is created.

## Use RegisterCustomRegistry API

Implement your kernel and schema (if required) using the OpKernel and OpSchema APIs (headers are in the include folder).
Create a CustomRegistry object and register your kernel and schema with this registry.
Register the custom registry with ONNXRuntime using RegisterCustomRegistry API.
See [this](https://github.com/microsoft/onnxruntime/blob/master/onnxruntime/test/framework/local_kernel_registry_test.cc) for an example. for an example.

## Contribute the operator to ONNXRuntime

This is for ops that are in the process of being proposed to ONNX. This way you don't have to wait for an approval from the ONNX team if the op is required in production today. See [this](https://github.com/microsoft/onnxruntime/blob/master/onnxruntime/contrib_ops) for an example.
