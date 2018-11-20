Adding a new op
===============

## A new op can be written and registered with ONNXRuntime in the following 3 ways
### 1. Using a dynamic shared library
* First write the implementation of the op and schema (if required) and assemble them in a shared library.
See [this](../onnxruntime/test/custom_op_shared_lib) for an example. Currently
this is supported for Linux only.

Example of creating a shared lib using g++ on Linux:
```g++ -std=c++14 -shared test_custom_op.cc -o test_custom_op.so -fPIC -I. -Iinclude/onnxruntime -L. -lonnxruntime -DONNX_ML -DONNX_NAMESPACE=onnx```

* Register the shared lib with ONNXRuntime.
See [this](../onnxruntime/test/shared_lib/test_inference.cc) for an example.

### 2. Using RegisterCustomRegistry API
* Implement your kernel and schema (if required) using the OpKernel and OpSchema APIs (headers are in the include folder).
* Create a CustomRegistry object and register your kernel and schema with this registry.
* Register the custom registry with ONNXRuntime using RegisterCustomRegistry API.

See
[this](../onnxruntime/test/framework/local_kernel_registry_test.cc) for an example.

### 3. Contributing the op to ONNXRuntime
This is mostly meant for ops that are in the process of being proposed to ONNX. This way you don't have to wait for an approval from the ONNX team
if the op is required in production today.
See [this](../onnxruntime/contrib_ops) for an example.
