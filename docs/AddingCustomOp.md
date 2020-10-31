Adding a new op
===============

## A new op can be written and registered with ONNXRuntime in the following 3 ways
### 1. Using the custom op API in the C/C++ APIs (onnxruntime_c_api.h)
* Create an OrtCustomOpDomain with the domain name used by the custom ops
* Create an OrtCustomOp structure for each op and add them to the OrtCustomOpDomain with OrtCustomOpDomain_Add
* Call OrtAddCustomOpDomain to add the custom domain of ops to the session options
See [this](../onnxruntime/test/shared_lib/test_inference.cc) for an example called MyCustomOp that uses the C++ helper API (onnxruntime_cxx_api.h).
You can also compile the custom ops into a shared library and use that to run a model via the C++ API. The same test file contains an example.
The source code for a sample custom op shared library containing two custom kernels is [here](../onnxruntime/test/testdata/custom_op_library/custom_op_library.cc).
See [this](../onnxruntime/test/python/onnxruntime_test_python.py) for an example called testRegisterCustomOpsLibrary that uses the Python API
to register a shared library that contains custom op kernels.
Currently, the only supported Execution Providers (EPs) for custom ops registered via this approach are the `CUDA` and the `CPU` EPs. 

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
