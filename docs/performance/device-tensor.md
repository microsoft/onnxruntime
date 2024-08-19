---
title: Device tensors
description: Custom device memory usage to reduce copies
parent: Performance
nav_order: 6
---
# Using device tensors in ONNX Runtime

Using device tensors can be a crucial part in building efficient AI pipelines, especially on heterogenous memory systems.
A typical example of such systems is any PC with a dedicated GPU.
While a [recent GPU](https://www.techpowerup.com/gpu-specs/geforce-rtx-4090.c3889) itself has a memory bandwidth of about 1TB/s, the interconnect [PCI 4.0 x16](https://en.wikipedia.org/wiki/PCI_Express) to the CPU can often be the limiting factor with only ~32GB/s.
Therefore it is often best to keep data local to the GPU as much as possible or hide slow memory traffic behind computation as the GPU is able to execute compute and PCI memory traffic simultaneously.

A typical use case for these scenarios where memory is already local to the inference device would be a GPU accelerated video processing of an encoded video stream which can be decoded with GPU decoders.
Another common case are iterative networks like diffusion networks or large language models for which intermediate tensors do not have to be copied back to CPU.
Tile based inference for high resolution images is another use-case where custom memory management is important to reduce GPU idle times during PCI copies. Rather than doing sequential processing of each tile it is possible to overlap PCI copies and processing on the GPU and pipeline work in that matter.

<img src="../../images/pipeline_pci_processing.png" alt="Image of sequential PCI->Processing->PCI and another image of it being interleaved."/>

## CUDA

CUDA in ONNX Runtime has two custom memory types.
`"CudaPinned"` and `"Cuda"` memory where [CUDA pinned](https://developer.nvidia.com/blog/how-optimize-data-transfers-cuda-cc/) is actually CPU memory which is directly accessible by the GPU allowing for fully asynchronous up and download of memory using [`cudaMemcpyAsync`](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY.html#group__CUDART__MEMORY_1g85073372f776b4c4d5f89f7124b7bf79).
Normal CPU tensors only allow for a synchronous downloads from GPU to CPU while CPU to GPU copies can always be executed asynchronous.

Allocating a tensor using the `Ort::Sessions`'s allocator is very straight forward using the [C++ API](https://onnxruntime.ai/docs/api/c/struct_ort_1_1_value.html#a5d35080239ae47cdbc9e505666dc32ec) which directly maps to the C API.
```c++
Ort::Session session(ort_env, model_path_cstr, session_options);
Ort::MemoryInfo memory_info_cuda("Cuda", OrtArenaAllocator, /*device_id*/0,
                                 OrtMemTypeDefault);
Ort::Allocator gpu_allocator(session, memory_info_cuda);
auto ort_value = Ort::Value::CreateTensor(
        gpu_allocator, shape.data(), shape.size(),
        ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16);
```

External allocated data can also be wrapped to an `Ort::Value` without copying it:
```c++
Ort::MemoryInfo memory_info_cuda("Cuda", OrtArenaAllocator, device_id,
                                 OrtMemTypeDefault);
std::array<int64_t, 4> shape{1, 4, 64, 64};
size_t cuda_buffer_size = 4 * 64 * 64 * sizeof(float);
void *cuda_resource;
CUDA_CHECK(cudaMalloc(&cuda_resource, cuda_buffer_size));
auto ort_value = Ort::Value::CreateTensor(
    memory_info_cuda, cuda_resource, cuda_buffer_size,
    shape.data(), shape.size(),
    ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);
```

These allocated tensors can then be used as [I/O Binding](../performance/tune-performance/iobinding.md) to eliminate copy ops on the network and move the responsibility to the user.
With such IO bindings more performance tunings are possible:
- due to the fixed tensor address, a CUDA graph can be captured to reduce CUDA launch latency on CPU
- due to either having fully asynchronous downloads to pinned memory or eliminating memory copies by using device local tensor, CUDA can run [fully asynchronous via a run option](../execution-providers/CUDA-ExecutionProvider.md#performance-tuning) on its given stream

To set the custom compute stream for CUDA, refer to the V2 option API exposing the `Ort[CUDA|TensorRT]ProviderOptionsV2*`opaque struct pointer and the function `Update[CUDA|TensorRT]ProviderOptionsWithValue(options, "user_compute_stream", cuda_stream);` to set it's stream member.
More details can be found in each execution provider doc.

If you want to verify your optimizations Nsight System helps to correlate CPU API and GPU execution of CUDA operations.
This will allow to verify as well if the desired synchronizations were made and no async operation falls back to synchronous execution.
It is also used in [this](https://www.nvidia.com/gtc/session-catalog/?search=S62336#/session/1695978753458001R4wk) GTC talk explaining optimal usage of device tensors.

### Python API

The Python API supports the same performance opportunities as the above mentioned C++ API.
[Device tensors](https://onnxruntime.ai/docs/api/python/api_summary.html#data-on-device) can be allocated as shown here.
Besides this the `user_compute_stream` can be set through this [API](https://onnxruntime.ai/docs/api/python/api_summary.html#onnxruntime.InferenceSession.set_providers)

```python
sess = onnxruntime.InferenceSession("model.onnx", providers=["TensorrtExecutionProvider"])
option = {}
s = torch.cuda.Stream()
option["user_compute_stream"] = str(s.cuda_stream)                    
sess.set_providers(["TensorrtExecutionProvider"], [option])
```

Enabling asynchronous execution in python is possible through the same [run option](https://onnxruntime.ai/docs/api/python/api_summary.html#runoptions) as on the C++ API.

## DirectML

Achieving the same behavior is possible through DirectX resources.
To run asynchronous processing, it is crucial to do the same management of execution streams as needed with CUDA.
For DirectX, this means managing the device and its command queue, which is possible through the C API.
Details of how to set the compute command queue are documented with the usage of [`SessionOptionsAppendExecutionProvider_DML1`](../execution-providers/DirectML-ExecutionProvider.md#usage).

If separate command queues are used for copy and compute, it is possible to overlap PCI copies and execution as well as make execution asynchronous.

```c++
#include <onnxruntime/dml_provider_factory.h>
Ort::MemoryInfo memory_info_dml("DML", OrtDeviceAllocator, device_id,
                                OrtMemTypeDefault);

std::array<int64_t, 4> shape{1, 4, 64, 64};
void *dml_resource;
size_t d3d_buffer_size = 4 * 64 * 64 * sizeof(float);
const OrtDmlApi *ort_dml_api;
Ort::ThrowOnError(Ort::GetApi().GetExecutionProviderApi(
                  "DML", ORT_API_VERSION, reinterpret_cast<const void **>(&ort_dml_api)));

// Create d3d_buffer using D3D12 APIs
Microsoft::WRL::ComPtr<ID3D12Resource> d3d_buffer = ...;

// Create the dml resource from the D3D resource.
ort_dml_api->CreateGPUAllocationFromD3DResource(d3d_buffer.Get(), &dml_resource);


Ort::Value ort_value(Ort::Value::CreateTensor(memory_info_dml, dml_resource,
                     d3d_buffer_size, shape.data(), shape.size(),
                     ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT));
```

A [single file sample](https://github.com/ankan-ban/HelloOrtDml/blob/main/Main.cpp) can be found on GitHub which shows how to manage and create copy and execution command queues.

### Python API

Although allocating DirectX inputs from Python might not be a major use case, the API is available. This can prove to be very beneficial, especially for intermediate network caches, such as key-value caching in large language models (LLMs).

```python
import onnxruntime as ort
import numpy as np

session = ort.InferenceSession("model.onnx",
                               providers=["DmlExecutionProvider"])

cpu_array = np.zeros((1, 4, 512, 512), dtype=np.float32)
dml_array = ort.OrtValue.ortvalue_from_numpy(cpu_array, "dml")

binding = session.io_binding()
binding.bind_ortvalue_input("data", dml_array)
binding.bind_output("out", "dml")
# if the output dims are known we can also bind a preallocated value
# binding.bind_ortvalue_output("out", dml_array_out)

session.run_with_iobinding(binding)
```
