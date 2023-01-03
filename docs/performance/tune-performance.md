---
title: Tune performance
parent: Performance
nav_order: 1
redirect_from: /docs/how-to/tune-performance
---

# ONNX Runtime Performance Tuning
{: .no_toc }

ONNX Runtime provides high performance across a range of hardware options through its [Execution Providers interface](../execution-providers) for different execution environments.

Along with this flexibility come decisions for tuning and usage. For each model running with each execution provider, there are settings that can be tuned (e.g. thread number, wait policy, etc) to improve performance.

This document covers basic tools and knobs that can be leveraged to find the best performance for your model and hardware.

## Contents
{: .no_toc }

* TOC placeholder
{:toc}

## Performance Tuning Tools

### ONNX GO Live Tool

The [ONNX Go Live "OLive" tool](https://github.com/microsoft/OLive) is a Python package that automates the process of accelerating models with ONNX Runtime (ORT). It contains two parts: (1) model conversion to ONNX with correctness checking (2) auto performance tuning with ORT. Users can run these two together through a single pipeline or run them independently as needed.

As a quickstart, please see the [notebook tutorials](https://github.com/microsoft/OLive/tree/master/notebook-tutorial) and [command line examples](https://github.com/microsoft/OLive/tree/master/cmd-example) 

### Profiling and Performance Report

The onnxruntime_perf_test.exe tool (available from the build drop) can be used to test various knobs. Please find the usage instructions using `onnxruntime_perf_test.exe -h`.

You can enable ONNX Runtime latency profiling in code:

```python
import onnxruntime as rt

sess_options = rt.SessionOptions()
sess_options.enable_profiling = True
```

If you are using the onnxruntime_perf_test.exe tool, you can add `-p [profile_file]` to enable performance profiling.

In both cases, you will get a JSON file which contains the detailed performance data (threading, latency of each operator, etc). This file is a standard performance tracing file, and to view it in a user-friendly way, you can open it by using chrome://tracing:

* Open Chrome browser
* Type chrome://tracing in the address bar
* Load the generated JSON file

To profile CUDA kernels, please add the cupti library to your PATH and use the onnxruntime binary built from source with `--enable_cuda_profiling`.
To profile ROCm kernels, please add the roctracer library to your PATH and use the onnxruntime binary built from source with `--enable_rocm_profiling`. 

Performance numbers from the device will then be attached to those from the host. For example:

```json
{"cat":"Node", "name":"Add_1234", "dur":17, ...}
{"cat":"Kernel", "name":"ort_add_cuda_kernel", dur:33, ...}
```

Here, the "Add" operator from the host initiated a CUDA kernel on device named "ort_add_cuda_kernel" which lasted for 33 microseconds.
If an operator called multiple kernels during execution, the performance numbers of those kernels will all be listed following the call sequence:

```json
{"cat":"Node", "name":<name of the node>, ...}
{"cat":"Kernel", "name":<name of the kernel called first>, ...}
{"cat":"Kernel", "name":<name of the kernel called next>, ...}
```

ONNX Runtime also offers a [tool](https://github.com/microsoft/onnxruntime/tree/master/tools/perf_view) to render the statistics as a summarized view in the browser.

## Using different Execution Providers

To learn more about different Execution Providers, see [Reference: Execution Providers](../execution-providers).

### Build the EP

**Python**

Official Python packages on PyPI only support the default CPU (MLAS) and default GPU (CUDA) execution providers. For other execution providers, you need to [build from source](../build/eps.md). The recommended instructions build the wheel with debug info in parallel.

For example:

`DNNL: ./build.sh --config RelWithDebInfo --use_dnnl --build_wheel --parallel`

`CUDA: ./build.sh --config RelWithDebInfo --use_cuda --build_wheel --parallel`

**C and C#**

Official releases on Nuget support default (MLAS) for CPU, and CUDA for GPU. For other execution providers, you need to build from source. Append `--build_csharp` to the instructions to build both C# and C packages.

For example:

`DNNL: ./build.sh --config RelWithDebInfo --use_dnnl --build_csharp --parallel`

`CUDA: ./build.sh --config RelWithDebInfo --use_cuda --build_csharp --parallel`

### Register the EP

In order to use the DNNL, CUDA, or TensorRT execution providers, you need to call the C API `OrtSessionOptionsAppendExecutionProvider`.

C API Example:

```c
  const OrtApi* g_ort = OrtGetApi(ORT_API_VERSION);
  OrtEnv* env;
  g_ort->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "test", &env)
  OrtSessionOptions* session_option;
  g_ort->OrtCreateSessionOptions(&session_options);
  g_ort->OrtSessionOptionsAppendExecutionProvider_CUDA(sessionOptions, 0);
  OrtSession* session;
  g_ort->CreateSession(env, model_path, session_option, &session);
```

C# API Example:

```c#
SessionOptions so = new SessionOptions();
so.GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_EXTENDED;
so.AppendExecutionProvider_CUDA(0);
var session = new InferenceSession(modelPath, so);
```

Python API Example:

```python
import onnxruntime as rt

so = rt.SessionOptions()
so.graph_optimization_level = rt.GraphOptimizationLevel.ORT_ENABLE_ALL
session = rt.InferenceSession(model, sess_options=so, providers=['CUDAExecutionProvider'])
```

## Which Execution Provider will provide the best performance?

Performance is dependent on the specific model you're trying to run, the session and run options you've selected, and of course, your specific hardware target. Below you'll find some more information that may be helpful to select the right Execution Provider.

### CUDA (Default GPU) or CPU?

The CPU version of ONNX Runtime provides a complete implementation of all operators in the ONNX spec. This ensures that your ONNX-compliant model can execute successfully. In order to keep the binary size small, common data types are supported for the ops. If you are using an uncommon data type that is not supported, you can file an issue and/or contribute a PR (see examples - [PR #2112](https://github.com/microsoft/onnxruntime/pull/2112), [PR #2034](https://github.com/microsoft/onnxruntime/pull/2034), [PR #1565](https://github.com/microsoft/onnxruntime/pull/1565)). Please make sure you provide details on usage justification.

Additionally, not all CUDA kernels are implemented, as these have been prioritized on an as-needed basis. This means that if your model contains operators that do not have a CUDA implementation, it will fall back to CPU. Switching between CPU and GPU can cause significant performance impact. If you require a specific operator that is not currently supported, please consider [contributing](https://github.com/microsoft/onnxruntime/tree/master/CONTRIBUTING.md) and/or [file an issue](https://github.com/microsoft/onnxruntime/issues) clearly describing your use case and share your model if possible.

### TensorRT or CUDA?

TensorRT and CUDA are separate execution providers for ONNX Runtime. On the same hardware, TensorRT will generally provide better performance; however, this depends on the specific model and whether the operators in the model can be supported by TensorRT. In cases where TensorRT cannot handle the subgraph(s), it will fall back to CUDA. Note that the TensorRT EP may depend on a different version of CUDA than the CUDA EP.

### TensorRT/CUDA or DirectML? 

DirectML is the hardware-accelerated DirectX 12 library for machine learning on Windows and supports all DirectX 12 capable devices (Nvidia, Intel, AMD). This means that if you are targeting Windows GPUs, using the DirectML Execution Provider is likely your best bet. This can be used with both the ONNX Runtime as well as [WinML APIs](https://docs.microsoft.com/en-us/windows/ai/windows-ml/api-reference).

## Tips for tuning performance

Below are some suggestions to try for tuning the performance of various EPs.

### Shared arena based allocator

Memory consumption can be reduced between multiple sessions by configuring the shared arena based allocation. See the `Share allocator(s) between sessions` section in the [C API documentation](../get-started/with-c.md).

### mimalloc allocator usage

ONNX Runtime supports overriding memory allocations using [mimalloc](https://github.com/microsoft/mimalloc), a fast, general-purpose allocator.

Depending on your model and usage, it can deliver single- or double-digit improvements in performance. The GitHub README page describes various scenarios on how mimalloc can be leveraged for performance tuning.

mimalloc is a submodule in the ONNX Runtime source tree. On Windows, one can employ the `--use_mimalloc` build flag which builds a static version of mimalloc and links it to ONNX Runtime. This redirects ONNX Runtime allocators and all new/delete calls to mimalloc.
Currently, there are no special provisions to employ mimalloc on Linux. It is recommended to use the LD_PRELOAD mechanism using pre-built binaries of mimalloc that you can build/obtain separately.

### Thread management

#### Set intra-op thread affinity

Sometimes, it may be beneficial to customize intra-op thread affinities, for example:
* There are multiple sessions run in parallel, customer might prefer their intra-op thread pools run on separate cores to avoid contention.
* Customer want to limit a intra-op thread pool to run on only one of the NUMA nodes to reduce overhead of cache miss.

For session intra-op thread pool, please read the [configuration](https://github.com/microsoft/onnxruntime/blob/68b5b2d7d33b6aa2d2b5cf8d89befb4a76e8e7d8/include/onnxruntime/core/session/onnxruntime_session_options_config_keys.h#L180) and consume it like:

```python
sess_op = SessionOptions()
sess_op.intra_op_num_threads = 3
sess_op.add_session_config_entry('session.intra_op_thread_affinities', '1;2')
sess = ort.InferenceSession('model.onnx', sess_op, ...)
```

For global thread pool, please read the [API](https://github.com/microsoft/onnxruntime/blob/68b5b2d7d33b6aa2d2b5cf8d89befb4a76e8e7d8/include/onnxruntime/core/session/onnxruntime_c_api.h#L3636) and [usage](https://github.com/microsoft/onnxruntime/blob/68b5b2d7d33b6aa2d2b5cf8d89befb4a76e8e7d8/onnxruntime/test/global_thread_pools/test_main.cc#L98).

#### Custom threading callbacks

Occasionally, customers might prefer to use their own fine-tuned threads for multithreading,
hence ORT offers thread creation and joining callbacks in the [C++ API](https://github.com/microsoft/onnxruntime/blob/master/include/onnxruntime/core/session/onnxruntime_cxx_api.h):

```c++
  std::vector<std::thread> threads;
  void* custom_thread_creation_options = nullptr;
  // initialize custom_thread_creation_options

  // On thread pool creation, ORT calls CreateThreadCustomized to create a thread
  OrtCustomThreadHandle CreateThreadCustomized(void* custom_thread_creation_options, OrtThreadWorkerFn work_loop, void* param) {
    threads.push_back(std::thread(work_loop, param));
    // configure the thread by custom_thread_creation_options
    return reinterpret_cast<OrtCustomThreadHandle>(threads.back().native_handle());
  }

  // On thread pool destruction, ORT calls JoinThreadCustomized for each created thread
  void JoinThreadCustomized(OrtCustomThreadHandle handle) {
    for (auto& t : threads) {
      if (reinterpret_cast<OrtCustomThreadHandle>(t.native_handle()) == handle) {
        // recycling resources ... 
        t.join();
      }
    }
  }

  int main(...) {
    ...
    Ort::Env ort_env;
    Ort::SessionOptions session_options;
    session_options.SetCustomCreateThreadFn(CreateThreadCustomized);
    session_options.SetCustomThreadCreationOptions(&custom_thread_creation_options);
    session_options.SetCustomJoinThreadFn(JoinThreadCustomized);
    Ort::Session session(*ort_env, MODEL_URI, session_options);
    ...
  }
```

For global thread pool:

```c++
  int main() {
    const OrtApi* g_ort = OrtGetApiBase()->GetApi(ORT_API_VERSION);
    OrtThreadingOptions* tp_options = nullptr;
    g_ort->CreateThreadingOptions(&tp_options);
    g_ort->SetGlobalCustomCreateThreadFn(tp_options, CreateThreadCustomized);
    g_ort->SetGlobalCustomThreadCreationOptions(tp_options, &custom_thread_creation_options);
    g_ort->SetGlobalCustomJoinThreadFn(tp_options, JoinThreadCustomized);
    // disable per-session thread pool, create a session for inferencing
    g_ort->ReleaseThreadingOptions(tp_options);
  }
```

Note that `CreateThreadCustomized` and `JoinThreadCustomized`, once being set, will be applied to both ORT intra op and inter op thread pools uniformly.

### Default CPU Execution Provider (MLAS)

For the default CPU execution provider, you can try the following knobs in the Python API to control the thread number:

```python
import onnxruntime as rt

sess_options = rt.SessionOptions()

sess_options.intra_op_num_threads = 2
sess_options.execution_mode = rt.ExecutionMode.ORT_SEQUENTIAL
sess_options.graph_optimization_level = rt.GraphOptimizationLevel.ORT_ENABLE_ALL
```

* Thread Count
  * `sess_options.intra_op_num_threads = 2` controls the number of threads to use to run the model.
* Sequential vs Parallel Execution
  * `sess_options.execution_mode = rt.ExecutionMode.ORT_SEQUENTIAL` controls whether the operators in the graph run sequentially or in parallel. Usually when a model has many branches, setting this option to `ORT_PARALLEL` will provide better performance.
  * When `sess_options.execution_mode = rt.ExecutionMode.ORT_PARALLEL`, you can set `sess_options.inter_op_num_threads` to control the
number of threads used to parallelize the execution of the graph (across nodes).
* Graph Optimization Level
  * `sess_options.graph_optimization_level = rt.GraphOptimizationLevel.ORT_ENABLE_ALL` enables all optimizations which is the default. Please see [onnxruntime_c_api.h](https://github.com/microsoft/onnxruntime/tree/master/include/onnxruntime/core/session/onnxruntime_c_api.h#L286) (enum `GraphOptimizationLevel`) for the full list of all optimization levels. For details regarding available optimizations and usage, please refer to the [Graph Optimizations](graph-optimizations.md) documentation.

### MKL_DNN/nGraph Execution Provider

MKL_DNN and nGraph depend on OpenMP for parallelization. For those execution providers, we need to use OpenMP environment variables to tune the performance.

The most widely used environment variables are:

* OMP_NUM_THREADS=n
  * Controls the thread pool size

* OMP_WAIT_POLICY=PASSIVE/ACTIVE
  * Controls whether thread spinning is enabled
  * PASSIVE is also called throughput mode and will yield CPU after finishing current task
  * ACTIVE will not yield CPU, instead it will have a while loop to check whether the next task is ready
  * Use PASSIVE if your CPU usage already high, and use ACTIVE when you want to trade CPU with latency

### IOBinding

When working with non-CPU execution providers, it's most efficient to have inputs (and/or outputs) arranged on the target device (abstracted by the execution provider used) prior to executing the graph (calling `Run()`). When the input is not copied to the target device, ORT copies it from the CPU as part of the `Run()` call. Similarly, if the output is not pre-allocated on the device, ORT assumes that the output is requested on the CPU and copies it from the device as the last step of the `Run()` call. This obviously eats into the execution time of the graph, misleading users into thinking ORT is slow when the majority of the time is spent in these copies. To address this, we've introduced the notion of IOBinding. The key idea is to arrange for inputs to be copied to the device and for outputs to be pre-allocated on the device prior to calling `Run()`. IOBinding is available in all our language bindings. Following are code snippets in various languages demonstrating the usage of this feature.

* C++

```c++
  Ort::Env env;
  Ort::Session session(env, model_path, session_options);
  Ort::IoBinding io_binding{session};
  auto input_tensor = Ort::Value::CreateTensor<float>(memory_info, input_tensor_values.data(), input_tensor_size, input_node_dims.data(), 4);
  io_binding.BindInput("input1", input_tensor);
  Ort::MemoryInfo output_mem_info{"Cuda", OrtDeviceAllocator, 0,
                                  OrtMemTypeDefault};
  
  // Use this to bind output to a device when the shape is not known in advance. If the shape is known you can use the other overload of this function that takes an Ort::Value as input (IoBinding::BindOutput(const char* name, const Value& value)).
  // This internally calls the BindOutputToDevice C API.

  io_binding.BindOutput("output1", output_mem_info);
  session.Run(run_options, io_binding);
```

* Python

Refer to the [Python API docs](https://onnxruntime.ai/docs/api/python).

* C#

Refer to [OrtIoBindingAllocationTest.cs](https://github.com/microsoft/onnxruntime/blob/master/csharp/test/Microsoft.ML.OnnxRuntime.Tests/OrtIoBindingAllocationTest.cs).

### Convolution-heavy models and the CUDA EP

ORT leverages CuDNN for convolution operations and the first step in this process is to determine which "optimal" convolution algorithm to use while performing the convolution operation for the given input configuration (input shape, filter shape, etc.) in each `Conv` node. This sub-step involves querying CuDNN for a "workspace" memory size and have this allocated so that CuDNN can use this auxiliary memory while determining the "optimal" convolution algorithm to use. By default, ORT clamps the workspace size to 32 MB which may lead to a sub-optimal convolution algorithm getting picked by CuDNN. To allow ORT to allocate the maximum possible workspace as determined by CuDNN, a provider option named `cudnn_conv_use_max_workspace` needs to get set (as shown below). Keep in mind that using this flag may increase the peak memory usage by a factor (sometimes a few GBs) but this does help CuDNN pick the best convolution algorithm for the given input. We have found that this is an important flag to use while using an fp16 model as this allows CuDNN to pick tensor core algorithms for the convolution operations (if the hardware supports tensor core operations). This flag may or may not result in performance gains for other data types (`float` and `double`).

* Python

```python
providers = [("CUDAExecutionProvider", {"cudnn_conv_use_max_workspace": '1'})]
sess_options = ort.SessionOptions()
sess = ort.InferenceSession("my_conv_heavy_fp16_model.onnx", sess_options=sess_options, providers=providers)
```

* C/C++

```c++
OrtCUDAProviderOptionsV2* cuda_options = nullptr;
CreateCUDAProviderOptions(&cuda_options);

std::vector<const char*> keys{"cudnn_conv_use_max_workspace"};
std::vector<const char*> values{"1"};

UpdateCUDAProviderOptions(cuda_options, keys.data(), values.data(), 1);

OrtSessionOptions* session_options = /* ... */;
SessionOptionsAppendExecutionProvider_CUDA_V2(session_options, cuda_options);

// Finally, don't forget to release the provider options
ReleaseCUDAProviderOptions(cuda_options);
```

* C#

```csharp
var cudaProviderOptions = new OrtCUDAProviderOptions(); // Dispose this finally

var providerOptionsDict = new Dictionary<string, string>();
providerOptionsDict["cudnn_conv_use_max_workspace"] = "1";

cudaProviderOptions.UpdateOptions(providerOptionsDict);

SessionOptions options = SessionOptions.MakeSessionOptionWithCudaProvider(cudaProviderOptions);  // Dispose this finally
```

### Convolution Input Padding in the CUDA EP

ORT leverages CuDNN for convolution operations. While CuDNN only takes 4-D or 5-D tensor as input for convolution operations, dimension padding is needed if the input is 3-D tensor. Given an input tensor of shape [N, C, D], it can be padded to [N, C, D, 1] or [N, C, 1, D]. While both of these two padding ways produce same output, the performance may be a lot different because different convolution algorithms are selected, especially on some devices such as A100. By default the input is padded to [N, C, D, 1]. A provider option named `cudnn_conv1d_pad_to_nc1d` needs to get set (as shown below) if [N, C, 1, D] is preferred.

* Python

```python
providers = [("CUDAExecutionProvider", {"cudnn_conv1d_pad_to_nc1d": '1'})]
sess_options = ort.SessionOptions()
sess = ort.InferenceSession("my_conv_model.onnx", sess_options=sess_options, providers=providers)
```

* C/C++

```c++
OrtCUDAProviderOptionsV2* cuda_options = nullptr;
CreateCUDAProviderOptions(&cuda_options);

std::vector<const char*> keys{"cudnn_conv1d_pad_to_nc1d"};
std::vector<const char*> values{"1"};

UpdateCUDAProviderOptions(cuda_options, keys.data(), values.data(), 1);

OrtSessionOptions* session_options = /* ... */;
SessionOptionsAppendExecutionProvider_CUDA_V2(session_options, cuda_options);

// Finally, don't forget to release the provider options
ReleaseCUDAProviderOptions(cuda_options);
```

* C#

```csharp
var cudaProviderOptions = new OrtCUDAProviderOptions(); // Dispose this finally

var providerOptionsDict = new Dictionary<string, string>();
providerOptionsDict["cudnn_conv1d_pad_to_nc1d"] = "1";

cudaProviderOptions.UpdateOptions(providerOptionsDict);

SessionOptions options = SessionOptions.MakeSessionOptionWithCudaProvider(cudaProviderOptions);  // Dispose this finally
```

### Using CUDA Graphs in the CUDA EP

NOTE: Please note that this feature is currently being offered in "preview" mode.

While using the CUDA EP, ORT supports the usage of [CUDA Graphs](https://developer.nvidia.com/blog/cuda-10-features-revealed/) to remove CPU overhead associated with launching CUDA kernels sequentially. To enable the usage of CUDA Graphs, use the provider option as shown in the samples below.
Currently, there are some constraints with regards to using the CUDA Graphs feature:

* Models with control-flow ops (i.e. `If`, `Loop` and `Scan` ops) are not supported.

* Usage of CUDA Graphs is limited to models where-in all the model ops (graph nodes) can be partitioned to the CUDA EP.

* The input/output types of models need to be tensors.

* Shapes of inputs/outputs cannot change across inference calls. Dynamic shape models are supported - the only constraint is that the input/output shapes should be the same across all inference calls.

* By design, [CUDA Graphs](https://developer.nvidia.com/blog/cuda-10-features-revealed/) is designed to read from/write to the same CUDA virtual memory addresses during the graph replaying step as it does during the graph capturing step. Due to this requirement, usage of this feature requires using IOBinding so as to bind memory which will be used as input(s)/output(s) for the CUDA Graph machinery to read from/write to (please see samples below).

* While updating the input(s) for subsequent inference calls, the fresh input(s) need to be copied over to the corresponding CUDA memory location(s) of the bound `OrtValue` input(s) (please see samples below to see how this can be achieved). This is due to the fact that the "graph replay" will require reading inputs from the same CUDA virtual memory addresses.

* Multi-threaded usage is currently not supported, i.e. `Run()` MAY NOT be invoked on the same `InferenceSession` object from multiple threads while using CUDA Graphs.

NOTE: The very first `Run()` performs a variety of tasks under the hood like making CUDA memory allocations, capturing the CUDA graph for the model, and then performing a graph replay to ensure that the graph runs. Due to this, the latency associated with the first `Run()` is bound to be high. Subsequent `Run()`s only perform graph replays of the graph captured and cached in the first `Run()`. 

* Python

```python
providers = [("CUDAExecutionProvider", {"enable_cuda_graph": '1'})]
sess_options = ort.SessionOptions()
sess = ort.InferenceSession("my_model.onnx", sess_options=sess_options, providers=providers)

providers = [("CUDAExecutionProvider", {'enable_cuda_graph': True})]
x = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)
y = np.array([[0.0], [0.0], [0.0]], dtype=np.float32)
x_ortvalue = onnxrt.OrtValue.ortvalue_from_numpy(x, 'cuda', 0)
y_ortvalue = onnxrt.OrtValue.ortvalue_from_numpy(y, 'cuda', 0)

session = onnxrt.InferenceSession("matmul_2.onnx", providers=providers)
io_binding = session.io_binding()

# Bind the input and output
io_binding.bind_ortvalue_input('X', x_ortvalue)
io_binding.bind_ortvalue_output('Y', y_ortvalue)

# One regular run for the necessary memory allocation and cuda graph capturing
session.run_with_iobinding(io_binding)
expected_y = np.array([[5.0], [11.0], [17.0]], dtype=np.float32)
np.testing.assert_allclose(expected_y, y_ortvalue.numpy(), rtol=1e-05, atol=1e-05)

# After capturing, CUDA graph replay happens from this Run onwards
session.run_with_iobinding(io_binding)
np.testing.assert_allclose(expected_y, y_ortvalue.numpy(), rtol=1e-05, atol=1e-05)

# Update input and then replay CUDA graph with the updated input
x_ortvalue.update_inplace(np.array([[10.0, 20.0], [30.0, 40.0], [50.0, 60.0]], dtype=np.float32))
session.run_with_iobinding(io_binding)
```

* C/C++

```c++
const auto& api = Ort::GetApi();

struct CudaMemoryDeleter {
  explicit CudaMemoryDeleter(const Ort::Allocator* alloc) {
    alloc_ = alloc;
  }

  void operator()(void* ptr) const {
    alloc_->Free(ptr);
  }
  
  const Ort::Allocator* alloc_;
};
  
// Enable cuda graph in cuda provider option.
OrtCUDAProviderOptionsV2* cuda_options = nullptr;
api.CreateCUDAProviderOptions(&cuda_options);
std::unique_ptr<OrtCUDAProviderOptionsV2, decltype(api.ReleaseCUDAProviderOptions)> rel_cuda_options(cuda_options, api.ReleaseCUDAProviderOptions);
std::vector<const char*> keys{"enable_cuda_graph"};
std::vector<const char*> values{"1"};
api.UpdateCUDAProviderOptions(rel_cuda_options.get(), keys.data(), values.data(), 1);

Ort::SessionOptions session_options;
api.SessionOptionsAppendExecutionProvider_CUDA_V2(static_cast<OrtSessionOptions*>(session_options), rel_cuda_options.get();


// Create IO bound inputs and outputs.
Ort::Session session(*ort_env, L"matmul_2.onnx", session_options);
Ort::MemoryInfo info_cuda("Cuda", OrtAllocatorType::OrtArenaAllocator, 0, OrtMemTypeDefault);
Ort::Allocator cuda_allocator(session, info_cuda);

const std::array<int64_t, 2> x_shape = {3, 2};
std::array<float, 3 * 2> x_values = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
auto input_data = std::unique_ptr<void, CudaMemoryDeleter>(cuda_allocator.Alloc(x_values.size() * sizeof(float)),
                                                           CudaMemoryDeleter(&cuda_allocator));
cudaMemcpy(input_data.get(), x_values.data(), sizeof(float) * x_values.size(), cudaMemcpyHostToDevice);

// Create an OrtValue tensor backed by data on CUDA memory
Ort::Value bound_x = Ort::Value::CreateTensor(info_cuda, reinterpret_cast<float*>(input_data.get()), x_values.size(),
                                              x_shape.data(), x_shape.size());

const std::array<int64_t, 2> expected_y_shape = {3, 2};
std::array<float, 3 * 2> expected_y = {1.0f, 4.0f, 9.0f, 16.0f, 25.0f, 36.0f};
auto output_data = std::unique_ptr<void, CudaMemoryDeleter>(cuda_allocator.Alloc(expected_y.size() * sizeof(float)),
                                                            CudaMemoryDeleter(&cuda_allocator));

// Create an OrtValue tensor backed by data on CUDA memory
Ort::Value bound_y = Ort::Value::CreateTensor(info_cuda, reinterpret_cast<float*>(output_data.get()),
                                              expected_y.size(), expected_y_shape.data(), expected_y_shape.size());

Ort::IoBinding binding(session);
binding.BindInput("X", bound_x);
binding.BindOutput("Y", bound_y);

// One regular run for necessary memory allocation and graph capturing
session.Run(Ort::RunOptions(), binding);

// After capturing, CUDA graph replay happens from this Run onwards
session.Run(Ort::RunOptions(), binding);

// Update input and then replay CUDA graph with the updated input
x_values = {10.0f, 20.0f, 30.0f, 40.0f, 50.0f, 60.0f};
cudaMemcpy(input_data.get(), x_values.data(), sizeof(float) * x_values.size(), cudaMemcpyHostToDevice);
session.Run(Ort::RunOptions(), binding);
```

* C#

Will be supported in future releases


## Troubleshooting performance issues

Here is a list of things to check when assessing performance issues:

* Have you enabled all [graph optimizations](graph-optimizations.md)? The official published packages do enable all by default but when building from source, check that these are enabled in your build.
* Have you searched through prior-filed [GitHub issues](https://github.com/microsoft/onnxruntime/issues) to see if your problem has been discussed previously? Please do this before filing new issues.
* If using CUDA or TensorRT, do you have the right versions of the dependent libraries installed? 

### I need help performance tuning for BERT models

For BERT models, sometimes ONNX Runtime cannot apply the best optimizations due to reasons such as framework version updates. We recommend trying out the [BERT optimization tool](https://github.com/microsoft/onnxruntime/tree/master/onnxruntime/python/tools/transformers), which reflects the latest changes in graph pattern matching and model conversions, and a set of [notebooks](https://github.com/microsoft/onnxruntime/tree/master/onnxruntime/python/tools/transformers/notebooks) to help get started.

### Why is the model graph not optimized even with graph_optimization_level set to ORT_ENABLE_ALL?

The ONNX model from IR_VERSION 4 only treats initializers that appear in graph input as non-constant. This may prevent some of the graph optimizations like const folding, operator fusion etc. Move initializers out of graph inputs if there is no need to override them, by either re-generating the model with the latest exporter/converter or with the tool [remove_initializer_from_input.py](https://github.com/microsoft/onnxruntime/tree/master/tools/python/remove_initializer_from_input.py).

### Why is my model running slower on GPU than on CPU?

Depending on which execution provider you're using, it may not have full support for all the operators in your model. Fallback to CPU ops can cause hits in performance speed. Moreover, even if an op is implemented by the CUDA execution provider, it may not necessarily assign/place the op to the CUDA EP due to performance reasons. To see the placement decided by ORT, turn on verbose logging and look at the console output.

### My converted TensorFlow model is slow - why?

NCHW and NHWC are two different memory layout for 4-D tensors.

Most TensorFlow operations used by a CNN support both NHWC and NCHW data format. The TensorFlow team suggests that on GPUs NCHW is faster but on CPUs NHWC is sometimes faster in TensorFlow. However, ONNX only supports NCHW. As a result, if the original model is in NHWC format, extra transposes may be added when the model is converted. The [tensorflow-onnx](https://github.com/onnx/tensorflow-onnx) converter does remove many of these transposes, but if this doesn't help sufficiently, consider retraining the model in the NCHW format.

### I am seeing high latency variance

On some platforms, onnxruntime may exhibit high latency variance during inferencing. This is caused by the constant cost model that onnxruntime uses to parallelize tasks in the thread pool.
For each task, the constant cost model will calculate a granularity for parallelization among threads, which stays constant to the end of the task execution. This approach can bring imbalanced load sometimes, causing high latency variance.
To mitigate this, onnxruntime provides a dynamic cost model which can be enabled as a session option:

```python
sess_options.add_session_config_entry('session.dynamic_block_base', '4')
```

Whenever set with a positive value, the onnxruntime thread pool will parallelize internal tasks with a decreasing granularity.
Specifically, assuming there is a function expected to run N number of times by the thread pool, with the dynamic cost model enabled, each thread in the pool will claim

```python
residual_of_N / (dynamic_block_base * num_of_threads)
```

whenever it is ready to run. So over a period of time, threads in the pool are likely to be better load balanced, thereby lowering the latency variance.

Due to the same reason, the dynamic cost model may also improve the performance for cases when threads are more likely be preempted.
Per our tests, by far the best configuration for dynamic_block_base is 4, which lowers the variance while keeping good performance.