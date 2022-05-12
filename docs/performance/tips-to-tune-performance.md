---
title: Tune performance > Tips
parent: Performance
nav_order: 4
description: Tips to tune Onnx Runtime performance in terms of reducing memory consumption, thread management, IO Binding, and customizing CUDA Execution Provider.
redirect_from: /docs/how-to/tune-performance
---
<div class="container">


<p id="tips"></p>

## Tips for Tuning Performance

Here are some tips for tuning the performance of ORT in terms of <a href="#memory">reducing memory consumption</a>, <a href ="#thread">thread management</a>, <a href="#iobinding">IO Binding</a>, and <a href="#customCUDA">customizing CUDA Execution Provider</a>.

Please refer to [Execution Provider](../execution-providers/index.md) specific performance tuning samples and tips for optimizing your OnnxRuntime performance.

<h3 id="memory">Memory Consumption Reduction</h3>

Here are some tips to reduce memory consumption and tune performance with ORT.

### 1. Shared arena-based allocator
Memory consumption can be reduced between multiple sessions by configuring the shared arena-based allocation. See the `Share allocator(s) between sessions` section in the [C API documentation](../get-started/with-c.md).

### 2. Mimalloc allocator

OnnxRuntime supports overriding memory allocations using mimalloc allocator, which is a general-purpose fast allocator. See [mimalloc github](https://github.com/microsoft/mimalloc). 
- Depending on your model and usage mimalloc can deliver single- or double-digit improvements. The GitHub README page describes various scenarios on how mimalloc can be leveraged to support your scenarios.
- mimalloc is a submodule in the OnnxRuntime source tree. On Windows, one can employ `--use_mimalloc` build flag which would build a static version of mimalloc and link it to OnnxRuntime. This would redirect OnnxRuntime allocators and all new/delete calls to mimalloc. Currently, there are no special provisions to employ mimalloc on Linux. This can be done via LD_PRELAOD mechanism using pre-built binaries that you can build/obtain separately.
 
<p><a href="#tips">Performance Tuning Tips</a></p>

<h3 id="thread">Thread Management</h3>

ONNX Runtime allows different [threading implementation](https://github.com/microsoft/onnxruntime/blob/master/docs/NotesOnThreading.md) choices for OpenMP or non-OpenMP. Here are some best practices for [thread management](https://github.com/microsoft/onnxruntime/blob/master/docs/FAQ.md#how-do-i-force-single-threaded-execution-mode-in-ort-by-default-sessionrun-uses-all-the-computers-cores) to customize your ONNX Runtime environment:

* If ORT is built with OpenMP, use the OpenMP env variable to control the number of IntraOp num threads.
* If ORT is not built with OpenMP, use the appropriate ORT API to control IntraOp num threads.
* InterOp num threads setting: 
    - is used only when parallel execution is enabled
    - is not affected by OpenMP settings
    - should always be set using the ORT APIs

### 1. Thread Management: Custom threading callbacks

ORT offers thread creation and joining callbacks using [C++ API](https://github.com/microsoft/onnxruntime/blob/master/include/onnxruntime/core/session/onnxruntime_cxx_api.h). This will allow customers to use their own fine-tuned threads for multithreading. 

Here is a code sample for ORT custom threading in C++.

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

For ORT global thread pool in C++, here is a code sample:

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

Note that the CreateThreadCustomized and JoinThreadCustomized settings will be applied to both the ORT IntraOp and the InterOp thread pools uniformly.

### 2. Thread Management: Default CPU Execution Provider (MLAS)

Microsoft Linear Algebra Subprogram (MLAS), the default execution provider, uses different knobs to control the thread number.

Here is a sample Python API code for the default CPU Execution Provider (MLAS).

```python
import onnxruntime as rt

sess_options = rt.SessionOptions()

sess_options.intra_op_num_threads = 2
sess_options.execution_mode = rt.ExecutionMode.ORT_SEQUENTIAL
sess_options.graph_optimization_level = rt.GraphOptimizationLevel.ORT_ENABLE_ALL
```

* Thread Count
  * `sess_options.intra_op_num_threads = 2` controls the number of threads to use to run the model
* Sequential vs. Parallel Execution
  * `sess_options.execution_mode = rt.ExecutionMode.ORT_SEQUENTIAL` controls whether the operators in the graph run sequentially or in parallel. Usually, when a model has many branches, setting this option to false will provide better performance.
  * When `sess_options.execution_mode = rt.ExecutionMode.ORT_PARALLEL`, you can set `sess_options.inter_op_num_threads` to control the number of threads used to parallelize the execution of the graph (across nodes).

* sess_options.graph_optimization_level = rt.GraphOptimizationLevel.ORT_ENABLE_ALL. Default is already ORT_ENABLE_ALL(99). Please see [onnxruntime_c_api.h](https://github.com/microsoft/onnxruntime/tree/master/include/onnxruntime/core/session/onnxruntime_c_api.h#L286)  (enum GraphOptimizationLevel) for the full list of all optimization levels. For details regarding available optimizations and usage please refer to the [Graph Optimizations Doc](graph-optimizations.md).

### 3. Thread Management: MKL_DNN/nGraph Execution Provider

Math Kernel Library for Deep Neural Networks (MKL_DNN) and nGraph (a C++ library for DNN) depend on OpenMp for parallelization. For those execution providers, we need to use the OpenMP environment variable to tune the performance. The most widely used environment variables are:

* OMP_NUM_THREADS=n
  * Controls the thread pool size

* OMP_WAIT_POLICY=PASSIVE/ACTIVE
  * Controls whether thread spinning is enabled
  * PASSIVE is also called throughput mode and will yield CPU after finishing current task
  * ACTIVE will not yield CPU, instead it will have a while loop to check whether the next task is ready
  * Use PASSIVE if your CPU usage already high, and use ACTIVE when you want to trade CPU with latency

<p><a href="#tips">Performance Tuning Tips</a></p>

<h3 id="iobinding">IO Binding</h3>

ONNX Runtime supports [Data-on-device](https://onnxruntime.ai/docs/api/python/api_summary.html#data-on-device). ORT allows custom data structure to support all data formats and allows users to place the **data** backing these on a device, for example, on a CUDA supported device. In ONNX Runtime, this is called IOBinding.

* When working with non-CPU execution providers, it is most efficient to have inputs (and/or outputs) arranged on the target device (abstracted by the execution provider used) prior to executing the graph (calling Run). When the input is not copied to the target device, ORT copies it from the CPU as part of the Run() call.
* Similarly, if the output is not pre-allocated on the device, ORT assumes that the output is requested on the CPU and copies it from the device as the last step of the Run() call. This obviously eats into the execution time of the graph misleading users into thinking ORT is slow when most of the time is spent in these copies. To address this issue, we've introduced the notion of IOBinding. The key idea is to arrange for inputs to be copied to the device and for outputs to be pre-allocated on the device prior to calling Run().

IO Binding is available in all the ORT language bindings. Here are the code snippets in various languages demonstrating the usage of this feature.

**C++ IOBinding**

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

**Python IOBinding**


Refer to the [Python API docs](https://onnxruntime.ai/docs/api/python). Follow the best practices on [ONNX Runtime Python binding](https://github.com/pybind/pybind11/blob/master/docs/faq.rst#someclass-declared-with-greater-visibility-than-the-type-of-its-field-someclassmember--wattributes).

You can refer to the Github issues that were resolved to optimize IO Binding.
- [Add IO Binding support for Bert Benchmark util](https://github.com/microsoft/onnxruntime/pull/10907)
- [ONNX GPU IO Binding](https://github.com/microsoft/onnxruntime/issues/8872)
- [IO Binding scenarios](https://github.com/microsoft/onnxruntime/pull/10651)


**C# IOBinding**

Refer to the [C# docs](https://github.com/microsoft/onnxruntime/blob/master/csharp/test/Microsoft.ML.OnnxRuntime.Tests/OrtIoBindingAllocationTest.cs)

<p><a href="#tips">Performance Tuning Tips</a></p>

<h3 id="customCUDA">CUDA Execution Providers</h3>

Tips to customize the CUDA Execution Provider are given in this article for <a href="#convolutionheavy">Convolution heavy models</a>, <a href="#convolutioninput">Convolution input padding</a>, and <a href="#cudagraphs">Using CUDA graphs</a>.

<p id="convolutionheavy"></p>

### 1. Convolution heavy models and the CUDA Execution Provider

ORT leverages CUDA Deep Neural Networks (CuDNN) for convolution operations.

- The first step in this process is to determine which "optimal" convolution algorithm to use while performing the convolution operation for the given input configuration (input shape, filter shape, and so on.) in each `Conv` node.
- The next step involves querying CuDNN for a "workspace" memory size and have this allocated so CuDNN can use this auxiliary memory while determining the "optimal" convolution algorithm to use. 
- By default, ORT clamps the workspace size to 32 MB which may lead to a sub-optimal convolution algorithm getting picked by CuDNN. To allow ORT to allocate the maximum possible workspace as determined by CuDNN, a provider option named `cudnn_conv_use_max_workspace` needs to get set (as shown in the following code snippet).
- Note that, using this flag may increase the peak memory usage by a factor (sometimes a few GBs) but this does help CuDNN pick the best convolution algorithm for the given input. We have found that this is an important flag to use while using an FP16 model as this allows CuDNN to pick tensor core algorithms for the convolution operations (if the hardware supports tensor core operations). This flag may or may not result in performance gains for other data types (`float` and `double`).

* Python

```python
providers = [("CUDAExecutionProvider", {"cudnn_conv_use_max_workspace": '1'})]
sess_options = ort.SessionOptions()
sess = ort.InferenceSession("my_conv_heavy_fp16_model.onnx",  sess_options = sess_options, providers=providers)
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

<p><a href="#customCUDA">CUDA EP</a> > <a href="#tips">Performance Tuning Tips</a> > <a href="#" id="back-to-top">Back to top</a></p>


<p id="convolutioninput"></p>

### 2. Convolution Input Padding in the CUDA EP

ORT leverages CuDNN for convolution operations. While CuDNN only takes 4-D or 5-D tensor as input for convolution operations, dimension padding is needed if the input is 3-D tensor.

Given an input tensor of shape [N, C, D], it can be padded to [N, C, D, 1] or [N, C, 1, D]. Both padding ways produce the same output.

However, the performance may differ due to the convolution algorithms selected, especially, on some devices such as A100. By default, the input is padded to [N, C, D, 1]. If [N, C, 1, D] is required, a provider option named `cudnn_conv1d_pad_to_nc1d` needs to be set (as shown below).

* Python

```python
providers = [("CUDAExecutionProvider", {"cudnn_conv1d_pad_to_nc1d": '1'})]
sess_options = ort.SessionOptions()
sess = ort.InferenceSession("my_conv_model.onnx",  sess_options = sess_options, providers=providers)
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

<p><a href="#customCUDA">CUDA EP</a> > <a href="#tips">Performance Tuning Tips</a> > <a href="#" id="back-to-top">Back to top</a></p>

<p id="cudagraphs"></p>

### 3. Using CUDA Graphs in the CUDA EP

NOTE: Please note that this feature is currently being offered in "preview" mode.

While using the CUDA EP, ORT supports the usage of [CUDA Graphs](https://developer.nvidia.com/blog/cuda-10-features-revealed/) to remove CPU overhead associated with launching CUDA kernels sequentially. To enable the usage of CUDA Graphs, use the provider option as shown in the samples below. Currently, there are some constraints with regards to using the CUDA Graphs feature which are listed here:

1. Models with control-flow ops, that is, models with `If`, `Loop`, and `Scan` ops are not supported

2. Usage of CUDA Graphs is limited to models where-in all the model ops (graph nodes) can be partitioned to the CUDA EP

3. The input/output types of models need to be tensors

4. Shapes of inputs/outputs cannot change across inference calls. Dynamic shape models are supported - the only constraint is that the input/output shapes should be the same across all inference calls

5. By design, [CUDA Graphs](https://developer.nvidia.com/blog/cuda-10-features-revealed/) is designed to read from/write to the same CUDA virtual memory addresses during the graph replaying step as it does during the graph capturing step. Due to this requirement, usage of this feature requires using IOBinding to bind memory which will be used as input(s)/output(s) for the CUDA Graph machinery to read from/write to (please refer the code samples given below)

6. While updating the input(s) for subsequent inference calls, the fresh input(s) need to be copied over to the corresponding CUDA memory location(s) of the bound `OrtValue` input(s) (please see samples below to see how this can be achieved). This is because of the "graph replay" will require reading inputs from the same CUDA virtual memory addresses

7. Multi-threaded usage is not supported currently, that is, `Run()` MAY NOT be invoked on the same `InferenceSession` object from multiple threads while using CUDA Graphs

NOTE: The very first `Run()` performs a variety of tasks under the hood like making CUDA memory allocations, capturing the CUDA graph for the model, and then performing a graph replay to ensure that the graph runs. Due to this, the latency associated with the first `Run()` is bound to be high. The subsequent `Run()`s only perform graph replays of the graph captured and cached in the first `Run()`.

* Python

```python
providers = [("CUDAExecutionProvider", {"enable_cuda_graph": '1'})]
sess_options = ort.SessionOptions()
sess = ort.InferenceSession("my_model.onnx",  sess_options = sess_options, providers=providers)

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

<p><a href="#customCUDA">CUDA EP</a> > <a href="#tips">Performance Tuning Tips</a> > <a href="#" id="back-to-top">Back to top</a></p>


</div>