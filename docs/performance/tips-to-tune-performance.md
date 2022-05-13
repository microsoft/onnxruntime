---
title: Performance Tuning Tips
parent: Performance
nav_order: 4
description: Tips to tune ONNX Runtime performance in terms of reducing memory consumption, thread management, IO Binding, and customizing CUDA Execution Provider.
redirect_from: /docs/how-to/tune-performance
---
<div class="container">

<p id="tips"></p>

# Tips for Tuning Performance

Here are some tips for tuning the performance of ORT in terms of <a href="#memory">reducing memory consumption</a>, <a href ="#thread">thread management</a>, and <a href="#iobinding">IO Binding</a>.

Please refer to [Execution Provider](../execution-providers/index.md) specific performance tuning samples and tips for additional ONNX Runtime best practices.

## Contents
{: .no_toc }

* TOC placeholder
{:toc}

<p id="memory"></p>

## Memory Consumption Reduction

Here are some tips to reduce memory consumption and tune performance with ORT.

### 1. Shared arena-based allocator
Memory consumption can be reduced between multiple sessions by configuring the shared arena-based allocation. See the `Share allocator(s) between sessions` section in the [C API documentation](../get-started/with-c.md).

### 2. Mimalloc allocator

OnnxRuntime supports overriding memory allocations using mimalloc allocator, which is a general-purpose fast allocator. See [mimalloc github](https://github.com/microsoft/mimalloc). 
- Depending on your model and usage mimalloc can deliver single- or double-digit improvements. The GitHub README page describes various scenarios on how mimalloc can be leveraged to support your scenarios.
- mimalloc is a submodule in the OnnxRuntime source tree. On Windows, one can employ `--use_mimalloc` build flag which would build a static version of mimalloc and link it to OnnxRuntime. This would redirect OnnxRuntime allocators and all new/delete calls to mimalloc. Currently, there are no special provisions to employ mimalloc on Linux. This can be done via LD_PRELAOD mechanism using pre-built binaries that you can build/obtain separately.
 
<p><a href="#">Back to top</a></p>

<p id="thread"></p>

## Thread Management

ONNX Runtime allows different [threading implementation](https://github.com/microsoft/onnxruntime/blob/master/docs/NotesOnThreading.md) choices for OpenMP or non-OpenMP. Here are some best practices for [thread management](https://github.com/microsoft/onnxruntime/blob/master/docs/FAQ.md#how-do-i-force-single-threaded-execution-mode-in-ort-by-default-sessionrun-uses-all-the-computers-cores) to customize your ONNX Runtime environment:

* If ORT is built with OpenMP, use the OpenMP env variable to control the number of IntraOp num threads.
* If ORT is not built with OpenMP, use the appropriate ORT API to control IntraOp num threads.
* InterOp num threads setting: 
    - is used only when parallel execution is enabled
    - is not affected by OpenMP settings
    - should always be set using the ORT APIs

<p><a href="#thread">Thread Management</a> > <a href="#">Back to top</a></p>

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


<p><a href="#thread">Thread Management</a> > <a href="#">Back to top</a></p>

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

<p><a href="#thread">Thread Management</a> > <a href="#">Back to top</a></p>

### 3. Thread Management: MKL_DNN/nGraph Execution Provider

Math Kernel Library for Deep Neural Networks (MKL_DNN) and nGraph (a C++ library for DNN) depend on OpenMp for parallelization. For those execution providers, we need to use the OpenMP environment variable to tune the performance. The most widely used environment variables are:

* OMP_NUM_THREADS=n
  * Controls the thread pool size

* OMP_WAIT_POLICY=PASSIVE/ACTIVE
  * Controls whether thread spinning is enabled
  * PASSIVE is also called throughput mode and will yield CPU after finishing current task
  * ACTIVE will not yield CPU, instead it will have a while loop to check whether the next task is ready
  * Use PASSIVE if your CPU usage already high, and use ACTIVE when you want to trade CPU with latency

<p><a href="#">Back to top</a></p>

<p id="iobinding"></p>

## IO Binding

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

<p><a href="#">Back to top</a></p>




</div>