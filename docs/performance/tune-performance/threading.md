---
title: Thread management
grand_parent: Performance
parent: Tune performance
nav_order: 3
---

# Thread management

## Contents
{: .no_toc }

* TOC placeholder
{:toc}


For the default CPU execution provider, setting defaults are provided to get fast inference performance. You can customize the performance using the following knobs in the API to control the thread count and other settings:

Python (Defaults):
```python
import onnxruntime as rt

sess_options = rt.SessionOptions()

sess_options.intra_op_num_threads = 0
sess_options.execution_mode = rt.ExecutionMode.ORT_SEQUENTIAL
sess_options.graph_optimization_level = rt.GraphOptimizationLevel.ORT_ENABLE_ALL
sess_options.add_session_config_entry("session.intra_op.allow_spinning", "1")
```


* INTRA Thread Count

  * Controls the _total_ number of INTRA threads to use to run the model. 
  * INTRA = parallelize computation _inside_ each operator
  * Default: (not specified or 0). `sess_options.intra_op_num_threads = 0`
    * INTRA Threads Total = Number of physical CPU Cores. Leaving at default also enables some affinitization (explained below)
    * E.g. 6-core machine (with 12 HT logical processors) = 6 total INTRA threads


* Sequential vs Parallel Execution

  * Controls whether _multiple_ operators in the graph (_across_ nodes) run sequentially or in parallel. 
  * Default: `sess_options.execution_mode = rt.ExecutionMode.ORT_SEQUENTIAL`
  * Usually when a model has many branches, setting this option to `ORT_PARALLEL` will provide better performance. This could also hurt performance on some models without many branches.
  * When `sess_options.execution_mode = rt.ExecutionMode.ORT_PARALLEL`, you can set `sess_options.inter_op_num_threads` to control the
number of threads used to parallelize the execution of the graph (_across_ nodes).


* Graph Optimization Level

  * Default: `sess_options.graph_optimization_level = rt.GraphOptimizationLevel.ORT_ENABLE_ALL` enables all optimizations. 
  * Please see [onnxruntime_c_api.h](https://github.com/microsoft/onnxruntime/tree/main/include/onnxruntime/core/session/onnxruntime_c_api.h#L286) (enum `GraphOptimizationLevel`) for the full list of all optimization levels. For details regarding available optimizations and usage, please refer to the [Graph Optimizations](../model-optimizations/graph-optimizations.md) documentation.

* Thread-Pool Spinning Behavior

  * Controls whether additional INTRA or INTER threads spin waiting for work. Provides faster inference but consumes more CPU cycles, resources, and power
  * Default: 1 (Enabled)


## Set number of intra-op threads

Onnxruntime sessions utilize multi-threading to parallelize computation _inside_ each operator.

By default with intra_op_num_threads=0 or not set, each session will start with the main thread on the 1st core (not affinitized). Then extra threads per additional physical core are created, and affinitized to that core (1 or 2 logical processors).

Customer could manually configure the total number of threads like:

[Python](https://onnxruntime.ai/docs/api/python/api_summary.html#onnxruntime.SessionOptions.intra_op_num_threads) (below) - [C/C++](https://onnxruntime.ai/docs/api/c/struct_ort_api.html) - [.NET/C#](https://onnxruntime.ai/docs/api/csharp/api/Microsoft.ML.OnnxRuntime.SessionOptions.html#Microsoft_ML_OnnxRuntime_SessionOptions_IntraOpNumThreads)
```python
sess_opt = SessionOptions()
sess_opt.intra_op_num_threads = 3
sess = ort.InferenceSession('model.onnx', sess_opt)
```

With the above configuration of 3 total threads, two extra threads will be created in the addtional INTRA pool, so along with the main calling thread, there will be three threads in total to participate in intra-op computation. However, if customer explicitly set the number of threads like showcased above, there will be no affinity set to any of the created thread.

In addition, Onnxruntime also allow customers to create a global intra-op thread pool to prevent overheated contentions among session thread pools, please find its usage [here](https://github.com/microsoft/onnxruntime/blob/68b5b2d7d33b6aa2d2b5cf8d89befb4a76e8e7d8/onnxruntime/test/global_thread_pools/test_main.cc#L98).

## Thread spinning behavior

Controls whether additional INTRA or INTER threads spin waiting for work. Provides faster inference but consumes more CPU cycles, resources, and power.

Example disabling spinning so WorkerLoop doesn't consume extra active cycles spinning waiting or attempting to steal work

[Python](https://onnxruntime.ai/docs/api/python/api_summary.html#onnxruntime.SessionOptions.add_session_config_entry) (below) - [C++](https://onnxruntime.ai/docs/api/c/struct_ort_api.html) - [.NET/C#](https://onnxruntime.ai/docs/api/csharp/api/Microsoft.ML.OnnxRuntime.SessionOptions.html#Microsoft_ML_OnnxRuntime_SessionOptions_AddSessionConfigEntry_System_String_System_String_) - [Keys](https://github.com/microsoft/onnxruntime/blob/main/include/onnxruntime/core/session/onnxruntime_session_options_config_keys.h#L90)
```python
sess_opt = SessionOptions()
sess_opt.AddConfigEntry(kOrtSessionOptionsConfigAllowIntraOpSpinning, "0")
sess_opt.AddConfigEntry(kOrtSessionOptionsConfigAllowInterOpSpinning, "0")
```

## Set number of inter-op threads

A inter-op thread pool is for parallelism _between_ operators, and will only be created when session execution mode set to parallel:

By default, inter-op thread pool will also have one thread per physical core.

[Python](https://onnxruntime.ai/docs/api/python/api_summary.html#onnxruntime.SessionOptions.inter_op_num_threads) (below) - [C/C++](https://onnxruntime.ai/docs/api/c/struct_ort_api.html) - [.NET/C#](https://onnxruntime.ai/docs/api/csharp/api/Microsoft.ML.OnnxRuntime.SessionOptions.html#Microsoft_ML_OnnxRuntime_SessionOptions_InterOpNumThreads)
```python
sess_opt = SessionOptions()
sess_opt.execution_mode  = ExecutionMode.ORT_PARALLEL
sess_opt.inter_op_num_threads = 3
sess = ort.InferenceSession('model.onnx', sess_opt)
```

## Set intra-op thread affinity

It is normally best to not set thread affinity and let the OS handle thread assignment for perf and power reasons. However, for certain scenarios, it may be beneficial to customize intra-op thread affinities, for example:
* There are multiple sessions run in parallel, customer might prefer their intra-op thread pools run on separate cores to avoid contention.
* Customer want to limit a intra-op thread pool to run on only one of the NUMA nodes to reduce overhead of expensive cache miss among nodes.

For session intra-op thread pool, please read the [configuration](https://github.com/microsoft/onnxruntime/blob/68b5b2d7d33b6aa2d2b5cf8d89befb4a76e8e7d8/include/onnxruntime/core/session/onnxruntime_session_options_config_keys.h#L180) and consume it like:

[Python](https://onnxruntime.ai/docs/api/python/api_summary.html#onnxruntime.SessionOptions.add_session_config_entry) (below) - [C++](https://onnxruntime.ai/docs/api/c/struct_ort_api.html) - [.NET/C#](https://onnxruntime.ai/docs/api/csharp/api/Microsoft.ML.OnnxRuntime.SessionOptions.html#Microsoft_ML_OnnxRuntime_SessionOptions_AddSessionConfigEntry_System_String_System_String_) - [Keys](https://github.com/microsoft/onnxruntime/blob/main/include/onnxruntime/core/session/onnxruntime_session_options_config_keys.h#L176)
```python
sess_opt = SessionOptions()
sess_opt.intra_op_num_threads = 3
sess_opt.add_session_config_entry('session.intra_op_thread_affinities', '1;2')
sess = ort.InferenceSession('model.onnx', sess_opt, ...)
```

For global thread pool, please read the [API](https://github.com/microsoft/onnxruntime/blob/68b5b2d7d33b6aa2d2b5cf8d89befb4a76e8e7d8/include/onnxruntime/core/session/onnxruntime_c_api.h#L3636) and [usage](https://github.com/microsoft/onnxruntime/blob/68b5b2d7d33b6aa2d2b5cf8d89befb4a76e8e7d8/onnxruntime/test/global_thread_pools/test_main.cc#L98).

## Numa support and performance tuning

Since release 1.14, Onnxruntime thread pool could utilize all physical cores that are available over NUMA nodes.
The intra-op thread pool will create an extra thread on every physical core (except the 1st core). E.g. assume there is a system of 2 NUMA nodes, each has 24 cores.
Hence intra-op thread pool will create 47 threads, and set thread affinity to each core.

For NUMA systems, it is recommended to test a few thread settings to explore for best performance, in that threads allocated among NUMA nodes might has higher cache-miss overhead when cooperating with each other. For example, when number of intra-op threads has to be 8, there are different ways to set affinity:

[Python]() (below) - [C++]() - [.NET/C#]()
```python
sess_opt = SessionOptions()
sess_opt.intra_op_num_threads = 8
sess_opt.add_session_config_entry('session.intra_op_thread_affinities', '3,4;5,6;7,8;9,10;11,12;13,14;15,16') # set affinities of all 7 threads to cores in the first NUMA node
# sess_opt.add_session_config_entry('session.intra_op_thread_affinities', '3,4;5,6;7,8;9,10;49,50;51,52;53,54') # set affinities for first 4 threads to the first NUMA node, and others to the second
sess = ort.InferenceSession('resnet50.onnx', sess_opt, ...)
```

Test showed that setting affinities to a single NUMA node has nearly 20 percent performance improvement aginst the other case.

## Custom threading callbacks

Occasionally, users may prefer to use their own fine-tuned threads for multithreading. ORT offers thread creation and joining callbacks in the [C++ API](https://github.com/microsoft/onnxruntime/blob/main/include/onnxruntime/core/session/onnxruntime_cxx_api.h):


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


Note that `CreateThreadCustomized` and `JoinThreadCustomized`, once  set, will be applied to both ORT intra op and inter op thread pools uniformly.



