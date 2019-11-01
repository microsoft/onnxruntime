# ONNX Runtime Performance Tuning

## Why do we need to tune performance?
ONNX Runtime is designed to be open and extensible with its concept of "Execution Provider" to represents different execution kernels. See the [design overview](./HighLevelDesign.md). 

ONNX Runtime supports a variety of execution providers across CPU and GPU: [see the list here](../README.md#high-performance).
For different models and different hardware, there is no silver bullet which can always perform the best. Even for a single execution provider, often there are several knobs that can be tuned (e.g. thread number, wait policy etc.).

This document covers basic tools and knobs that can be leveraged to find the best performance for your model and hardware.

## Is there a tool to help with performance tuning?
Yes, the onnxruntime_perf_test.exe tool (available from the build drop) can be used to test various knobs. Please find the usage instructions using `onnxruntime_perf_test.exe -h`.

Additionally, the [ONNX Go Live "OLive" tool](https://github.com/microsoft/OLive) provides an easy-to-use pipeline for converting models to ONNX and optimizing performance with ONNX Runtime. The tool can help identify the optimal runtime configuration to get the best performance on the target hardware for the model.

## Using different execution providers

### Python API
Official Python packages on Pypi only support the default CPU (MLAS) and default GPU (CUDA) execution providers. For other execution providers, you need to build from source. Please refer to the [build instructions](../BUILD.md). The recommended instructions build the wheel with debug info in parallel.

For example: 

`MKLDNN:		 ./build.sh --config RelWithDebInfo --use_mkldnn --build_wheel --parallel`

` CUDA:	     ./build.sh --config RelWithDebInfo --use_cuda  --build_wheel --parallel`


### C and C# API
Official release (nuget package) supports default (MLAS) and MKL-ML for CPU, and CUDA for GPU. For other execution providers, you need to build from source. Append `--build_csharp` to the instructions to build both C# and C packages.

For example:

`MKLDNN:		 ./build.sh --config RelWithDebInfo --use_mkldnn --build_csharp --parallel`

`CUDA:	     ./build.sh --config RelWithDebInfo --use_cuda  --build_csharp --parallel`

In order to use MKLDNN, nGraph, CUDA, or TensorRT execution provider, you need to call the C API OrtSessionOptionsAppendExecutionProvider. Here is an example for the CUDA execution provider:

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
session = rt.InferenceSession(model, sess_options=so)
session.set_providers(['CUDAExecutionProvider'])
```
## How to tune performance for a specific execution provider?

### Default CPU Execution Provider (MLAS)
The default execution provider uses different knobs to control the thread number.

For the default CPU execution provider, you can try following knobs in the Python API:
```python
import onnxruntime as rt

sess_options = rt.SessionOptions()

sess_options.intra_op_num_threads = 2
sess_options.execution_mode = rt.ExecutionMode.ORT_SEQUENTIAL
sess_options.graph_optimization_level = rt.GraphOptimizationLevel.ORT_ENABLE_ALL
```

* Thread Count
  * `sess_options.intra_op_num_threads = 2` controls the number of threads to use to run the model
* Sequential vs Parallel Execution
  * `sess_options.execution_mode = rt.ExecutionMode.ORT_SEQUENTIAL` controls whether then operators in the graph should run sequentially or in parallel. Usually when a model has many branches, setting this option to false will provide better performance.
  * When `sess_options.execution_mode = rt.ExecutionMode.ORT_PARALLEL`, you can set `sess_options.inter_op_num_threads` to control the

number of threads used to parallelize the execution of the graph (across nodes).
* sess_options.graph_optimization_level = rt.GraphOptimizationLevel.ORT_ENABLE_ALL. Default is ORT_ENABLE_BASIC(1). Please see [onnxruntime_c_api.h](../include/onnxruntime/core/session/onnxruntime_c_api.h#L241)  (enum GraphOptimizationLevel) for the full list of all optimization levels. For details regarding available optimizations and usage please refer to the [Graph Optimizations Doc](../docs/ONNX_Runtime_Graph_Optimizations.md).

### MKL_DNN/nGraph/MKL_ML Execution Provider
MKL_DNN, MKL_ML and nGraph all depends on openmp for parallization. For those execution providers, we need to use the openmp enviroment variable to tune the performance.

The most widely used enviroment variables are:

* OMP_NUM_THREADS=n
  * Controls the thread pool size

* OMP_WAIT_POLICY=PASSIVE/ACTIVE
  * Controls whether thread spinning is enabled
  * PASSIVE is also called throughput mode and will yield CPU after finishing current task
  * ACTIVE will not yield CPU, instead it will have a while loop to check whether the next task is ready
  * Use PASSIVE if your CPU usage already high, and use ACTIVE when you want to trade CPU with latency



## Profiling and Performance Report

You can enable ONNX Runtime latency profiling in code:

```python
import onnxruntime as rt

sess_options = rt.SessionOptions()
sess_options.enable_profiling = True
```
If you are using the onnxruntime_perf_test.exe tool, you can add `-p [profile_file]` to enable performance profiling.

In both cases, you will get a JSON file which contains the detailed performance data (threading, latency of each operator, etc). This file is a standard performance tracing file, and to view it in a user friendly way, you can open it by using chrome://tracing:
* Open chrome browser
* Type chrome://tracing in the address bar
* Load the generated JSON file
