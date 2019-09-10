# How to tune ONNX Runtime Performance?

## Why do we need to tune performance?
ONNX Runtime is designed to be open and scalable, it created the concept of "Execution Provider" to represents different execution kernels. 

ONNX Runtime right now supports 4 CPU execution providers, which are, default(MLAS), MKL-ML, MKLDNN and nGraph. For nVidia GPU, we support CUDA and TensorRT execution providers. (Technically, MKL-ML is not an formal execution provider since it can only enabled by using build options and does not support GetCapability interface.)
For different models and different hardware, there is no silver bullet which can always perform the best. And even for a single execution provider, many times you have several knobs to tune, like thread number, wait policy etc.

This document will document some basic tools and knobs you could leverage to find the best performace for your model and your hardware.


## How do I use different execution providers?
**Please be kindly noted that this is subject to change. We will try to make it consistent and easy to use across different language bindings in the future.**

### Python API
For official python package which are released to Pypi, we only support the default CPU (MLAS) and default GPU execution provider. If you want to get other execution providers,
you need to build from source.

Here are the build instructions:
* MKLDNN:		 ./build.sh --config RelWithDebInfo --use_mkldnn --build_wheel --parallel
* MKLML:		 ./build.sh --config RelWithDebInfo --use_mklml --build_wheel --parallel
* nGraph:		 ./build.sh --config RelWithDebInfo --use_ngraph  --build_wheel --parallel
* CUDA:	     ./build.sh --config RelWithDebInfo --use_cuda  --build_wheel --parallel
* TensorRT:	 ./build.sh --config RelWithDebInfo --use_tensorrt --build_wheel --parallel

### C and C# API
Official release (nuget package) supports default (MLAS) and MKL-ML for CPU, and CUDA for GPU. For other execution providers, you need to build from source.

Similarly, here are the cmds to build from source. --build_csharp will build both C# and C package.
* MKLDNN:		 ./build.sh --config RelWithDebInfo --use_mkldnn --build_csharp --parallel
* MKLML:		 ./build.sh --config RelWithDebInfo --use_mklml --build_csharp --parallel
* nGraph:		 ./build.sh --config RelWithDebInfo --use_ngraph  --build_csharp --parallel
* CUDA:	     ./build.sh --config RelWithDebInfo --use_cuda  --build_csharp --parallel
* TensorRT:	 ./build.sh --config RelWithDebInfo --use_tensorrt --build_csharp --parallel

In order to use MKLDNN, nGraph, CUDA, or TensorRT execution provider, you need to call C API OrtSessionOptionsAppendExecutionProvider. Here is one example for CUDA execution provider:

C API Example:
```c
  OrtEnv* env;
  OrtInitialize(ORT_LOGGING_LEVEL_WARNING, "test", &env)
  OrtSessionOptions* session_option = OrtCreateSessionOptions();
  OrtProviderFactoryInterface** factory;
  OrtCreateCUDAExecutionProviderFactory(0, &factory);
  OrtSessionOptionsAppendExecutionProvider(session_option, factory);
  OrtReleaseObject(factory);
  OrtCreateSession(env, model_path, session_option, &session);
```

C# API Example:
```c#
SessionOptions so = new SessionOptions();
so.AppendExecutionProvider(ExecutionProvider.MklDnn);
var session = new InferenceSession(modelPath, so);
```

## How to tune performance for a specific execution provider?

### Default CPU Execution Provider (MLAS)
Default execution provider use different knobs to control thread number. Here are some details:
** Please kindly noted that those are subject to change in the future**

For default CPU execution provider, you can try following knobs in Python API:
```python
import onnxruntime as rt

sess_options = rt.SessionOptions()

sess_options.session_thread_pool_size=2
sess_options.enable_sequential_execution=True
sess_options.set_graph_optimization_level(2)
```
* sess_options.session_thread_pool_size=2 controls how many thread do you want to use to run your model
* sess_options.enable_sequential_execution=True controls whether you want to run operators in your graph sequentially or in parallel. Usually when your model has many branches, set this option to false will give you better performance.
* sess_options.set_graph_optimization_level(2). Default is 1. Please see [onnxruntime_c_api.h](../include/onnxruntime/core/session/onnxruntime_c_api.h#L241)  (enum GraphOptimizationLevel) for the full list of all optimization levels.

### MKL_DNN/nGraph/MKL_ML Execution Provider
MKL_DNN, MKL_ML and nGraph all depends on openmp for parallization. For those execution providers, we need to use openmp enviroment variable to tune the performance.

The most widely used enviroment variables are:
* OMP_NUM_THREADS=n
* OMP_WAIT_POLICY=PASSIVE/ACTIVE

As you can tell from the name, OMP_NUM_THREADS controls the thread pool size, while OMP_WAIT_POLICY controls whether enable thread spinning or not. 
OMP_WAIT_POLICY=PASSIVE is also called throughput mode, it will yield CPU after finishing current task. OMP_WAIT_POLICY=ACTIVE will not yield CPU, instead it will have a while loop to check
whether next task is ready or not. Use PASSIVE if your CPU usage already high, use ACTIVE when you want to trade CPU with latency.

## Is there a tool to help tune the performance easily?
Yes, we have created a tool named onnxruntime_perf_test.exe, and you find it at the build drop.
You can use this tool to test all those knobs easily. Please find the usage of this tool by onnxruntime_perf_test.exe -h

## How to enable profiling and view the generated JSON file?

You can enable ONNX Runtime latency profiling in code:

```python
import onnxruntime as rt

sess_options = rt.SessionOptions()
sess_options.enable_profiling = True
```
Or, if you are using the onnxruntime_perf_test.exe tool, you can add -p [profile_file] to enable performance profiling.

In both ways, you will get a JSON file, which contains the detailed performance data (threading, latency of each operator, etc). This file is a standard performance tracing file, and to view it in a user friendly way, you can open it by using chrome://tracing:
* Open chrome browser
* Type chrome://tracing in the address bar
* Load the generated JSON file

