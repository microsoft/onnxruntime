---
title: Tune performance
parent: How to
nav_order: 3
---

# ONNX Runtime Performance Tuning
{: .no_toc }

ONNX Runtime provides high performance across a range of hardware options through its [Execution Providers interface](../reference/execution-providers) for different execution environments.

Along with this flexibility comes decisions for tuning and usage. For each model running with each execution provider, there are settings that can be tuned (e.g. thread number, wait policy, etc) to improve performance.

This document covers basic tools and knobs that can be leveraged to find the best performance for your model and hardware.

## Contents
{: .no_toc }

* TOC placeholder
{:toc}

## Performance Tuning Tools

### ONNX GO Live Tool
{: .no_toc }

The [ONNX Go Live "OLive" tool](https://github.com/microsoft/OLive) is an easy-to-use pipeline for converting models to ONNX and optimizing performance with ONNX Runtime. The tool can help identify the optimal runtime configuration to get the best performance on the target hardware for the model.

As a quickstart, please see the notebooks: [Python](https://github.com/microsoft/OLive/blob/master/notebook/Convert_Models_and_Tune_Performance_with_OLive_Python_SDK.ipynb), [Docker images](https://github.com/microsoft/OLive/blob/master/notebook/Convert_Models_and_Tune_Performance_with_OLive_Docker_Images.ipynb)

### Profiling and Performance Report
{: .no_toc }

The onnxruntime_perf_test.exe tool (available from the build drop) can be used to test various knobs. Please find the usage instructions using `onnxruntime_perf_test.exe -h`.

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

## Using different Execution Providers

To learn more about different Execution Providers, see [Reference: Execution Providers](../reference/execution-providers).

### Build the EP
{: .no_toc }

**Python**

Official Python packages on Pypi only support the default CPU (MLAS) and default GPU (CUDA) execution providers. For other execution providers, you need to [build from source](../how-to/build/eps.md). The recommended instructions build the wheel with debug info in parallel.

For example:

`DNNL:		 ./build.sh --config RelWithDebInfo --use_dnnl --build_wheel --parallel`

`CUDA:	     ./build.sh --config RelWithDebInfo --use_cuda  --build_wheel --parallel`

**C and C#**

Official releases on Nuget support default (MLAS) for CPU, and CUDA for GPU. For other execution providers, you need to build from source. Append `--build_csharp` to the instructions to build both C# and C packages.

For example:

`DNNL:		 ./build.sh --config RelWithDebInfo --use_dnnl --build_csharp --parallel`

`CUDA:	     ./build.sh --config RelWithDebInfo --use_cuda  --build_csharp --parallel`


### Register the EP
{: .no_toc }

In order to use DNNL, CUDA, or TensorRT execution provider, you need to call the C API OrtSessionOptionsAppendExecutionProvider.

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

## Which Execution Provider will provide the best performance? 

Performance is dependent on the specific model you're trying to run, the session and run options you've selected, and of course, your specific hardware target. Below you'll find some more information that may be helpful to select the right Execution Provider.

### CUDA (Default GPU) or CPU?
{: .no_toc }
The CPU version of ONNX Runtime provides a complete implementation of all operators in the ONNX spec. This ensures that your ONNX-compliant model can execute successfully. In order to keep the binary size small, common data types are supported for the ops. If you are using an uncommon data type that is not supported, you can file an issue and/or contribute a PR (see examples - [PR #2112](https://github.com/microsoft/onnxruntime/pull/2112), [PR #2034](https://github.com/microsoft/onnxruntime/pull/2034), [PR #1565](https://github.com/microsoft/onnxruntime/pull/1565)). Please make sure you provide details on usage justification.

Additionally, not all CUDA kernels are implemented, as these have been prioritized on an as-needed basis. This means that if your model contains operators that do not have a CUDA implementation, it will fall back to CPU. Switching between CPU and GPU can cause significant performance impact. If you require a specific operator that is not currently supported, please consider [contributing](https://github.com/microsoft/onnxruntime/tree/master/CONTRIBUTING.md) and/or [file an issue](https://github.com/microsoft/onnxruntime/issues) clearly describing your use case and share your model if possible.

### TensorRT or CUDA?
{: .no_toc }
TensorRT and CUDA are separate execution providers for ONNX Runtime. On the same hardware, TensorRT will generally provide better performance; however, this depends on the specific model and whether the operators in the model can be supported by TensorRT. In cases where TensorRT cannot handle the subgraph(s), it will fall back to CUDA. Note that the TensorRT EP may depend on a different version of CUDA than the CUDA EP.

### TensorRT/CUDA or DirectML? 
{: .no_toc }
DirectML is the hardware-accelerated DirectX 12 library for machine learning on Windows and supports all DirectX 12 capable devices (Nvidia, Intel, AMD). This means that if you are targeting Windows GPUs, using the DirectML Execution Provider is likely your best bet. This can be used with both the ONNX Runtime as well as [WinML APIs](../reference/api/winrt-api.md).

## Tuning performance
Below are some suggestions for things to try for various EPs for tuning performance. 

### Shared arena based allocator
Memory consumption can be reduced between multiple sessions by configuring the shared arena based allocation. See the `Share allocator(s) between sessions` section in the [C API documentation](../reference/api/c-api).

### Thread management

* If ORT is built with OpenMP, use the OpenMP env variable to control the number of intra op num threads.
* If ORT is not built with OpenMP, use the appropriate ORT API to control intra op num threads.
* Inter op num threads (used only when parallel execution is enabled) is not affected by OpenMP settings and should
always be set using the ORT APIs.

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
  * `sess_options.execution_mode = rt.ExecutionMode.ORT_SEQUENTIAL` controls whether the operators in the graph run sequentially or in parallel. Usually when a model has many branches, setting this option to false will provide better performance.
  * When `sess_options.execution_mode = rt.ExecutionMode.ORT_PARALLEL`, you can set `sess_options.inter_op_num_threads` to control the
number of threads used to parallelize the execution of the graph (across nodes).

* sess_options.graph_optimization_level = rt.GraphOptimizationLevel.ORT_ENABLE_ALL. Default is already ORT_ENABLE_ALL(99). Please see [onnxruntime_c_api.h](https://github.com/microsoft/onnxruntime/tree/master/include/onnxruntime/core/session/onnxruntime_c_api.h#L241)  (enum GraphOptimizationLevel) for the full list of all optimization levels. For details regarding available optimizations and usage please refer to the [Graph Optimizations Doc](../resources/graph-optimizations.md).

### MKL_DNN/nGraph Execution Provider

MKL_DNN and nGraph depend on openmp for parallelization. For those execution providers, we need to use the openmp environment variable to tune the performance.

The most widely used environment variables are:

* OMP_NUM_THREADS=n
  * Controls the thread pool size

* OMP_WAIT_POLICY=PASSIVE/ACTIVE
  * Controls whether thread spinning is enabled
  * PASSIVE is also called throughput mode and will yield CPU after finishing current task
  * ACTIVE will not yield CPU, instead it will have a while loop to check whether the next task is ready
  * Use PASSIVE if your CPU usage already high, and use ACTIVE when you want to trade CPU with latency

## Troubleshooting performance issues

The answers below are troubleshooting suggestions based on common previous user-filed issues and questions. This list is by no means exhaustive and there is a lot of case-by-case fluctuation depending on the model and specific usage scenario. Please use this information to guide your troubleshooting, search through previously filed issues for related topics, and/or file a new issue if your problem is still not resolved.


### Performance Troubleshooting Checklist
{: .no_toc }

Here is a list of things to check through when assessing performance issues.
* Are you using OpenMP? OpenMP will parallelize some of the code for potential performance improvements. This is not recommended for running on single threads.
* Have you enabled all [graph optimizations](../resources/graph-optimizations.md)? The official published packages do enable all by default, but when building from source, check that these are enabled in your build.
* Have you searched through prior filed [Github issues](https://github.com/microsoft/onnxruntime/issues) to see if your problem has been discussed previously? Please do this before filing new issues.
* If using CUDA or TensorRT, do you have the right versions of the dependent libraries installed? 

### I need help performance tuning for BERT models
{: .no_toc }

For BERT models, sometimes ONNX Runtime cannot apply the best optimization due to reasons such as framework version updates. We recommend trying out the [BERT optimization tool](https://github.com/microsoft/onnxruntime/tree/master/onnxruntime/python/tools/transformers), which reflects the latest changes in graph pattern matching and model conversions, and a set of [notebooks](https://github.com/microsoft/onnxruntime/tree/master/onnxruntime/python/tools/transformers/notebooks) to help get started.

### Why is the model graph not optimized even with graph_optimization_level set to ORT_ENABLE_ALL?
{: .no_toc }

The ONNX model from IR_VERSION 4 only treats initializers that appear in graph input as non-constant. This may fail some of the graph optimizations, like const folding, operator fusion and etc. Move initializers out of graph inputs if there is no need to override them, by either re-generating the model with latest exporter/converter or with the tool [remove_initializer_from_input.py](https://github.com/microsoft/onnxruntime/tree/master/tools/python/remove_initializer_from_input.py).

### Why is my model running slower on GPU than CPU?
{: .no_toc }

Depending on which execution provider you're using, it may not have full support for all the operators in your model. Fallback to CPU ops can cause hits in performance speed. Moreover even if an op is implemented by the CUDA execution provider, it may not necessarily assign/place the op to the CUDA EP due to performance reasons. To see the placement decided by ORT, turn on verbose logging and look at the console output.

### My converted Tensorflow model is slow - why?
{: .no_toc }

NCHW and NHWC are two different memory layout for 4-D tensors.

Most TensorFlow operations used by a CNN support both NHWC and NCHW data format. The Tensorflow team suggests that on GPU NCHW is faster but on CPU NHWC is sometimes faster in Tensorflow. However, ONNX only supports NCHW. As a result, if the original model is in NHWC format, when the model is converted extra transposes may be added. The [tensorflow-onnx](https://github.com/onnx/tensorflow-onnx) and [keras-onnx](https://github.com/onnx/keras-onnx) converters do remove many of these transposes, but if this doesn't help sufficiently, consider retraining the model using NCHW.
