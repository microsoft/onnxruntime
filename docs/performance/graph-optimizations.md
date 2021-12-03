---
title: Graph optimizations
parent: Performance
nav_order: 3
redirect_from: /docs/how-to/graph-optimizations
---

# Graph Optimizations in ONNX Runtime
{: .no_toc }

ONNX Runtime provides various graph optimizations to improve performance. Graph optimizations are essentially graph-level transformations, ranging from small graph simplifications and node eliminations to more complex node fusions and layout optimizations.

Graph optimizations are divided in several categories (or *levels*) based on their complexity and functionality. They can be performed either *online* or *offline*. In online mode, the optimizations are done before performing the inference, while in offline mode, the runtime saves the optimized graph to disk. ONNX Runtime provides Python, C#, C++, and C APIs to enable different optimization levels and to choose between offline vs. online mode.

Below we provide details on the optimization levels, the online/offline mode, and the various APIs to control them.

## Contents
{: .no_toc }

* TOC placeholder
{:toc}

## Graph Optimization Levels

Graph optimizations are divided into three levels:

1. Basic
2. Extended
3. Layout Optimizations

The optimizations belonging to one level are performed after the optimizations of the previous level have been applied (e.g., extended optimizations are applied after basic optimizations have been applied).

**All optimizations are enabled by default.**

### Basic Graph Optimizations

These are semantics-preserving graph rewrites which remove redundant nodes and redundant computation. They run before graph partitioning and thus apply to all the execution providers. Available basic graph optimizations are as follows:

* Constant Folding: Statically computes parts of the graph that rely only on constant initializers. This eliminates the need to compute them during runtime.

* Redundant node eliminations: Remove all redundant nodes without changing the graph structure. The following such optimizations are currently supported:
  * Identity Elimination
  * Slice Elimination
  * Unsqueeze Elimination
  * Dropout Elimination

* Semantics-preserving node fusions : Fuse/fold multiple nodes into a single node. For example, Conv Add fusion folds the Add operator as the bias of the Conv operator. The following such optimizations are currently supported:
  * Conv Add Fusion
  * Conv Mul Fusion
  * Conv BatchNorm Fusion
  * Relu Clip Fusion
  * Reshape Fusion

### Extended Graph Optimizations

These optimizations include complex node fusions. They are run after graph partitioning and are only applied to the nodes assigned to the CPU or CUDA execution provider. Available extended graph optimizations are as follows:

| Optimization                    | Execution Provider | Comment                                                                     |
|---------------------------------|--------------------|-----------------------------------------------------------------------------|
| GEMM Activation Fusion          | CPU                |                                                                             |
| Matmul Add Fusion               | CPU                |                                                                             |
| Conv Activation Fusion          | CPU                |                                                                             |
| GELU Fusion                     | CPU or CUDA        |                                                                             |
| Layer Normalization Fusion      | CPU or CUDA        |                                                                             |
| BERT Embedding Layer Fusion     | CPU or CUDA        | Fuse BERT embedding layer, layer normalization and attention mask length    |
| Attention Fusion*               | CPU or CUDA        |                                                                             |
| Skip Layer Normalization Fusion | CPU or CUDA        | Fuse bias of fully connected layer, skip connection and layer normalization |
| Bias GELU Fusion                | CPU or CUDA        | Fuse bias of fully connected layer and GELU activation                      |
| GELU Approximation*             | CUDA               | Disabled by default. Enable with [kOrtSessionOptionsEnableGeluApproximation](https://cs.github.com/microsoft/onnxruntime/blob/175acf08f470db0bb2e4b8eefe55cdeb87c8b132/include/onnxruntime/core/session/onnxruntime_session_options_config_keys.h?q=kOrtSessionOptionsEnableGeluApproximation#L52) |


<details>
  <summary>* Approximations (click to expand)</summary>
  <p>
  To optimize performance of <a href="https://en.wikipedia.org/wiki/BERT_(language_model)">BERT</a>, approximation is used in GELU Approximation and Attention Fusion for CUDA execution provider. The impact on accuracy is negligible based on our evaluation: F1 score for a BERT model on SQuAD v1.1 is almost same (87.05 vs 87.03).
  </p>
</details>

### Layout Optimizations

These optimizations change the data layout for applicable nodes to achieve higher performance improvements. They are run after graph partitioning and are only applied to nodes assigned to CPU execution provider. Available layout optimizations are as follows:

* NCHWc Optimizer: Optimizes the graph by using NCHWc layout instead of NCHW layout.

## Online/Offline Mode

All optimizations can be performed either online or offline. In online mode, when initializing an inference session, we also apply all enabled graph optimizations before performing model inference. Applying all optimizations each time we initiate a session can add overhead to the model startup time (especially for complex models), which can be critical in production scenarios. This is where the offline mode can bring a lot of benefit. In offline mode, after performing graph optimizations, ONNX Runtime serializes the resulting model to disk. Subsequently, we can reduce startup time by using the already optimized model and disabling all optimizations.

**Notes**:

* When running in offline mode, make sure to use the exact same options (e.g., execution providers, optimization level) and hardware as the target machine that the model inference will run on (e.g., you cannot run a model pre-optimized for a GPU execution provider on a machine that is equipped only with CPU).
* When layout optimizations are enabled, the offline mode can only be used on compatible hardware to the environment when the offline model is saved. For example, if model has layout optimized for AVX2, the offline model would require CPUs that support AVX2.

## Usage

### Levels

ONNX Runtime defines the `GraphOptimizationLevel` enum to determine which of the aforementioned optimization levels will be enabled. Choosing a level enables the optimizations of that level, as well as the optimizations of all preceding levels. For example, enabling Extended optimizations, also enables Basic optimizations. The mapping of these levels to the enum is as follows:

* GraphOptimizationLevel::ORT_DISABLE_ALL -> Disables all optimizations
* GraphOptimizationLevel::ORT_ENABLE_BASIC -> Enables basic optimizations
* GraphOptimizationLevel::ORT_ENABLE_EXTENDED -> Enables basic and extended optimizations
* GraphOptimizationLevel::ORT_ENABLE_ALL -> Enables all available optimizations including layout optimizations

### Offline mode

To enable serialization of the optimized model to disk, set the SessionOptions option `optimized_model_filepath`.

#### Python API Example
```python
import onnxruntime as rt

sess_options = rt.SessionOptions()

# Set graph optimization level
sess_options.graph_optimization_level = rt.GraphOptimizationLevel.ORT_ENABLE_EXTENDED

# To enable model serialization after graph optimization set this
sess_options.optimized_model_filepath = "<model_output_path\optimized_model.onnx>"

session = rt.InferenceSession("<model_path>", sess_options)
```

#### C API Example
```c
  const OrtApi* Ort::g_api = OrtGetApi(ORT_API_VERSION);
  OrtEnv* env;
  g_ort->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "test", &env);
  OrtSessionOptions* session_options;
  g_ort->CreateSessionOptions(&session_options)

  // Set graph optimization level
  g_ort->SetSessionGraphOptimizationLevel(session_options, ORT_ENABLE_EXTENDED);

  // To enable model serialization after graph optimization set this
  const wchar_t* optimized_model_path = L"optimized_model_path";
  g_ort->SetOptimizedModelFilePath(session_options, optimized_model_path);

  OrtSession* session;
  const wchar_t* model_path = L"model_path";
  g_ort->CreateSession(env, model_path, session_option, &session);
```

#### C# API Example
```c#
SessionOptions so = new SessionOptions();

// Set graph optimization level
so.GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_EXTENDED;

// To enable model serialization after graph optimization set this
so.OptimizedModelFilePath = "model_output_path\optimized_model.onnx"

var session = new InferenceSession(modelPath, so);
```

#### C++ API Example
```c++
Ort::SessionOptions session_options;

// Set graph optimization level
session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

// To enable model serialization after graph optimization set this
session_options.SetOptimizedModelFilePath("optimized_file_path");

auto session_ = Ort::Session(env, "model_file_path", session_options);
```
