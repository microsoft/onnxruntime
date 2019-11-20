# Graph Optimizations in ONNX Runtime

ONNX Runtime provides various graph optimizations to improve model performance. Graph optimizations are essentially graph-level transformations, ranging from small graph simplifications and node eliminations to more complex node fusions and layout optimizations.

Graph optimizations are divided in several categories (or *levels*) based on their complexity and functionality. They can be performed either *online* or *offline*. In online mode, the optimizations are done before performing the inference, while in offline mode, the runtime saves the optimized graph to disk. ONNX Runtime provides Python, C#, C++, and C APIs to enable different optimization levels and to choose between offline vs. online mode.

Below we provide details on the optimization levels, the online/offline mode, and the various APIs to control them.

## Graph Optimization Levels

Graph optimizations are divided in three levels:
* Basic
* Extended
* Layout Optimizations

The optimizations belonging to one level are performed after the optimizations of the previous level have been applied (e.g., extended optimizations are applied after basic optimizations have been applied).

### Basic Graph Optimizations

These are semantics-preserving graph rewrites which remove redundant nodes and redundant computation. These optimizations are enabled by default. They run before graph partitioning and thus apply to all the execution providers. Available basic graph optimizations are as follows:

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

### Extended Graph Optimizations

These optimizations include complex node fusions. They are run after graph partitioning and are only applied to the nodes assigned to the CPU execution provider. Available extended graph optimizations are as follows:

* GEMM Activation Fusion
* Matmul Add Fusion
* Conv Activation Fusion
* GELU Fusion

### Layout Optimizations

These optimizations change the data layout for applicable nodes to achieve higher performance improvements. They are run after graph partitioning and are only applied to nodes assigned to CPU execution provider. Available layout optimizations are as follows:

* NCHWc Optimizer: Optimizes the graph by using NCHWc layout instead of NCHW layout.

## Online/Offline Mode

All optimizations can be performed either online or offline. In online mode, when initializing an inference session, we also apply all enabled graph optimizations before performing model inference. Applying all optimizations each time we initiate a session can add overhead to the model startup time (especially for complex models), which can be critical in production scenarios. This is where the offline mode can bring a lot of benefit. In offline mode, after performing graph optimizations, ONNX Runtime serializes the resulting model to disk. Subsequently, when new inference sessions are created for this model, we can instead use the already optimized model to reduce startup time.

**Notes**: 

* When running in offline mode, make sure to use the exact same options (e.g., execution providers, optimization level) and hardware as the target machine that the model inference will run on (e.g., you cannot run a model pre-optimized for a GPU execution provider on a machine that is equipped only with CPU).
* When layout optimizations are enabled, the offline mode can only be used on compatible hardware to the environment when the offline model is saved. For example, if model has layout optimized for AVX2, the offline model would require CPUs that support AVX2.

## Usage

### General Note
**Levels**: 
ONNX Runtime defines the `GraphOptimizationLevel` enum to determine which of the aforementioned optimization levels will be enabled. Choosing a level enables the optimizations of that level, as well as the optimizations of all preceding levels. For example, enabling Extended optimizations, also enables Basic optimizations. The mapping of these levels to the enum is as follows:

* GraphOptimizationLevel::ORT_DISABLE_ALL -> Disables all optimizations
* GraphOptimizationLevel::ORT_ENABLE_BASIC -> Enables basic optimizations
* GraphOptimizationLevel::ORT_ENABLE_EXTENDED -> Enables basic and extended optimizations
* GraphOptimizationLevel::ORT_ENABLE_ALL -> Enables all available optimizations including layout optimizations

**Online/Offline Mode**:
To enable serialization of the optimized model to disk, set the SessionOptions option `optimized_model_path` to the desired path where the optimized model will be stored.

### Python API Usage
```python
import onnxruntime as rt

sess_options = rt.SessionOptions()

# Set graph optimization level
sess_options.graph_optimization_level = rt.GraphOptimizationLevel.ORT_ENABLE_EXTENDED

# To enable model serialization after graph optimization set this
sess_options.optimized_model_filepath = "<model_output_path\optimized_model.onnx>"

session = rt.InferenceSession("<model_path>", sess_options)
```

### C API Example:
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

### C# API Example:
```c#
SessionOptions so = new SessionOptions();

// Set graph optimization level
so.GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_EXTENDED;

// To enable model serialization after graph optimization set this
so.OptimizedModelFilePath = "model_output_path\optimized_model.onnx"

var session = new InferenceSession(modelPath, so);
```

### C++ API Example:
```c++
Ort::SessionOptions session_options;

// Set graph optimization level
session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

// To enable model serialization after graph optimization set this
session_options.SetOptimizedModelFilePath("optimized_file_path");

auto session_ = Ort::Session(env, "model_file_path", session_options);
```
