# Graph Optimizations in ONNX Runtime

ONNX Runtime provides various graph level optimizations to improve model performance. Graph optimizations are essentially graph level rewrites ranging from small graph simplifications and node eliminations to more complex node fusions and layout optimizations. 

Graph optimizations are divided in several levels\categories based on their complexity and function. They can be performed both online and offline. ONNX Runtime provides Python, C#, C++ and C apis to enable different optimization levels and to choose the offline vs online mode. In offline mode the runtime saves the optimized graph to the disk. **Note**: When running in offline mode make sure to use the exact same options and hardware as the target machine.

## Graph Optimization Levels

Graph optimizations are divided in 3 levels
* Basic
* Extended
* Layout Optimizations

### Basic Graph Optimizations

These are semantics preserving graph rewrites which remove redundant nodes and redundant computations. These optimizations are enabled by default. They run before graph partitioning and thus apply to all the execution providers. Available basic graph optimizations are as follows:

* Constant Folding : Statically computes parts of the graph that rely only on constant initializers. This eliminates the need to compute them during runtime.

* Redundant node eliminations : Removes all redeundant nodes without changing the graph structure
  * Identity Elimination
  * Slice Elimination
  * Unsqueeze Elimination
  * Dropout Elimination

* Semantics preserving node fusions : Fuses\folds multiple nodes into 1 node. For example Conv Add fusion folds the Add operator as the bias of the Conv operator.
  * Conv Add Fusion
  * Conv Mul Fusion
  * Conv BatchNorm Fusion
  * Relu Clip Fusion

### Extended Graph Optimizations

These optimizations include complex node fusions. They are run after graph partitioning and are only applied to the nodes assigned to CPU execution provider. Available extended graph optimizations are as follows:

* GEMM Activation Fusion
* Matmul Add Fusion
* Conv Activation Fusion
* GELU Fusion

### Layout Optimizations

These optimizations change the data layout for applicable nodes to achieve higher performance improvements. They are run after graph partitioning and are only applied to nodes assigned to CPU execution provider. Available layout optimizations are as follows:

* NCHWc Optimizer : Optimizes the graph by using NCHWc layout instead of NCHW layout.

## Usage

### General Note
Levels: 
ONNX Runtime defines GraphOptimizationLevel enum for these levels. Each level when enabled, enables all the preceeding levels too. Mapping of these levels to enum is as follows:

GraphOptimizationLevel::ORT_DISABLE_ALL -> Disables all optimizations
GraphOptimizationLevel::ORT_ENABLE_BASIC -> Enables basic optimizations
GraphOptimizationLevel::ORT_ENABLE_EXTENDED -> Enables basic and extended optimizations
GraphOptimizationLevel::ORT_ENABLE_ALL -> Enables all avialable optimizations including layout optimizations

Modes:
All these optimizations can be run offline too. For production scenarios where model startup time is critical this can bring a lot of benefit. In offline mode, after running graph optimizations onnx runtime serializes the resulting model to the disk. **Note**: When running in offline mode make sure to use the exact same options(execution providers, optimization level etc...) and hardware as the target machine.

To enable model serialization set the SessionOptions option `optimized_model_path`.

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
