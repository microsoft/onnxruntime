---
title: EP Context Design
description: ONNX Runtime EP Context Cache Feature Design
parent: Execution Providers
nav_order: 17
redirect_from: /docs/reference/execution-providers/EP-Context-Design
---

# OnnxRuntime EP Context Cache Feature Design
{: .no_toc }

## Contents
{: .no_toc }

* TOC placeholder
{:toc}

## Background

ONNX Runtime **Execution Providers (EPs)** enable users to **run ONNX models on various hardware accelerators** powered by backend SDKs (e.g., **QNN**, **OpenVINO**, **Vitis AI**, etc.).<br/>
Execution Providers **convert the ONNX model into the graph format** required by the backend SDK and **compile it into the format needed by the hardware**.<br/>
In the **NPU domain**, this **conversion and compilation process** can be time-consuming, especially for **LLM models**, sometimes taking **tens of minutes** to complete. This significantly impacts the **user experience** during session creation.<br/>
To **eliminate** the repeated overhead of model conversion and compilation, most backend SDKs offer a feature to **dump the pre-compiled model into a binary file**.<br/>
- The **pre-compiled model** can be directly loaded by the backend SDK and executed on the target device.
- This greatly **reduces session creation time** and improves overall efficiency.
To support this optimization, ONNX Runtime introduced a **contributor operator** called `EPContext` in the **MS domain**.

## EPContext Op Schema

Op domain: com.microsoft<br/>
Node inputs & outputs: variadic<br/>

Attributes table below:<br/>

|Attributes           |Data type|Description                                                                                               |
|---------------------|---------|----------------------------------------------------------------------------------------------------------|
|main_context         |int64    |**1 (default)**: This node **references EP context content** that contains the **graph associated with this node**.<br/>**0**: The node does **not reference any EP context content**. Instead, it expects to retrieve the graph from another node where this field is set to **1**.<br/>Some EPs support **a single context containing multiple graphs**.<br/>The `EPContext` node with `main_context = 1` refers to the **primary context**.<br/>This context contains multiple graphs, which can be **referenced by other nodes** with `main_context = 0`.|
|ep_cache_context     |string   |The **payload of the EP context** if `embed_mode = 1`, or the **path to the context file** if `embed_mode = 0`.<br/>The path is **relative to the ONNX model file** and can be either a **file name** or a **subfolder/file name**.|
|embed_mode           |int64    |**1 (default)**: `ep_cache_context` contains the **payload of the context content**.<br/>**0**: `ep_cache_context` contains the **file path to the context binary**.|
|ep_sdk_version       |string   |Optional. The **SDK version** used to **generate this node**.|
|onnx_model_filename  |string   |Optional. Original Onnx model file name.|
|hardware_architecture|string   |Optional. Hardware architecture.|
|partition_name       |string   |Optional. OnnxRuntime partitioned graph name.|
|source               |string   |Optional. The **source identifier** used to **generate this node**.<br/>This should be a **unique key defined by the EP**, allowing ONNX Runtime to support **multiple `EPContext` nodes** running with different EPs.<br/>For example:<br/>**QNN EP** only accepts nodes with `source = QNN` or `QnnExecutionProvider`.<br/>**OpenVINO EP** only accepts nodes with `source = OpenVINOExecutionProvider`.|
|notes                |string   |Optional. Additional information required by specific EP.|
|max_size             |int64    |Optional. The **maximum size** in the context, with usage **dependent on the EP**.<br/>Defaults to **0**.|

<p align="center"><img width="60%" src="../../images/EP_context_node.png" alt="EP Context node example"/></p>

## OnnxRuntime Session Options Related to EP Context Cache Generation And Inference

|Session option             |Description                                                                                               |
|---------------------------|----------------------------------------------------------------------------------------------------------|
|ep.context_enable          |Used **only for EP context model generation**.<br/>**1**: Enables ONNX Runtime to **dump the context cache model**.<br/>**0 (default)**: **Disables** context model dumping.|
|ep.context_file_path       |Specifies the **file path** for the **dumped model**.<br/>**Default:** `original_file_name_ctx.onnx` for **context model generation**.<br/>For **model inference**:<br/>If the user loads the model from a **memory buffer** and the **EP context binary** is located outside the ONNX model, this option must be set.<br/>ONNX Runtime EP uses this path to **determine the folder location**, combining it with `ep_cache_context` (which points to the **context binary path**) to construct the **absolute path** to the context binary file.|
|ep.context_embed_mode      |Used **only for context model generation**.<br/>**1**: Dumps the **EP context content directly into the ONNX model**, stored inside the `ep_cache_context` node attribute.<br/>**0 (default)**: Dumps the **EP context content into a separate file** and stores the **file name** in the ONNX model.<br/>The **file path** is tracked in the `ep_cache_context` node attribute.|
|ep.context_node_name_prefix|Used **only for context model generation**.<br/>Specifies the **prefix for the `EPContext` node name** (also used as the `partition_name` attribute and internal graph name).<br/>Ensures **uniqueness across nodes** when multiple `EPContext` nodes are combined into a **single model**, preventing naming conflicts.<br/>The EP can also apply this prefix to the **`ep_graph` name** inside the converted EP context binary.|
|session.model_external_initializers_file_folder_path|This is not specific to the **EPContext** design. Generally, for models with external data, when loading the model from a **memory buffer**, the session loses track of the model's name and path, making it unable to locate the external data file. Use this configuration to specify the **folder path** for the external data files.<br/>All external data files should be placed within the **same folder**.|
|ep.context_model_external_initializers_file_name|Used **only for context model generation**.<br/>This configuration is used when some nodes are partitioned on the **CPU EP** and those nodes have **external initializers**. When generating the **EP context model**, the new model **should not rely on the old external data file** used by the source ONNX model.<br/>Use this setting when **dumping the EP context model** with an external initializers file.<br/>If specified, all initializers will be placed inside the **external data file**.<br/>Otherwise, all initializers will be embedded inside the **generated ONNX file**.<br/>By default, this option is **not set**, meaning all initializers will be included within the ONNX file.|

## EP Context Cache Model Generation Workflow

### EP Interface `GetEpContextNodes()` for Generating the EP Context Cache Model

Generating the **partitioned graph** directly within the Execution Provider (EP) code is challenging, as the EP lacks a complete view of the entire partitioned graph. To address this, ONNX Runtime introduces a new **Execution Provider interface**: `GetEpContextNodes()`.

```cpp
virtual const InlinedVector<const Node*> GetEpContextNodes() const {
  return InlinedVector<const Node*>();
}
```

- This API returns an **array of pointers** to `EPContext` nodes.  
- Execution Providers should implement this interface if they need to **generate the context cache model**. Otherwise, they can leave it unimplemented.  
- It is the **EP's responsibility** to create the `EPContext` nodes along with their dependencies (e.g., the context binary file if `embed_mode = 0`).  
- The **ONNX Runtime GraphPartitioner** uses this interface to retrieve the `EPContext` nodes and generate the **partitioned ONNX model**.
[EP context model generation code details here](https://github.com/microsoft/onnxruntime/blob/544bdd60730270f49f6a5baafdff54065f626776/onnxruntime/core/framework/graph_partitioner.cc#L646-L750)


### EP Context Cache Model Generation Guidelines
**OnnxRuntime EPs** should adhere to the following guidelines to create the **EP context cache model** and maintain a unified user interface:

- **Ownership**
  - The **Execution Provider (EP)** is responsible for **creating the EPContext node** along with its dependencies.
  - The **ONNX Runtime framework** is responsible for **generating the EP context ONNX model** using the `EPContext` node list provided by the EP.

- **Lifetime**
  - The lifetime of `EPContext` nodes begins at least when the EP calls compile and ends when the EP is destroyed.

- **ep.context_enable**
  - ONNX Runtime creates the EP context cache model if `ep.context_enable = 1`.
  - Otherwise, if `ep.context_enable = 0` (default), ONNX Runtime follows the standard workflow without generating a cache model.

- **ep.context_file_path**
  - If `ep.context_file_path` is not provided, ONNX Runtime generates the output model file name by replacing `.onnx` in the original input model file name with `_ctx.onnx`.
  - If `ep.context_file_path` is specified, ONNX Runtime uses the provided file path. The EP should also use this path to determine the folder location for dumping the compiled EP context binary file when `ep.context_embed_mode = 0`.
  - **Note:** `ep.context_file_path` is required when loading the model from a **memory buffer**, as ONNX Runtime cannot retrieve the original model file path in this scenario.

- **ep.context_embed_mode**
  - `1`: Embeds the EP context content directly into the ONNX model.
  - `0` (default): Dumps the EP context content into a **separate file** (EP context binary file).
    - There should be a single EP context binary, even if multiple partitioned subgraphs exist. If the EP cannot achieve this in the short term, please note it on the EP webpage. In such cases, users will need to determine the necessary files for production deployment by iterating through all primary `EPContext` nodes (nodes with `embed_mode=1`) and extracting the file paths from the **node attribute** `ep_cache_context`.
    - The EP context binary file name should be `[model_name]_[ep].bin`. 
    - The EP records the context binary file name in the **EPContext node attribute** `ep_cache_context`.  
    - The context binary file must be located in the **same directory** as the dumped ONNX model file.  
    - The file path recorded in the EPContext node is a **relative path** to the ONNX model file.  
    - **Note:** Subfolders are allowed.

- **ep.context_node_name_prefix**
  - If the user wants to add a **custom prefix** to the EPContext node name (also applied to the `partition_name` attribute and graph name), the EP should provide this capability when generating EPContext nodes.
  - This is useful when combining multiple EPContext nodes from different models into a **single model**, where there is a risk of **node name or graph name conflicts** across models.
  - The EP should support multiple EP contexts within a single model, enabling users to **merge and interconnect EPContext nodes** generated from different models.

- **Source model with external data**
<br/>    When the source model relies on an external data file, ONNX uses a relative path to locate that file. Therefore, the external data file must reside in the same directory as the source model. However, newly generated models **should not depend** on any original source files. This approach is driven by several considerations:
  - All newly generated files should be located in the same directory.
  - There's no guarantee that the output files will be generated in the same directory as the source files.
  - The `EPContext` design allows a model to be partitioned by multiple EPs, each compiling its own `EPContext` nodes. A unified and standardized process helps avoid data duplication.
  - Some EPs may need to copy weights from the source into their context binaries to satisfy specific data layout requirements.
  - For subgraphs that fall back to the ONNX Runtime CPU EP, all weight data will, by default, be embedded directly into the newly generated `[model_name]_ctx.onnx` model. If `ep.context_model_external_initializers_file_name` is set, then all weight data will instead be saved to the specified external initializers file.


### Usage Scenario Code Examples

**Generate the EPContext model by creating session from model path:**
```
    Ort::SessionOptions so;

    // Enable EPContext ONNX model dumping
    so.AddConfigEntry(kOrtSessionOptionEpContextEnable, "1");

    // Add the execution provider (using QNN as an example)
    so.AppendExecutionProvider("QNN", provider_options);

    // Create the session to dump the `_ctx.onnx` model
    Ort::Session session1(env, "./model1.onnx", so);
```

**Generate the EPContext model by creating session from model in memory buffer:**<br/>
Similar to the C API CreateSessionFromArray, the example below creates an ONNX Runtime session from a model stored in a memory array, causing the session to lose track of the model's name and path.
To generate the EPContext model, you must specify the file path using: `ep.context_file_path`.
```
    // Read model file into buffer array
    std::vector<char> buffer;
    ReadFileToBuffer("./model1.onnx", buffer);

    Ort::SessionOptions so;

    // Enable EPContext ONNX model dumping
    so.AddConfigEntry(kOrtSessionOptionEpContextEnable, "1");

    // Specify the generated EPContext model file path using option ep.context_file_path
    so.AddConfigEntry(kOrtSessionOptionEpContextFilePath, "./model_ctx.onnx");

    // Add the execution provider (using QNN as an example)
    so.AppendExecutionProvider("QNN", provider_options);


    // Create the session to dump the `_ctx.onnx` model
    Ort::Session session1(env, buffer.data(), buffer.size(), so);
```

**Generate the EPContext model by creating session from model in memory buffer, and model has external weights:**<br/>
Create the session from memory array, and the model depend on external data. The session requires `session.model_external_initializers_file_folder_path` to figure out the external data location, and same with previously example, `ep.context_file_path` to set the file path for the generated EPContext model.
```
    // Read model file into buffer array
    std::vector<char> buffer;
    ReadFileToBuffer("./model_folder/model1.onnx", buffer);

    Ort::SessionOptions so;

    // Enable EPContext ONNX model dumping
    so.AddConfigEntry(kOrtSessionOptionEpContextEnable, "1");

    // Specify the generated EPContext model file path using option ep.context_file_path
    so.AddConfigEntry(kOrtSessionOptionEpContextFilePath, "./model_folder/model_ctx.onnx");

    // Specify the external data folder path using option session.model_external_initializers_file_folder_path
    so.AddConfigEntry(kOrtSessionOptionsModelExternalInitializersFileFolderPath, "./external_data_folder/");

    // Add the execution provider (using QNN as an example)
    so.AppendExecutionProvider("QNN", provider_options);


    // Create the session to dump the `_ctx.onnx` model
    Ort::Session session1(env, buffer.data(), buffer.size(), so);
```
Note: If there is a **subgraph fallback** on the **CPU EP** that depends on external data, the generated EPContext model **should not rely on the original external data file** used by the base model. By default, the EPContext model **embeds all external data** directly into the generated ONNX file. If you need to store weights in an external file, set `ep.context_model_external_initializers_file_name`. This option forces all initializers to be saved in the specified external file.

## Inference Workflow for EP Context Cache Models

ONNX Runtime EPs that support loading models with `EPContext` nodes should follow the workflow and rules below for model inference:

- **Model Identification**
  - The EP should first determine whether the model contains `EPContext` nodes.
    - If no `EPContext` nodes are present, the EP follows its normal inference workflow.
    - If the model contains `EPContext` nodes:
      - The EP should inspect the `source` node attribute of all `EPContext` nodes to verify if any of them are intended for the current EP (i.e., the `source` attribute matches the key expected by the EP).
      - The EP should only partition the `EPContext` nodes where the `source` attribute matches the key required by the EP.
      - The EP loads the cached context from the matched `EPContext` nodes.
  
- **Handling External Context Binaries (embed_mode = 0)**
  When the `EPContext` cache model is generated with `embed_mode = 0`, the context binary is stored as a separate file alongside the ONNX model in the same folder.
  - ONNX Runtime retrieves the relative path of the context binary file from the `ep_cache_context` attribute of the `EPContext` node.
  - **For models loaded from a file path:**
    - The EP should determine the folder path of the input model file and combine it with the relative path to construct the full path to the context binary file.
  - **For models loaded from a memory buffer:**
    - Since the EP cannot derive the model's folder path, the user must specify the session option `ep.context_file_path`.
    - The EP uses `ep.context_file_path` to determine the folder path and combines it with the relative path to construct the full path to the context binary file.

- **Support for Multiple Primary `EPContext` Nodes (`main_context = 1`)**
  - The EP should support multiple primary `EPContext` nodes without any limitations.
  - The EP must be capable of loading all EP context binary buffers/files specified in the `ep_cache_context` attributes of the `EPContext` nodes, deserializing them, managing the `ep_graphs`, and selecting the appropriate one for execution.

- **Error Handling During EP Context Binary Loading**

  The EP or its backend SDK should be capable of detecting common failure scenarios (including but not limited to the following). In such cases, the EP should return a status with the `INVALID_GRAPH` status code:

  - Detect mismatches between the driver version and the version required by the EP context binary; return an error if they are incompatible.
  - Detect mismatches between the runtime SDK version and the version used to generated the EP context binary; return an error if they are incompatible.
  - Return an error if loading the EP context binary fails for any reason.


<p align="center"><img width="60%" src="../../images/EP_context_nodes_with_different_eps.png" alt="EP Context nodes with different EPs"/></p>

### Usage Scenario Code Examples

**Create inference session from pre-compiled EPContext model:**<br/>
Create the session from model file path. If there is external EP context binary file, the session can figure out the binary file path from the model file path.
```
    Ort::SessionOptions so;

    // Add EP, take QNN for example
    so.AppendExecutionProvider("QNN", provider_options);

    // Create sessions to load from the _ctx.onnx model
    Ort::Session session1(env, "model1_ctx.onnx", so);

    session1.run(...);
```

**Create inference session from pre-compiled EPContext model in memory buffer:**<br/>
Creating a session from a memory buffer of the model causes the session to lose track of the model's name and path. To resolve this, you must set: `ep.context_file_path`.
- The session uses this path to identify the folder location.
- With the EP context binary file name from the `EPContext` node, the session constructs the full path to the final EP context binary file.
```
    // Read model file into buffer array
    std::vector<char> buffer;
    ReadFileToBuffer("./model_folder/model_ctx.onnx", buffer);

    Ort::SessionOptions so;

    // Specify the EPContext model file path using option ep.context_file_path
    so.AddConfigEntry(kOrtSessionOptionEpContextFilePath, "./model_path/model_ctx.onnx");

    // Add EP, take QNN for example
    so.AppendExecutionProvider("QNN", provider_options);

    // Create sessions to load from the buffer
    Ort::Session session1(env, buffer.data(), buffer.size(), so);

    session1.run(...);
```

## EPContext with Weight Sharing

### Weight Sharing in Onnx Domain
In ONNX, weight sharing refers to multiple ONNX models with external weights pointing to the same external weight file. These models use the same tensor names, allowing them to reference the same tensor data.
<p align="center"><img width="50%" src="../../images/Onnx_weight_sharing.png" alt="Weight sharing across Onnx models"/></p>

### Weight Sharing in EP Domain with EPContext
EP weight sharing is enabled using a pre-generated EP context binary/blob.
To do this, users must **generate the context binary offline** (Ahead Of Time).
- Some EPs require specific platforms, such as **Linux x86_64** and/or **Windows x86_64**. Please refer to the specific EP page for details.
- The EP context binary contains **multiple graphs** that share the **same tensors**.

<p align="center"><img width="50%" src="../../images/EP_weight_sharing.png" alt="Weight sharing in EP context binary"/></p>

The EP or backend SDK should be capable of converting and compiling the graph as described above.
- The EP or SDK should identify identical weights from the existing EP context generated by previously compiled graphs.
- When new graphs are compiled into the EP context, they should reuse existing weights if they are recognized as identical.
For example, in `[model_name]_[ep].bin`, `tensor1_1` from `ep_graph1` and `tensor2_1` from `ep_graph2` are identical and both point to the same data offset, `tensor_data1`.

### EPContext Model Generation with Weight Sharing Workflow
<p align="center"><img width="90%" src="../../images/EP_weight_sharing_workflow.png" alt="Weight sharing workflow"/></p>

Each ONNX Runtime session is associated with an ONNX model. Models that share weights are grouped into a model group, while ONNX Runtime sessions with common properties are organized into a session group. ONNX Runtime introduces two session options: `ep.share_ep_contexts` and `ep.stop_share_ep_contexts` to facilitate session grouping.
- All ONNX Runtime sessions within the session group should have `ep.share_ep_contexts` enabled.
- The final ONNX Runtime session uses `ep.stop_share_ep_contexts` to indicate that it is the last session in the group.
Note: A single ONNX model may contain multiple `EPContext` nodes, depending on the graph partitioning result. However, for simplicity, each model is shown with only one `EPcontext` node here.

### Implementation Guidelines for EPContext Model Generation with Weight Sharing
- Shared Workspace Creation:
<br/>    The first session creates a shared workspace (e.g., EP Singleton) to share resources with other sessions.
- EP Context Binary File Naming:
<br/>    The EP context binary file name is determined by the first session and stored in the shared workspace (e.g., EP Singleton) for use across session groups.
<br/>    The EP context binary file name should be `[model1_name]_[ep].bin`.
- Graph Compilation:
<br/>    All sessions in the session group compile their graphs into the shared resource. 
- `EPContext` Model Generation:
<br/>    Each session in the session group creates an `EPContext` ONNX model. The EP generates an `EPContext` node that references the EP context binary file name. The ONNX Runtime framework then dumps the `EPContext` ONNX model.
- Final EP Context Binary File Generation:
<br/>    The last session (the one with `ep.stop_share_ep_contexts` enabled) in the session group generates the final EP context binary file using the name stored in the shared workspace.
- Shared Workspace Cleanup:
<br/>    The last session clears the shared workspace. An empty shared workspace indicates that the next session to run is the first session.
- Number of Files Generated:
<br/>    For N source models that share weights, a total of N+1 files should be generated.
<br/>    The generated files are `model1_ctx.onnx`, `...`, `modeln_ctx.onnx`, `[model1_name]_[ep].bin`.

#### User Code Example
```
    Ort::SessionOptions so;

    // Enable EPContext ONNX model dumping
    so.AddConfigEntry(kOrtSessionOptionEpContextEnable, "1");

    // Enable EP context sharing across sessions
    so.AddConfigEntry(kOrtSessionOptionShareEpContexts, "1");

    // Add the execution provider (using QNN as an example)
    so.AppendExecutionProvider("QNN", provider_options);

    // Create the first session to dump the model1_ctx.onnx file
    Ort::Session session1(env, "model1.onnx", so);

    // Mark the last session by enabling ep.stop_share_ep_contexts
    so.AddConfigEntry(kOrtSessionOptionStopShareEpContexts, "1");

    // Create the last session to dump the model2_ctx.onnx file and generate the [model1_name]_[ep].bin
    Ort::Session session2(env, "model2.onnx", so);
```

#### General Tool for EPContext Model Generation with Weight Sharing
OnnxRuntime provides the [ep_weight_sharing_ctx_gen](https://github.com/microsoft/onnxruntime/tree/main/onnxruntime/test/ep_weight_sharing_ctx_gen) tool to automate the weight-sharing workflow. This tool handles the entire process. This tool is specifically designed for **weight sharing** scenarios, streamlining the `EPContext` model generation process.
Example command line:
```
./ep_weight_sharing_ctx_gen -e qnn -i "soc_model|60 htp_graph_finalization_optimization_mode|3" ./model1.onnx,./model2.onnx
```
It creates two Onnx models (`model1_ctx.onnx`, `model2_ctx.onnx`) and one QNN context binary file (`[model1_name]_[ep].bin`).

### Inference Sessions from EPContext Models with Weight Sharing
To use the dumped EPContext models with weight sharing enabled, ONNX Runtime inference sessions must have **resource sharing** activated. This is done by setting the session option: 
```
    ep.share_ep_contexts = 1
```

#### Implementation Guidelines for Inferencing from EPContext Models with Weight Sharing
- Create the first OnnxRuntime inference session
  - Set session option: `ep.share_ep_contexts=1`.
  - Load the `model1_ctx.onnx` model.
  - The shared workspace is initially empty.
  - The EP loads `[model1_name]_[ep].bin` and deserializes the binary to retrieve all graphs (e.g., `ep_graph1`, `ep_graph2`).
  - The `EPContext` node in model1_ctx.onnx specifies the use of `ep_graph1`.
  - The session uses `ep_graph1` for inference.
  - The remaining graphs (`ep_graph2`) are placed into the shared workspace for future sessions.
- Create the Second ONNX Runtime Inference Session
  - Set session option: `ep.share_ep_contexts=1`.
  - Load the `model2_ctx.onnx` model.
  - The `EPContext` node in `model2_ctx.onnx` specifies the use of `ep_graph2`.
  - The shared workspace already contains `ep_graph2`.
  - The EP **skips loading** `[model1_name]_[ep].bin` since the required graph is already available in the shared workspace.
  - The session **moves `ep_graph2` from the shared workspace to the current session**, making it **no longer accessible** from the shared workspace.
- Session Cleanup Best Practices
  - To avoid issues during concurrent execution, it is recommended to **destroy the sessions in reverse order** (i.e., destroy the second session before the first session).
  - This ensures proper resource management and prevents potential conflicts with shared resources.

#### User Code Example
```
    Ort::SessionOptions so;
    // enable ep.share_ep_contexts
    so.AddConfigEntry(kOrtSessionOptionShareEpContexts, "1");

    // Add EP, take QNN for example
    so.AppendExecutionProvider("QNN", provider_options);

    // Create sessions to load from the _ctx.onnx models with resource sharing enabled
    Ort::Session session1(env, "model1_ctx.onnx", so);	
    Ort::Session session2(env, "model2_ctx.onnx", so);

    session1.run(...);
    session2.run(...);
```
