# Graph Partitioning with Annotations and Memory Constraints

ONNX Runtime automatically partitions a model graph across the execution providers (EPs) registered with a session. This page describes the advanced partitioning features introduced for controlling how nodes are assigned to devices and how GPU memory consumption is managed during partitioning.

## Overview

Large models may exceed the memory capacity of a single accelerator (e.g., a CUDA GPU). These features allow you to:
1. Annotate model layers so that specific parts of the model are directed to specific devices (CPU, GPU, NPU).
2. Collect per-node memory statistics during a profiling run.
3. Set a memory budget for an EP so that ONNX Runtime only places nodes on the accelerator until the budget is exhausted; remaining nodes are then eligible for assignment by the subsequent EPs in the session's provider list (often CPU, but not necessarily).

Together, these form a two-phase workflow: profile the model once to collect memory data, then partition it in production using that data and a memory limit.

## Layer Assignment Annotations

### Concept

Each node in an ONNX model can carry a metadata property called `layer_ann` (layering annotation). This is a free-form string that identifies which logical layer or group the node belongs to. At session creation time, you provide a configuration that maps annotation patterns to target devices. ONNX Runtime then uses these mappings to pre-assign nodes to the corresponding EPs before the normal capability-based partitioning runs.

### Annotating the Model

Annotations are stored in each ONNX `NodeProto`'s `metadata_props` field with the key `layer_ann`. You can add them manually using the ONNX Python API:

```python
import onnx
model = onnx.load("model.onnx")
for node in model.graph.node:
    # Assign a layer annotation based on your own logic
    entry = next((prop for prop in node.metadata_props if prop.key == "layer_ann"), None)
    if entry is None:
        entry = node.metadata_props.add()
        entry.key = "layer_ann"
    entry.value = "encoder_layer_0"  # your annotation string

onnx.save(model, "model_annotated.onnx")
```

### Annotating with Olive

Olive provides built-in support for adding layer annotations during ONNX conversion via the `CaptureLayerAnnotations` pass (added in [PR #2361](https://github.com/microsoft/Olive/pull/2361)). You supply a `layer_annotations` dictionary where each **key** is the annotation string to write into `layer_ann`, and each **value** is a list of node-name substrings. During ONNX export, every node whose name contains one of the substrings receives the corresponding `layer_ann` metadata property. If multiple substrings match, the first one in iteration order wins.

Many exported transformer models use consistent node naming patterns. For example, node names often include recurring substrings such as `embed_tokens`, `self_attn`, `mlp`, or `norm`. Because `CaptureLayerAnnotations` matches node-name substrings, these patterns can be used to group related nodes into logical layers and write the corresponding `layer_ann` values during export. Adjust the substrings to match the naming conventions in your own model.

#### Step 1 — Create the workflow config file

Save the following as `annotate_model.json`:

```json
{
  "input_model": {
    "type": "HfModel",
    "model_path": "microsoft/Phi-3.5-mini-instruct"
  },
  "passes": {
    "capture_annotations": {
      "type": "CaptureLayerAnnotations",
      "layer_annotations": {
        "embedding_layer": ["embed_tokens"],
        "attention_layer": ["self_attn", "q_proj", "k_proj", "v_proj", "o_proj"],
        "mlp_layer": ["mlp", "gate_proj", "up_proj", "down_proj"],
        "norm_layer": ["norm", "layernorm"]
      }
    },
    "conversion": {
      "type": "OnnxConversion",
      "target_opset": 16,
      "save_as_external_data": true,
      "all_tensors_to_one_file": true
    }
  },
  "log_severity_level": 1,
  "output_dir": "models/annotated_phi3"
}
```

The `layer_annotations` dictionary maps annotation names to node-name substring patterns:
- Any node whose name contains `"embed_tokens"` → annotated as `"embedding_layer"`
- Any node whose name contains `"self_attn"`, `"q_proj"`, etc. → annotated as `"attention_layer"`
- And so on for `"mlp_layer"` and `"norm_layer"`.

You can also use `ModelBuilder` instead of `OnnxConversion` — both paths apply the annotations automatically:

```json
    "conversion": {
      "type": "ModelBuilder",
      "precision": "int4"
    }
```

#### Step 2 — Run the workflow

```bash
pip install 'olive-ai[auto-opt]'
olive run --config annotate_model.json
```

This will:
1. Download the model from Hugging Face.
2. Store the `layer_annotations` mapping inside `model_attributes` (via `CaptureLayerAnnotations`).
3. Convert to ONNX and stamp every matching node with `metadata_props["layer_ann"]` set to the corresponding annotation name.
4. Write the annotated ONNX model to `models/annotated_phi3/`.

#### Verifying the annotations

You can verify that the annotations were applied:

```python
import onnx

model = onnx.load("models/annotated_phi3/model.onnx", load_external_data=False)
for node in model.graph.node:
    for prop in node.metadata_props:
        if prop.key == "layer_ann":
            print(f"{node.name}: {prop.value}")
```

The annotated model is now ready for use with the ORT session options described below.

### Configuring Layer Assignment at Runtime

Use the session option `session.layer_assignment_settings` to tell ONNX Runtime how to map annotations to devices.

```
device1(annotation1, annotation2, ...); device2(=annotation3, annotation4, ...)
```

- `device`: a recognized device designator, matched against the execution providers registered in the session. The supported designators are:

| Device Designator | Meaning | Examples of EPs That May Bind |
|:------------------|:--------|:-----------------------------|
| `cpu` | Any CPU device | CPUExecutionProvider |
| `gpu` | Anything discovered and designated as `OrtHardwareDeviceType_GPU` | CUDA EP, ROCm EP, DirectML EP running on discrete or integrated GPUs |
| `cuda` | NVIDIA CUDA GPU. A device that is `OrtHardwareDeviceType_GPU` and NVIDIA is a vendor. Same as `gpu:nvidia`. | CUDAExecutionProvider |
| `dml` | DirectX-compatible GPU | DMLExecutionProvider |
| `npu` | Neural processing unit | Qualcomm QNN EP, Intel NPU EP |
| `fpga` | FPGA-backed accelerators | Custom plugin EPs |
| `accelerator` | Catch-all for any non-CPU device | Any vendor EP |
| `gpu:<vendor>` | Vendor specialization | `gpu:nvidia`, `gpu:amd`, `gpu:intel` |
| `gpu:<index>` | Specific device index | GPU 0, GPU 1 |

- `annotation`: string to match against the `layer_ann` value on each node.
- `=` prefix: denotes an exact match. Without `=`, the annotation is treated as a prefix match (any node whose `layer_ann` starts with the string will match).
- Prefix rules have higher priority than exact-match rules. Within the same match type, priority is left-to-right.
- Multiple device rules are separated by `;`.

```python
import onnxruntime as ort

opts = ort.SessionOptions()

# Nodes annotated with layer_ann starting with "encoder" go to GPU,
# nodes with exact annotation "final_output" go to CPU.
opts.add_session_config_entry(
    "session.layer_assignment_settings",
    "gpu(encoder); cpu(=final_output)"
)

session = ort.InferenceSession("model_annotated.onnx", opts,
                               providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
```

#### Targeting a Specific GPU

On a machine with multiple GPUs, use `gpu:<index>` to direct annotations to a particular device:

```python
opts.add_session_config_entry(
    "session.layer_assignment_settings",
    "gpu:1(encoder, decoder); cpu(=postprocess)"
)
```

> **Note:** ONNX Runtime currently allows only one instance of a given EP type per session, so you cannot split annotations across multiple GPUs in a single session. The `gpu:<index>` designator selects *which* GPU the single registered EP targets.

Nodes that do not match any rule fall through to the normal EP capability-based assignment.

> **Unmatched device designators:** If a device designator in the settings does not match any EP registered in the session, ONNX Runtime logs a warning and skips that rule. Nodes covered by the skipped rule are not pre-assigned and fall through to the normal capability-based partitioning.

> **Note — Annotations vs. actual placement:** An annotation expresses a *preference*, not a guarantee. If the target EP does not have a registered kernel for a node (for example, a particular data-type / opset-version combination is not implemented in the CUDA EP), that node will not be placed on the requested device. Instead it falls through to the next EP in the provider list that can handle it.

## Capacity-Aware Partitioning (implemented for CUDA)

When running models on a CUDA GPU with limited memory, you can set a memory budget so ONNX Runtime stops assigning nodes to the CUDA EP once the estimated memory consumption reaches the limit. Nodes are considered in topological order and assignment halts at the first node that would exceed the budget — ONNX Runtime does not search ahead for smaller nodes that might still fit. Remaining nodes are then eligible for assignment by the subsequent EPs in the session's provider list (often CPU, but not necessarily).

### Step 1: Collect Memory Statistics (Profiling Run)

Run the model once with memory statistics collection enabled. This records per-node allocation data to a CSV file.

```python
import onnxruntime as ort
import numpy as np

opts = ort.SessionOptions()
# Disable memory patterns for accurate per-node measurement
opts.enable_mem_pattern = False
opts.add_session_config_entry(
    "session.collect_node_memory_stats_to_file",
    "node_memory_stats.csv"
)

session = ort.InferenceSession("model.onnx", opts,
                               providers=["CUDAExecutionProvider", "CPUExecutionProvider"])

def make_concrete_shape(shape, default_dim=1):
    """ORT input shapes may contain symbolic dims or None (e.g. batch size).
    Replace those with a small concrete value for profiling.

    For the most accurate memory statistics, use the largest input shapes
    your production workload will encounter.  For example, if your service
    runs with a maximum batch size of 8, pass default_dim=8 so the profiled
    allocations reflect that peak usage."""
    return tuple(
        dim if isinstance(dim, int) and dim > 0 else default_dim
        for dim in shape
    )

# Run inference at least once to collect statistics.
# For models with dynamic inputs, prefer real sample inputs or model-appropriate
# concrete shapes instead of relying on the declared ORT input shape directly.
input_data = {
    inp.name: np.zeros(make_concrete_shape(inp.shape), dtype=np.float32)
    for inp in session.get_inputs()
}
session.run(None, input_data)
```

This produces a CSV file with columns:
`#name,input_sizes,initializers_sizes,total_dynamic_sizes,total_temp_allocations`

In this example, `node_memory_stats.csv` is a relative path. Relative paths are resolved against the model's directory when the model was loaded from a filesystem path. If you provide an absolute path, that path is used as-is. If the model was not loaded from a filesystem path (for example, it was loaded from bytes), the output file is written relative to the current working directory.

Multiple `session.run()` calls update the stats with the maximum values observed per node.

> **What if the GPU cannot hold the entire model?**  The profiling run itself requires the model to fit in GPU memory because the CUDA EP must execute each node to record its actual allocations. If the model exceeds GPU capacity during profiling, reduce the input dimensions (e.g., use a smaller batch size) so that the run completes. The resulting per-node statistics will still be representative of relative node costs and can be used to set the memory budget for subsequent production sessions.

### Step 2: Partition with a Memory Budget

In a subsequent session, provide the memory limit and the stats file to enable capacity-aware partitioning.

```python
import onnxruntime as ort

opts = ort.SessionOptions()

# Format: "memory_limit_in_kb,stats_filename"
# Set a 4 GB limit and use the stats from the profiling run
opts.add_session_config_entry(
    "session.resource_cuda_partitioning_settings",
    "4194304,node_memory_stats.csv"
)

session = ort.InferenceSession("model.onnx", opts,
                               providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
```

ONNX Runtime processes nodes in topological order, accumulating estimated memory. When the cumulative cost exceeds the budget, assignment to the CUDA EP halts immediately — remaining nodes are not considered even if they would individually fit within the budget. Those nodes are eligible for assignment by the subsequent EPs in the session's provider list.

Because assignment follows topological order, groups of nodes that you would prefer to offload (for example, MoE expert blocks) may appear at arbitrary positions in the graph. If you want specific node groups to have the lowest priority for device placement, combine the memory budget with layer annotations: annotate the nodes you want to offload to CPU explicitly, and let the capacity-aware partitioner handle the rest.

### Ad-Hoc Mode (No Stats File)

If you do not have pre-recorded statistics, you can specify only a memory limit. ONNX Runtime will estimate per-node cost from initializer sizes and static output shapes, applying a 1.5x safety multiplier.

This mode is less accurate than using pre-recorded stats but provides a quick way to constrain GPU memory without a profiling run.

```python
# Memory limit only, no stats file (note the trailing comma)
opts.add_session_config_entry(
    "session.resource_cuda_partitioning_settings",
    "4194304,"
)
```

### Setting Format Summary
The value of `session.resource_cuda_partitioning_settings` is a comma-separated pair:

| Format | Meaning |
|:------|:-------|
| `<limit_kb>,<stats_file>` | Use both memory limit and pre-recorded stats |
| `<limit_kb>,` | Memory limit only (ad-hoc estimation) |
| `,<stats_file>` | Stats only (no explicit limit) |
| `,` | Neither (EP attempts auto-detection) |

The stats file path follows the same resolution rules described above: relative paths are resolved against the model's directory, absolute paths are used as-is.

## Combining Both Features
Layer annotations and capacity-aware partitioning can be used together. When both are configured:
- Layer annotations provide the initial node-to-device mapping.
- The capacity-aware partitioner enforces the memory budget, potentially overriding assignments that would exceed the GPU memory limit.

This combination gives you fine-grained control: use annotations to express logical model structure, and let the memory budget act as a safety net.

```python
opts = ort.SessionOptions()

opts.add_session_config_entry(
    "session.layer_assignment_settings",
    "gpu(encoder, decoder); cpu(=postprocess)"
)

opts.add_session_config_entry(
    "session.resource_cuda_partitioning_settings",
    "4194304,node_memory_stats.csv"
)

session = ort.InferenceSession("model_annotated.onnx", opts,
                               providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
```
