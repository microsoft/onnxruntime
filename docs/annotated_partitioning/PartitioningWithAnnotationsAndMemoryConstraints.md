# Graph Partitioning with Annotations and Memory Constraints

ONNX Runtime automatically partitions a model graph across the execution providers (EPs) registered with a session. This page describes the advanced partitioning features introduced for controlling how nodes are assigned to devices and how GPU memory consumption is managed during partitioning.

## Overview

Large models may exceed the memory capacity of a single accelerator (e.g., a CUDA GPU). These features allow you to:
1. Annotate model layers so that specific parts of the model are directed to specific devices (CPU, GPU, NPU).
2. Collect per-node memory statistics during a profiling run.
3. Set a memory budget for an EP so that ONNX Runtime only places nodes on the accelerator until the budget is exhausted; remaining nodes are then eligible for assignment by the subsequent EPs in the session's provider list (often CPU, but not necessarily).

Together, these form a two-phase workflow: profile the model once to collect memory data, then partition it in production using that data and a memory limit.

The sections below document implemented features. The final
[proposed automatic placement design](#proposed-automatic-region-aware-cpugpu-placement) describes a future,
opt-in framework planner; the existing settings do not enable that design.

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

### Choosing Block-Instance Annotations

The Olive example above groups nodes by operation category. An annotation such as `attention_layer`
can identify attention nodes across many transformer blocks, but does not identify an individual block
or provide an ordered sequence of placement boundaries.

For block-level placement, label each block instance separately and include its attention, MLP,
normalization, and residual-path nodes as appropriate. For example, an exporter whose node names contain
`/layers.0/`, `/layers.1/`, and `/layers.2/` could use this `layer_annotations` mapping:

```json
{
  "decoder.block.0": ["/layers.0/"],
  "decoder.block.1": ["/layers.1/"],
  "decoder.block.2": ["/layers.2/"]
}
```

Inspect the exported graph to verify membership and dependency edges; a name match alone does not
establish a complete block. Delimited name patterns avoid accidentally matching block 1 with block 10.
For annotation-based device rules, use exact matches such as `=decoder.block.1` when that distinction
matters. Embeddings, output heads, shared nodes, and unannotated nodes need separate placement treatment.

These labels already support manual device assignment. They do **not** make the current partitioner
infer block order, guarantee atomic block placement, or automatically search for a fitting boundary.
The proposed planner below would validate and use such groups as candidate-generation metadata.

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

### Name-Based Layer Assignment (No Model Modification)

For models that already have structured node names (most HuggingFace exports, ONNX models produced by PyTorch, etc.), you can skip the annotation step entirely. The session option `session.name_based_layer_assignment` performs **substring matching** directly against `Node::Name()`:

```
device1(pattern1, pattern2, ...); device2(pattern3, pattern4, ...)
```

- **Substring matching:** A pattern matches if it appears *anywhere* in the node name. For example, `layers.0/` matches `/model/layers.0/self_attn/q_proj/MatMul`.
- **Longest match wins:** When multiple patterns match the same node name, the longest pattern takes priority. For example, `layers.10/` wins over `layers.1/` for a node named `/model/layers.10/...`.
- **No `=` prefix:** The exact-match qualifier (`=`) from annotation-based syntax is rejected with an error. All patterns are treated as substrings.
- **Same device designators:** The device portion uses the same device designators as `session.layer_assignment_settings` (see table above).

```python
import onnxruntime as ort

opts = ort.SessionOptions()

# Assign layers 0–7 to GPU, layers 8–15 to CPU based on node names
opts.add_session_config_entry(
    "session.name_based_layer_assignment",
    "gpu(layers.0/, layers.1/, layers.2/, layers.3/, layers.4/, layers.5/, layers.6/, layers.7/); "
    "cpu(layers.8/, layers.9/, layers.10/, layers.11/, layers.12/, layers.13/, layers.14/, layers.15/)"
)

session = ort.InferenceSession("model.onnx", opts,
                               providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
```

**Tips for writing patterns:**
- Include the trailing `/` in layer patterns (e.g., `layers.1/` instead of `layers.1`) to avoid `layers.1` accidentally matching `layers.10`, `layers.11`, etc.
- Use [Netron](https://netron.app/) to inspect your model's node names and identify suitable substrings.
- Nodes that do not match any pattern fall through to normal EP capability-based assignment (typically CPU).

**Mutual exclusivity with annotation-based matching:** The `session.name_based_layer_assignment` and `session.layer_assignment_settings` options are **mutually exclusive** — setting both will return an error. Use annotation-based matching for models that carry explicit `layer_ann` metadata annotations, or name-based matching for unmodified models with structured node names. If you need fine-grained exceptions (e.g., force one specific node to CPU), add the node's name pattern to the name-based config instead of mixing the two approaches.

**No subgraph inheritance:** Unlike annotation-based matching (where unannotated subgraph nodes inherit their parent's device assignment), name-based matching treats every node independently. Since node names are dense (virtually every node has a name encoding its structural position), inheritance is unnecessary — each node matches on its own name.

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

Stats files may contain only a subset of graph nodes. For backward compatibility, a node missing from a supplied stats file has zero accounting cost; ONNX Runtime does not mix ad-hoc estimates into a profile-based accounting session.

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

For dynamic-shape models, `session.max_shape_override` supplies planning shapes:

```python
opts.add_session_config_entry(
    "session.max_shape_override",
    "input_ids:[8,4096];attention_mask:[8,4096]"
)
```

ONNX Runtime propagates these values through a separate shape-inference graph. The resulting shapes
are used for ad-hoc dynamic-output sizes and Level-1 workspace estimates, so they directly affect the
hard partitioning budget and can change CUDA/CPU assignment. They do not constrain runtime inputs and
are estimation hints rather than guaranteed upper bounds.

### Setting Format Summary
The value of `session.resource_cuda_partitioning_settings` is a comma-separated pair:

| Format | Meaning |
|:------|:-------|
| `<limit_kb>,<stats_file>` | Use both memory limit and pre-recorded stats |
| `<limit_kb>,` | Memory limit only (ad-hoc estimation) |
| `,<stats_file>` | Stats only (no explicit limit) |
| `,` | Neither (EP attempts auto-detection) |

The stats file path follows the same resolution rules described above: relative paths are resolved against the model's directory, absolute paths are used as-is.

### Interaction with NHWC Layout (`prefer_nhwc`)

EPs that prefer the NHWC data layout — for example, the CUDA EP when it is created with the provider option `prefer_nhwc=1` (CUDA plugin EP: session option `ep.cuda.prefer_nhwc_layout=1`) — partition layout-sensitive subgraphs in two passes:

1. **First pass (tentative):** The EP tags the nodes it could claim so the layout transformer can rewrite them into the NHWC (`com.microsoft.nhwc`) form.
2. **Second pass (final):** After the layout transform, the EP runs capability detection again and fuses/optimizes the rewritten nodes. Some first-pass nodes may be dropped here (for example, a node whose NHWC form is not actually supported), in which case they fall back to a later EP.

Because the first-pass tags are tentative, ONNX Runtime performs complete second-pass support discovery without budget gating, probes the accountant-aware capability grouping without budget truncation, and rebuilds reservations for the reconciled survivor set so shared initializers are charged exactly once. Final budget admission is retried if truncation changes that survivor set. Only final assignments remain committed. This keeps the accumulated memory estimate accurate when `prefer_nhwc` is combined with `session.resource_cuda_partitioning_settings`, prevents displaced or dropped nodes from consuming phantom budget, and prevents later survivors from being reclaimed at zero cost after an early budget stop.

## Combining Features
Layer annotations OR name-based assignment can be combined with capacity-aware partitioning. Note that annotation-based and name-based matching are **mutually exclusive** — you cannot use both simultaneously.

When a layer assignment option (either annotation-based or name-based) is configured together with the capacity-aware partitioner:
- The layer assignment option expresses the desired device placement.
- The capacity-aware partitioner enforces the memory budget, potentially overriding assignments that would exceed the GPU memory limit.

This gives you fine-grained control: use annotations or name patterns to express logical model structure, and let the memory budget act as a safety net.

```python
opts = ort.SessionOptions()

# Name-based assignment (no model modification needed)
opts.add_session_config_entry(
    "session.name_based_layer_assignment",
    "gpu(layers.0/, layers.1/, layers.2/, layers.3/); cpu(layers.4/, layers.5/, layers.6/, layers.7/)"
)

# Memory budget as a safety net
opts.add_session_config_entry(
    "session.resource_cuda_partitioning_settings",
    "4194304,node_memory_stats.csv"
)

session = ort.InferenceSession("model.onnx", opts,
                               providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
```

## Proposed: Automatic Region-Aware CPU/GPU Placement

> **Status: proposed, not implemented.** This section describes an opt-in extension, not additional
> semantics of `session.resource_cuda_partitioning_settings` or the layer-assignment options.
> The current additive cutoff is not a whole-session physical GPU memory guarantee.

The goal is to select a static CPU/GPU placement and enforceable dispatch contracts under a declared
GPU memory budget **before** materializing device weights and kernels. CPU offloading here means CPU
execution of selected regions with their required weights on the host, not CPU-resident weights that
are repeatedly streamed to GPU kernels during `Run()`.

This section owns graph-region discovery, fusion-aware execution candidates, candidate CPU/GPU cuts, and
partitioner integration through final placement commitment. Dispatch-contract semantics and whole-session
lifetime/backing accounting are defined in the companion
[policy/workspace design proposal (PR #32944)](https://github.com/microsoft/onnxruntime/pull/32944),
which proposes `policy_based_workspace_estimation.md`. The
[constrained-environment roadmap](future_directions_constrained_env.md) provides broader context.
This is pre-commit selection, not late rollback of already materialized layers or retry after OOM.

### Terminology: Graph Regions, Execution Candidates, and Placement

Separate **subgraph identification**, **implementation choice**, and **device assignment**.

Here, a **subgraph** is a subset of an existing graph's nodes and the edges between them, with an
input/output boundary. In ORT, the term is also used for nested control-flow graphs (such as `If`
branches or a `Loop` body) and node sets offered by EPs for fusion/compilation. A subgraph used for
planning does not require a separate `Graph` object, fusion, or EP/device assignment.

| Term | Meaning in this proposal |
|:-----|:-------------------------|
| **Graph region** (or **region**) | A validated subgraph, such as a decoder block, used to organize the search. It does not identify a device, require one kernel, or by itself make the subgraph indivisible. |
| **Region discovery / region formation** | Identifying and validating those subgraphs and their dependencies from annotations, configured names, or graph structure. This answers "which nodes form each planning subgraph?" without assigning them to CPU/GPU or rewriting them. |
| **Fusion-aware execution candidate** | An EP's offer to execute a particular source-node set using a specified implementation recipe, with its interface, eligibility conditions, contracts, and requirements. It can describe standalone kernels, a fused kernel, or a compiled region; it is not yet a selected assignment. |
| **Candidate CPU/GPU cut** (or **assignment boundary**) | A possible graph frontier separating CPU-assigned and GPU-assigned work in a tentative plan. This is the meaning of boundary search below, not merely the input/output boundary of a region or execution candidate. |
| **Candidate plan** | A provisional whole-graph combination of CPU/GPU assignments, compatible execution candidates, selected contracts, transfers, and backing strategy. Unlike an individual execution candidate, it must cover the complete graph. |
| **Placement** | Assignment of executable work to particular EPs/devices, such as CPU EP or CUDA EP. A tentative placement is a trial assignment; final placement is committed only after verification. It is distinct from allocating or uploading the associated storage. |

Use **graph region** for a subgraph's planning role rather than "placement region", and
**region formation** for the subgraph-identification step.
Reserve **placement** for CPU/GPU assignment. A graph-region boundary, an execution candidate's
input/output interface, and a CPU/GPU assignment boundary need not coincide.

### Initial Scope and Compatibility

Start with CUDA plus CPU, bounded planning shapes, at most one concurrently active `Run()` per session,
and a validated single-stream execution plan. Planning does not require running inference or a warm-up.
Keep the workload envelope fixed: reducing context length, batch size, or cache capacity is not a
placement alternative. Caller-guaranteed workload bounds and serialization can satisfy these conditions;
additional ORT-side workload enforcement and bounded kernel fallback coverage remain deferred work in
the companion proposal.

Existing provider priority, support checks, and annotation/name-based preferences remain unchanged
outside the opt-in mode. The new mode must distinguish preferences from explicit hard constraints;
hard-pinned placement and dispatch contracts restrict the search, and unsupported pins fail with a
diagnostic. Preserve CUDA's CPU-preferred shape-computation subgraphs. A CPU alternative is valid only
when every moved node has CPU support for its actual operator, opset, and data type, and the resulting
host-memory requirements are acceptable. A GPU-supported fp16/bf16 region is not automatically CPU-supported.

The initial selector can use a deterministic ordinal preference, such as avoiding additional offload
when a fitting contract change preserves GPU placement. It must not claim an optimal latency without
comparable performance evidence. Concurrent Runs, capture-retained pools, control flow, and compiling
EPs require additional models or explicit rejection of unsupported budget-planning modes.

### Discover Graph Regions Separately from EP Capabilities

ORT has no universal semantic layer unit. **Region discovery** identifies subgraphs to organize candidate
plans; it does not decide where those subgraphs execute. An EP capability instead describes nodes an EP
can execute, possibly as a fused or compiled region. The two need not have the same boundaries: one
decoder-block region may contain many single-node CUDA capabilities or several fused capabilities.
Conversely, an indivisible capability can constrain otherwise proposed region boundaries.

Use the following discovery hierarchy:

1. Prefer per-block-instance `layer_ann` metadata, with exporter-supplied structure where available.
2. Otherwise use configured, validated node-name grouping. Do not guess a block index from arbitrary
   numeric substrings or assume exporter naming conventions are universal.
3. Optionally form a bounded set of graph regions using graph frontiers when structure is unavailable.
   These are graph-based heuristics, not recovered semantic layers; otherwise retain fixed-placement
   contract selection and report that automatic block-boundary search is unavailable.

Maintain stable region identity and membership through supported transformations. Order regions from
graph dependencies, using explicit block indices as validated hints rather than ordering names
lexicographically. Check that contracting proposed groups does not introduce dependency cycles; reject
or merge incompatible groups. Respect mandatory fusion/compilation and QDQ constraints, and do not cut
an accepted indivisible capability. Potential optional fusion does not by itself make a group a no-cut
region. Shared nodes, embeddings, output heads, and nodes outside groups must have explicit treatment
in every complete candidate plan.

The existing `LayeringIndex` maps device rules to node membership and filters EP graph views. It is
useful for applying a selected placement, but is not itself a region-discovery or boundary-search engine,
and does not establish semantic block order.

### Discover Fusion-Aware Execution Candidates

**Execution-candidate discovery** asks each EP "how could you execute these source nodes?" rather than
identifying planning subgraphs or committing device assignments. **Fusion-aware** means considering the
resulting implementation after an eligible fusion/lowering, not only the original nodes' standalone
support and memory estimates. Extend the planning-safe `GetCapability()` path to describe these
implementations before deciding their placement. A candidate may use standalone kernels, a registered
fused kernel, or a compiled subgraph; it need not correspond to one device kernel. An EP may support a
pattern only as a fused region without supporting its constituent nodes separately.

Each EP returns only candidates it owns. ORT combines CUDA candidates with independently discovered
CPU candidates; the CUDA EP does not propose CPU EP assignments. Planning queries identify the graph
revision, workload bounds, device/configuration, and user constraints. The following information is
conceptual planning metadata, not an existing `ComputeCapability` payload or a finalized public API:

| Metadata | Meaning and responsibility |
|:---------|:---------------------------|
| `sub_graph` membership and boundaries | Identify the source work covered by this offer, including external value/initializer dependencies. These are not necessarily the nodes remaining after lowering, and one planning region can contain several capabilities. |
| `candidate_id` and `graph_revision` | Identify the particular offer from an EP instance and the source graph version against which it was discovered. They prevent applying a stale match; they are not sufficient cache keys without the relevant query context. |
| `alternative_view_ids` | Identify alternative sets of capabilities produced together, such as standalone and optimized views. A shared capability can belong to several views. |
| `compatibility_constraints` | Express exclusions, required companions, boundary/layout agreements, and indivisibility. Disjoint source membership alone does not establish compatibility. |
| `implementation_recipe` | Identify the reproducible implementation: standalone kernels, a registered fused kernel, or compiled-region lowering. A standalone recipe may require no rewrite. |
| `application_handle / recipe_state` | Retain the prepared match, replacement/lowering choice, and dependencies needed to apply that implementation without independently selecting a different one. This can be CPU-side metadata, not a compiled executable or constructed kernel. |
| `planning_graph or graph_effects_description` | Expose resulting tensors, boundaries, shapes/types, aliases, and storage effects. Prefer a compact description when sufficient; a planning graph does not require a full graph copy for every candidate. |
| `contract_id` and EP-owned resolved state | Identify an enforceable execution policy for this implementation, including the permitted runtime choices. The ID references retained, versioned state, not just an intent string that may later resolve differently. |
| Eligibility, workload coverage, and explicit-failure conditions | Distinguish static support conditions, admitted runtime states, and states that must fail instead of escaping to an unaccounted implementation. A bounded contract need not guarantee successful execution for every request. |
| `allocation_requirements[]` | Describe implementation-owned persistent/prepack storage, initialization scratch, and runtime workspace, with resource identity, location, size bound, alignment, lifetime, and established sharing/alias relationships. ORT resolves these into whole-session physical backing. |
| `bound_status / unresolved_terms` | Record whether the requirements are sufficiently bounded for the query and which terms remain unknown or unsupported. A candidate may be executable yet lack enough information for strict-budget admission. |

A conceptual internal representation is:

```text
ComputeCapability
|-- sub_graph
|   `-- source-node membership and supplied/derived boundary inputs/outputs
`-- planning (optional)
    |-- candidate_id + graph_revision
    |-- alternative_view_ids + compatibility_constraints
    |-- implementation_recipe
    |   |-- standalone kernel, fused-kernel, or compiled-region lowering
    |   |-- application_handle / recipe_state
    |   `-- planning_graph or graph_effects_description
    `-- contracts[]
        |-- contract_id + EP-owned resolved contract state
        |-- eligibility / workload coverage / explicit-failure conditions
        |-- allocation_requirements[]
        `-- bound_status / unresolved_terms
```

The current `ComputeCapability` supplies `sub_graph`, `nodes_to_optimize`, and `optimization_func`;
the optional planning payload above is proposed. It is an information model, not a requirement to
embed every object in the capability: shared descriptors can be referenced by IDs or retained handles.
The optional payload preserves legacy capabilities outside budget-planned mode. A missing payload is
not a zero-cost estimate or sufficient evidence for strict-budget admission without a proven adapter.
Boundary information can be supplied or derived from the graph; the sketch does not imply that every
boundary field already exists on `IndexedSubGraph`.

Prepared recipe and resolved contract state need explicit validity, retention, and release rules,
including discarded candidates and initialization failures. Their IDs do not permit retaining borrowed
graph/node views beyond the views' documented lifetimes.

The corresponding planning query could look like:

```text
GetCapability(graph_view, kernel_lookup, optimizer_registry, query_context)
    -> execution_candidates[]

query_context:
    mode = Planning
    graph_revision
    workload_bounds
    device/configuration
    user_constraints
```

These are illustrative internal planning interfaces, not the current `GetCapability()` signature or
finalized C++/C API layouts. Existing discovery/accountant integration needs adaptation; plugin ABI
support must provide equivalent semantics through versioned C-compatible interfaces, not export these
C++ objects directly.

Returned requirements are evaluated for that query context and selected recipe/contract, not universal
per-node estimates. Changes to the graph revision, placement boundaries, workload, configuration, or
constraints require invalidating/re-querying affected descriptors unless their reuse is proven valid.

The recipe and requirement query must describe the same implementation. The implementation owner
supplies kernel-private sizing, preferably reusing runtime sizing helpers. ORT accounts for graph
tensors, transfers, and global backing/lifetimes; a shared framework optimizer must describe storage
its rewrite creates without duplicating the target kernel's scratch formulas. A workspace scalar alone
does not describe the transformed execution plan or whole-session peak.

For example, replacing `A -> temporary T -> B` with `fused_AB` may eliminate materialized `T` while
introducing packed weight `P`. The description must identify those effects and any coexistence of
source weights, `P`, initialization scratch, and runtime workspace. A compiler can keep private
operations opaque if its interface and resource description are sufficient for planning.

For an initial implementation, an EP can expose a standalone view and a preferred optimized view of a
queried tentative region, where supported:

```text
standalone view:         {A}, {B}, {C}
preferred optimized view: {A, B fused}, {C}
```

Here `{A, B, C}` can remain one graph region in both views. Its grouping has not changed; the EP has
offered different ways to implement its work. Neither view alone decides whether that region runs on
CPU or GPU.

These are alternative sets of capabilities, not necessarily two `ComputeCapability` objects. ORT
selects a non-overlapping, complete cover across EPs that respects each view's compatibility constraints.
For example, two disjoint candidates may still require incompatible layouts at their connecting
boundary; combining them requires a described conversion or a different candidate selection.
The preferred view is not "all possible fusion": patterns can overlap, and a standalone view is valid
only where standalone execution is supported. Two views are a bounded starting point, not a claim of
optimality or a proof that no fitting implementation exists.

A placement boundary may prevent optional fusion. Re-query/refine the affected region and account for
the unfused intermediates, implementation requirements, and transfers instead of retaining the fused
estimate. Once selected, an indivisible fused/compiled candidate must be accepted wholly or rejected.
Already-fused input nodes remain atomic unless supported expansion or rebuilding is available.

### Integrate L1 Estimates and Policy-Based Route-Aware Contracts

The companion L1/policy work supplies the sizing foundation; the adapters, capability payload, and
selected-contract transport below are proposed framework integration. Reuse the existing sizing and
policy logic rather than introducing a second estimator with separate formulas.

#### Reuse L1 Requirements Separately from Budget Acceptance

The existing CUDA path combines support discovery, L1 estimation, `ComputeResourceCount()`, and an
additive threshold decision. In planning mode, L1 instead describes each offered implementation's
requirements before ORT chooses a complete placement. The intended split is:

```text
Shared implementation sizing and policy logic
    |-- legacy summary: Level1MemoryEstimate
    |       `-- existing resource accountant and additive acceptance
    `-- planning descriptor: recipe + contract + allocation requirements
            `-- provisional whole-session execution/backing evaluator
```

`Level1MemoryEstimate` is a compatibility summary, not a whole-session memory bound:

| Existing L1 field | Planning interpretation |
|:------------------|:------------------------|
| `runtime_workspace_bytes` | Execution workspace, with resource identity, alignment, and operation/Run lifetime. `nullopt` means unknown, not zero. |
| `persistent_prepack_bytes` | Additional packed destinations retained for the session, distinct from original initializers already represented in the graph. |
| `initialization_scratch_bytes` | Temporary construction/PrePack storage, including its coexistence with source weights and packed destinations. |
| `runtime_transient_bytes` | Runtime-preparation storage, such as lazy profiling scratch, with preparation lifetime. Taking the maximum of preparation and execution scratch requires established non-overlap. |

When reusing an existing L1 estimator, associate its results with the implementation, selected contract,
workload bounds, and allocation lifetimes they describe. Use the implementation's allocation
requirements, not the aggregate `ComputeResourceCount()` result, which already includes graph-related
costs accounted for separately by ORT. Missing allocation information remains unresolved; default zero
values alone do not establish a zero-memory requirement.

Here, a compatibility adapter means this conversion code; it need not be a separate component or API.

For a changed/fused implementation, query its own recipe and contract. Do not assume that adding the
source nodes' L1 estimates bounds the replacement. Current reservation consolidation preserves
bookkeeping; it does not establish the replacement's requirements, coverage, or runtime permissions.

| Mode and available information | Treatment |
|:-------------------------------|:----------|
| Ordinary partitioning without planning metadata | Preserve existing discovery/accounting behavior. |
| Budget planning with complete metadata | Evaluate the described requirements and graph effects. |
| Budget planning with a proven compatibility adapter | Construct equivalent planning information, for example for an unchanged singleton implementation with trusted L1 sizing and known lifetimes. |
| Budget planning without adequate metadata or adapter | Keep requirements unresolved; exclude the candidate from strict-budget feasibility and report failure if no complete feasible plan remains. |

Profiles and generic safety margins are not substitutes for missing analytical bounds. A null resource
accountant is not the adapter: current CUDA also skips its L1 queries in that path, so estimation must
be callable independently of additive acceptance.

#### Resolve Policy Intent into an EP-Owned Contract

A policy name such as `memory` is an intent supplied to the EP's resolver, not a complete contract.
The resolved policy, together with its workload/device conditions, defines the contract:

```text
Candidate implementation + workload bounds + device/configuration + allowed intent
    -> shared policy resolver and complete route-aware sizing
    -> resolved phase-specific route permissions
       + coverage / explicit-failure conditions
       + allocation requirements / bound status
    -> one planning contract
```

The [GQA committed-policy work (PR #33316)](https://github.com/microsoft/onnxruntime/pull/33316) provides
a useful foundation: `GQAWorkspaceAggregate` contains the workspace, persistent-prepack, and
initialization-scratch sizes together with `committed_backends`. A planning
adapter should retain their relationship rather than discard the permissions after extracting bytes.
The proposed `contract_id` references that resolved state. ORT need not learn GQA backend enums or
route-selection formulas, and no CUDA-specific masks need to cross the public plugin ABI.

For a GQA contract `C`, the existing soundness obligation remains:

```text
W_L1(C) = max over admitted workload states x:
              max over routes r permitted by C in phase(x):
                  CompleteWorkspace(x, r)
```

`CompleteWorkspace` includes all coexisting route preparation, conversion/preservation, and backend
scratch. Separate persistent and initialization requirements are retained rather than collapsed into
this runtime scalar. For example, where eligibility supports these sets, a contract could permit:

```text
prefill:         {Flash}
decode:          {XQA, Flash}
chunked prefill: {Flash}
```

L1 bounds both decode routes even when XQA is preferred. Runtime can fall back to Flash, but cannot
escape to unfused attention unless that route is also permitted and bounded. Phase admission follows
the actual classifier's possible outcomes, including device-resident length inputs, not assumptions
about the semantic prompt stage. The workload/enforcement assumptions stated above still apply.

The EP resolves route permissions and owns runtime selection within them; ORT selects candidate and
contract IDs when evaluating the whole session. Offer only eligible, user-permitted contracts. The
first implementation can expose one contract for the configured policy; automatic comparison of a
small menu can follow without changing the evaluator. Preserve hard policy restrictions and legacy
defaults rather than silently overriding them. A policy named `memory` is a local selection rule,
not proof of the smallest session footprint once persistent storage and initialization are included.

If a permitted route lacks an adequate bound, the contract remains unresolved. For example, the GQA
aggregate cannot establish a graph-free bound for a committed cuDNN route. Strict-budget planning
must not silently remove that route or substitute a different policy; any alternative must be
explicitly offered and allowed. A change that alters graph effects or interfaces must also be reflected
in the recipe description, not hidden behind a different contract ID.

#### Bind the Selected Contract through L2 and Runtime

Selection must retain the candidate/recipe/contract identities, resolved permissions, workload/device
conditions, and plan generation through application, verification, and kernel initialization. Kernel
construction must consume the selected binding, not independently re-resolve an intent string to a
different policy. Exact post-transformation or PrePack facts may refine requirements only through
validated reconciliation with the selected contract.

L2 may tighten requirements without broadening permissions or changing the accepted backing
assumptions. Larger allocations, different lifetimes, or a different implementation require a verified
revised plan before allocation/use, or an explicit initialization failure. Runtime dispatch remains
inside the committed permissions and fails explicitly when no permitted route is available. Existing
GQA policy enforcement is the foundation; the proposed capability-to-kernel binding and whole-session
checks are not already supplied by that enforcement or by post-PrePack verification alone.

### Select Placement-Sensitive Optimizations through EP Recipes

In budget-planned mode, the EP selects its execution recipe; ORT schedules its application and verifies
the result. Gate unplanned placement-sensitive transformations, rather than banning optimizers because
their implementation lives in the framework. EPs can reuse shared implementations through
`GraphOptimizerRegistry`; they do not need to duplicate those optimizers.

Separate **describing/selecting L2+ effects** from **applying graph rewrites**. A complete recipe
provides implementation requirements and enough logical graph effects for ORT to evaluate the result
without executing the transformation. Query kernel-private bounds from the workload, configuration,
and selected implementation/contract; build the provisional execution/backing plan from the described
tensors, aliases, and boundaries. Do not use graph mutation as the normal way to discover each
candidate's requirements.

Assignment-dependent selectors can use a tentative EP-ownership view or an isolated candidate graph.
This supplies same-EP eligibility context without finalizing assignments. Evaluate placement, L2+
recipes, and dispatch contracts jointly: moving a boundary can change which recipes are eligible and
requires refreshing affected descriptions. L2+ planning therefore precedes final placement, while
concrete rewriting normally occurs only when finalizing a provisional winner.

Common normalization can run before discovery and establish the planning graph revision. Optional
post-partition fusion/layout optimizations must participate through selected recipes or be disabled in
this mode. Required casts, copies, and correctness transformations remain framework responsibilities,
but their execution and memory effects must be included before final budget acceptance. Any later
transformation that changes the selected implementation or backing assumptions requires re-evaluation.
Here, L2 graph optimization is distinct from the companion proposal's Level-2 kernel requirement
verification.

The existing `ComputeCapability::nodes_to_optimize` and `optimization_func` are application hooks,
not pre-assignment requirement descriptions. Extend their planning integration so ORT can evaluate the
recipe without invoking its mutating callback. After selecting a provisional winner, apply only its
selected callbacks with tentative EP ownership established, then reconcile and verify before final
commitment. Each callback must apply the described recipe and update the capability consistently;
it must not independently select a different fusion or implementation.

If a recipe cannot describe its effects sufficiently, it may use temporary metadata-only lowering or
an isolated planning-graph transformation to produce that description. This is an implementation
fallback, not a requirement to run the existing L2+ pipeline for every trial. Unknown requirements
remain unresolved. Neither this fallback nor ordinary discovery, provisional rewriting, and verification
may execute inference or bulk-materialize device weights or kernels.

### Generate a Small Family of CPU/GPU Assignment Boundaries

**Candidate boundaries** here are candidate CPU/GPU cuts: trial device-assignment frontiers used to
generate candidate plans. They are not new subgraphs or fusion recipes. For a validated ordered
chain of `L` blocks, start with a CPU prefix and GPU suffix:

```text
cut k: block[0 .. k-1] on CPU | block[k .. L-1] on GPU
k = 0 .. L
```

This defines at most `L + 1` cuts, rather than `2^L` independent block assignments. Fixed nodes, provider
support, and indivisible regions can exclude cuts or create unavoidable device islands. Generate cuts
on demand near the current boundary instead of storing a full node-placement copy for every cut.

A CPU/GPU assignment boundary is a graph frontier, not just the output of block `k-1`. Two adjacent
graph regions both assigned to CUDA do not create an inter-device transfer merely because they are
different regions. For each trial assignment, identify **every** tensor edge crossing devices,
including residual/skip connections, shared consumers, and graph I/O. Include necessary copies,
staging backing, casts/layout conversions, and synchronization in the provisional plan; deduplicate
physical allocations separately from edge enumeration.

Moving a region to CPU does not necessarily release all of its apparent GPU bytes: a shared initializer
may still have a GPU consumer, and transfers can create new backing or prolong lifetimes. Do not assume
memory is monotonic in the number of CPU blocks. Interpolation or binary search is appropriate only
after validating that property for the candidate family; otherwise use a bounded, deterministic search.
Additional families, such as explicitly identified expert regions, can be added later without attempting
all node-placement combinations.

### Evaluate Candidate Plans with a Provisional Session Plan

`GetCapability()` can see connectivity and shape metadata, but not the finalized placement, transfer
nodes, execution schedule, allocation reuse, and requirements across all EPs. Its local node estimates
cannot by themselves decide which placement has the lowest whole-session backing. This stage evaluates
complete candidate plans assembled from graph regions, assignment boundaries, and EP execution
candidates, not an isolated fusion candidate.

The framework should use this loop:

```text
discover support, regions, recipe descriptions, and small EP-authored contract menus
    -> evaluate the preferred logical plan for tentative placement/recipes/contracts
    -> diagnose the initialization or Run peak and generate targeted alternatives
    -> cheap-score alternatives; build logical execution/backing plans for a bounded shortlist
    -> choose a provisional winner under the configured objective
    -> apply its selected recipes on a candidate graph with tentative EP ownership
    -> reconcile transformations and verify the resulting execution/backing plan
    -> commit the verified final placement/contracts/backing plan
    -> materialize within the accepted initialization plan
```

First try lower-memory, already enforceable contracts on peak-driving nodes when the objective prefers
retaining GPU placement. Include bounded bundles across tied peaks: reducing one node's workspace can
save nothing if another node still determines a shared scratch pool's size. Then trial nearby region
cuts when contract changes do not fit. Do not enumerate the Cartesian product of all node contract menus.

Cache pure requirement calculations by graph revision, relevant geometry/boundaries, workload,
device/configuration, recipe, and contract, but never merge allocation identity merely because two
nodes have the same geometry. Reuse region/dependency summaries to cheaply prioritize trials; use the
real execution/allocation strategy to verify finalists. Cheap scores and partial summaries are not
budget-feasibility evidence.

Each finalist needs a provisional placement-specific plan covering physical allocation roots and
lifetimes: weights/prepack coexistence, persistent KV/I/O/output backing, activations and copies, node
workspace, initialization scratch, and retained allocator pools. Account asynchronous completion,
alignment, and non-pool overhead/reserve, not just logical live-tensor bytes. The budget test is the
maximum of initialization and Run backing peaks plus disjoint overhead/reserve, under the companion
proposal's ownership and workload assumptions. CUDA's `gpu_mem_limit` is an arena limit, not this entire
session-associated device budget.

Per-implementation L1 bounds are inputs to this backing plan; neither summing all L1 peaks nor taking
their maximum is a generic budget test.

The EP owns route eligibility and requirement formulas; the framework consumes contract descriptors
without knowing CUDA route names. Unknown sizes or unsupported lifetime assumptions remain unresolved,
not zero. A fixed-contract placement search is a useful first slice; contract alternatives can be added
without changing the whole-plan evaluator.

Bound the number of trials/refinements, cache graph revisions, and use deterministic tie-breaks.
Exhausting the search means **no feasible plan was found within the available candidates and limits**,
not proof that the model cannot fit. Report the evaluated peak breakdown, unresolved terms, selected
boundary, and search limits. A found plan is conditional on the modeled workload/allocator assumptions,
not an unconditional device-wide no-OOM guarantee.

### Separate `GetCapability()` Discovery from Budget Acceptance

The opt-in flow needs a planning-safe discovery path. CUDA currently performs support discovery and
additive budget gating in `GetCapability()`; simply passing a null resource accountant also skips its
current L1 estimator queries. Refactor requirement discovery separately from acceptance instead of
using that shortcut.

| Stage | Responsibility |
|:------|:---------------|
| Discover | EPs expose executable candidates, optimization recipes, candidate contracts, and allocation requirements without committing resource charges or materializing device kernels. Discover alternatives before applying the selected-placement filter. |
| Plan | The framework evaluates placement/recipes/contracts jointly using tentative EP-ownership views and recipe-described logical graphs, including transfers and execution/backing plans. Check user constraints and both initialization/Run peaks without normally rewriting each trial graph. |
| Apply provisional winner | Apply only the provisional winner's selected recipes on a candidate graph with tentative EP ownership. This concretizes an already evaluated description without committing final placement. |
| Reconcile and verify | Validate the transformed graph against the selected recipes, rediscover affected regions when needed, and rebuild/check transfer, execution, and backing plans, including required copies/casts and both memory peaks. |
| Commit final placement | Finalize the selected EP eligibility/assignments and freeze graph revision, recipes, contract IDs, transfer boundaries, and backing strategy. Commit resource reservations only for the verified final survivors. |
| Materialize and verify | Load device initializers, create kernels, and PrePack within the accepted initialization plan. Reconcile exact L2 requirements with the committed contracts and backing plan before their allocations/use. |

**The final placement decision is the commit step, after selected optimizations and budget verification,
before device materialization.** Choosing a provisional candidate is not final acceptance. Use an
isolated candidate graph or a reliable rebuild mechanism so rejected rewrites do not contaminate other
trials. If recipe consistency, support, or budget verification fails, reject the candidate and try another
within the search limit, or report failure explicitly. A recipe mismatch requires corrected discovery
and re-evaluation, not acceptance of an unmodeled implementation.

Accurate discovery should make final verification a consistency check, not the first opportunity to
discover the optimization's memory costs. Committed EP assignments alone are insufficient: the
optimization recipes, dispatch contracts, transfers, and physical backing assumptions must also remain
consistent through materialization and runtime.

This requires framework changes: `SequentialPlanner::CreatePlan()` currently runs in session-state
finalization after partitioning. Reuse its allocation/release/stream machinery in a provisional planning
path rather than pretending the existing `GetCapability()` call already has that global view.

Some EPs compile or create external state during capability discovery or partitioning. Those EPs need
an explicit planning-safe discovery/materialization separation before joining this search; repeatedly
calling an arbitrary EP's current `GetCapability()` is not a generic dry run.

Current layering fallback makes annotation-directed nodes available to later EPs when the requested EP
does not claim them. In budget-planned mode, that change invalidates the candidate: rebuild and
re-evaluate it within the refinement limit, or fail explicitly. Do not silently accept different
placement, fusion, dispatch permissions, transfers, or allocation lifetimes after checking the budget.
The final executable graph, including inserted copy/cast nodes, must match the verified plan.

L2 verification after kernel construction/PrePack cannot alone prevent initialization OOM. Requirements
must be bounded before those allocations; materialization must honor the accepted backing strategy.
A larger L2 requirement needs a verified revised plan before additional allocation/use, or an explicit
initialization error. Warn-only continuation and allocator growth beyond the plan are not hard-budget
enforcement.

### Preserve the Same Planning Semantics across the Plugin EP ABI

The internal C++ and plugin EP interfaces should normalize to the same candidate/recipe/contract model.
The existing plugin ABI exposes `OrtEp::GetCapability()`, support-building functions such as
`EpGraphSupportInfo_AddSingleNode()` and `EpGraphSupportInfo_AddNodesToFuse()`, and `OrtEp::Compile()`.
These do not provide the proposed planning protocol or a C ABI equivalent of `optimization_func`;
the current plugin adapter does not use `GraphOptimizerRegistry`.

Preserve existing signatures and legacy behavior. Add optional, version-negotiated planning and
selected-plan application/materialization support, with append-only API-table extensions and checks
before accessing newly introduced members. Use opaque handles and C-compatible descriptors/builders,
not `ComputeCapability`, STL containers, or `std::function` across the ABI. The adapter should carry the
selected candidate, recipe, and contract identities into application/compilation so the plugin cannot
silently choose an unmodeled implementation.

ORT-owned descriptors can copy candidate metadata; EP-owned opaque recipe and contract state need
explicit retention/release rules covering discarded candidates and initialization failures. Specify
counted arrays, allocator ownership, handle validity, and `OrtStatus*` failure behavior. Do not retain
borrowed graph/node views beyond their documented lifetimes.

A compiling EP can apply its internal recipe during compilation of the selected region without exposing
every internal rewrite to the host, provided the result honors the planned interface, contracts, and
allocation bounds. A registered-kernel EP that needs host graph rewrites requires selected-plan
application through a scoped, validated edit interface or an ORT-registered recipe, not unrestricted
access to ORT's internal C++ `Graph`.

An older plugin or missing planning callback means unsupported planning, not zero memory cost. Admit
it to strict budget planning only through a proven compatibility adapter; otherwise reject that
configuration explicitly while preserving its legacy non-planning path. The initial CUDA/CPU slice
need not implement all plugin extensions. Concrete callback signatures, descriptor layouts, ownership,
and negotiation belong in a focused follow-up API proposal; this section specifies required semantics,
not finalized public APIs.

### Implementation Slices and Validation

Build on the companion proposal's fixed-placement lifetime report and enforceable backing plan before
adding automatic offloading:

1. Discover/validate regions, executable candidates, and optimization recipes; report their complete
   transfer frontiers without changing placement.
2. Evaluate explicit CPU/GPU cuts under fixed enforceable contracts with provisional plans and host
   support/memory checks, including selected-recipe application and verification before final commitment.
3. Add bounded automatic boundary selection, then small contract-menu alternatives and peak-targeted
   bundles, with immutable bindings and final-assignment verification.

Test incomplete or overlapping groups, block 1 versus block 10 matching, numeric block order, residual
edges crossing several blocks, shared GPU weights after CPU offload, unsupported CPU dtypes/opsets,
and boundaries crossing indivisible capabilities. Planner tests should cover initialization-dominated
peaks, scratch reuse/tied peaks, transfer-induced non-monotonic memory, transformed/fused bindings,
fusion-only support, optional fusion lost across a cut, stale recipe/graph revisions, mismatched optimizer
results, rejected-candidate isolation, unexpected EP fallback, and search-limit exhaustion. Plugin
extensions need coverage for older/missing callbacks, selected-contract propagation, recipe/contract-state cleanup,
and explicit unsupported-planning diagnostics. Distinguish recipe-description queries from mutating
application callbacks, and check agreement between logical candidate plans and concretely transformed
graphs. Cover L1 adapter completeness, unknown versus proven-zero requirements, resolved-policy identity,
phase/route fallback closure, policy pins, and rejection of unresolved route bounds. Validate modeled
physical backing against measured device peaks for representative admitted workloads; measurements
calibrate/validate the model rather than prove coverage of unobserved shapes or routes.
