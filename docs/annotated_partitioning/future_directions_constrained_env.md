# Future Directions: Constrained Environment Partitioning

## Context

Today's annotation-based partitioning requires each node to carry a `layer_ann` metadata property. The `LayeringIndex` matches these annotations against user-supplied rules (prefix trie + exact match) to assign nodes to devices. The `IResourceAccountant` optionally enforces memory budgets.

The goal: make ORT as easy to use as ollama for running large models on machines with limited GPU memory — automatic or near-automatic layer offloading without requiring model producers to annotate every node.

---

## Direction 1: Name-Based Substring Matching (No Annotation Step)

### Idea

Skip the annotation metadata entirely. Instead, match directly against **node names** using substrings/patterns from the configuration. MS Foundry models (and most HuggingFace exports) already encode layer structure in node names:

```
/model/layers.0/self_attn/q_proj/MatMul
/model/layers.0/self_attn/k_proj/MatMul
/model/layers.15/mlp/gate_proj/MatMul
/model/embed_tokens/Gather
/model/norm/LayerNormalization
```

A config like `gpu(layers.0, layers.1, ..., layers.15); cpu(layers.16, ..., layers.31)` would partition without any model modification.

### How to Approach

1. **Extend `LayeringRuleMatcher` to operate on node names directly.** Today the matcher looks at `node.GetMetadata("layer_ann")`. Add a parallel path that runs the same prefix/substring matching against `node.Name()`. This is a small change — the trie infrastructure already exists.

2. **Config via a separate session option (same syntax).** Rather than introducing a new qualifier into the existing `kOrtSessionOptionsLayerAssignmentSettings` syntax, add a parallel session option that uses **the same `device(prefix1, prefix2, ...); ...` grammar** but matches against `Node::Name()` instead of node metadata:

   ```cpp
   // Existing (annotation-based, matches node metadata 'layer_ann'):
   static const char* const kOrtSessionOptionsLayerAssignmentSettings =
       "session.layer_assignment_settings";

   // NEW (name-based, matches Node::Name() via substring/prefix):
   static const char* const kOrtSessionOptionsNameBasedLayerAssignment =
       "session.name_based_layer_assignment";
   ```

   Usage stays identical — only the matching target differs:
   ```
   # Annotation-based (existing):
   session.layer_assignment_settings = "cuda(encoder_layer, attention); cpu(embed)"

   # Name-based (new, same syntax):
   session.name_based_layer_assignment = "cuda(layers.0, layers.1); cpu(layers.16)"

   # Range expressions (future extension, not currently supported):
   session.name_based_layer_assignment = "cuda(layers.[0-15]); cpu(layers.[16-31])"
   ```

   This approach:
   - Keeps the existing parser/grammar unchanged (reuse `LayeringRuleMatcher`)
   - Makes intent explicit — users opt into name-based matching deliberately
   - Both options can coexist (annotation-based takes priority if both match a node)
   - No risk of breaking existing annotation-based workflows

3. **Build index at load time.** During `InferenceSession::Initialize()`, after graph is loaded but before partitioning:
   - If config contains name-based rules, iterate all nodes once
   - Build `NodeIndex → RuleIndex` map using substring/prefix matching on `Node::Name()`
   - Feed this into the existing `LayeringIndex` infrastructure (same downstream flow)

4. **Range expressions (future extension).** The current `LayeringRuleMatcher` supports only exact match (`=prefix`) and prefix match. It does **not** support range syntax today. For transformer models with numbered layers, a future extension could add range support:
   ```
   cuda(layers.[0-15]); cpu(layers.[16-31])
   ```
   This would avoid enumerating 32+ layer prefixes manually, but requires new parsing logic. Until then, users must enumerate each layer prefix explicitly or use a broad prefix like `layers.` that captures all layers for a single device.

### Advantages

- **Zero model modification** — works with any model that has structured naming
- **Reuses existing partitioning infrastructure** — only the index-building step changes
- **User-friendly** — users can inspect node names with Netron and write rules directly
- **Composable with resource accounting** — can combine name-based assignment with memory budgets

### Risks / Open Questions

- **Name stability**: Node names aren't guaranteed stable across exports. Mitigated by prefix/substring matching rather than exact names.
- **Unnamed nodes / nodes created by graph transformers**: See detailed analysis below.
- **Subgraph nodes**: Control flow (If/Loop) subgraph nodes may have different naming conventions.

### Handling Nodes Created by Graph Transformers

#### Pre-partitioning transformers (Level 1)

Level 1 optimizers run **before** partitioning. With annotation-based matching, these transformers propagate annotations to new nodes via the `AddNode(..., annotation_source)` overload, which copies `GetLayeringAnnotation()` from an original node. The name-based approach needs an analogous story.

**Key insight: new node names ARE derivative of original names.** Verified in the codebase:

- `Graph::GenerateNodeName(base_name)` takes a base string and ensures uniqueness by appending `_token_<N>` only on collision.
- Transformers construct the base name from original node(s):
  - `layer_norm_fusion.cc`: `GenerateNodeName(mul_node.Name() + "/LayerNormFusion/")`
  - `matmul_add_fusion.cc`: `GenerateNodeName(matmul_node.Name() + "/MatMulAddFusion")`
  - `attention_fusion.cc`: `GenerateNodeName("Attention")` ← **exception — generic name**

So if the original node was `/model/layers.5/self_attn/q_proj/MatMul`, the fused node typically becomes something like `/model/layers.5/self_attn/q_proj/MatMul/MatMulAddFusion` — which still contains the original layer prefix and will match substring rules like `layers.5`.

**This means name-based matching is naturally robust to pre-partitioning fusions** — no explicit annotation-copying step is needed, because the substring match against the derivative name still hits the same rules. This is actually **simpler** than the annotation-based approach.

**Edge cases to handle:**
1. **Generic names** (e.g., `"Attention"` without incorporating the original name): Some fusions create nodes with generic names that don't contain layer-identifying substrings. In general, name-based partitioning can only be used when node names contain representative strings suitable for layer matching. Two options exist:
   - **Annotation fallback**: Use annotation-based assignment for these nodes, or update the transformer to follow the derivative naming convention.
   - **Exact-match rule**: If the user is willing to assign *all* nodes with a given generic name to the same device, they can add an exact-match rule (e.g., `cuda(=Attention)`). This works without annotations, but applies uniformly — all nodes matching that name go to the specified device.
2. **Multiple source nodes**: When a fusion merges N nodes from potentially different layers, the resulting name typically uses one of them as the base. If the merged nodes span layer boundaries, the fused node will match whichever layer the chosen base name belongs to. This mirrors annotation-based behavior (annotation is copied from one source node).

**Recommendation:** No special machinery needed for pre-partitioning transformers. The derivative naming convention already preserves matchability. Document this as a convention that transformer authors should follow: always pass an original node's name as the base to `GenerateNodeName()`.

#### Post-partitioning transformers (Level 2+)

Level 2+ optimizers run **after** partitioning, when EP assignments have already been made. These transformers already copy the EP assignment from original nodes:

```cpp
new_node.SetExecutionProviderType(original_node.GetExecutionProviderType());
```

**No action needed for name-based matching here.** By the time Level 2+ transformers run, partitioning is complete. The node names are irrelevant — only the EP assignment matters, and that's already propagated correctly.

---

## Direction 2: Minimize Allocations for Static-Shape Models

### Goal

For models with fully static shapes (common in transformer inference with fixed batch/sequence dimensions), ORT should minimize or eliminate runtime memory allocations. When all tensor shapes are known at `Initialize()` time, the runtime can pre-compute exact memory requirements and pre-allocate everything upfront — no arena overhead, no per-`Run()` allocation calls, deterministic memory usage.

Additionally, by knowing exact memory requirements before execution begins, ORT can **minimize the chance of running into OOM** — if the total memory needed exceeds device capacity, the session can fail at `Initialize()` time with a clear error rather than crashing mid-inference with an opaque allocation failure.

This brings ORT's allocation efficiency on par with specialized runtimes like llama.cpp, while retaining ORT's generality for arbitrary model architectures.

### Reference: What llama.cpp Does

llama.cpp exploits that transformer inference has **fully deterministic memory usage**:
- All weight tensors are known at load time
- KV cache is pre-allocated for max sequence length
- Intermediate activation buffers have shapes determined by `(batch, seq_len, hidden_dim)` — all known in advance
- Workspace/temp buffers are known per-op and pre-planned

This means the runtime computes **exactly** how much memory is needed before running — zero allocation calls during inference.

### What ORT Already Does

Before discussing gaps, it's important to note what ORT already provides:

- **Shape inference runs during `Initialize()`** — specifically during `graph.Resolve()` after graph optimizations but before memory planning. Shape info is populated on all `NodeArg` objects.
- **Memory pattern pre-allocation** — When `EnableMemPattern` is set and shapes are static, `TensorAllocatorWithMemPattern` computes exact allocation offsets/sizes via `OrtValuePatternPlanner`, then pre-allocates a single large buffer per device via `Reserve()` (bypasses arena, calls device allocator directly). Intermediate buffers are reused based on liveness analysis.
- **Initializers can bypass arena** — When `use_device_allocator_for_initializers` is set, initializers are loaded via `Reserve()` (direct device allocation) during session state finalization, bypassing the arena's binning/coalescing logic. Without this option, initializers allocate through the arena like any other buffer.
- **BFC Arena** — Best-Fit-with-Coalescing allocator provides fast (O(log n)) sub-allocation. Allocations are cheap, but it suffers from memory waste due to power-of-two growth and chunk granularity.

**What's already outside the arena:**
- Initializers → `Reserve()` (direct device allocator)
- Memory pattern buffers (activations) → `Reserve()` (direct device allocator)

**What's still arena-allocated:**
- **Runtime temp/workspace buffers** — kernels call `GetScratchBuffer<T>()` during `Compute()`, which allocates from the device allocator (arena by default). These are ephemeral (freed when kernel completes) and their sizes are only known at execution time.

**Note on arena and temp buffer reuse:** While the BFC arena wastes memory due to chunk granularity, it does provide **automatic reuse** for temp buffers during sequential execution — subsequent kernels reuse the same arena memory for their scratch needs without actual device allocations.

**CUDA mempool as an alternative:** ORT already supports replacing the BFC arena with native CUDA memory pools (`cudaMallocFromPoolAsync`), enabled via:
```cpp
session_options.add_session_config_entry("ep.cudaexecutionprovider.arena.use_cuda_mempool", "1");
```
This provides stream-aware pooling managed by the CUDA driver, with less memory waste than BFC. Since `GetScratchBuffer()` uses the same device allocator as activations (resolved via `SessionState::GetAllocator(device)` — keyed by `OrtDevice` only, not by purpose), enabling mempool automatically benefits temp buffers too. A separate temp-only allocator would require architectural changes to `AllocatorMap` (currently not feasible without significant refactoring).

**Pre-allocated space for temp buffers (alternative approach):** If workspace sizes can be pre-computed (see Phase A below), temp buffers could be served from the same pre-allocated memory pattern buffer used for activations. Since workspace is live only during its kernel's execution, it participates naturally in liveness-based offset planning — no arena needed at all. This is the more principled solution: solve the workspace size problem first, then temp memory becomes part of the static plan.

### IResourceAccountant Precision

`IResourceAccountant::ComputeResourceCount()` already returns **exact sizes when shapes are static**:
- Initializer sizes: exact via `GetSizeInBytesFromTensorProto()`
- Output tensor sizes: exact when all dimensions are known via `GetSizeInBytesFromTensorTypeProto()`
- Weight deduplication: tracked via `pending_weights_`/`committed_weights_` to avoid double-counting

**Note on actual memory consumption:** While these give exact logical tensor sizes, actual device memory will be rounded up to page/alignment boundaries — either per allocation or for a single large buffer. The reported sizes are a lower bound; real usage includes alignment overhead.

It applies a **1.5x safety multiplier** to account for unknowable temp/workspace allocations. This multiplier exists because temp buffer sizes are discovered only at runtime — no kernel declares its workspace needs in advance. **For static-shape models where `DeclareWorkspaceRequirements` (Phase A below) has been implemented for all relevant kernels, this multiplier becomes unnecessary and should be bypassed** — exact workspace sizes are known at planning time, eliminating the need for a safety margin.

**Per-node accounting (not per-layer)**: `IResourceAccountant` tracks costs at the node/subgraph level — `nodes_costs` in `IndexedSubGraph` is for accounting after an EP claims nodes (single nodes or fused groups), not for layering. It has no concept of "layer" as defined by the layering index.

**Do we need per-layer aggregation?** Likely not as a first-class feature in `IResourceAccountant`.

The current budget enforcement cuts off EP placement at the individual node level — once the cumulative budget is exceeded, subsequent nodes are rejected regardless of layer boundaries. **This is intentional**: atomic rollback of already-placed nodes within a layer was explicitly rejected during the previous implementation phase due to complexity and because it would require re-running `GetCapability()` with different node sets.

The layering index already controls the *order* in which nodes are presented to the EP (layer by layer), so in practice the budget cutoff tends to land near layer boundaries. If exact layer-boundary cuts are desired in the future, it would need to be a separate mechanism (e.g., pre-computing per-layer costs and making accept/reject decisions at the layer level before calling `GetCapability()`), not a change to the accountant.

For debugging/UX purposes, per-layer summaries can be computed externally by summing node costs grouped by their layer annotation — no accountant changes needed.

### The Remaining Gap: Runtime Temp Buffers

The 1.5x multiplier and arena waste both stem from the same root cause: **kernels allocate workspace at runtime without prior declaration.**

Examples of how workspace sizes are determined:
- **cuDNN Convolution**: Workspace depends on algorithm selection (`cudnn_fe_graph->get_workspace_size()`)
- **cuDNN RNN**: Queries cuDNN at runtime (`cudnnGetRNNTempSpaceSizes()`)
- **Attention kernels**: Computed from runtime parameters (`batch_size * seq_len * num_heads * head_size`)

The `SequentialExecutionPlan` tracks only activation tensors and initializers — workspace buffers are ephemeral and never planned statically. There is no `GetWorkspaceSize()` virtual method on `OpKernel`.

To eliminate both the multiplier and arena waste for temp buffers, two problems must be solved:
1. **Learn temp buffer sizes in advance** — e.g., a `DeclareWorkspaceRequirements(shapes)` method on kernels, queryable during `Initialize()` when shapes are static
2. **Allocate temp buffers outside the arena** — once sizes are known, include them in the memory pattern plan alongside activations

### What ORT Is Missing

| Capability | llama.cpp | ORT Today | Gap |
|-----------|-----------|-----------|-----|
| Static memory plan | Yes — computed at load | Yes — `MemoryPatternGroup` + `Reserve()` | ✓ Activations + initializers already bypass arena |
| Pre-allocated activation buffers | Yes — fixed slots | Yes — memory patterns with liveness reuse | ✓ Already exists for static shapes |
| Workspace pre-computation | Yes — known per op | No — kernels discover at runtime | Need `DeclareWorkspaceRequirements()` on kernels |
| Workspace outside arena | Yes — part of static plan | No — `GetScratchBuffer()` uses arena | Need to include workspace in memory pattern plan |
| Zero-copy weight transfer | mmap + `cudaMemcpy` at load per layer | mmap + `cudaMemcpy` at load per partition | ✓ Same model — not a gap |

**Note on weight transfer:** Both llama.cpp and ORT use the same approach: **static partitioning at load time, no dynamic weight swapping during inference.**

- **llama.cpp**: User sets `-ngl N` (number of GPU layers). At load time, those N layers' weights are `cudaMemcpy`'d from mmap'd file to GPU. Remaining layers stay in host memory. No runtime swapping — this is performant because there is zero weight transfer overhead during token generation.
- **ORT (with constrained partitioning)**: The layering index + `IResourceAccountant` determines which nodes run on GPU. At `Initialize()` time, only those nodes' initializers are copied to device from mmap'd external data. Remaining weights stay in host memory.

**Best practice for constrained environments:** Model weights should be stored as **external data on disk** (not embedded in protobuf). This ensures:
1. ORT memory-maps the file — minimal host memory overhead during loading.
2. Only GPU-partitioned nodes' weights are copied to device — no OOM as long as partitioning respects the budget.
3. CPU-partitioned nodes' weights remain accessible via mmap without requiring a separate host allocation.

Since partitioning is decided once at `Initialize()` time and all required device weights are resident before `Run()`, there is no need for dynamic layer loading/offloading during inference.

### How to Approach

#### Phase A: Workspace Pre-declaration (`DeclareWorkspaceRequirements`)

The core missing piece. Today no kernel declares its temp buffer needs before `Compute()`. To close this gap, introduce a method on `OpKernel` that returns workspace descriptors — each with a size and a key for later retrieval.

**Analogy to PrePack:** This mechanism is similar to the existing `PrePack()` pattern — both are called once during session state finalization (not during `Run()`), both store results that are reused across all subsequent runs. `PrePack()` pre-processes weight data; `DeclareWorkspaceRequirements()` pre-computes workspace layout.

**Interface:**

```cpp
struct WorkspaceRequirement {
  size_t size_bytes;        // Size of this workspace buffer
  int slot_id;             // Kernel-defined slot identifier (0, 1, 2, ...)
                           // Unique within a single kernel instance
};

// Optional override on OpKernel (called during FinalizeSessionState):
virtual Status DeclareWorkspaceRequirements(
    gsl::span<const TensorShape> input_shapes,
    InlinedVector<WorkspaceRequirement>& requirements) const {
  return Status::OK();  // Default: no declaration (fall back to arena)
}
```

A kernel can declare multiple workspace slots (e.g., attention needs separate Q transpose buffer, output buffer, seqlens buffer). The `slot_id` is defined by the kernel author and is stable across calls — it identifies *which* buffer within that kernel's logic.

**Key constraint:** Multiple nodes may use the same kernel class. Each node instance gets its own set of workspace slots. The unique key for retrieval is `(NodeIndex, slot_id)` — the framework supplies `NodeIndex`, the kernel supplies `slot_id`.

**Memory reuse via liveness-based offset planning:**

Workspace buffers are live only during their kernel's execution step. This means workspaces from non-overlapping steps can share the same physical memory — exactly the same liveness analysis already used for activation tensors. The offset planner assigns overlapping offsets to workspaces whose liveness intervals don't intersect:

```
Step 0: Node A workspace (slots 0,1) → offsets [0, 4096]
Step 1: Node B workspace (slot 0)    → offset [0]  ← reuses Node A's memory
Step 2: Node C workspace (slots 0,1) → offsets [0, 8192]  ← reuses again
```

Peak workspace memory = max over all steps of (sum of workspace slots for that step), not the sum of all workspaces across all nodes.

**Concurrency model (multiple concurrent `Run()` calls):**

The existing memory pattern system already handles this correctly:
- The **pattern** (offset/size map) is computed once during `Initialize()` and cached in `SessionState` — shared, read-only.
- The **actual buffer** is allocated per-`Run()` by each `ExecutionFrame` using the pattern as a blueprint.
- Each concurrent `Run()` gets its own `ExecutionFrame` with its own workspace buffer — no sharing, no synchronization needed.

Workspace pre-allocation follows the same model:
- `DeclareWorkspaceRequirements()` is called during `FinalizeSessionState()` → produces a workspace offset plan (shared, immutable).
- Each `Run()` allocates a workspace buffer of `peak_workspace_size` bytes and uses offsets from the plan.
- Concurrent runs each get their own buffer — safe without locks.

**Note on CUDA:** In practice, concurrent `Run()` on the same CUDA session is uncommon (users don't typically do this). But the design should remain thread-safe by following the same per-run buffer pattern.

**Planning flow (during FinalizeSessionState):**

1. For each kernel in the execution plan (when shapes are static), call `DeclareWorkspaceRequirements()` with the inferred input shapes.
2. Record `{NodeIndex, slot_id} → size_bytes` in the execution plan.
3. Run liveness analysis: workspace for node N is live only during step N's execution.
4. Compute offsets (same algorithm as activation patterns) → yields `peak_workspace_size` and per-slot offsets.
5. Store workspace pattern in `SessionState` (like memory patterns).

**Per-Run retrieval (during Compute):**

Each `ExecutionFrame` allocates a workspace buffer of `peak_workspace_size` via `Reserve()` and provides offset-based access:

**Alternative A: Transparent fallback in GetScratchBuffer**

Modify `GetScratchBuffer<T>(slot_id, size, stream)` to check for a pre-planned buffer first:

```cpp
template <typename T>
IAllocatorUniquePtr<T> GetScratchBuffer(int slot_id, size_t count_or_bytes, Stream* stream) const {
  // Check if workspace was pre-planned for this node + slot
  void* preallocated = context_.GetPreallocatedWorkspace(slot_id);
  if (preallocated) {
    // Return non-owning pointer (buffer lifetime managed by the frame)
    return IAllocatorUniquePtr<T>(static_cast<T*>(preallocated), [](T*){});
  }
  // Fall back to arena (dynamic shapes, or DeclareWorkspaceRequirements not implemented)
  return IAllocator::MakeUniquePtr<T>(allocator_, count_or_bytes, false, stream);
}
```

Pro: Minimal kernel code changes — just add `slot_id` parameter. Con: Overloads `GetScratchBuffer` semantics; non-owning vs owning pointer distinction is subtle.

**Alternative B: Separate retrieval path**

Keep `GetScratchBuffer()` unchanged for arena allocation. Add a new method:

```cpp
// In OpKernelContext:
void* GetPreallocatedWorkspace(int slot_id) const;
// Returns nullptr if not pre-planned → kernel must call GetScratchBuffer() instead

// Kernel usage:
void* ws = context->GetPreallocatedWorkspace(0);
if (!ws) {
  scratch_buffer_ = GetScratchBuffer<void>(workspace_size, stream);
  ws = scratch_buffer_.get();
}
```

Pro: Clear separation, no ambiguity about ownership. Con: Kernels need explicit fallback logic (but this is a one-time pattern per kernel).

**Compatibility with dynamic shapes:** Both alternatives are opt-in. If `DeclareWorkspaceRequirements()` is not overridden or returns empty (dynamic shapes), everything falls back to `GetScratchBuffer()` → arena, exactly as today. Same kernel binary works for both static and dynamic models.

**Incremental adoption:** Start with the highest-impact ops (attention, convolution, GEMM) which account for the majority of workspace. Less common ops continue using the arena with a reduced safety multiplier in `IResourceAccountant`.

**Buffer strategy:** Workspace offsets can share the activation buffer (liveness doesn't overlap — workspace is live only during its step, activations may span steps). Alternatively, a separate workspace buffer is simpler initially and easier to account for in memory limits.

##### EP Plugin C ABI Surface for Workspace Pre-declaration

In the plugin architecture, `DeclareWorkspaceRequirements` crosses the C ABI boundary. This section defines the concrete API additions.

**Declaration side — new optional function pointer on `OrtKernelImpl`:**

```c
// Added to OrtKernelImpl (optional, like PrePackWeight):
ORT_API2_STATUS(DeclareWorkspaceRequirements,
    _In_ OrtKernelImpl* this_ptr,
    _In_reads_(num_inputs) const int64_t* const* input_shapes,   // shape per input
    _In_reads_(num_inputs) const size_t* input_shape_ranks,      // rank per input
    _In_ size_t num_inputs,
    _Out_writes_all_(max_slots) OrtWorkspaceSlot* slots,         // pre-allocated by ORT
    _In_ size_t max_slots,                                        // capacity (e.g., 8)
    _Out_ size_t* num_slots);                                     // actual count filled

// Slot descriptor (C struct, no inheritance):
typedef struct OrtWorkspaceSlot {
  int slot_id;          // Kernel-defined, stable identifier (0, 1, 2, ...)
  size_t size_bytes;    // Required size for this slot
} OrtWorkspaceSlot;
```

If `DeclareWorkspaceRequirements` is NULL on the `OrtKernelImpl`, ORT skips the kernel during workspace planning (falls back to arena at runtime).

**Retrieval side — new function in `OrtEpApi`:**

```c
// Added to OrtEpApi (called by plugin kernels during Compute):
ORT_API2_STATUS(KernelContext_GetPreallocatedWorkspace,
    _In_ const OrtKernelContext* context,
    _In_ int slot_id,
    _Outptr_result_maybenull_ void** buffer);   // NULL if not pre-planned
```

Returns a pointer into the pre-allocated workspace buffer at the offset computed during planning. Returns NULL if no workspace was pre-planned for this kernel+slot (dynamic shapes, or kernel didn't declare). The pointer is valid for the duration of the `Compute()` call.

**Slot ID provisioning — how kernels define unique slot_ids:**

Slot IDs are **kernel-author-defined constants**, not dynamically allocated. Each kernel class defines its slots as an enum or set of constants in its implementation:

```cpp
// Example: CUDA Attention kernel (inside the plugin DLL)
namespace cuda {
class AttentionKernel : public OrtKernelImplBase {
  // Slot IDs are private constants — stable across versions, used as array indices
  static constexpr size_t kSlotQTranspose = 0;
  static constexpr size_t kSlotKTranspose = 1;
  static constexpr size_t kSlotVTranspose = 2;
  static constexpr size_t kSlotSoftmaxWorkspace = 3;
  static constexpr size_t kNumSlots = 4;

  OrtStatus* DeclareWorkspaceRequirements(...) override {
    slots[kSlotQTranspose] = {kSlotQTranspose, batch * heads * seq * head_dim * sizeof(half)};
    slots[kSlotKTranspose] = {kSlotKTranspose, batch * heads * seq * head_dim * sizeof(half)};
    slots[kSlotVTranspose] = {kSlotVTranspose, batch * heads * seq * head_dim * sizeof(half)};
    slots[kSlotSoftmaxWorkspace] = {kSlotSoftmaxWorkspace, cudnn_workspace_size};
    *num_slots = kNumSlots;
    return nullptr;
  }

  OrtStatus* Compute(OrtKernelContext* ctx) override {
    void* q_buf = nullptr;
    // Uses pre-planned workspace if available, falls back to arena otherwise
    api_->KernelContext_GetScratchBuffer(ctx, kSlotQTranspose, q_transpose_size, &q_buf);
    // ... use q_buf ...
  }
};
}  // namespace cuda
```

**Key design properties:**

| Property | Design Choice | Rationale |
|----------|--------------|-----------|
| Slot ID scope | Per kernel *instance* (node) | Same kernel class on different nodes gets separate buffers; ORT disambiguates via `(NodeIndex, slot_id)` |
| Slot ID assignment | Static constants in kernel code | No registry, no runtime allocation, no cross-kernel coordination needed |
| Slot ID range | `[0, max_slots)` — small integers | Simple array indexing in the offset plan; `max_slots` = 8 is generous for any single kernel |
| Uniqueness guarantee | Kernel author's responsibility | Same convention as `input_index` in `PrePackWeight` — the kernel knows its own buffer layout |
| Stability across versions | Expected (like enum values) | Slot IDs are internal to the kernel; not exposed to users or other kernels |

**Where state lives:**

| State | Location | Lifetime |
|-------|----------|----------|
| Slot definitions (id + size) | Returned by `DeclareWorkspaceRequirements` → stored in `ExecutionPlan` | Session lifetime (computed once at `Initialize()`) |
| Offset map `{(NodeIndex, slot_id) → offset}` | `SessionState::workspace_pattern_` (new field, analogous to `mem_patterns_`) | Session lifetime (shared, read-only) |
| Peak workspace size per EP/device | `SessionState::workspace_pattern_` | Session lifetime |
| Actual workspace buffer | `ExecutionFrame` (allocated per-`Run()` via `Reserve()`) | Single `Run()` invocation |

**No global slot registry needed.** Unlike input indices which are defined by the ONNX op schema, slot IDs are entirely internal to the kernel implementation. Two different kernel classes can both use `slot_id=0` without conflict — the framework always qualifies with `NodeIndex`. This means:
- No coordination between kernel authors
- No registration step during plugin initialization
- No versioning concerns (IDs never cross the plugin boundary as semantic values)

#### Phase B: Eliminate Arena for Static-Shape Models

Once workspace is pre-declared, **all** allocations for a static-shape model are known at `Initialize()` time:
- Initializers → already `Reserve()` (done)
- Activations → already memory-pattern `Reserve()` (done)
- Workspace → new, via Phase A

At this point, the BFC arena serves no purpose for the main execution path. The session could:
1. Pre-allocate exact memory per device (sum of pattern peak + workspace peak)
2. Use offset-based addressing for all buffers
3. Disable the arena entirely for this session (save memory waste from chunk granularity)

Runtime temp buffers from ops that don't implement `DeclareWorkspaceRequirements()` can still fall back to a small arena.

### Custom Executable: Purpose and Scope

A minimal custom executable (CLI tool) serves three purposes:

1. **Code example and test bed.** Demonstrates how to configure and exercise the constrained-environment features (name-based partitioning, memory budgets, static allocation mode) end-to-end using the ORT C/C++ API. Acts as a living integration test that exercises the full pipeline without depending on GenAI or external frameworks.

2. **Interactive LLM demo (llama.cpp-style UX).** Loads a transformer ONNX model, manages the decode loop (prompt → KV cache → token sampling → output), and interacts with the user via stdin/stdout. This showcases ORT's ability to run large models on constrained hardware with the same user experience as llama.cpp — but backed by ORT's general-purpose runtime.

3. **Primitive GenAI replacement for testing.** For the narrow case of single-model, single-user, greedy/top-k text generation, the executable can replace GenAI as a simpler alternative that doesn't pull in the full GenAI dependency. It is **not** a production replacement for GenAI (no batching, no beam search, no speculative decoding) — it is a minimal harness for validating that the partitioning and memory features work correctly on real models.

**What the executable handles (application-level):**
- Token encode/decode (via sentencepiece or tokenizers library)
- KV cache allocation and rotation (fixed max sequence length)
- Autoregressive decode loop (feed output token back as next input)
- Session configuration: name-based layer assignment, memory budget, static shapes

**What ORT handles (session-level, no executable changes needed):**
- Graph partitioning across devices (Direction 1 + `IResourceAccountant`)
- Static memory pre-allocation (Phase A + B)
- Kernel execution, data transfers, stream synchronization

**Feasibility: no fundamental ORT blockers.** The existing session API (`CreateSession` → `Run` with named I/O) is sufficient for an autoregressive decode loop. KV cache management is feeding output tensors back as inputs — the same pattern GenAI uses over the same C API.

| Concern | Status | Notes |
|---------|--------|-------|
| Tokenizer | External dependency | ORT core has no tokenizer. Options: link sentencepiece, use onnxruntime-extensions, or bundle minimal BPE |
| KV cache rotation | Straightforward | Pre-allocate `(batch, heads, max_seq, head_dim)`, feed `past_key_values` outputs back as inputs each step |
| Decode loop | Trivial | Run session → extract logits → sample token → repeat |
| Model format | Constraint | Requires decoder-style ONNX export with explicit KV cache I/O (HuggingFace optimum exports provide this) |
| Partitioning | This design | Direction 1 + `IResourceAccountant` |
| Static allocation | Phase A+B | Fixed `max_seq_len` makes all decode-phase shapes static |

The executable would be ~500–1000 LOC (excluding tokenizer): configure session options, set up KV cache tensors, run the generate loop. The tokenizer is the only non-trivial external dependency.

---

## Recommended Roadmap

```
Near-term (low effort, high value):
├── 1. Name-based matching in LayeringRuleMatcher
│     - Add 'name:' prefix qualifier to config syntax
│     - Match against Node::Name() instead of metadata
│     - Support range expressions for numbered layers
│
├── 2. Precise memory estimation for static-shape models
│     - Run shape inference during Initialize() when shapes are known
│     - Compute exact per-layer memory in IResourceAccountant
│
Mid-term (medium effort):
├── 3. Auto-partitioning with memory budget only
│     - User specifies "6GB GPU budget"
│     - ORT computes optimal layer split automatically
│     - Combines (1) + (2)
│
├── 4. Static allocation mode
│     - Pre-allocate all buffers when shapes are known
│     - Eliminate per-Run() allocation overhead
│
Long-term (high effort, ollama-parity):
├── 5. Layer prefetch pipeline
│     - Stream weights CPU↔GPU during execution
│     - Enables running models larger than GPU memory
│
└── 6. Integration with GenAI
      - KV cache-aware memory planning
      - Continuous batching + layer offload coordination
```

---

## Key Insight

The fundamental difference between ORT and llama.cpp for this use case is **generality vs specialization**. llama.cpp knows it's running a transformer with sequential layers. ORT handles arbitrary graphs. The trick is to **detect** when a model is transformer-like (sequential layers, static shapes) and engage a specialized execution path — without losing generality for other model types.

Direction 1 (name-based matching) is the lowest-friction win: it makes the existing annotation system accessible without model modification. Direction 2 (static pre-allocation + auto-splitting) is what closes the gap with ollama but requires more infrastructure work, particularly around shape-aware memory planning at partition time.
