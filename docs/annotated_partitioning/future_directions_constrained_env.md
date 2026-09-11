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

1. **Add a new `SubstringMatcher` for node-name matching.** Today the `LayeringRuleMatcher` supports exact match and prefix match (via a trie that walks from position 0 of the input string). Neither mode works for node names: a node named `/model/layers.5/self_attn/q_proj/MatMul` does not *start with* `layers.5` — the identifying substring appears in the middle. Name-based matching fundamentally requires **substring** search. The existing trie infrastructure is irrelevant here — a new, simpler matching approach is needed (see "Substring Matching Implementation" below).

2. **Config via a separate session option (same grammar, different matcher).** Rather than introducing a new qualifier into the existing `kOrtSessionOptionsLayerAssignmentSettings` syntax, add a parallel session option that uses **the same `device(pattern1, pattern2, ...); ...` grammar** but performs **substring matching** against `Node::Name()` instead of prefix/exact matching against node metadata:

   ```cpp
   // Existing (annotation-based, matches node metadata 'layer_ann'):
   static const char* const kOrtSessionOptionsLayerAssignmentSettings =
       "session.layer_assignment_settings";

   // NEW (name-based, matches Node::Name() via substring):
   static const char* const kOrtSessionOptionsNameBasedLayerAssignment =
       "session.name_based_layer_assignment";
   ```

   Usage stays identical — only the matching target and algorithm differ:
   ```
   # Annotation-based (existing, prefix/exact match against node metadata):
   session.layer_assignment_settings = "cuda(encoder_layer, attention); cpu(embed)"

   # Name-based (new, substring match against Node::Name()):
   session.name_based_layer_assignment = "cuda(layers.0/, layers.1/); cpu(layers.16/)"

   # Range expressions (future extension, not currently supported):
   session.name_based_layer_assignment = "cuda(layers.[0-15]); cpu(layers.[16-31])"
   ```

   This approach:
   - Keeps the existing parser/grammar unchanged (reuse the `device(pattern1, pattern2, ...); ...` syntax)
   - Uses a **new `SubstringMatcher`** (not the existing trie-based `LayeringRuleMatcher`) for the actual matching
   - Makes intent explicit — users opt into name-based matching deliberately
   - The two options are **mutually exclusive** — setting both returns an error
   - No risk of breaking existing annotation-based workflows

3. **Build index at load time.** During `InferenceSession::Initialize()`, after graph is loaded but before partitioning:
   - If config contains name-based rules, iterate all nodes once
   - Build `NodeIndex → RuleIndex` map using substring matching on `Node::Name()`
   - Feed this into the existing `LayeringIndex` infrastructure (same downstream flow)

4. **Range expressions (future extension).** The config grammar does **not** support range syntax today. For transformer models with numbered layers, a future extension could add range support:
   ```
   cuda(layers.[0-15]); cpu(layers.[16-31])
   ```
   This would avoid enumerating 32+ layer prefixes manually, but requires new parsing logic. Until then, users must enumerate each layer prefix explicitly or use a broad prefix like `layers.` that captures all layers for a single device.

### Substring Matching Implementation

The existing `LayeringRuleMatcher` uses a **trie** for prefix matching — it walks the input string from position 0 and checks if any trie path matches a prefix of the input. This only works when the pattern appears at the **start** of the matched string.

For node names, patterns appear in the **middle**:
```
Pattern:    "layers.5"
Node name:  "/model/layers.5/self_attn/q_proj/MatMul"
                    ^^^^^^^^ — match at position 7, not position 0
```

The trie is useless here. A new `SubstringMatcher` class is needed.

#### Design: Flat vector + `std::string::find`

The simplest correct approach:

```cpp
class SubstringMatcher {
 public:
  explicit SubstringMatcher(const LayeringRules& rules);

  /// Returns the index of the best matching rule for the given node name.
  /// "Best" = longest pattern that appears as a substring in the name.
  std::optional<size_t> Match(std::string_view node_name) const;

 private:
  // Sorted by pattern length descending — longest patterns checked first.
  // First match wins (longest-match priority).
  struct PatternEntry {
    std::string pattern;
    size_t rule_index;
  };
  InlinedVector<PatternEntry> patterns_;  // sorted longest-first
};
```

**Match algorithm:**
```cpp
std::optional<size_t> SubstringMatcher::Match(std::string_view node_name) const {
  for (const auto& entry : patterns_) {
    if (node_name.find(entry.pattern) != std::string_view::npos) {
      return entry.rule_index;
    }
  }
  return std::nullopt;
}
```

**Why longest-match-first ordering:**

Without it, `layers.1` (a substring of `layers.10`, `layers.11`, ..., `layers.19`) would incorrectly match nodes from layers 10–19. By checking longer patterns first, `layers.10` matches before `layers.1` gets a chance. Users should include the path separator for unambiguous matching: `layers.1/` won't match `layers.10/...`.

**Performance:** With ~64 patterns and node names < 200 chars, this is O(P × N) per node where P = number of patterns and N = name length. Total cost for a 1000-node model: ~64 × 200 × 1000 = ~12M character comparisons. This completes in microseconds on modern hardware and runs only once during `Initialize()`. No optimization (Aho-Corasick, etc.) is warranted.

**Priority semantics:**

| Scenario | Behavior |
|----------|----------|
| Single match | Return that rule's index |
| Multiple matches (different lengths) | Longest pattern wins |
| Multiple matches (same length, different rules) | First rule in config order wins (stable sort by length, preserving config order as tiebreaker) |
| No match | Return `nullopt` → node goes to fallback EP (CPU) |

**Integration with `LayeringIndex`:**

`LayeringIndex` owns either a `LayeringRuleMatcher` (annotation mode) or a `SubstringMatcher` (name-based mode) — the two are mutually exclusive. The `ProcessGraph` method branches based on which mode is active:

```cpp
void LayeringIndex::ProcessGraph(const Graph& graph, std::optional<size_t> parent_layer_id) {
  for (const auto& node : graph.Nodes()) {
    std::optional<size_t> matched_rule_idx;

    if (substring_matcher_) {
      // Name-based mode: substring match against node name, no inheritance.
      // Node names are dense, so each node is matched independently.
      matched_rule_idx = substring_matcher_->Match(node.Name());
    } else {
      // Annotation-based mode: prefix/exact match against metadata,
      // with subgraph inheritance for unannotated nodes.
      const std::string& annotation = node.GetLayeringAnnotation();
      if (!annotation.empty()) {
        matched_rule_idx = matcher_.Match(annotation);
      }
      if (!matched_rule_idx && parent_layer_id) {
        matched_rule_idx = parent_layer_id;
      }
    }

    if (matched_rule_idx) {
      node_to_layering_index_[node.Index()] = *matched_rule_idx;
    }
  }
}
```

**Why mutual exclusivity (not priority/fallback):**

The two modes have fundamentally different inheritance semantics. Annotations are sparse — nodes without annotations inherit from their subgraph parent to maintain device consistency. Names are dense — virtually every node has a name, so inheritance is unnecessary and would incorrectly override name-based matches in subgraphs. Making the modes mutually exclusive keeps the semantics simple and predictable.

### Advantages

- **Zero model modification** — works with any model that has structured naming
- **Reuses existing partitioning infrastructure** — only the index-building and matching steps change
- **User-friendly** — users can inspect node names with Netron and write rules directly
- **Composable with resource accounting** — can combine name-based assignment with memory budgets

### Risks / Open Questions

- **Name stability**: Node names aren't guaranteed stable across exports. Mitigated by prefix/substring matching rather than exact names.

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
   - **Substring rule**: Add a substring pattern (e.g., `cuda(Attention)`) to assign all nodes whose name contains `Attention`. Note that name-based assignment does not support the '=' exact-match qualifier — all patterns are substrings.
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

### Dynamic Shapes in Transformer Models

Nearly all transformer models are *exported* with dynamic batch + sequence_length — this is the default in PyTorch `torch.onnx.export`, Hugging Face Optimum, and Olive. However, for constrained deployment the picture is different:

- **Fixed-shape re-export is standard for edge/embedded:** batch=1, seq_len=128 (or a few discrete lengths like 128/256/512). This is standard practice for TensorRT, CoreML, and QNN deployments.
- **LLM serving keeps shapes dynamic** (variable prompts, KV-cache growth). But these are typically high-VRAM scenarios (A100/H100), not constrained environments.
- **Vision transformers** (ViT, DINO, etc.) have fixed patch sequences — only batch is dynamic, and fixing batch=1 yields fully static shapes.

**Implication for pre-allocation:** The target audience of `pre_allocate_execution_buffers` — embedded, edge, single-model-per-device — typically *can* use static shapes. Models are re-exported with fixed dimensions as part of the deployment pipeline. The dynamic-shape case (LLM serving with variable seq_len) lives in a different deployment tier where VRAM budget is less critical than throughput and the arena allocator handles repeated allocations efficiently.

**Implication for workspace estimation:** Even with dynamic shapes, `EstimateWorkspace` (Level 1) and `DeclareWorkspaceRequirements` (Level 2) remain valuable for *budget decisions* — they can use configured input-shape limits (max batch, max seq_len from model config) to estimate how many nodes fit on the device. However, ordinary shape inference does not prove that every operator's output shape is monotonic with respect to its input dimensions. For example, a `Slice` with negative indices can produce a smaller output when its input dimension increases. A shape inferred from maximum inputs is therefore an estimation hint, not a guaranteed upper bound. Pre-allocation must retain runtime bounds checks and an allocation fallback rather than relying on the hint alone to prevent OOM.

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

**CUDA mempool as an alternative:** ORT supports replacing the BFC arena with native CUDA memory pools (`cudaMallocFromPoolAsync`). This is enabled via the EP-scoped arena configuration key `arena.use_cuda_mempool` (e.g., `"ep.cuda.arena.use_cuda_mempool" = "1"` in session config). This provides stream-aware pooling managed by the CUDA driver, with less memory waste than BFC. Since `GetScratchBuffer()` uses the same device allocator as activations (resolved via `SessionState::GetAllocator(device)` — keyed by `OrtDevice` only, not by purpose), enabling mempool automatically benefits temp buffers too. A separate temp-only allocator would require architectural changes to `AllocatorMap` (currently not feasible without significant refactoring).

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

#### The Chicken-and-Egg Problem: Workspace Estimation vs EP Assignment

**Problem statement:** To make precise memory budget decisions during `GetCapability()`, `IResourceAccountant` needs workspace sizes per node. But workspace sizes come from kernels, which don't exist until *after* EP assignment (kernels are created during `Compile()`/session state finalization — same reason `PrePack` happens late). At decision time, you can't ask a kernel that doesn't exist yet.

**Why this matters:** The goal is not to fail gracefully — it's to **avoid failure entirely**. Today, models either OOM on device or trigger heavy VRAM thrashing on Windows. The partitioning must be conservative enough to prevent this, while accurate enough to maximize GPU utilization.

**Solution: Two-level estimation with post-assignment verification**

**Level 1 — Static workspace estimation function (at partitioning time):**

When a kernel is registered (via `KernelRegistry_AddKernel`), optionally provide a **static estimation function** — a class-level function that takes node info and returns a conservative workspace estimate without needing a kernel instance:

```c
// Registered alongside the kernel definition in KernelRegistry_AddKernel:
typedef OrtStatus*(ORT_API_CALL* OrtKernelWorkspaceEstimateFunc)(
    _In_ const OrtEpApi* api,          // for querying node attributes/shapes/device props
    _In_ const OrtNode* node,          // the specific node being evaluated
    _In_ const OrtEp* ep,              // EP instance (for device properties like SM count)
    _Out_ size_t* estimated_workspace_bytes);
```

The function uses `api->Node_GetAttribute*()` and `api->Node_GetInputShape()` to access the node's attributes and input shapes, and `api->Ep_GetDeviceProperty()` for GPU hardware properties — everything needed to compute workspace without a kernel instance.

**Can the estimate be precise (not just conservative)?**

Depends on the kernel:

| Kernel | Workspace depends on | Available at GetCapability()? | Precise estimate? |
|--------|---------------------|-------------------------------|-------------------|
| **Attention (Flash)** | shapes + `num_heads` attr + `device_prop.multiProcessorCount` | ✓ All available (EP has device_prop) | **YES — exact** |
| **Conv (cuDNN)** | cuDNN `build_plans(handle)` with tensor shapes + conv params | ✓ EP has handle; shapes/attrs available from node | **YES — exact** (with `HEUR_MODE_A`) |
| **GEMM/MatMul** | No workspace | N/A | N/A (returns 0) |

For **attention**, workspace is determined by `get_num_splits_and_buffer_sizes()` which is pure arithmetic given `(batch, seq, heads, head_size, multiProcessorCount)`. The EP already has `multiProcessorCount` from `cudaGetDeviceProperties()` which runs during EP construction (before `GetCapability()`). So the estimate can be **exact**.

For **cuDNN-based ops** (Conv), the workspace depends on which algorithm cuDNN selects via `build_plans(handle)`. However, a cuDNN handle is just a lightweight context object (`cudnnCreate` + `cudnnSetStream`) — the EP already owns one from construction time. With static shapes, all inputs to `build_plans()` are known: tensor dimensions, conv parameters (from node attributes), and the handle. The `CUDNN_HEUR_MODE_A` (fast heuristic) used by ORT is essentially a lookup + arithmetic — not actual GPU profiling. So the estimation function **can call `build_plans()` and get the exact workspace size**. This makes Conv estimates **precise too**.

The reason `build_plans()` currently runs during first `Compute()` is historical: ORT didn't have a pre-execution workspace declaration phase, and shapes weren't known until runtime. With static shapes and the estimation function pattern, this computation can move earlier.

The estimation function accesses the handle by casting `OrtEp*` to the EP's concrete type (safe because the function is EP-specific code registered by that EP):

```cpp
auto* cuda_ep = static_cast<const CudaEp*>(ep);  // plugin path
cudnnHandle_t handle = cuda_ep->GetCudnnHandle();
```

**Can it be the same function as DeclareWorkspaceRequirements?**

Not the same function pointer (different signatures — one has a kernel instance, one doesn't). But the **core computation logic can be a shared static helper** called from both:

```cpp
// Illustrative BSHD Attention formula: one float accumulation value per element.
// A real kernel substitutes its exact formula but keeps this checked-arithmetic pattern.
static std::optional<size_t> ComputeIllustrativeBshdAttentionWorkspace(
    int64_t batch, int64_t sequence, int64_t heads, int64_t head_size) {
  const int64_t dimensions[] = {batch, sequence, heads, head_size};
  // A zero extent fixes this product at zero even if another extent is unknown.
  for (const int64_t dimension : dimensions) {
    if (dimension == 0) {
      return 0;
    }
  }

  for (const int64_t dimension : dimensions) {
    if (dimension < 0) {
      return std::nullopt;
    }
  }
  try {
    // size_t matches the allocator contract; SafeInt rejects values above the platform's SIZE_MAX.
    SafeInt<size_t> bytes{sizeof(float)};
    for (const int64_t dimension : dimensions) {
      bytes *= SafeInt<size_t>(dimension);
    }
    return static_cast<size_t>(bytes);
  } catch (const OnnxRuntimeException&) {
    return std::nullopt;
  }
}

// Level 1's graph parser extracts the illustrative BSHD input shape.
static std::optional<size_t> EstimateAttentionWorkspace(
    gsl::span<const int64_t> input_shape) {
  if (input_shape.size() != 4) {
    return std::nullopt;
  }
  return ComputeIllustrativeBshdAttentionWorkspace(
      input_shape[0], input_shape[1], input_shape[2], input_shape[3]);
}

// Level 2 (has kernel instance; called during FinalizeSessionState):
Status Attention::DeclareWorkspaceRequirements(
    gsl::span<const WorkspaceInputShape> input_shapes,
    InlinedVector<WorkspaceRequirement>& reqs) const {
  reqs.clear();
  const TensorShape* input_shape = GetWorkspaceInputShape(input_shapes, 0).GetShape();
  if (input_shape == nullptr || input_shape->NumDimensions() != 4) {
    // Missing/unusable metadata is not a session-initialization error. Schema validation is
    // separate; retain the runtime allocation fallback.
    return Status::OK();
  }

  const auto total = ComputeIllustrativeBshdAttentionWorkspace(
      (*input_shape)[0], (*input_shape)[1], (*input_shape)[2], (*input_shape)[3]);
  if (!total.has_value()) {
    return Status::OK();  // Valid partial/large shape: retain runtime allocation fallback.
  }
  if (*total != 0) {
    reqs.push_back({*total, kSlotFlashWorkspace, /*alignment_bytes=*/0});
  }
  return Status::OK();
}
```

Both wrappers call the same checked helper and therefore produce identical results. Production helpers
may also take already-extracted device properties, but every product, sum, conversion, and rounding
operation must remain inside checked arithmetic.

**For cuDNN-based ops**, the estimation function can also be precise — it calls `build_plans()` using the EP's handle and the node's shapes/attributes. Level 2 re-check serves as a diagnostic safety net — if the post-fusion total exceeds the budget, a warning is logged indicating that the Level 1 estimate was too optimistic (e.g., cuDNN returning different workspace sizes due to driver version differences or fusion changing the algorithm selection).

**KernelCreateInfo and registration macros:**

Today `KernelCreateInfo` contains `{kernel_def, kernel_create_func, status}`. To add the estimation function:

```cpp
struct KernelCreateInfo {
  std::unique_ptr<KernelDef> kernel_def;
  KernelCreateFn kernel_create_func;
  OrtKernelWorkspaceEstimateFunc workspace_estimate_func;  // NEW — may be nullptr
  Status status;
};
```

The existing `ONNX_OPERATOR_TYPED_KERNEL_EX` macro doesn't need changes — it produces `KernelCreateInfo` via `BuildKernelCreateInfo<>()`. A new macro variant adds the estimation function for kernels that implement it:

```cpp
// New macro: ONNX_OPERATOR_TYPED_KERNEL_EX_WITH_ESTIMATE
// Same as ONNX_OPERATOR_TYPED_KERNEL_EX but also registers a workspace estimation function.
#define ONNX_OPERATOR_TYPED_KERNEL_EX_WITH_ESTIMATE(                                          \
    name, domain, ver, type, provider, builder, estimate_fn, ...)                             \
  class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(provider, domain, ver, type, name);             \
  template <>                                                                                 \
  KernelCreateInfo                                                                            \
  BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(                                \
      provider, domain, ver, type, name)>() {                                                 \
    return KernelCreateInfo(                                                                  \
        builder.SetName(#name)                                                                \
            .SetDomain(domain)                                                                \
            .SinceVersion(ver)                                                                \
            .Provider(provider)                                                               \
            .Build(),                                                                         \
        static_cast<KernelCreatePtrFn>(                                                       \
            [](FuncManager&, const OpKernelInfo& info,                                        \
               std::unique_ptr<OpKernel>& out) -> Status {                                    \
              out = std::make_unique<__VA_ARGS__>(info);                                       \
              return Status::OK();                                                            \
            }),                                                                               \
        estimate_fn);                                                                         \
  }
```

This requires a new `KernelCreateInfo` constructor overload:

```cpp
struct KernelCreateInfo {
  std::unique_ptr<KernelDef> kernel_def;
  KernelCreateFn kernel_create_func;
  OrtKernelWorkspaceEstimateFunc workspace_estimate_func;  // NEW — may be nullptr
  Status status;

  // Existing constructor (unchanged — sets workspace_estimate_func to nullptr):
  KernelCreateInfo(std::unique_ptr<KernelDef> definition,
                   KernelCreateFn create_func)
      : kernel_def(std::move(definition)),
        kernel_create_func(create_func),
        workspace_estimate_func(nullptr) {}

  // New constructor with estimation function:
  KernelCreateInfo(std::unique_ptr<KernelDef> definition,
                   KernelCreateFn create_func,
                   OrtKernelWorkspaceEstimateFunc estimate_func)
      : kernel_def(std::move(definition)),
        kernel_create_func(create_func),
        workspace_estimate_func(estimate_func) {}

  KernelCreateInfo(KernelCreateInfo&& other) noexcept
      : kernel_def(std::move(other.kernel_def)),
        kernel_create_func(std::move(other.kernel_create_func)),
        workspace_estimate_func(other.workspace_estimate_func) {}

  KernelCreateInfo() = default;
};
```

**Usage example** (registering a CUDA Attention kernel with estimation):

```cpp
// In cuda_contrib_kernels.cc:
ONNX_OPERATOR_TYPED_KERNEL_EX_WITH_ESTIMATE(
    Attention,                            // name
    kMSDomain,                            // domain
    1,                                    // ver
    float,                                // type
    kCudaExecutionProvider,               // provider
    KernelDefBuilder()                    // builder
        .TypeConstraint("T", DataTypeImpl::GetTensorType<float>())
        .InputMemoryType(OrtMemTypeCPUInput, {3, 4}),
    &cuda::EstimateAttentionWorkspace,    // estimate_fn  ← NEW argument
    cuda::Attention<float>);              // kernel class (__VA_ARGS__)
```

Kernels without estimation continue using `ONNX_OPERATOR_TYPED_KERNEL_EX` unchanged — their `workspace_estimate_func` is `nullptr`, and the budget logic applies the 1.5x multiplier as today. The migration is opt-in, kernel by kernel.

**Integration with GetCapability and the resource budget:**

The estimation function is called during budget enforcement — by the EP directly (in-tree) or by the host bridge (plugin). The result is combined with the base cost from `IResourceAccountant`.

**Multiplier handling — non-member helper approach:**

`ComputeResourceCount()` currently applies a 1.5x multiplier to approximate workspace for kernels without estimation functions. With precise workspace estimates available, the multiplier must be skipped. Rather than changing `ComputeResourceCount()`'s signature, we move the multiplier out and into a non-member helper that encapsulates the budget decision:

```cpp
// Non-member helper (e.g., in resource_accountant_helpers.h):
// Called by both in-tree GetCapability and the plugin host bridge.
ResourceCount ComputeNodeCostForBudget(
    IResourceAccountant& accountant,
    const Node& node,
    std::optional<ResourceCount> workspace_estimate) {
  // ComputeResourceCount returns base cost: outputs + initializers (dedup'd)
  // NO multiplier — multiplier is now applied here when needed
  ResourceCount base_cost = accountant.ComputeResourceCount(node);

  if (workspace_estimate.has_value()) {
    // Precise workspace known — add it directly, no multiplier
    return AddResourceCounts(base_cost, *workspace_estimate);
  }
  // No workspace estimate — apply heuristic multiplier (1.5x)
  return ApplyWorkspaceHeuristic(base_cost);
}

// Multiplier as an explicit utility:
ResourceCount ApplyWorkspaceHeuristic(ResourceCount base) {
  size_t bytes = std::get<0>(base);
  return ResourceCount{static_cast<size_t>(bytes * 1.5)};
}
```

**Design rationale:**
- `ComputeResourceCount()` signature is **unchanged** — it returns the raw base cost (outputs + initializers with dedup). The 1.5x multiplier moves out of the accountant into this helper.
- The helper is the **single decision point** for both code paths (in-tree and plugin host bridge). No duplicated logic.
- `ApplyWorkspaceHeuristic()` makes the multiplier explicit and testable. It can be adjusted (e.g., per-EP or per-op-type) without changing any interface.
- The helper integrates naturally with the existing budget check pattern:

```cpp
// Usage in GetCapability (both paths):
auto total_cost = ComputeNodeCostForBudget(*accountant, node, workspace_estimate);
auto would_be_consumed = AddResourceCounts(consumed, total_cost);

if (has_budget && ResourceCountExceeds(would_be_consumed, budget)) {
    accountant->SetStopAssignment();
    break;
}

consumed = would_be_consumed;
sub_graph->SetAccountant(accountant);
sub_graph->AppendNodeCost(total_cost);
```

**Why this is clean with committed/uncommitted weights:**

- **Weight dedup is unaffected.** `ComputeResourceCount()` handles pending/committed weight tracking internally. The workspace estimate is purely additive — it's not a weight, so it doesn't participate in dedup.
- **`AppendNodeCost()` stores the combined total.** When `AccountForNode()` runs later (during `TryAssignNodes()`), it adds the stored cost (base + workspace) to `consumed_amount` and commits the weights. The workspace portion just inflates the per-node cost.
- **`CommitWeightsForNode()` only touches initializers.** Workspace is a separate addend, not tracked in weight sets.
- **`ResetForNewPass()` is fine.** The workspace estimate is stateless — recomputed fresh from node shapes each call, no state to carry across passes.

If no estimation function is registered for a kernel, the helper applies the 1.5x multiplier as today (unchanged behavior).

**Example estimation function (CUDA Conv):**

```cpp
OrtStatus* EstimateConvWorkspace(const OrtEpApi* api, const OrtNode* node,
                                  const OrtEp* ep, size_t* out) {
    // Get input shape (X: NCHW)
    const int64_t* x_shape = nullptr;
    size_t x_rank = 0;
    OrtStatus* status = api->Node_GetInputShape(node, 0, &x_shape, &x_rank);
    if (status) return status;
    if (x_rank < 3) {
        return api->CreateStatus(ORT_INVALID_ARGUMENT,
                                 "Conv: input X must be at least rank 3");
    }

    // Get weight shape (W: [M, C/group, kH, kW, ...])
    const int64_t* w_shape = nullptr;
    size_t w_rank = 0;
    status = api->Node_GetInputShape(node, 1, &w_shape, &w_rank);
    if (status) return status;
    if (w_rank != x_rank) {
        return api->CreateStatus(ORT_INVALID_ARGUMENT,
                                 "Conv: weight rank must match input rank");
    }

    // Get conv attributes (all optional — defaults to empty/zeros per ONNX spec)
    const int64_t* pads = nullptr;
    size_t pads_count = 0;
    status = api->Node_GetAttributeInts(node, "pads", &pads, &pads_count);
    if (status) return status;  // distinguishes "not present" (OK + nullptr) from error

    const int64_t* strides = nullptr;
    size_t strides_count = 0;
    status = api->Node_GetAttributeInts(node, "strides", &strides, &strides_count);
    if (status) return status;

    const int64_t* dilations = nullptr;
    size_t dilations_count = 0;
    status = api->Node_GetAttributeInts(node, "dilations", &dilations, &dilations_count);
    if (status) return status;

    // EP-specific: cast to concrete type to access cuDNN handle
    auto* cuda_ep = static_cast<const CudaEp*>(ep);
    cudnnHandle_t handle = cuda_ep->GetCudnnHandle();
    if (!handle) {
        return api->CreateStatus(ORT_RUNTIME_EXCEPTION,
                                 "Conv: cuDNN handle not available on EP");
    }

    // Build cuDNN frontend graph and query workspace
    // (same logic as CreateCudnnFeExecutionPlan but without storing state)
    auto graph = BuildConvFrontendGraph(x_shape, x_rank, w_shape, w_rank,
                                         pads, pads_count, strides, strides_count,
                                         dilations, dilations_count);
    if (!graph) {
        return api->CreateStatus(ORT_RUNTIME_EXCEPTION,
                                 "Conv: failed to build cuDNN frontend graph");
    }

    auto plan_status = graph->build_plans(handle, cudnn_frontend::BuildPlanPolicy_t::HEURISTICS_ONLY);
    if (plan_status) {
        return api->CreateStatus(ORT_RUNTIME_EXCEPTION,
                                 plan_status.get_message());
    }

    *out = graph->get_workspace_size();
    return nullptr;  // success
}
```

This produces the **exact same result** as the kernel would compute during `DeclareWorkspaceRequirements` — same handle, same shapes, same algorithm selection. The shared logic is the cuDNN frontend graph construction.

**Why no public API for device properties or library handles:** The estimation function is registered BY the EP for ITS kernels — it's EP-specific code running in the EP's own DLL. It can safely cast `OrtEp*` to its concrete type (e.g., `CudaEp*`) to access device_prop, cuDNN handles, etc. This is the same pattern kernels already use: `static_cast<const CUDAExecutionProvider*>(info.GetExecutionProvider())->GetDeviceProp()`. No generic `Ep_GetCudnnHandle` or `Ep_GetDeviceIntProperty` API is needed — that would be CUDA-specific pollution of the universal EP interface.

**Level 2 — Post-fusion budget re-check (before InsertCast and MemcpyTransformer):**

The `TransformGraph()` pipeline has a natural insertion point after EP-specific optimizers but before the transformers that bake in EP boundaries:

```
L1 optimizers → Partition (GetCapability) → L2/L3 EP-specific optimizers → [HERE] → InsertCastTransformer → L4 → MemcpyTransformer
```

At `[HERE]`:
- Nodes are assigned to EPs ✓
- EP-specific fusions (ConvRelu, FusedMatMul, etc.) have already been applied ✓
- The graph reflects the *actual* ops that will become kernels ✓
- Cast nodes have NOT been inserted yet ✓ (no fp16↔fp32 casts at boundaries)
- Memcpy nodes have NOT been inserted yet ✓ (boundaries can still move)
- Kernels do NOT exist yet ✗ (cannot call `DeclareWorkspaceRequirements`)

**Why before InsertCastTransformer:** The InsertCastTransformer inserts fp16↔fp32 Cast nodes at EP boundaries where input/output types don't match. If we offload nodes *after* Cast insertion, we'd leave orphaned Cast nodes at the old boundary and need new ones at the new boundary — similar to the MemcpyTransformer problem. By running before both, any Cast and Memcpy nodes are inserted at the final (post-offload) boundaries.

Since kernels don't exist, we call the **same `EstimateWorkspace()` functions** from Level 1 — but now on the post-fusion graph. This eliminates the only meaningful gap between Level 1 and Level 2: fused ops that didn't exist at `GetCapability()` time now have their own estimation functions registered alongside their kernel definitions.

**Algorithm:**

1. For each node assigned to the constrained EP, look up its `OrtKernelWorkspaceEstimateFunc` from the kernel registry (same registry that will later be used to create the kernel).
2. Call the estimation function on the (possibly fused) node → get workspace.
3. Re-run the budget check: `base_cost + workspace` for all assigned nodes.
4. If total ≤ budget → proceed to InsertCastTransformer/MemcpyTransformer normally.
5. If total > budget → **log a warning** and proceed. Do NOT attempt to offload nodes.

**Why warn-only (no runtime offload):**

The earlier design attempted tail-node offloading at this stage — walking backward through GPU-assigned nodes and reassigning them to CPU. In practice, this is problematic:

- **For bf16/fp16 models** (the dominant constrained-VRAM use case): CPU EP lacks kernels for most bf16/fp16 compute ops (MatMul, Attention, LayerNorm). The offload loop would hit a non-offloadable node almost immediately and accomplish nothing.
- **For fp32 CNN models** (where CPU *could* handle offloaded ops): The performance cost of GPU→CPU→GPU data transfers typically outweighs the memory benefit of offloading a few tail nodes.
- **Complexity vs value**: Offload logic (type checking, contiguous-tail constraint, boundary correctness) adds significant code for a feature that rarely fires and rarely helps.

The correct fix for a Level 2 budget overrun is to **improve Level 1 accuracy** — make the estimation functions precise enough that post-fusion re-check merely confirms (not corrects) the budget. Level 2 serves as a **diagnostic safety net**: if the warning fires, it indicates the Level 1 estimate was too optimistic, and the estimation function for the offending kernel(s) should be improved.

```cpp
// Pseudo-code in TransformGraph, after L2/L3, before InsertCastTransformer:
if (level2_total > budget) {
    size_t overrun = level2_total - budget;
    LOGS(logger, WARNING)
        << "Post-fusion budget re-check: EP '" << ep_type
        << "' exceeds memory budget by " << overrun << " bytes. "
        << "Level 1 estimation was too optimistic. "
        << "Consider improving workspace estimation for fused ops. "
        << "Proceeding — runtime OOM may occur.";
}
```

This keeps the pipeline simple: Level 2 is purely observational (re-check + warn), not interventional. If the warning fires in testing, the developer improves the relevant `EstimateWorkspace()` function. In production, the budget was validated at Level 1 and Level 2 divergence should be rare.

**Why Level 2 exists separately from Level 1:**

Level 1 (during `GetCapability()`) operates on the **pre-fusion** graph. It estimates workspace for the original unfused nodes (Conv, Relu separately). Level 2 operates on the **post-fusion** graph (ConvRelu as a single node). The two can diverge when:
- A fused op's workspace differs from the sum of its parts (common — fusion often reduces workspace)
- Level 2/3 optimizers add or remove nodes (e.g., constant folding eliminates a node entirely)

For most LLM models (which are repetitive transformer blocks with minimal fusion opportunity), Level 1 and Level 2 will agree. Level 2 matters more for CNN models with heavy fusion (Conv+BN+Relu patterns).

**When static shapes are unavailable:**

If the model has dynamic shapes and no usable shape hint, the estimation function cannot compute workspace. In this case:
- Level 1 returns no estimate, and Level 2 returns `Status::OK()` with an empty requirements list.
  Valid but unestimable metadata is never fatal to session initialization.
- `ComputeNodeCostForBudget()` falls back to the 1.5x heuristic multiplier on base cost.
- The user may need to **tune the memory budget by trial and error** — setting a conservative budget and adjusting based on observed OOM or under-utilization. This is analogous to llama.cpp's `-ngl` flag: the user picks a layer count and adjusts based on whether it fits.
- `session.max_shape_override` can provide input-shape hints (for example, maximum batch and sequence length). ORT propagates them through a disposable shadow graph for Level 1 and Level 2 estimation without changing runtime input constraints. As noted above, propagated shapes are estimates rather than proven upper bounds.

**Proposed plugin C ABI for Level 1 (deferred):**

This PR does not add or wire the following C ABI; it is a future design sketch.

```c
// Workspace estimation function type (no kernel instance needed):
typedef OrtStatus*(ORT_API_CALL* OrtKernelWorkspaceEstimateFunc)(
    _In_ const OrtEpApi* api,
    _In_ const OrtNode* node,
    _In_ const OrtEp* ep,
    _Out_ size_t* estimated_workspace_bytes);

// Extension to KernelRegistry_AddKernel in OrtEpApi:
ORT_API2_STATUS(KernelRegistry_AddKernelV2,
    _Inout_ OrtKernelRegistry* registry,
    _In_ const OrtKernelDef* kernel_def,
    _In_ OrtKernelCreateFunc create_func,
    _In_opt_ void* create_func_state,
    _In_opt_ OrtKernelWorkspaceEstimateFunc workspace_estimate_func);  // NEW — may be NULL
```

**Required OrtEpApi additions for the estimation function to query node info:**

```c
// Query input shape from a node's NodeArg (populated by shape inference):
ORT_API2_STATUS(Node_GetInputShape,
    _In_ const OrtNode* node,
    _In_ size_t input_index,
    _Outptr_result_maybenull_ const int64_t** shape,  // NULL if dynamic
    _Out_ size_t* rank);

// Query integer attribute from a node:
ORT_API2_STATUS(Node_GetAttributeInt,
    _In_ const OrtNode* node,
    _In_ const char* attr_name,
    _Out_ int64_t* value);

// Query integer array attribute:
ORT_API2_STATUS(Node_GetAttributeInts,
    _In_ const OrtNode* node,
    _In_ const char* attr_name,
    _Outptr_ const int64_t** values,
    _Out_ size_t* count);
```

**Device properties and library handles** (cuDNN, cuBLAS, etc.) are accessed by casting `OrtEp*` to the EP's concrete type inside the estimation function — no public API needed (see examples above).

---

### Implementation Across Kernel Types and GetCapability Paths

ORT has **three distinct kernel authoring scenarios** and **two GetCapability architectures**. The workspace estimation and declaration APIs must work correctly in each combination.

#### Three Kernel Types

| Type | Description | Registration mechanism | Examples |
|------|-------------|----------------------|----------|
| **In-tree** | C++ kernels compiled into the ORT binary | `BuildKernelCreateInfo<>()` via macros (`ONNX_OPERATOR_TYPED_KERNEL_EX`) | All CPU kernels, legacy CUDA EP kernels |
| **Plugin (shared source)** | Same C++ source as in-tree, compiled into EP plugin DLL, uses adapter layer | `KernelRegistry_AddKernel` C ABI, with `CudaKernelAdapter<T>` bridging | CUDA EP plugin kernels |
| **Pure ABI** | Kernels written directly against the C ABI (`OrtKernelImpl`) | `KernelRegistry_AddKernel` C ABI, `OrtKernelImpl` function pointers | Third-party EP plugin kernels |

#### Two GetCapability Architectures

| Architecture | Resource budgeting location | Workspace estimation call site |
|-------------|---------------------------|-------------------------------|
| **In-tree** | Inside `CUDAExecutionProvider::GetCapability()` — EP owns the loop, calls `resource_accountant->ComputeResourceCount(node)`, makes accept/reject decisions | EP calls estimation function directly in its loop |
| **Plugin bridge** | In `PluginExecutionProvider::GetCapability()` (the C++ host wrapper) — plugin EP only proposes candidates, host does budgeting after plugin returns | Host calls estimation function during budget enforcement |

**Critical difference:** In the plugin path, the plugin's `GetCapabilityImpl` returns a list of "I support these nodes" without resource checks. The **host bridge** (`ep_plugin_provider_interfaces.cc`) then iterates those nodes in topological order, calls `resource_accountant->ComputeResourceCount(node)` for each, and enforces the budget — halting assignment when the threshold is exceeded. The plugin never sees the accountant directly.

#### Implementation: `OrtKernelWorkspaceEstimateFunc` (Level 1 — at partitioning time)

**In-tree path:**

```cpp
// In CUDAExecutionProvider::GetCapability() loop (in-tree only):
const KernelCreateInfo* kci = kernel_lookup.LookUpKernel(node);
std::optional<size_t> workspace_estimate;
if (kci && kci->workspace_estimate_func) {
    size_t ws = 0;
    // In-tree: pass IExecutionProvider* — func casts to CUDAExecutionProvider*
    kci->workspace_estimate_func(this, node, &ws);
    workspace_estimate = ws;
}
// Use non-member helper for budget decision:
auto total_cost = ComputeNodeCostForBudget(*resource_accountant, node, workspace_estimate);
// ... budget check with total_cost ...
```

The estimation function for in-tree kernels is a static member function. It casts the EP pointer to `CUDAExecutionProvider*` to access `GetDeviceProp()` and `PerThreadDefaultCudnnHandle()` — exactly the same pattern kernels already use.

**Plugin bridge path:**

```cpp
// In PluginExecutionProvider::GetCapability() host-side budget loop
// (ep_plugin_provider_interfaces.cc):
for (const auto& node_grouping : api_graph_support_info.node_groupings) {
    const Node& internal_node = node_grouping.nodes[0]->GetInternalNode();

    // Look up workspace estimate function from kernel registry
    const KernelCreateInfo* kci = kernel_lookup.LookUpKernel(internal_node);
    std::optional<size_t> workspace_estimate;
    if (kci && kci->workspace_estimate_func) {
        size_t ws = 0;
        // Plugin path: registered via KernelRegistry_AddKernelV2
        // Function casts OrtEp* to its concrete type internally
        OrtStatus* est_status = kci->workspace_estimate_func(
            &ep_api_, ep_node->ToExternal(), ort_ep_.get(), &ws);
        if (est_status) { OrtApis::ReleaseStatus(est_status); }
        else { workspace_estimate = ws; }
    }

    // Same non-member helper as in-tree:
    auto total_cost = ComputeNodeCostForBudget(*resource_accountant, internal_node,
                                               workspace_estimate);
    // ... budget check with total_cost ...
}
```

**Pure ABI path (third-party EP):**

Same as plugin bridge — the estimation function is registered via `KernelRegistry_AddKernelV2` and called by the host during budget enforcement. The kernel author provides the function pointer at registration time:

```c
// Third-party EP kernel registration:
OrtStatus* MyConvEstimate(const OrtEpApi* api, const OrtNode* node,
                           const OrtEp* ep, size_t* out) {
    // Cast to concrete EP type to access device-specific state:
    auto* my_ep = static_cast<const MyGpuEp*>(ep);
    // ... compute workspace from node shapes + my_ep->device_properties ...
}

// During EP's RegisterKernels callback:
ep_api->KernelRegistry_AddKernelV2(registry, conv_kernel_def, CreateConvKernel,
                                    /*state=*/nullptr, &MyConvEstimate);
```

#### Implementation: `DeclareWorkspaceRequirements` (Level 2 — after kernel creation)

**Current in-tree path:**

```cpp
// In include/onnxruntime/core/framework/op_kernel.h:
[[nodiscard]] virtual Status DeclareWorkspaceRequirements(
    gsl::span<const WorkspaceInputShape> input_shapes,
    InlinedVector<WorkspaceRequirement>& requirements) const {
  requirements.clear();
  return Status::OK();  // Default: no workspace declared
}
```

In-tree kernels override this method like `PrePack()`, and `FinalizeSessionState()` calls it after
kernel instances are created. The authoritative positional three-state input contract and lockstep
signature-replacement decision are specified once under
[Phase A](#phase-a-workspace-pre-declaration-declareworkspacerequirements).

**Plugin and pure-C ABI paths are deferred.** The adapter `OpKernel` header mirrors the in-tree
signature and default no-op only so shared kernel source continues to compile in the plugin build.
There is no `CudaKernelAdapter` forwarding implementation, no `OrtKernelImpl` function pointer, and
no host-side bridge for this virtual. In particular, the plugin host does not invoke
`DeclareWorkspaceRequirements`; plugin-compiled kernels currently keep the adapter default no-op and
continue allocating workspace dynamically.

#### Summary: Where Each Piece Lives

| Component | In-tree | Plugin (shared source) | Pure ABI |
|-----------|---------|----------------------|----------|
| **Workspace estimation func** | Static member on kernel class; stored in `KernelCreateInfo::workspace_estimate_func` | Same static function, registered via `KernelRegistry_AddKernelV2` | C function pointer, registered via `KernelRegistry_AddKernelV2` |
| **Who calls estimation** | EP's `GetCapability()` loop via `ComputeNodeCostForBudget()` helper | Host bridge via same `ComputeNodeCostForBudget()` helper | Host bridge (same) |
| **DeclareWorkspaceRequirements** | Virtual override on core `OpKernel` | Adapter default no-op only; forwarding deferred | Not present; C-ABI design deferred |
| **Who calls DeclareWorkspace** | In-tree `FinalizeSessionState()` | Nobody across the plugin boundary | Nobody |
| **Device property access** | `static_cast<CUDAExecutionProvider*>(ep)->GetDeviceProp()` | `static_cast<const CudaEp*>(ep)->GetDeviceProp()` | `static_cast<const MyEp*>(ep)->GetDeviceProps()` |
| **cuDNN handle access** | `static_cast<CUDAExecutionProvider*>(ep)->PerThreadDefaultCudnnHandle()` | `static_cast<const CudaEp*>(ep)->GetCudnnHandle()` | N/A (EP-specific) |

#### Key Design Principle

The **estimation function** signature differs between in-tree and plugin paths:

- **In-tree:** `static size_t EstimateWorkspace(const IExecutionProvider* ep, const Node& node)` — C++ types, direct EP access
- **Plugin/ABI:** `OrtStatus* EstimateWorkspace(const OrtEpApi*, const OrtNode*, const OrtEp*, size_t*)` — C ABI, opaque types

But both compute the same result. For shared-source kernels, both wrappers call one checked,
graph-type-free math helper, ensuring identical estimates without carrying in-tree graph types across
the plugin boundary.

**Implementation-confirmed boundary (issue #29775 Phase-A pilot, PR #29811, MatMulNBits):** the split above
is sharper than "same function, different signature" — it is two functions with different reusability.
`ComputeFpAIntBGemmWorkspaceSize(m, n, k, sm, multiProcessorCount)` takes only plain shape/arch integers
(no ORT graph types) and is the part that can be shared verbatim by a future plugin implementation. The
logic that extracts `m`/`n`/`k` from `const Node&`/`NodeArg`/`TensorShape` (in-tree types only) is a
separate, thinner wrapper that plugin code cannot reuse and must re-implement against whatever shape
representation the C ABI exposes (`OrtNode`/`Node_GetInputShape`). Keep any future shared-source kernel's
math core free of in-tree-only graph types for exactly this reason — see the
`workspace-estimation-shared-header` skill for the related header-boundary pitfall this pilot hit.

---

#### Phase A: Workspace Pre-declaration (`DeclareWorkspaceRequirements`)

The core missing piece. Today no kernel declares its temp buffer needs before `Compute()`. To close this gap, introduce a method on `OpKernel` that returns workspace descriptors — each with a size and a key for later retrieval.

**Analogy to PrePack:** This mechanism is similar to the existing `PrePack()` pattern — both are called once during session state finalization (not during `Run()`), both store results that are reused across all subsequent runs. `PrePack()` pre-processes weight data; `DeclareWorkspaceRequirements()` pre-computes workspace layout.

**Interface:**

```cpp
struct WorkspaceRequirement {
  size_t size_bytes;        // Size of this workspace buffer
  int slot_id;             // Kernel-defined slot identifier (0, 1, 2, ...)
                           // Unique within a single kernel instance
  size_t alignment_bytes;   // 0 = allocator default is sufficient. See prose below.
};

// Optional override on OpKernel (called during FinalizeSessionState):
virtual Status DeclareWorkspaceRequirements(
    gsl::span<const WorkspaceInputShape> input_shapes,
    InlinedVector<WorkspaceRequirement>& requirements) const {
  requirements.clear();
  return Status::OK();  // Default: no declaration (fall back to arena)
}
```

`WorkspaceInputShape` is positional and presence-aware:

- `Missing` means the optional input was omitted.
- `PresentWithShape` includes scalars, zero extents, and partial shapes. Every unknown dimension is
  represented by `-1`. Negative proto dimensions also normalize to `-1`; multiple `-1` dimensions
  are independent unknowns and do not imply symbolic equality.
- `PresentWithoutShape` means the input exists but rank/dimension metadata is unavailable.

The span aligns with `Node::InputDefs()` and `OpKernelContext::Input(i)`; implicit inputs are excluded.
Callers preserve internal and trailing optional-input holes, and kernels may ignore missing/shapeless
inputs that are unrelated to their workspace formula.

The former `gsl::span<const TensorShape>` pilot signature was removed rather than retained as an
overload. This direct C++ signature replacement is intentional and requires a lockstep rebuild of the
core and in-tree providers. The virtual is unreleased; stable binary compatibility is provided by the
C API, which was not changed for this pilot.

`WorkspaceInputShape` deliberately carries no shape-provenance field in Phase A. Consumers cannot
distinguish executable-graph shapes from values propagated from `session.max_shape_override`.
Before Phase B uses declarations for allocation, the planner design must decide whether provenance
is required and define the corresponding runtime bound check and allocation fallback. This PR does
not add speculative provenance state.

An empty Level-2 requirements list currently means that no requirement can be declared and the
kernel retains its dynamic allocation fallback. Zero-sized requirements are also omitted. A future
planner must define the semantics of an explicit, proven-zero slot before the interface gains a zero
marker; this PR intentionally adds neither a marker nor new provenance state.

**`alignment_bytes` (added in PR #29811):** `0` requests the allocator's default alignment. Any
nonzero value is a binding alignment requirement on the returned slot buffer (or operator-owned root)
that a planner must honor. The stacked PA/PMHA declaration uses an explicit 256-byte root alignment;
this ensures that its relatively aligned internal subregions are also absolutely aligned. A plain
`size_t` with a `0` sentinel is used rather than `std::optional<size_t>` because this POD is meant to
be usable across a plugin-DLL boundary eventually, and `std::optional`'s layout is not guaranteed
stable across compilers/STL versions the way a scalar with a sentinel value is.

**Shape source wiring:** framework callers resolve presence-aware entries from two shape sources:

- Rank/dimension metadata already present on the executable graph is retained, including partial
  shapes with `-1` unknown dimensions.
- For dynamic models, `session.max_shape_override` is applied to a disposable shadow graph. Normal ORT
  shape inference propagates those input hints to intermediate values without changing executable-graph
  metadata or runtime input validation. These propagated values are estimates, not proven upper
  bounds, because operator shape transformations are not necessarily monotonic.

Level 1 consumes the current shadow result during `GetCapability()`. It reuses that result for the second
capability pass when layout transformation reports no modification and rebuilds it after known graph
mutations. Level 2 creates a final result after partitioning because EP fusion and transformation can
change NodeArgs, generated schemas, and subgraph identities. Reusing a pre-mutation result across those
stages would be incorrect; deeper caching requires a reliable graph-mutation generation rather than
assuming equal topology. The resolver still calls kernels with `Missing` or `PresentWithoutShape`
entries; each kernel decides whether the inputs needed by its formula are estimable. Returning no
requirements preserves the existing dynamic allocation fallback.

This Phase-A pilot only gathers declarations. Plugin C-ABI forwarding, workspace offset
planning/allocation, and integration with resource accounting remain deferred; no declaration changes
partition budgets or replaces the runtime scratch-allocation path yet. Phase-B planner/accounting
integration remains in the scope of #32071.

A kernel can declare multiple workspace slots. The `slot_id` is defined by the kernel author and is
stable across calls; `(NodeIndex, slot_id)` identifies one internal buffer for one kernel instance.
The interface capability and the current planner capability are intentionally distinct.

##### Current Phase-B strategy in #32071

#32071 integrates opted-in workspace with ORT's existing activation `MemoryPattern`; it does not
introduce a separate `WorkspacePattern` or allocate a persistent execution buffer during
`FinalizeSessionState()`:

1. Level 1 contributes an operator workspace reservation to partition resource accounting.
2. After kernel creation and prepacking, Level 2 is compared with that reservation and registers
   usable requirements in the sequential execution plan.
3. Each registered `(NodeIndex, slot_id)` receives a negative synthetic pattern ID, disjoint from
   graph `OrtValue` indices.
4. On the first run for a feed-shape memory-pattern key, the execution frame traces the synthetic
   workspace allocation/free lifetime. The kernel still receives no planned pointer and falls back to
   `GetScratchBuffer()`.
5. ORT caches one pattern containing both activation and workspace blocks. On later runs with a
   compatible feed-shape key, the execution frame allocates the normal per-run pattern backing buffer
   and returns `backing_buffer + workspace_offset`. The planned offset and returned pointer must
   satisfy the requirement's binding `alignment_bytes` value, or the allocator's default alignment
   when that value is zero.
6. If the pattern is unavailable or the runtime request exceeds the declared capacity,
   `GetPreallocatedWorkspace(slot_id, requested_bytes)` returns no pointer and the kernel retains its
   dynamic `GetScratchBuffer()` fallback.

This reduces allocator traffic and lets workspace reuse memory occupied by non-overlapping
activations after the warm-up run. It does not protect the first run from OOM, guarantee that the
per-run backing-buffer allocation succeeds, or provide initialize-time persistent allocation.

**Current one-slot limitation:** `WorkspaceRequirement` and
`DeclareWorkspaceRequirements()` retain generic multi-slot support, but #32071 opts in only kernels
that both explicitly override `SupportsPreallocatedWorkspace()` and declare exactly one slot. A
declaration alone is not planner opt-in. This is sufficient for the current MatMulNBits pilot and both
packed-attention operators. PackedAttention can expose one 256-byte-aligned, operator-owned root with
two internal regions:

```text
[0, aligned projection end): projection region and alignment padding
[aligned projection end, root end): selected Attention backend workspace
```

PMHA, whose Q/K/V inputs are already projected, naturally exposes one attention root. PA/PMHA root
retrieval and slicing and the `SupportsPreallocatedWorkspace()` opt-in must land atomically. Until
then, PA retains its two dynamic projection and Attention allocations, PMHA retains its one dynamic
Attention allocation, and Level 1 and Level 2 agree on each operator's single-root declaration. This
does not require multi-slot planner support.

**Future stronger mode:** allocating and retaining the complete execution buffer during session
initialization would be a separate feature. Such a mode would need trustworthy static or bounded
shapes, runtime bound enforcement, a concurrency policy, and no dynamic-allocation fallback before it
could claim that a successful `Initialize()` prevents allocation-time OOM during `Run()`. #32071 does
not provide that guarantee.

##### EP Plugin C ABI Surface for Workspace Pre-declaration

The issue #29775 Phase-A pilot does **not** add this surface. Plugin/C-ABI forwarding remains deferred:

- The adapter `OpKernel` declaration mirrors the in-tree C++ signature and default no-op so
  plugin-compiled shared source builds, but the host does not bridge or invoke that virtual.
- `OrtKernelImpl` and `OrtEpApi` have no workspace-declaration/retrieval additions from this pilot.
- Plugin kernels therefore continue to use the existing dynamic scratch-allocation path.

A future C-ABI design must preserve the Phase-A positional input-shape contract rather than collapsing
omitted inputs, rank-0 tensors, and absent shape metadata into ambiguous shape/rank arrays. It must
also define ownership/lifetime for returned slot descriptors, retrieval fallback behavior, and
versioning before any host bridge is implemented.

That future work also requires DLL-boundary validation of shared pure-math helpers, a separate plugin
build/test configuration, and explicit host-side forwarding. None of those pieces is implemented by
the pilot, and the presence of the adapter default must not be interpreted as host support.

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
| Tokenizer | Reuse from ORT Extensions | See tokenizer strategy below |
| KV cache rotation | Straightforward | Pre-allocate `(batch, heads, max_seq, head_dim)`, feed `past_key_values` outputs back as inputs each step |
| Decode loop | Trivial | Run session → extract logits → sample token → repeat |
| Model format | Constraint | Requires decoder-style ONNX export with explicit KV cache I/O (HuggingFace optimum exports provide this) |
| Partitioning | This design | Direction 1 + `IResourceAccountant` |
| Static allocation | Phase A+B | Fixed `max_seq_len` makes all decode-phase shapes static |

**Tokenizer strategy — borrowing from ORT Extensions:**

[ORT Extensions](https://github.com/microsoft/onnxruntime-extensions) already implements production-quality tokenizers in C++:
- **BPE** (GPT-2, LLaMA-3, Phi, Mistral) — `onnxruntime_extensions/tokenizer/bpe_tokenizer.cc`
- **SentencePiece** (LLaMA-1/2, T5, mT5) — wraps the SentencePiece C++ library
- **WordPiece** (BERT, DistilBERT) — `onnxruntime_extensions/tokenizer/wordpiece_tokenizer.cc`

For the custom executable, we can **extract the tokenizer C++ code directly** from ORT Extensions rather than taking a full dependency on the extensions DLL. The tokenizer logic is self-contained: it reads a vocabulary/merge file, applies the algorithm (BPE merge loop, SentencePiece unigram, or WordPiece greedy match), and produces token IDs. No ONNX graph execution is involved.

**Practical approach:**
1. Copy the relevant tokenizer source files (BPE tokenizer is ~500 LOC + vocab loading) into the demo executable's source tree.
2. Strip the ORT Extensions custom-op registration wrapper — keep only the core `Encode(string) → vector<int>` and `Decode(vector<int>) → string` logic.
3. Load the tokenizer model file (e.g., `tokenizer.json` from HuggingFace, or `tokenizer.model` for SentencePiece) at startup alongside the ONNX model.

This gives us a battle-tested tokenizer with no additional runtime dependency — just a few source files compiled into the executable. The code is already Apache-2.0 licensed (same as ORT).

The executable would be ~500–1000 LOC (excluding tokenizer): configure session options, set up KV cache tensors, run the generate loop. With the borrowed tokenizer code, the total grows to ~1500–2000 LOC but remains self-contained with zero external dependencies beyond ORT itself.

---

## Recommended Roadmap

```
Near-term (low effort, high value):
├── 1. Name-based matching via session.name_based_layer_assignment  [DONE]
│     - Separate session option with substring matching against Node::Name()
│     - SubstringMatcher with longest-match-wins priority
│     - Mutually exclusive with annotation-based matching (setting both options returns INVALID_ARGUMENT)
│
├── 2. Precise per-node memory estimation
│     - Static workspace estimation functions registered per kernel type
│     - IResourceAccountant uses exact output sizes + workspace estimates
│     - Eliminates 1.5x multiplier for kernels with estimation functions
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
