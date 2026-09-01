# BNHS Value layout for GroupQueryAttention

Status: partially implemented (see [Sequencing](#8-sequencing))
Last updated: 2026-09-01

## Motivation

Some execution providers can execute `com.microsoft.GroupQueryAttention` (GQA) more efficiently when
the Value KV-cache is laid out as `BNHS` — `(batch, num_heads, head_size, seq)` — rather than the
`BNSH` layout the operator schema mandates. The second attention matmul (`attn_weights @ V`) becomes
an NT gemm, which maps better onto some hardware.

The GQA schema cannot simply change: it is a stable contrib op, and most EPs are BNSH-only. This
design lets an application discover an EP's preference, allocate its KV-cache accordingly, and tell
ORT — without changing the operator schema.

The approach is to keep the GQA node BNSH and move the layout conversion into the graph, where an
EP compiler can absorb it:

```
past_value (BNHS, graph input) -> Transpose[0,1,3,2] -> GQA -> Transpose[0,1,3,2] -> present_value (BNHS, graph output)
```

An EP that prefers BNHS fuses the whole `Transpose -> GQA -> Transpose` sequence into a single op
that reads V as BNHS and aliases `past_value`/`present_value` to one buffer. For that EP the
transposes are notation, not work. An EP that cannot fuse them executes them for real: still
correct, but slow (see [Fallback cost](#fallback-cost)).

## Design contract

| Item | Decision |
|---|---|
| GQA schema | **Unchanged.** The node is always BNSH. No new attribute, no `ContribOperators.md` regeneration. |
| Scope | **Value only.** `past_key`, `present_key` and `k_scale` are untouched. |
| Meaning of the session key | Layout of the KV-cache buffers **at the main-graph boundary** — what the application binds to `past_value` and reads from `present_value`. |
| Mechanism | `Transpose(perm=[0,1,3,2])` between graph input `past_value` and GQA input 4; and between GQA output 2 and graph output `present_value`. |
| Consumer | The EP compiler fuses the sequence into one op that reads BNHS V and aliases past/present to one buffer. |
| Fallback | A non-fusing EP executes the transposes: correct, slow. Diagnosed by a warning, not an error. |
| Scope of application | **Main graph only** (`graph_level == 0`). Subgraphs (BeamSearch decoder body, Loop) are out of scope — the boundary there is not the application's. |
| Precondition | The two Value operands are judged independently. An operand that is not application visible (`past_value` not a non-initializer graph input, or `present_value` not a graph output) is skipped with a warning; the other operand of the same node is still converted. An operand that *is* application visible but cannot be converted fails session initialization — see 3.5. |

GQA operand indices, from [`docs/ContribOperators.md`](../ContribOperators.md#commicrosoftgroupqueryattention):
`past_value` = input **4**, `present_value` = output **2**.

## 1. EP advertises its preference

No new C API is required. `OrtApi::EpDevice_EpMetadata`
(`include/onnxruntime/core/session/onnxruntime_c_api.h`, C++ `ConstEpDevice::EpMetadata()`)
already returns an `OrtKeyValuePairs`. This is a well-known-key contract only.

**1.1** Add to `include/onnxruntime/core/session/onnxruntime_ep_device_ep_metadata_keys.h`:

```cpp
// Preferred layout for the GroupQueryAttention Value KV-cache at the graph boundary.
// Values: "BNSH" (batch, num_heads, seq, head_size) or "BNHS" (batch, num_heads, head_size, seq).
// If absent, "BNSH" is assumed. The application passes the chosen layout to the session via
// kOrtSessionOptionsGqaValueLayout.
static const char* const kOrtEpDevice_EpMetadataKey_GqaPreferredValueLayout =
    "gqa_preferred_value_layout";
```

A single value, not a list. If "supports both, prefers X" is needed later, the value can become a
comma-separated preference list with the first entry preferred — backward compatible with a
single-value reader.

**1.2** The compiling EP's factory populates the key in `GetSupportedDevices` before calling
`CreateEpDevice`. EPs without GQA support omit it.

**1.3** Add the key to `onnxruntime/test/autoep/library/example_plugin_ep/ep_factory.cc`, next to
the existing `"supported_devices"` entry, so there is a test fixture.

**1.4** Language bindings need no change. Python `get_ep_devices()` and C# `OrtEpDevice.EpMetadata`
already surface arbitrary metadata.

## 2. Application communicates the choice

**2.1** Add to `include/onnxruntime/core/session/onnxruntime_session_options_config_keys.h`:

```cpp
// Layout of the GroupQueryAttention Value KV-cache tensors (past_value input / present_value
// output) as bound by the application. "BNSH" (default) or "BNHS".
// When "BNHS", ORT inserts Transpose nodes so the GQA node still sees BNSH; an EP that prefers
// BNHS is expected to fuse Transpose->GQA->Transpose. Query the EP's preference via the
// "gqa_preferred_value_layout" OrtEpDevice metadata key.
// Applies to every GQA node in the model.
static const char* const kOrtSessionOptionsGqaValueLayout = "session.gqa_value_layout";
```

**2.2** Validate at session initialization. Anything other than `"BNSH"` or `"BNHS"` returns
`ORT_INVALID_ARGUMENT` with the offending value in the message. No silent fallback. Note
`ORT_RETURN_IF_NOT` produces `ORT_FAIL`, so this needs an explicit `ORT_MAKE_STATUS(..., INVALID_ARGUMENT, ...)`.

**Status codes.** The two situations are deliberately distinguishable, because they call for
different responses from an application:

| Situation | Code |
|---|---|
| Unrecognized option value; option set on an ORT format model | `ORT_INVALID_ARGUMENT` — the caller passed something wrong |
| Recognized value, but this model's topology or cache format cannot satisfy it (section 3.5) | `ORT_FAIL` — the option is fine, the model is not, so falling back to BNSH may work |

## 3. The graph transform

New files `onnxruntime/core/optimizer/gqa_value_layout_transformer.{h,cc}`. Both
`onnxruntime/core/optimizer/*.cc` and `*.h` are globbed by `cmake/onnxruntime_optimizer.cmake`, so
**no CMake change is needed**.

### 3.1 Header

```cpp
class GqaValueLayoutTransformer : public GraphTransformer {
 public:
  // converted_boundaries collects the graph inputs / outputs this run converted, for the
  // post-partition diagnostic in 4.2.
  explicit GqaValueLayoutTransformer(GqaValueLayoutBoundaries* converted_boundaries = nullptr) noexcept
      : GraphTransformer("GqaValueLayoutTransformer"), converted_boundaries_(converted_boundaries) {}

 private:
  Status ApplyImpl(Graph&, bool& modified, int graph_level, const logging::Logger&) const override;

  GqaValueLayoutBoundaries* const converted_boundaries_;
};
```

It is constructed only when the layout is BNHS, so it needs no configuration beyond the boundary
collector.

`ShouldOnlyApplyOnce()` is deliberately **not** overridden. Re-running has to be safe regardless,
because a model saved with `session.optimized_model_filepath` already carries the transform and may
be reloaded into a new session with the option still set — a fresh `Apply` that the override would not
guard. Section 3.4 is what provides that guarantee, and leaving the default keeps it under test.

### 3.2 Algorithm

```
ApplyImpl(graph, modified, graph_level, logger):
  if graph_level != 0: return OK          // main graph only; do NOT Recurse

  // pass 1: classify every node, mutating nothing (3.6)
  nodes_to_transform = []                 // (node index, plan) pairs
  for node in graph.Nodes():
    if node.OpType() != "GroupQueryAttention" or node.Domain() != kMSDomain: continue
    ORT_RETURN_IF_ERROR(ClassifyNode(graph, node, logger, out plan))   // 3.4, 3.5
    if plan.AnythingToDo(): nodes_to_transform.append((node.Index(), plan))

  // pass 2: rewire. TransformNode() has no failure modes.
  for (node_index, plan) in nodes_to_transform:
    TransformNode(graph, *graph.GetNode(node_index), plan, converted_boundaries_)
    modified = true


// plan.convert_past_value / plan.convert_present_value are set per operand, so a node with only one
// application-visible Value operand converts just that side.
TransformNode(graph, node, plan):
  // ---- input side ----
  if plan.convert_past_value:
    NodeArg* boundary = node.MutableInputDefs()[4]    // graph input, declared BNSH today
    NodeArg& bnsh = graph.GetOrCreateNodeArg(
        graph.GenerateNodeArgName(boundary->Name() + "_bnsh"), boundary->TypeAsProto());
    graph.AddNode(..., "Transpose", ..., {boundary}, {&bnsh}, ..., kOnnxDomain)
        .AddAttribute("perm", {0, 1, 3, 2});
    graph_utils::ReplaceNodeInput(node, 4, bnsh);
    SwapLastTwoDims(*boundary);        // graph input now declares BNHS

  // ---- output side ----
  if plan.convert_present_value:
    NodeArg* boundary = node.MutableOutputDefs()[2]   // graph output, declared BNSH today
    NodeArg& bnsh = graph.GetOrCreateNodeArg(
        graph.GenerateNodeArgName(boundary->Name() + "_bnsh"), boundary->TypeAsProto());
    node.MutableOutputDefs()[2] = &bnsh;              // retarget GQA output 2 first, so the
                                                      // boundary never has two producers
    graph.AddNode(..., "Transpose", ..., {&bnsh}, {boundary}, ..., kOnnxDomain)
        .AddAttribute("perm", {0, 1, 3, 2});
    SwapLastTwoDims(*boundary);        // graph output now declares BNHS
```

On shapes: the **new** `_bnsh` NodeArgs inherit the original (BNSH) type and shape and need no
adjustment — `Graph::Resolve` confirms them. It is the **boundary** NodeArgs whose declared shapes
are swapped to BNHS. This is what keeps `InferenceSession::ValidateInputsOutputs` happy at `Run`
time: it hard-fails on any static dimension mismatch, and `head_size` is essentially always static
in exported models.

`SwapLastTwoDims(NodeArg&)`: if the arg has no declared shape, no-op (an unshaped input accepts any
shape). Otherwise require rank 4 and `SetShape` a copy with dims 2 and 3 exchanged. The type is
already set, so `SetType` is not needed — but note the ordering constraint documented in
`include/onnxruntime/core/graph/node_arg.h` if that ever changes.

Neither `Graph::SetInputs` nor `Graph::SetOutputs` is called. The set of graph inputs and outputs is
unchanged; only the NodeArgs' shapes and their producer/consumer wiring change.

### 3.3 `v_scale` requires no change

An earlier revision of this design called for transposing the `PER_CHANNEL` `v_scale` from
`[1, num_heads_k, 1, head_size]` to `[1, num_heads_k, head_size, 1]`. That is not needed.

`v_scale` is consumed by the GQA node, and the GQA node operates entirely in BNSH after the
transform: its `past_value` operand is the Transpose output and its `present_value` operand is the
Transpose input, both BNSH. The scale therefore still has to be broadcastable to a BNSH tensor,
exactly as it is today. The only BNHS tensors in the graph are the boundary NodeArgs, and `Transpose`
does not consume scales.

This does mean the application supplies `v_scale` in the model-declared
`[1, num_heads_k, 1, head_size]` shape regardless of the cache layout it chose, which is worth
stating in the user-facing documentation. `k_scale` is likewise unaffected.

### 3.4 Idempotency

Required, because a model saved via `session.optimized_model_filepath` already contains the
transposes and the BNHS boundary. Reloading it with the key still set would insert a second pair and
swap the boundary back to a BNSH declaration while the application still feeds BNHS — broken, and
broken quietly.

The two Value operands are classified **independently**, from the graph structure — not from a
metadata marker, which does not survive the ORT-format round trip reliably. `ClassifyPastValue` and
`ClassifyPresentValue` each return one `OperandStatus`:

| Status | Meaning | Effect |
|---|---|---|
| `kAbsent` | the node does not have this operand | nothing to do |
| `kConverted` | already routed through a `Transpose(perm=[0,1,3,2])` to or from an application boundary | nothing to do |
| `kConvertible` | sits at an application boundary and is not converted yet | convert this operand |
| `kOutOfScope` | present, but not a boundary the application binds | skip with a warning (3.5) |

Per-operand rather than per-node, because the two sides are genuinely independent: the GQA node stays
BNSH on both sides whatever happens, so converting only the operand that is application visible leaves
a coherent graph. A node with an internal `past_value` and an exported `present_value` gets the
`present_value` side converted; skipping the whole node would leave an application-visible output in
BNSH after the session accepted BNHS.

There is exactly one inconsistent combination: one operand `kConverted` while the other is
`kConvertible`. Both were equally convertible, so a half-converted node means the graph was edited by
hand or produced by a build that failed part way; the boundaries no longer agree with each other and
converting the remainder cannot repair that, so it is an error (3.5). Any other pairing is legitimate —
`kConverted` next to `kAbsent` or `kOutOfScope` is a fully converted node. Note that requiring *both*
operands to be converted before treating a node as done would be wrong: a prefill-only model has no
`past_value`, and a model can omit the `present_value` output, so for those the one operand present is
the whole conversion.

### 3.5 Scope of the option, and why the rest is an error

The option describes the layout of the buffers the **application binds**. That gives one legitimate
skip and one class of hard failure, and the distinction matters because it is the option's external
contract: if the application is told BNHS, every boundary it can see must actually be BNHS.

**Skip (warning), by design.** An operand classified `kOutOfScope` is a Value cache the application
never touches: a `past_value` that is not in `Graph::GetInputs()`, or a `present_value` that is not a
graph output. It keeps BNSH, and ORT logs a warning naming the node and the operand. Nothing
observable to the application changes, so this is a documented scope limit rather than a failure. It
is recorded in the `kOrtSessionOptionsGqaValueLayout` comment. The skip is **per operand**: the other
operand of the same node is still converted if it is application visible.

Note the visibility test is `Graph::GetInputs()`, which excludes initializers, not
`GetInputsIncludingInitializers()`. A `past_value` backed by an initializer that is *not* a graph
input is baked into the model and can never be bound, so it is `kOutOfScope`.

**Error at session initialization.** Anything else means an application-visible boundary would stay
BNSH while the application believes it is BNHS, and would bind buffers in the wrong layout. Silently
skipping would make the option self-inconsistent, so `ClassifyNode` returns an error for:

- **A shared boundary.** A `past_value` graph input read by more than one node, or a `present_value`
  graph output that is also consumed inside the graph. A boundary NodeArg is shared state: swapping
  its declared shape is visible to every node that reads or writes it, but only the node being
  processed gets rewired through a Transpose. For a shared `past_value`, converting the first node
  flips the graph input to BNHS while the second still reads it as BNSH, and processing the second
  swaps the declared shape back, undoing the first. For a `present_value` with internal consumers,
  those consumers silently receive BNHS where they expect BNSH.
- **A partially converted node.** One operand `kConverted` while the other is `kConvertible` (see
  3.4). Both were equally convertible, so this means the graph was edited by hand or produced by a
  build that failed part way; the boundary layouts no longer agree with each other and converting the
  remainder cannot repair that.
- **An overridable-initializer `past_value`.** An initializer that is also declared a graph input can
  be overridden by a feed, so the application may bind it — but its baked-in data stays BNSH whatever
  happens to the declared shape. Swapping the shape alone would either fail `Graph::Resolve` on the
  initializer/NodeArg mismatch or, when the feed is omitted, hand the default BNSH buffer to a
  Transpose that reads it as BNHS. The message points at the two fixes: drop the initializer, or
  transpose it when producing the model.
- **4-bit KV cache.** When `v_quant_type != "NONE" && kv_cache_bit_width == 4`, V is `uint8` with two
  4-bit values packed along `head_size`. A byte-wise `Transpose` cannot transpose sub-byte-packed
  data, and the declared-shape swap would be wrong as well. A fusing EP never executes the Transpose
  so it may be fine there, but the CPU fallback would be silently incorrect. Rejected until the
  packing semantics under BNHS are defined — see [Open items](#open-items).
- **A Value cache tensor that is not rank 4** (a declared shape of any other rank; an undeclared shape
  imposes no constraint and is fine).

**Check order matters.** `ValidateCacheFormat` (the 4-bit check) runs *before* the layout-state switch
in `ClassifyNode`, not after it. A 4-bit cache is unsupported whether this run would insert the
Transposes or a previous one already did, so recognizing an already-converted node and returning early
would let such a model initialize and then execute the invalid byte-wise transpose on a non-fusing EP. The rank check stays after the switch, because it only constrains a conversion this
run is about to perform.

Supporting shared boundaries would mean converting each boundary once and rewiring every BNSH user of
it, which is more than this design needs for the single-cache-per-layer models it targets.

### 3.6 Validate the whole graph, then convert

`ApplyImpl` runs two passes:

1. `ClassifyNode` over every GQA node, mutating nothing, collecting the indices to convert.
2. `TransformNode` over the collected indices.

The split matters because the errors in 3.5 are fatal to session initialization. Converting as the
walk proceeds would leave earlier nodes rewired and the graph unresolved when a later node fails —
`GraphTransformer::Apply` skips `Resolve()` when `ApplyImpl` returns an error. Validating first means
the graph is either fully converted or byte-for-byte as it was loaded.

It also removes an ordering dependency: every node is judged against the original graph, so a verdict
does not depend on the topological order or on producer/consumer bookkeeping staying accurate
mid-rewrite. `TransformNode` has no failure modes at all — `SwapLastTwoDims` is infallible because
`ValidateSwappableShape` already established the rank in pass 1.

## 4. Wiring into the session

**4.1** In `InferenceSession::TransformGraph` (`onnxruntime/core/session/inference_session.cc`),
immediately after the Level1 `ApplyTransformers` call and before `partitioner.Partition`:

```cpp
ORT_RETURN_IF_ERROR_SESSIONID_(
    graph_transformer_mgr_.ApplyTransformers(graph, TransformerLevel::Level1, *session_logger_));

#if !defined(ORT_MINIMAL_BUILD)
if (session_options_.config_options.GetConfigOrDefault(kOrtSessionOptionsGqaValueLayout, "BNSH") == "BNHS") {
  GqaValueLayoutTransformer gqa_value_layout{};
  ORT_RETURN_IF_ERROR_SESSIONID_(apply_transformer_once(gqa_value_layout, *session_logger_, graph));
}
#endif
```

`apply_transformer_once` is the existing lambda in `TransformGraph`; this mirrors how
`EnsureUniqueDQForNodeUnit` is invoked just above the Level1 call.

This placement is deliberate and buys two properties:

- **Runs at optimization level 0.** `InferenceSession::AddPredefinedTransformers` gates registration
  on `graph_optimization_level >= level`, so a transformer registered through
  `optimizer_utils::GenerateTransformers` is silently absent at `ORT_DISABLE_ALL`. A direct call
  bypasses that gate. It also bypasses `optimizers_to_disable_`, which is correct: this is a
  correctness-affecting transform, not an optimization.
- **The pattern reaches the EP intact.** `TransposeOptimizer` is the *last* Level1 transformer
  (`onnxruntime/core/optimizer/graph_transformer_utils.cc`) and its job is moving, merging and
  cancelling Transpose nodes. Running after it means nothing perturbs `Transpose -> GQA -> Transpose`
  before `GetCapability`. The Level2 `TransposeOptimizer` is CPU-EP-filtered and runs
  post-partitioning, so it only touches transposes that fell back to CPU — harmless, possibly
  helpful.

Nothing is registered in `graph_transformer_utils.cc`, and no `GenerateTransformersForMinimalBuild`
counterpart is needed.

**4.2 Fusion diagnostic.** After `partitioner.Partition`,
`ReportUnfusedGqaValueLayoutTransposes(graph, boundaries, logger)` logs a WARNING for each converted
boundary whose Transpose survived, naming the boundary tensor and the EP that left it behind. This is
the difference between a diagnosable perf cliff and an invisible one.

The check is anchored on the **boundaries**, not on the GQA nodes. `GqaValueLayoutTransformer` records
the graph input and output names it converted into a `GqaValueLayoutBoundaries`, which the caller then
passes here; the check asks whether the graph input's consumer, or the graph output's producer, is
still a value-layout Transpose. Graph input and output names are stable across partitioning, which is
what makes them a usable anchor.

Searching from the GQA node instead would miss the case that matters most. A compiling EP may claim
only the GQA node, so `GraphPartitioner` replaces it with a fused node while leaving both Transposes
in place — both full-cache copies still execute, but there is no GQA node left to search from and the
old implementation reported nothing. Conversely, when the EP fuses the whole sequence, the boundary
connects straight to the fused node and nothing is reported, which is correct.

**Skipped when saving an ORT format model.** That path runs the partitioner in
`GraphPartitioner::Mode::kAssignOnly`, which deliberately leaves the original nodes in place rather
than compiling or fusing them, so every boundary would be reported as unfused even though the EP will
fuse the pattern when the saved model is loaded.

### 4.3 Minimal builds

`gqa_value_layout_transformer.cc` is not in the minimal or extended-minimal source lists in
`cmake/onnxruntime_optimizer.cmake`, so it is not compiled there. That is safe because every
reference to `GqaValueLayoutTransformer` and `ReportUnfusedGqaValueLayoutTransposes` sits inside
`TransformGraph`, which is itself inside a `#if !defined(ORT_MINIMAL_BUILD)` block, and
`adjust_global_compile_flags.cmake` defines `ORT_MINIMAL_BUILD` for extended minimal builds as well.
That guard opens well above `TransformGraph` and closes well below it, with no intervening `#else`, so
the whole function -- including both GQA blocks -- is excluded. The declaration in
`inference_session.h` is inside the same guard, which it has to be for the two to agree at all.
A minimal build reaches the ORT format path instead, which is handled in section 5.

The value constants `kGqaValueLayoutBNSH` / `kGqaValueLayoutBNHS` live in the transformer header and
are header-only `constexpr`, so `PartitionOrtFormatModel` can use them in a minimal build without
pulling in the translation unit. The header is therefore included unconditionally: guarding the
include on the build flavour would leave those constants undefined in a pure minimal build, where
`PartitionOrtFormatModel` is compiled and needs them.

### Fallback cost

When the transposes are not fused, each generated token costs two full transposing copies of the
Value cache per layer, and past/present buffer sharing is lost (the GQA kernel can no longer append
in place into the application's `max_sequence_length` buffer), roughly doubling KV-cache memory. For
a 32-layer model at 4k context this dwarfs the attention math itself. It is correct, but it is not a
configuration anyone should ship; hence the warning in 4.2.

## 5. ORT-format path

`PartitionOrtFormatModel` does not go through `TransformGraph`, so `.ort` models receive no
insertion. Silently ignoring the option there is not safe: with dynamic or coincidentally square
cache dimensions the application's BNHS buffers pass input validation and the model computes on
transposed data, producing wrong results with no error.

`PartitionOrtFormatModel` therefore **rejects** the option outright, with `ORT_INVALID_ARGUMENT`. An
ORT format model that had the transform applied at conversion time already carries the BNHS boundary
shapes in the model itself and must be loaded without setting the option.

That check is deliberately **not** guarded on the build flavour. A minimal build serves ORT format
models only — `InferenceSession::Initialize` refuses anything else — so this is the only place the
option can be caught there, and guarding it would let a minimal build silently ignore the option.

Adding real support — running the transformer on the ORT format path — would require the transformer
in minimal builds (see 4.3) and is deferred until a consumer needs it.

## 6. Tests

**6.1** `onnxruntime/test/optimizer/gqa_value_layout_transformer_test.cc` (globbed by
`cmake/onnxruntime_unittests.cmake`, no CMake change):

- No-op when the key is absent, and when it is `"BNSH"`.
- Transposes inserted on both sides with `perm == [0,1,3,2]`; GQA input 4 and output 2 rewired.
- Graph input `past_value` and graph output `present_value` declared shapes have dims 2 and 3
  swapped; the intermediate `_bnsh` args retain BNSH.
- Idempotency: running the transformer twice produces the same graph as running it once.
- `past_value` absent (prefill-only model) yields only the output-side transpose, and
  `present_value` absent yields only the input-side transpose.
- Mixed visibility, both directions: `past_value` behind an Identity converts only the `present_value`
  side, and `present_value` behind an Identity converts only the `past_value` side. Each runs two
  passes so the mixed result is also shown to be idempotent. When neither operand is visible the node
  is left alone.
- An overridable-initializer `past_value` is rejected (section 3.5).
- Errors, per section 3.5: a `past_value` graph input shared by two GQA nodes; a `present_value` graph
  output also consumed inside the graph; a node with the layout applied to only one operand; a 4-bit
  quantized Value cache.
- An already-converted model is left alone, and an already-converted **4-bit** model is
  still rejected. The second case is what pins down the check order in 3.5; verified to fail when
  `ValidateCacheFormat` runs after the layout-state switch.
- The post-partition diagnostic (4.2) reports both boundaries when the Transposes survive with no GQA
  node present -- the compiling-EP case -- and reports nothing when they were fused away. The fixture
  asserts it contains no GQA node, so it cannot silently stop covering the regression.
- The graph is left untouched when validation fails (section 3.6). Two independent GQA nodes, one
  convertible and one not, asserted for both build orders — `GetNodesInTopologicalOrder()` does not
  follow insertion order for independent nodes, and only the order that presents the convertible node
  first catches a transformer that mutates while it validates. Verified to fail against a single-pass
  implementation.

Session-level tests cover the plumbing that a graph-level test cannot reach, by loading a serialized
model into an `InferenceSessionWrapper`:

- The transform is applied at `ORT_DISABLE_ALL`. This is the test that pins down the placement
  decision in 4.1; a registered level 1 optimizer would be skipped entirely at that level.
- No transposes are inserted for the default `"BNSH"` value.
- An invalid value fails session initialization with `ORT_INVALID_ARGUMENT` (code asserted, not just
  the message).
- An ORT format model fails session initialization with `ORT_INVALID_ARGUMENT` when the option is set, and loads normally when it
  is not.

Subgraphs are not covered by a test. `ApplyImpl` returns immediately for `graph_level != 0` and never
calls `Recurse`, so a subgraph node is unreachable by construction; a test would exercise the
transformer base class rather than this transformer.

**6.2** End-to-end numerical parity on the CPU EP, in the same test file:

- `BnhsMatchesBnshOnCpu` — the same model run with a BNSH boundary and with a BNHS boundary fed a
  pre-transposed cache produces bit-identical `output`, and identical `present_value` after
  transposing back. The past caches carry a pattern that varies along both swapped dimensions and the
  sequence lengths are set so the kernel reads them, otherwise the comparison would pass with a
  broken transpose. Explicit guards assert the compared tensors are neither constant nor
  transpose-invariant.
- `BnhsWithAliasedCacheBufferMatchesSeparateBuffersOnCpu` — one buffer bound to both `past_value` and
  `present_value` via `IOBinding`, as a decode loop would. The reference is the same BNHS model with
  separate buffers, not the BNSH session: aliasing under BNSH hands the CPU kernel an aliased past and
  present so it takes its shared-buffer path, while under BNHS the operands are the transpose
  intermediates, so comparing the two would compare two different kernel implementations. Chained
  with `BnhsMatchesBnshOnCpu`, this still covers the full claim.

Note the CPU kernel's shared-buffer path was observed to produce different `present_value` contents
than its non-shared path for the same inputs (a zero where the caller's buffer held data). That is
pre-existing behavior on the BNSH path, unrelated to this change, and was not investigated here.

**6.3** `onnxruntime/test/autoep/` — the new metadata key round-trips from the example plugin EP
through `EpDevice_EpMetadata`.

**6.4** On the compiling EP — fusion actually fires (assert that the GQA node and both Transposes
land on that EP, i.e. the 4.2 warning does *not* trigger), and multi-token decode with a single
aliased buffer bound to both `past_value` and `present_value` matches the CPU BNSH reference.

**6.5** Negative — an invalid session key value is rejected at session initialization.

## 7. Documentation

- The two header comments are the primary reference.
- The plugin-EP author guide gains the metadata-key contract and a description of what an EP must
  fuse in order to benefit.
- A short note wherever KV-cache binding is documented for genai-style consumers: query the EP, set
  the session key, allocate the cache BNHS, bind one buffer to both `past_value` and
  `present_value`.
- `docs/ContribOperators.md`: **no change** — the operator schema is untouched.

## 8. Sequencing

| PR | Contents | Status |
|---|---|---|
| 1-3 | Both public keys, validation, example-EP metadata, autoep test, transformer, `TransformGraph` wiring, fusion diagnostic, 6.1 unit tests, docs | Implemented |
| 3b | CPU-fallback numerical parity tests (6.2), sole-ownership guards (3.6), ORT format rejection (5) | Implemented |
| 4 | Compiling EP advertises the key, implements the fusion, 6.4 tests | Not started (EP-side) |
| 5 | Real ORT format support (running the transformer on that path) | Deferred |

## Open items

1. **4-bit packed V cache under BNHS.** Currently planned as a hard error (3.5). To support it we
   must define whether the packing axis follows `head_size` or becomes the (now-minor) `seq` axis,
   and the declared shape has to encode that choice. Worth deciding before PR 2 lands, since it
   turns a validation rule into a code path.
2. **Heterogeneous sessions.** The session key is session-wide. If one EP fuses and another does
   not, the non-fusing EP's layers hit the 4.2 warning path with no per-node escape. Acceptable
   initially; a per-EP override key would be the escape hatch if this becomes real.
3. **Shared boundaries.** The guards in 3.6 decline to transform a boundary with more than one user.
   Supporting them means transforming each boundary once and rewiring every BNSH user, which matters
   only for models that share one Value cache across GQA nodes.
4. **Fusion pattern contract.** The EP compiler's match criteria should be written down explicitly —
   in particular whether it tolerates non-adjacent Transposes, and whether it requires `perm` to be
   literally `[0,1,3,2]` versus any last-two-dimension swap. The 4.1 placement guarantees adjacency
   today, but pinning the contract protects against future transformer churn.
