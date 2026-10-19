# Post-partition runtime fusion for canonical ONNX

Status: **engineering design for review, not approved for implementation** (October 15, 2026).
Tracks [#33403](https://github.com/microsoft/onnxruntime/issues/33403).
No native build, dependency installation, GPU allocation, model execution, or runtime probe was
performed for this design. Test names, commands, numerical gates, and performance budgets below are
proposed acceptance criteria, not results. Maintainer review and acceptance are still required.

## 1. Relationship to the original proposal

This document refines the
[pinned Extensible Runtime Fusion proposal](https://github.com/microsoft/onnxruntime/blob/587ef1bf3467935b083740a52ac8fcba3785b2a6/docs/design/Extensible_Runtime_Fusion.md).
That file is absent from the default-branch base used here; the proposal branch is a reference,
not an implementation dependency or a branch to merge.

The north star remains:

> Mobius exports canonical standard ONNX. Olive owns quantization policy. Compiler EPs retain
> visibility and compile their regions. Implementation packages own optional kernel-fusion patterns
> and adapters. ORT owns generic registration, validation, ownership enforcement, and rewriting.

The canonical artifact is authoritative; a saved optimized model is a target-specific derivative.
Existing contrib artifacts and private transformer registration remain supported. Kernel-internal
improvements behind an unchanged operator contract already need no re-export. The additional proof
required here is a **matcher/adapter package update with identical canonical artifact hashes**.

### Scope

Start with existing CUDA kernels: FP16 dense dynamic-cache GQA and stateful convolution, then a
separately gated whole-`Scan` recurrence milestone. No new EP, kernel, quantization format,
Flash-Next/MTP support, universal optimizer migration, automatic offload/defusion search, or
transparent retry after state mutation. CPU and native WebGPU are architectural extensibility
requirements, not first-release implementations. Compiler EPs need not adopt this mechanism.
Qwen3.8-27B is a pilot, not a model-specific runtime API.

## 2. Evidence ledger and baseline discipline

All runtime claims below refer to **R**, unless a difference is explicitly noted. Links are immutable
research references, not claims that proposed interfaces exist.

| Key | Exact source |
|---|---|
| R | [ORT main `1988019b6e23ac4def4628cddd603c1bfb504fe8`](https://github.com/microsoft/onnxruntime/tree/1988019b6e23ac4def4628cddd603c1bfb504fe8) |
| M | [Mobius main `b1e9c8e9dec1dfa206183f957e77d07cfb734df2`](https://github.com/onnxruntime/mobius/tree/b1e9c8e9dec1dfa206183f957e77d07cfb734df2) |
| G | [GenAI main `991e3560e63b8a26806d4ec7d702c04e75fb6e51`](https://github.com/microsoft/onnxruntime-genai/tree/991e3560e63b8a26806d4ec7d702c04e75fb6e51) |
| F | [Fusion proposal `587ef1bf3467935b083740a52ac8fcba3785b2a6`](https://github.com/microsoft/onnxruntime/blob/587ef1bf3467935b083740a52ac8fcba3785b2a6/docs/design/Extensible_Runtime_Fusion.md) |
| P | [Placement proposal `e7604f130fef274e68fb2e410936550b832f254e`](https://github.com/microsoft/onnxruntime/blob/e7604f130fef274e68fb2e410936550b832f254e/docs/annotated_partitioning/PartitioningWithAnnotationsAndMemoryConstraints.md) |
| V | [OpenVINO plugin `7463f66e48d09fe5150af8c76520fbf75e08c490`](https://github.com/onnxruntime/onnxruntime-ep-openvino/tree/7463f66e48d09fe5150af8c76520fbf75e08c490); compiler-EP comparison only |

The PR base is default-branch commit `327d11ab6c22e387acf793dd9de8caf406055ca9`.
Source comparison with R found no changes in the cited session hook, public API, registry selection,
Scan registration, or GQA/LinearAttention/GatedDeltaNet kernel files. Relevant differences are the
accountant's new `INT32_MAX` validation for the CUDA GQA total-sequence bound and updates in the GQA
workspace bounds/estimate helpers. These do not supply target re-estimation for arbitrary rewrites.
Do not combine P's proposed reconciliation with R and describe the result as implemented.
Any implementation PR must repeat the comparison against its actual base and publish its build pin.
V's `CreateEpFactories`/`ReleaseEpFactory` lifetime patterns are useful references. Its
GetCapability/Compile and AddNodesToFuse path owns compiled regions; it is not a public extension
that borrows existing CUDA kernels. Creating a competing EP solely to reach those kernels is rejected.

### Current behavior versus missing mechanism

| Evidence at R | Implication |
|---|---|
| [`inference_session.cc:1590-1602,1896-1952,2015-2087`](https://github.com/microsoft/onnxruntime/blob/1988019b6e23ac4def4628cddd603c1bfb504fe8/onnxruntime/core/session/inference_session.cc#L1896-L1952) | L1 precedes partition; mutation accounting is installed after partition; L2+ and casts repeat; copies follow. |
| [`graph_transformer_utils.cc:413-426,464-467`](https://github.com/microsoft/onnxruntime/blob/1988019b6e23ac4def4628cddd603c1bfb504fe8/onnxruntime/core/optimizer/graph_transformer_utils.cc#L413-L426), [`graph_transformer_mgr.cc:38-59`](https://github.com/microsoft/onnxruntime/blob/1988019b6e23ac4def4628cddd603c1bfb504fe8/onnxruntime/core/optimizer/graph_transformer_mgr.cc#L38-L59) | Ordinary transformer registration order is not a once-only extension stage. |
| [`group_query_attention_fusion.cc:413-440`](https://github.com/microsoft/onnxruntime/blob/1988019b6e23ac4def4628cddd603c1bfb504fe8/onnxruntime/core/optimizer/group_query_attention_fusion.cc#L413-L440) | Existing GQA fusion starts from contrib GQA, not canonical Attention. |
| [`attention_fusion.cc:1030-1046`](https://github.com/microsoft/onnxruntime/blob/1988019b6e23ac4def4628cddd603c1bfb504fe8/onnxruntime/core/optimizer/attention_fusion.cc#L1030-L1046), [`session_state.cc:242-310`](https://github.com/microsoft/onnxruntime/blob/1988019b6e23ac4def4628cddd603c1bfb504fe8/onnxruntime/core/framework/session_state.cc#L242-L310) | Rewrites can retain EP assignment; finalization later binds registered kernels. |
| [`inference_session.cc:1113-1129`](https://github.com/microsoft/onnxruntime/blob/1988019b6e23ac4def4628cddd603c1bfb504fe8/onnxruntime/core/session/inference_session.cc#L1113-L1129) | Private C++ registration is not an external binary contract. Preserve its existing enable behavior. |
| [`onnxruntime_c_api.h:6130-6653,8210-8244`](https://github.com/microsoft/onnxruntime/blob/1988019b6e23ac4def4628cddd603c1bfb504fe8/include/onnxruntime/core/session/onnxruntime_c_api.h#L6130-L6653) | Public inspection and pre-finalization Model Editor are not a post-partition rewrite transaction. |
| [`onnxruntime_experimental_c_api.inc:21-37`](https://github.com/microsoft/onnxruntime/blob/1988019b6e23ac4def4628cddd603c1bfb504fe8/include/onnxruntime/core/session/onnxruntime_experimental_c_api.inc#L21-L37) | Experimental discovery exists, but no fusion registration entry exists. |
| [`kernel_registry_manager.cc:60-104`](https://github.com/microsoft/onnxruntime/blob/1988019b6e23ac4def4628cddd603c1bfb504fe8/onnxruntime/core/framework/kernel_registry_manager.cc#L60-L104) | Custom registries have precedence; an operator name alone proves neither implementation nor eligibility. |

## 3. Integration and graph lifecycle (D1)

For the fixed-placement pilot, propose an explicit call in `TransformGraph`, after partition and
installation of mutation-accounting callbacks, immediately before the existing outer L2+ loop
(before R's line 2015):

```text
canonical load -> shared preprocessing / L0-L1 -> ordinary partition and EP assignment
 -> install mutation accounting -> extension snapshot / match / validate / commit ONCE
 -> existing repeated L2/L3 -> casts -> L4 [existing loop] -> final copies
 -> final graph/resource admission -> kernel creation / PrePack / declarations -> Run
```

Enable only with explicit session configuration and optimization level at least L2. Registration
alone is not enablement. Disabled, absent, or Basic-only extensions do not receive callbacks or
change normal optimization. Explicit required-pattern configuration at Basic fails configuration
validation rather than pretending to satisfy that requirement.

V1 supports ordinary ONNX inference initialization. ORT-format loading, compile-only sessions,
training, online re-partitioning, and graph replay from an optimized cache are initially unsupported
extension entry paths: optional extension use is diagnosed as not applicable; required use fails.
This avoids accidentally modifying assign-only serialization behavior.

Core freezes a session snapshot generation and graph IDs after partition. Traverse graphs in stable
parent-before-child order; skip descendants of compiler-owned opaque regions. All callbacks see
this same generation before any proposal is committed. A parent whole-Scan proposal conflicts with
every descendant proposal. No callback runs again when L2 repeats or an emitted contrib node exposes
another optimization. New body graphs are not a new matching round. A later explicitly supported
reinitialization would require new identities and a new generation, not reuse of stale handles.

Match the graph actually present here, not the exporter input. Publish positive and near-match
fixtures at this boundary, including L0/L1 normalization and partition assignments. Later transforms
may change replacement nodes; trace them and revalidate kernel/resource provenance on the final
graph. Retain the existing private `RegisterGraphTransformer` path unchanged.

### Future fusion-aware placement integration

The fixed-placement path above is not the future planner's final-assignment contract. As
[chilo-ms notes](https://github.com/microsoft/onnxruntime/issues/33403#issuecomment-6226599402),
assignment-dependent matching may use tentative EP ownership. A planner must know eligible
implementation recipes, their logical graph effects and allocation requirements before accepting
a placement; source-only sizing could otherwise offload a region whose fused target would fit.
The separately proposed planning flow is:

```text
candidate EP assignment view
 -> read-only recipe / effect / resource queries
 -> planner chooses placement and compatible recipes
 -> rewrite only selected regions
 -> verify transformed graph and memory feasibility
 -> finalize ownership -> instantiate device resources
```

Keep recipe description separate from rewrite application. Queries must not mutate the live graph,
construct target kernels, upload weights or materialize trial device resources; evaluating another
placement must not require executing every candidate transformation. A future planning context must
identify its tentative ownership view, graph revision and workload/contract bounds, so recipes and
estimates from a discarded trial cannot be reused for another assignment.

V1's once-only callbacks still inspect a fixed assigned snapshot; the current C table does not
claim this planning-mode context. Any future adapter/context extensions require separate API,
lifetime and resource-owner approval with the
[region-aware placement proposal #33372](https://github.com/microsoft/onnxruntime/pull/33372).
Neither the planner nor repeated planning queries are added to this pilot. Required-pattern policy
and preference for eligible fusion remain in force; this is not per-session speed benchmarking.

## 4. Proposed experimental C boundary (D2)

**Every `OrtRF*` declaration below is proposed, not shipped.** These declarations specify the V1
wire contract; implementation headers must add normal ORT annotations and complete API documentation.
Use `ORT_API_CALL`, C-compatible records, UTF-8 strings, `OrtStatus*`, and existing public type
handles. No Graph, STL, protobuf object, provider pointer, CUDA stream, or kernel factory crosses it.

### Discovery and configuration

Approve an experimental factory registered through existing
[`GetExperimentalFunction`](https://github.com/microsoft/onnxruntime/blob/1988019b6e23ac4def4628cddd603c1bfb504fe8/include/onnxruntime/core/session/onnxruntime_c_api.h#L7580-L7600).
The proposed name is `OrtApi_GetRuntimeFusionApi_SinceV<N>`, where N is the actual introduction
API version, not an invented currently available symbol. The exact typed factory is:

```c
typedef struct OrtRFContext OrtRFContext;
typedef struct OrtRFCollector OrtRFCollector;
typedef struct OrtRFEpToken OrtRFEpToken;
typedef struct OrtRuntimeFusionApi OrtRuntimeFusionApi;

typedef OrtStatus* (ORT_API_CALL* OrtRFGetApiFn)(
    uint32_t abi_version, const OrtRuntimeFusionApi** out);
```

Retrieve `OrtApi` using the normal `OrtApiBase` version check, then resolve and cast the named
experimental function to exactly this typedef. Missing name means unavailable; unknown table
version returns `ORT_NOT_IMPLEMENTED` without changing `out`. Neither a new global exported symbol
nor a new EP entry point is assumed. Exact V1 table match is required. Any incompatible field or
semantic change gets a new table version; a changed factory signature gets a new experimental name.
No indefinite cross-release ABI promise. Publish a tested ORT/EP/build/device/package matrix.
Promotion, retirement, and a possible compatibility bridge require separate C API maintainer review.

Applications may call static registration directly. For a DSO, explicitly call
`RegisterCustomOpsLibrary_V2` with an application-approved absolute library path; its existing
`RegisterCustomOps(options, api_base)` entry point discovers this factory and registers fusions.
A fusion-only library need not register any model-visible op. Never scan directories, consult
model metadata for executable paths, or automatically load a package. Native code is trusted
application code, not sandboxed by graph validation.

### Records and read-only inspection

All extensible input/output records begin with `size_t struct_size`; omitted future fields are
not read, unknown flags and nonzero reserved fields are errors. V1 requires the documented V1
minimum size; field offsets and calling convention are platform ABI-specific. `size_t` counts
are checked for overflow before allocation. Empty arrays use count zero; optional value references
use kind 0, not a magic node index. IDs are session/snapshot-scoped, not persistent model names.

```c
typedef uint64_t OrtRFGraphId;
typedef uint64_t OrtRFNodeId;
typedef uint64_t OrtRFValueId;

typedef struct OrtRFBytes {
  const void* data;
  size_t size;
} OrtRFBytes;

typedef struct OrtRFValueRef {
  size_t struct_size;
  uint32_t kind;                 /* 0 absent, 1 snapshot value, 2 proposal-local value */
  OrtRFValueId id;
} OrtRFValueRef;

typedef struct OrtRFAttribute {
  size_t struct_size;
  const char* name;
  uint32_t kind;                 /* 0 unrepresented, 1 int64, 2 float, 3 bytes,
                                   4 ints, 5 floats, 6 tensor */
  int64_t integer;
  float real;
  OrtRFBytes bytes;
  const int64_t* integers;
  size_t integer_count;
  const float* reals;
  size_t real_count;
  const OrtValue* tensor;        /* CPU dense constant; ORT copies, never takes ownership */
} OrtRFAttribute;

typedef struct OrtRFNodeView {
  size_t struct_size;
  OrtRFNodeId id;
  const char* name;
  const char* domain;
  const char* op_type;
  int64_t since_version;
  const OrtRFEpToken* ep;
  uint32_t flags;                /* bit 0 executable, bit 1 compiler-owned opaque */
  const OrtRFValueRef* inputs;   /* kind 0 absent, kind 1 snapshot; never kind 2 */
  size_t input_count;
  const OrtRFValueRef* outputs;
  size_t output_count;
  const OrtRFAttribute* attributes;
  size_t attribute_count;
  const OrtRFGraphId* subgraphs; /* attribute order, with names available through GraphInfo */
  size_t subgraph_count;
} OrtRFNodeView;

typedef struct OrtRFGraphInfo {
  size_t struct_size;
  OrtRFGraphId id;
  OrtRFGraphId parent;           /* 0 for root */
  OrtRFNodeId parent_node;       /* 0 for root */
  const char* attribute_name;   /* owning node's graph attribute */
  size_t node_count;
  const OrtRFValueId* inputs;
  size_t input_count;
  const OrtRFValueId* outputs;
  size_t output_count;
  const OrtRFValueId* captures;  /* resolved outer-scope values */
  size_t capture_count;
} OrtRFGraphInfo;

typedef struct OrtRFValueView {
  size_t struct_size;
  OrtRFValueId id;
  OrtRFGraphId owner;
  const char* name;
  const OrtTypeInfo* type;      /* borrowed, not released by extension */
  uint32_t flags;               /* input, output, initializer, outer capture bits */
  const OrtValue* constant;     /* optional borrowed read-only CPU constant */
} OrtRFValueView;
```

V1 replacement attributes exclude graphs, sparse tensors, and lists of tensors/strings. Those
patterns are unsupported, not silently coerced. `NodeAt` represents unsupported source attributes
with kind 0 and their names; graph attributes are inspected through the named subgraph records.
A matcher requiring an unrepresented attribute cannot prove eligibility and must decline it.
Reading initializer payloads is explicit and subject to session byte limits; metadata inspection
must not materialize all 27B weights. `ValueInfo.constant` may be null even for an initializer;
`ReadConstant` returns a bounded borrowed view or a reason it cannot be exposed.
Existing initializer IDs can always be referenced without copying their entire payload.

### Replacement, registration, and function table

```c
typedef struct OrtRFNodeSpec {
  size_t struct_size;
  uint64_t local_node_id;
  const char* domain;
  const char* op_type;
  int64_t opset_version;        /* requested domain import; core resolves actual schema */
  const OrtRFValueRef* inputs;
  size_t input_count;
  const OrtRFValueRef* outputs;
  size_t output_count;
  const OrtRFAttribute* attributes;
  size_t attribute_count;
  uint32_t kernel_origin;      /* V1: 1 = target EP built-in registration only */
} OrtRFNodeSpec;

typedef struct OrtRFLocalValue {
  size_t struct_size;
  uint64_t local_id;
  const OrtTypeInfo* type;
  const OrtValue* initializer; /* optional new bounded CPU constant, deep-copied */
} OrtRFLocalValue;

typedef struct OrtRFProposal {
  size_t struct_size;
  const char* candidate_id;    /* unique within package/revision/snapshot */
  const char* pattern_contract;
  uint64_t snapshot_generation;
  OrtRFGraphId graph;
  OrtRFNodeId anchor;
  const OrtRFEpToken* target;
  uint32_t scope;              /* 0 flat; 1 bounded whole-Scan */
  const OrtRFNodeId* consumed_nodes;
  size_t consumed_count;
  const OrtRFValueId* boundary_inputs;
  size_t boundary_input_count;
  const OrtRFValueId* boundary_outputs;
  size_t boundary_output_count;
  const OrtRFLocalValue* local_values;
  size_t local_value_count;
  const OrtRFNodeSpec* nodes;   /* topologically ordered replacement DAG */
  size_t node_count;
  const char* resource_recipe; /* provider-approved target recipe key, not a byte assertion */
} OrtRFProposal;

typedef struct OrtRFShapeBound {
  size_t struct_size;
  const char* input_name;
  const int64_t* min_dimensions; /* finite nonnegative bounds; same rank */
  const int64_t* max_dimensions; /* finite nonnegative bounds, including state/cache */
  size_t rank;
} OrtRFShapeBound;

typedef struct OrtRFConfig {
  size_t struct_size;
  uint32_t enabled;
  uint32_t resource_mode;       /* 0 baseline; 1 budgeted */
  const char* const* required_patterns; /* qualified package-id/pattern-contract */
  size_t required_pattern_count;
  const OrtRFShapeBound* bounds;
  size_t bound_count;
  size_t max_proposals;
  size_t max_descriptor_bytes;
  size_t max_constant_bytes;
} OrtRFConfig;

typedef struct OrtRFExtension {
  size_t struct_size;
  uint32_t abi_version;
  const char* package_id;
  const char* package_revision;
  const char* target_ep_type;   /* filter, not an EP instance identity */
  uint64_t required_features;  /* V1 bit 0 = whole-Scan; unknown bits rejected */
  void* package_state;
  void (ORT_API_CALL* Retain)(void*);
  void (ORT_API_CALL* Release)(void*);
  OrtStatus* (ORT_API_CALL* CreateSessionState)(void* package_state, void** out);
  void (ORT_API_CALL* ReleaseSessionState)(void*);
  OrtStatus* (ORT_API_CALL* Match)(
      void* session_state, const OrtRFContext*, OrtRFCollector*);
} OrtRFExtension;

struct OrtRuntimeFusionApi {
  size_t struct_size;
  uint32_t abi_version;
  uint64_t supported_features;
  OrtStatus* (ORT_API_CALL* Configure)(OrtSessionOptions*, const OrtRFConfig*);
  OrtStatus* (ORT_API_CALL* RegisterExtension)(OrtSessionOptions*, const OrtRFExtension*);
  OrtStatus* (ORT_API_CALL* SnapshotInfo)(
      const OrtRFContext*, uint64_t* generation, OrtRFGraphId* root, OrtRFGraphId* current);
  OrtStatus* (ORT_API_CALL* GetSessionConfig)(
      const OrtRFContext*, const OrtRFConfig** out);
  OrtStatus* (ORT_API_CALL* GraphInfo)(
      const OrtRFContext*, OrtRFGraphId, const OrtRFGraphInfo** out);
  OrtStatus* (ORT_API_CALL* NodeAt)(
      const OrtRFContext*, OrtRFGraphId, size_t topological_index, const OrtRFNodeView** out);
  OrtStatus* (ORT_API_CALL* ValueInfo)(
      const OrtRFContext*, OrtRFValueId, const OrtRFValueView** out);
  OrtStatus* (ORT_API_CALL* ReadConstant)(
      const OrtRFContext*, OrtRFValueId, size_t byte_limit, const OrtValue** out);
  OrtStatus* (ORT_API_CALL* Submit)(OrtRFCollector*, const OrtRFProposal*);
};
```

Disabled is the default configuration. When enabling, supply nonzero caps; recommended starting
limits are 8,192 proposals, 64 MiB total descriptor storage, and 1 MiB of new/read constant payload.
These are limits, not reservations. The runtime may reject tighter platform limits explicitly.
`Configure` copies and replaces configuration atomically; options become immutable to the session
at initialization. Conflicting duplicate package ID/revision registrations fail rather than
silently changing precedence.

Before the first `Match`, core validates configuration against the loaded graph and freezes the
session's effective configuration. `GetSessionConfig` exposes that immutable snapshot, including
validated workload bounds, resource mode, required patterns and limits. ORT owns the returned
record and all nested strings/arrays; they are read-only borrows valid only during the current
callback, not handles to mutable `OrtSessionOptions`. Do not release, modify or retain them.
Every graph callback in a session sees the same effective configuration. Reconfiguring cloned
options must not alter another clone or an already-created session, even when registration and
package state are shared. `Match` must read this session view rather than infer bounds from shared
`package_state`, symbolic shape names or an earlier Run.

Graph, node, and value IDs are nonzero and unique within a snapshot, including descendants. The
context identifies its current graph; `SnapshotInfo` returns both root and current, while each callback's
collector restricts submissions to the graph being visited. `NodeAt` order is stable with node-ID
tie breaking. Core associates EP tokens with session-owned provider instances, supplies only tokens
for applicable EPs, and compares identity, never just the provider type string. The package can
compare tokens during the callback but cannot dereference or retain them. No proposal can invent
ownership. An eventual multi-instance EP implementation must preserve this invariant.

Replacement outputs referring to snapshot IDs must be precisely the original boundary values;
local IDs cannot collide with them. This expresses the source-to-target value mapping without
renaming observable values. Initializers referenced by snapshot ID retain graph/session ownership.
Shared initializers and outer captures are never deleted while another user exists. New constants
are copied into ORT-owned storage; packages cannot supply paths, executable callbacks, or device
pointers through a proposal. Each emitted compute node uses the target token; shape/host adapters
must already be supported on that EP or be left outside the consumed region.

### Ownership, threading, and errors

- Configure/RegisterExtension validate and copy atomically. On failure options are unchanged and
  no reference is retained. Strings, arrays, types, and submitted constants are copied by ORT
  before the call returns. Snapshot views, status message inputs, tokens, and collectors are
  borrowed only for the callback. Extensions must not call Release on borrowed handles.
- Successful registration retains package state once per shared registration object. Options
  clones share that immutable object and its DSO lease, preserving registration order and copied
  configuration. Each session creates separate mutable session state, even from cloned options.
  A failed `CreateSessionState` leaves `out` untouched; the package cleans its partial allocation.
- Within one session, callbacks are serialized and non-reentrant, including parent/body visits.
  Different sessions may initialize concurrently, including Create/Release callbacks. Shared
  package state must be thread-safe. No background callback may outlive its call.
- Destroy session callback state before releasing its registration lease. Release the last package
  reference while the library is still loaded, then unload. Releasing original options must not
  unload code used by clones or sessions. Application-managed/static code must remain loaded for
  that same interval; registration does not make manually unloading it safe.
- Loader integration must stage all fusion registrations made by one `RegisterCustomOps` invocation
  and publish only when it succeeds. On failure, release staged state before releasing the DSO.
  This is a **new required loader integration**, not an assertion that today's loader is
  transactional for arbitrary custom-op side effects. Fusion-only V1 conformance tests cover it.
- Null status means success. Returned errors are created through the negotiated OrtApi; ownership
  transfers to the caller, which calls `ReleaseStatus` (ORT does so for callback errors).
  No exceptions cross the ABI; release callbacks cannot fail. Failed calls do not modify out
  parameters. Malformed records and callback errors fail initialization.
- `Submit` success means copied and queued, not selected. Well-formed but unsupported candidates
  are recorded later as non-matches. Required patterns must have at least one accepted candidate;
  this is not a promise that every matching region fuses. Per-region mandatory coverage is deferred.

| Runtime / package combination | V1 outcome |
|---|---|
| Factory available, table version 1, qualified runtime/EP/build tuple | Register; eligibility still depends on graph/kernel validation. |
| Older runtime without the experimental name, or loading omitted by application | Application may run without an optional package; mandatory use fails before session creation. |
| Factory present but requested table unsupported, incompatible tuple, failed registration | Explicit error, no partial registration; do not reinterpret another table layout. |
| New runtime preserving exact V1 table and a newly qualified tuple | Compatible only after package conformance qualification; no blanket future-version promise. |

A failed explicit library load or callback is not silently swallowed as "optional." Optional
unavailability means the application elects not to register after its availability check.

Shape bounds are an application-declared workload envelope in **both** modes, not assumptions
inferred from one example. Require `0 <= min <= max` per dimension and reject unknown input names,
duplicate bounds and inconsistent ranks. Core validates all submitted feeds/bindings against the
accepted envelope before launching any kernel, including kernels before a recurrent region.
`Match` obtains these validated bounds through `GetSessionConfig`; graph type metadata alone
does not encode their finite minima/maxima. Missing bounds remain unknown, not unrestricted
permission to fuse.
Eligibility must hold for the entire fixed-shape or bounded envelope. No arbitrary package Run
callback, dynamic match/defuse, or data-dependent fallback is added.
If eligibility cannot be proven from topology, types, fixed shapes and declared shape bounds, leave
the region unfused. For example a session allowing `B=2,P>0,S>1` cannot receive GQA just because
its first call is prefill. Use a B=1 envelope, or separate B=2 prefill (`P=0`) and decode (`S=1`)
sessions with the same canonical file. A general B=2 dynamic/chunked session stays unfused.
Mask contents and arbitrary length-tensor values cannot be certified by shape bounds.

## 5. Host validation and commitment (D3)

1. Freeze registries and snapshot identities. Collect bounded proposals without mutating the graph.
   Enforce count/byte caps and unique candidate identities; duplicate identities are malformed.
2. Validate IDs, current generation, graph scope, acyclicity, schemas/import versions, type/shape
   compatibility, constants, and every use crossing the region boundary. Preserve all escaping
   outputs, including graph outputs and descendant captures. Missing observable state is not dead
   just because it has no ordinary local consumer. Do not raise an existing domain import and
   reinterpret surviving nodes; reject incompatible requests. Adding an otherwise absent target
   domain import is allowed only after validating all emitted schemas.
3. Require every consumed executable node to belong to the same target instance. CPU-resident
   initializer/shape inputs do not authorize consuming CPU compute. Never enter a compiler-owned
   compiled node/body, cross EPs, or reassign source nodes to force eligibility.
4. Resolve replacement kernels through the **actual session registry precedence**, without kernel
   construction. Validate schema/type constraints, build pruning, host-input requirements, device
   capabilities, and package-qualified operator contract. V1 declines any custom-registry override
   of an emitted kernel, even one with the same op name; it does not bypass normal lookup. Core
   records the selected built-in registration privately and checks it again at finalization.
5. In budgeted mode, obtain target-specific coverage before selection/commit (section 8).
6. Order candidates by application registration ordinal, then package ID/revision, candidate ID
   (UTF-8 byte order), then anchor ID. Greedily choose non-overlapping valid proposals. Consumption
   closure, including Scan descendants, defines overlap. Resource/semantic rejection is not a
   speed competition; no per-session benchmark or original-versus-fused cost tournament.
7. Validate the accepted **combined** replacement against a disposable graph representation before
   touching the live graph. This catches cycles and boundaries introduced by interactions between
   individually valid proposals. ORT, not the package, materializes nodes and assigns their EP.
8. Commit once, update identities/accounting, and Resolve. Unexpected commit/Resolve failure aborts
   and discards initialization. V1 promises no live-graph rollback, reconstruction, repartition, or
   execution retry. No partially mutated graph or recurrent state can continue.

Unsupported kernels, masks, ownership, or resource coverage detected before commitment leave the
original region intact with a reason, unless a required pattern cannot be satisfied. Missing
canonical execution support is still an ordinary session failure, not something fusion hides.
Distinguish `NO_MATCH`, `EP_MISMATCH`, `OPAQUE_REGION`, `UNSUPPORTED_MASK/STATE/TYPE`,
`KERNEL_MISSING/OVERRIDDEN`, `RESOURCE_UNKNOWN/INSUFFICIENT`, `OVERLAP`, `STALE_SNAPSHOT`,
`INVALID_PROPOSAL`, and `COMMIT_FAILED`. Stale/malformed proposals and callback failures are errors,
not silent non-matches.

Diagnostics identify artifact digest, package/revision, pattern contract, graph/generation, consumed
and emitted nodes, target instance token's diagnostic ID, selected registration, rejection reason,
and later transformations. Default logging includes no tensor payload, library search path, or
weight data. An opt-in trace records timing and resource provenance.

## 6. Pilot source-to-target contracts (D4)

The mappings below are package pattern contracts, not model names in the core API. They require
structural and semantic qualification; registration of a kernel establishes feasibility only.
Canonical ONNX inputs/outputs remain unchanged, including all cache and recurrent state values.
Adapters are ordinary existing operations in the replacement DAG, not hidden package execution.

### 6.1 Canonical Attention to GroupQueryAttention

Use M's `_attention.py::_apply_attention` and
[`rewrite_rules/_group_query_attention.py::AttentionToGQA:455-645`](https://github.com/onnxruntime/mobius/blob/b1e9c8e9dec1dfa206183f957e77d07cfb734df2/src/mobius/rewrite_rules/_group_query_attention.py#L455-L645)
as prior art, not as a semantic oracle.
Qwen35Attention's offset Q/K normalization, partial MRoPE, projection packing, and sigmoid output
gate stay outside the first region. Target `com.microsoft::GroupQueryAttention:1`, `do_rotary=0`,
no packed QKV, no local window, no quantized cache, no absorbed normalization.
CUDA activation registration is FP16/BF16, not FP32
([R registration](https://github.com/microsoft/onnxruntime/blob/1988019b6e23ac4def4628cddd603c1bfb504fe8/onnxruntime/contrib_ops/cuda/bert/group_query_attention.cc#L56-L84)).
V1 qualifies **FP16 only**; FP32 remains a baseline/non-match. BF16 is a later independently gated
package contract even though the kernel is registered.

| Canonical boundary | Emitted boundary / condition |
|---|---|
| Q `[B,Hq,S,D]`, K/V `[B,Hkv,S,D]` or proven equivalent 3-D form | Transpose/reshape to query `[B,S,Hq*D]`, key/value `[B,S,Hkv*D]`; inputs 0/1/2. Require `Hq % Hkv == 0`, equal K/V head dimensions, proven head ordering. |
| `q_num_heads`, `kv_num_heads`, Attention scale | GQA `num_heads=Hq`, `kv_num_heads=Hkv`; materialize exact scale, including default `1/sqrt(D)`. Do not change scaling or softcap/softmax semantics. |
| Dynamic past K/V `[B,Hkv,P,D]` | Inputs 3/4; present outputs 1/2 `[B,Hkv,P+S,D]`. No paged/static/windowed/sliding/borrowed cache, quantization, or fixed-capacity tail semantics. |
| Per-row total valid length `L_b` | `seqlens_k[b] = L_b - 1`, INT32 device input 5; scalar `total_sequence_length = max_b L_b`, INT32 CPU input 6. These are not a mask tensor or past lengths. |
| Causal current chunk | GQA attribute `causal=1` (not the internal `is_unidirectional` name); prove query position i attends keys through `P+i`. Reject top-left/bottom-right alignment disagreement. |
| Y and cache outputs | Output 0 is `[B,S,Hq*D]`; restore canonical Y layout and preserve exact present-K/V value IDs. |
| Optional Attention diagnostics | Reject an observed QK/softmax/LSE or other unsupported output. CUDA GQA cannot return QK; do not erase observable outputs. |

**First mask envelope:** uniform dense rows, one shared `P`, all `S` query tokens valid, `L_b=P+S`,
and no explicit mask or a structurally proven redundant causal mask. Allow prefill `P=0` and
single-token decode for dense uniform B; continuation chunks `P>0,S>1` require **B=1** under the
[existing helper constraint](https://github.com/microsoft/onnxruntime/blob/1988019b6e23ac4def4628cddd603c1bfb504fe8/onnxruntime/contrib_ops/cpu/bert/group_query_attention_helper.h#L429-L445).
`B>1,P>0,S>1` is a non-match even with equal lengths. Prove INT32 arithmetic before narrowing: `0 <= P`,
`1 <= S`, `P+S <= INT32_MAX`, and all shape/offset products fit their kernel interfaces.
Derive host placement from the selected kernel definition; copies are inserted by normal ORT
logic and included in resource/performance accounting. Never pull arbitrary device data to the
host during Match or infer runtime mask contents from a symbolic shape. A graph input mask that
happens to contain ones in a test is not structurally proven redundant. V1 declines arbitrary
runtime mask inputs; supporting a checked runtime predicate before any state mutation would require
a separately approved adapter/error contract. A workload promise alone must not erase mask semantics.

For padding, map a standard prefix-valid mask to `L_b` only under a **separate qualified contract**
that proves query alignment, cache padding, output values for padded queries, and per-row lengths.
V1 rejects different per-row valid lengths, left/interior padding, arbitrary additive bias,
noncausal mode, unknown mask provenance, and fully masked rows. Finite large-negative values are
not automatically equivalent to `-infinity`. Mask conversion must preserve original error/bounds
semantics; `Cast(INT64->INT32)` alone is not range validation. Unproven cases remain unfused.

The smallest blocker is a tiny post-partition Attention fixture with `P>0,S>1`, plus a ragged
near-match, recording source ownership and causal/output equality. Acceptance of uniform dense
GQA does not approve broader padding, BF16, cache aliases, or absorbed rotary.

### 6.2 Causal convolution

M's [`functions/causal_conv.py:21-145`](https://github.com/onnxruntime/mobius/blob/b1e9c8e9dec1dfa206183f957e77d07cfb734df2/src/mobius/functions/causal_conv.py#L21-L145)
supplies the source history/Conv/Swish form. Target existing
`com.microsoft::CausalConvWithState:1`:

| Source | Target |
|---|---|
| Current x `[B,C,S]`, history h `[B,C,K-1]`, `Concat(h,x)` along sequence | input 0 = x; optional input 3 = h; missing history means zero history only if source agrees. |
| Depthwise Conv weight `[C,1,K]`, optional bias `[C]` | inputs 1/2; `group=C`, stride 1, dilation 1, and exactly the source causal padding/cropping. |
| `z * Sigmoid(z)` after bias | `activation="swish"`; do not absorb differently ordered gating. |
| Trailing `K-1` values of **raw** concatenated history/current input | present_state output 1, not post-convolution or activated values. Output 0 keeps x's shape. |

V1 requires `ndim=1,channels_last=0,dilation=1,state_window=0`, dense FP16, no extra padding/stride
or side uses of consumed intermediates. A Concat already providing history plus a valid Conv
must not also receive another left pad. Check every Slice axis/start/end/step and Conv attribute,
not just the operator names. For `S<K-1`, next state includes surviving old history:
`tail_{K-1}(Concat(h,x))`. Include `S=1,2` and long prefill fixtures with nonzero h.
Do not trust M's historical comment that CUDA lacks this op; verify the selected build registry.
Omitted bias is an absent optional edge, not an invented zero tensor.
[R's CUDA registration](https://github.com/microsoft/onnxruntime/blob/1988019b6e23ac4def4628cddd603c1bfb504fe8/onnxruntime/contrib_ops/cuda/bert/causal_conv_with_state.cc#L16-L43)
and [schema](https://github.com/microsoft/onnxruntime/blob/1988019b6e23ac4def4628cddd603c1bfb504fe8/onnxruntime/core/graph/contrib_ops/bert_defs.cc#L4518-L4596)
are the target contract, not the old comment.

### 6.3 Whole-Scan recurrence

CUDA registers standard Scan, including opsets 23-24
([R `scan.cc:49-153`](https://github.com/microsoft/onnxruntime/blob/1988019b6e23ac4def4628cddd603c1bfb504fe8/onnxruntime/core/providers/cuda/controlflow/scan.cc#L49-L153)).
Host orchestration is not CPU EP assignment. M's standard-only LinearAttention lowering yields
shape/head adaptation and a Scan
([M lowering](https://github.com/onnxruntime/mobius/blob/b1e9c8e9dec1dfa206183f957e77d07cfb734df2/src/mobius/functions/linear_attention.py#L132-L235));
fusing arithmetic inside one iteration is insufficient.

Add only `scope=whole-Scan`: the proposal consumes the entire Scan node in its enclosing graph,
plus explicitly listed flat adapters if eligible. Core, not the package, derives its recursive
executable consumption closure. No general nested-graph editing API or replacement graph attribute
is introduced. Inspect the body read-only and require:

- Forward scan input/output directions, proven scan axes, matching carried-state count/order,
  no early termination, and exactly the supported recurrence.
- Complete outer captures and invariant weights/gates; no mutation or unaccounted side effects.
- CUDA ownership for the parent Scan and every consumed executable descendant, same instance.
  Reject CPU-parent/CUDA-body, mixed body, compiler-owned body, and nested control flow.
- Every externally observable sequence/state output is reconstructed, not merely the final token.
- Retire descendant reservation/profile/identity entries **before** destroying the body graphs;
  invalidate snapshot handles at commitment. Verify no Scan remains in the accepted region.

First target is `com.microsoft::LinearAttention:1`, `update_rule="gated_delta",state_window=0`.
Keep Q/K L2 normalization, Qwen gate/softplus arithmetic, sigmoid beta, precision casts and output
gating outside. Match their outputs at the recurrence boundary without moving a cast across an
accumulation. After head expansion, let H be the recurrence/value head count:

```text
q,k: [B,S,H*Dk]  v: [B,S,H*Dv]  g,beta: [B,S,H]
past/present S: [B,H,Dk,Dv] (K-major; state dtype must match the canonical recurrence)
A_t = exp(g_t) * S_(t-1)
delta_t = beta_t * (v_t - transpose(A_t) * k_t)
S_t = A_t + outer(k_t, delta_t)
y_t = scale * transpose(S_t) * q_t
```

| Canonical value | LinearAttention |
|---|---|
| Adapted Q/K/V after exact head repetition/order | inputs 0/1/2, 3-D packed; `q_num_heads=kv_num_heads=H` in the first inverse-GQA-adapted contract. |
| Scan incoming K-major state | input 3 `past_state`; never transpose merely because equal axes make it look valid. |
| Effective log decay and beta | inputs 4/5; preserve scalar-per-head broadcast and dtype. |
| Ordered Scan sequence output and final state | outputs 0/1, restore sequence axis/layout, preserving boundary IDs. |

Require the exact source cast schedule and matching state type; a source FP32 recurrence is not
permission to store state in FP16. In particular, the separate schema type variable for state
does **not** prove CUDA mixed-type support: the
[existing kernel reads/writes state through its activation template T](https://github.com/microsoft/onnxruntime/blob/1988019b6e23ac4def4628cddd603c1bfb504fe8/onnxruntime/contrib_ops/cuda/bert/linear_attention.cc#L180-L203).
For the pilot's FP32 state, Q/K/V/g/beta at the recurrence boundary must also be FP32, with
existing source casts outside the replacement and the same output cast back to FP16.
If M's actual lowered cast schedule cannot be mapped this way, P5 blocks this target: do not add
casts that change rounding merely to make the kernel eligible. Initial state may be nonzero. Runtime must not silently
reinterpret layout, reset state between chunks, or omit a state update on failure.

**GatedDeltaNet is later, not a rename.** Its state is FP32 V-major `[B,Hv,Dv,Dk]`, its token/head
input layout and optional packing differ, it can absorb Q/K normalization and Qwen gates, and it
declares possible initial/final state aliasing. It also allocates runtime plan scratch.
Explicit transpose/cast/resource/alias proofs would be needed for that wider envelope. Qualify
unequal `Dk,Dv` and nonsymmetric states first; the real model's equal 128 axes hide transpositions.

The [G Qwen35 builder](https://github.com/microsoft/onnxruntime-genai/blob/991e3560e63b8a26806d4ec7d702c04e75fb6e51/src/python/py/models/builders/qwen.py)
defaults `linear_attn_op=linear_attention` and also accepts `gated_delta_net`. The reference
recipe here fixes **linear_attention, nonpaged mode, state_window=0**. Do not mix those state ABIs.
Actual exported pattern survival, body ownership and numerical equivalence remain unverified;
the structural probe in section 10 blocks recurrence implementation, not merely its release.

### 6.4 Official pilot identity, not a qualification claim

Official configuration:
[`Qwen/Qwen3.8-27B@1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0/config.json`](https://huggingface.co/Qwen/Qwen3.8-27B/blob/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0/config.json).
Required upstream raw SHA256:
`191e0af232104ed8b65258cf3fb2b842e288008baca7633c11b82a1ac7203aab` (4,312 bytes).
M records this provenance in
[`qwen38-27b-standard-onnx.md:4-16`](https://github.com/onnxruntime/mobius/blob/b1e9c8e9dec1dfa206183f957e77d07cfb734df2/docs/design/qwen38-27b-standard-onnx.md#L4-L16)
and supplies a [reformatted config fixture](https://github.com/onnxruntime/mobius/blob/b1e9c8e9dec1dfa206183f957e77d07cfb734df2/testdata/configs/qwen3_8-27b.json#L1-L100).
The reformatted fixture's byte hash is not the upstream raw hash. Direct raw retrieval/hash
recomputation was not available for this design; require independent verification before M5.

This is **qwen3_5**, not qwen4_exp/Flash-Next: 64 decoder layers, 48 linear plus 16 full attention,
dense MLP, hidden/intermediate `5120/17408`, vocabulary `248320`; full-attention heads `24/4`
with head dimension `256`; linear key/value heads `16/48` with `Dk=Dv=128`; convolution kernel 4.
Official dtype is BF16 with FP32 recurrent state. The pilot's FP16 conversion is a separately
qualified derivative policy, not a claim that official BF16 has already been qualified.
MRoPE is interleaved with sections `[11,11,10]`, partial factor `0.25`, theta `10000000`;
maximum positions `262144`, RMS epsilon `1e-6`.

[Mobius #758](https://github.com/onnxruntime/mobius/issues/758) records historical qualification,
not ModelBuilder equivalence or resolution of all official FP16/HF differences. No private cached
ONNX package, weights, reports or successful full-model run is assumed available.

## 7. State and compatibility invariants

Canonical graph I/O is the state API. No extension-owned hidden persistent recurrent/cache state is
allowed. Normal Run, I/O binding, cancellation, asynchronous completion, and existing allocation
rules apply. New fusion adds no retry guarantee. A failed Run may have modified an allowed aliased
buffer under the existing op contract; the application must discard/reset that state, not rerun
unfused on it. The first dynamic-cache contract excludes past/present aliasing; recurrence/convolution
must preserve the source's observable ownership semantics and cannot newly overwrite a live input.

Downstream transformations must not silently change dtype, state layout or fusion contract.
Resource checks use the actual final implementation and alias plan, not a preferred kernel.
Serialized derivatives record canonical digest, package and API revision, runtime/EP/build/device
constraints, and graph/resource profile provenance. They are invalidated when those constraints
change; never label them canonical or reload them as evidence of package-independent matching.
Legacy contrib graphs and private transformer enablement receive regression coverage, not migration
requirements. Supporting standard CPU/WebGPU fallback means leaving a valid unsupported region
unchanged; it does not imply every standard op has a kernel on every build.

## 8. Resources: two modes, no inferred hard guarantee (D5)

At R,
[`ConsolidateWorkspaceReservations:25-79`](https://github.com/microsoft/onnxruntime/blob/1988019b6e23ac4def4628cddd603c1bfb504fe8/onnxruntime/core/framework/resource_accountant.cc#L25-L79)
sums source reservation categories and takes a scratch maximum; it does not estimate a new target.
The partial CSV path
[`resource_accountant.cc:184-188`](https://github.com/microsoft/onnxruntime/blob/1988019b6e23ac4def4628cddd603c1bfb504fe8/onnxruntime/core/framework/resource_accountant.cc#L184-L188)
can return zero for a missing node. This is not a complete upper bound.
Graph callbacks remap/remove entries by graph identity. Kernel declarations collected after
creation/PrePack
([`session_state.cc:2127-2280`](https://github.com/microsoft/onnxruntime/blob/1988019b6e23ac4def4628cddd603c1bfb504fe8/onnxruntime/core/framework/session_state.cc#L2127-L2280))
do not make every missing or underreserved entry a strict failure.

### Baseline pilot

Preserve ordinary execution/diagnostics and measure actual allocations, including initialization.
No hard capacity promise. Mark consolidated reservations as inherited-source/unknown-target, or
clear their target-bound interpretation; do not present them as new target estimates.
GQA has sizing/declaration support; do not infer it for LinearAttention or
GatedDeltaNet from registry availability. Stateful headers do not expose an equivalent declaration
override. Baseline is the first delivery; budgeted fusion is separately gated.

### Budgeted integration

The resource owner must approve a target recipe keyed by actual selected kernel/build/device,
attributes, dtypes, shape/workload bounds, state/alias behavior, and concurrency. A package supplies
a recipe key, not authoritative byte counts. Unknown is distinct from zero. The admission flow is:

1. Before candidate commitment, evaluate target coverage against declared finite workload bounds.
   Replace source charges with target charges (deduplicate shared constants) and conservatively
   check the combined plan. Unknown/underreserved candidates stay unfused with a reason; an unmet
   required pattern fails. Do not alter the user's EP budget.
2. After **all** L2+/cast/copy transformations, rebuild or revalidate coverage for the final graph,
   including unaffected source nodes and every inserted adapter/copy. If coverage is invalidated
   or no longer fits, fail initialization **even for optional fusion**. No graph rollback or
   automatic unfused retry.
3. Admit the whole initialization/execution envelope **before initializer upload, kernel
   construction, PrePack, or another budget-consuming initialization step**. Include initialization peaks, not just Run.
   Declarations after creation validate the recipe but are not the preventive gate.
4. Verify actual declarations/allocations against admitted limits; unexpected excess is a fatal
   initialization/Run error, never permission to allocate first and silently revise the budget.
   Workload-bound violations must be detected before launching any state-mutating kernel.

| Category | Bound and lifetime required |
|---|---|
| Weights/initializers | Device/host residency, shared storage deduplication, persistent packed copies, and old/new overlap during packing. |
| Initialization | Packing/conversion scratch, target plan creation, allocator retention and simultaneous session initialization. |
| Workspace/activations | Selected dispatch variants, streams, overlapping lifetimes, alignment and allocator overhead; not simply the sum of source workspaces. |
| Cache/state | Caller-owned and ORT-owned buffers, dynamic growth up to the bound, old and new buffers, layout adapters, alias-dependent preservation scratch. |
| Transfers | Staging/pinned host buffers, copy outputs and asynchronous completion lifetimes. |
| External/retained storage | Existing session/EP allocations, allocator pools and application-bound buffers; explicit ownership and charge domain. |

GQA's exactly-one-K/V-pair alias case requires preservation scratch. Model it or reject it; two
identical past/present shapes do not establish identical memory requirements. Initially exclude
uncovered alias plans. GatedDeltaNet plan scratch and any recurrence initialization must have
explicit coverage before budgeted use. Missing fused names in a CSV are never free.

Profiles are valid only for their graph digest/revision, kernel registration and package contract,
runtime/EP/device build, dtype, shape bounds, dispatch/alias policy and workload/concurrency envelope.
A graph-removal callback retires old entries; it does not create a target estimate. Whole-Scan
removal retires descendants before their graph identities can dangle.

This is a **session/EP-accounted bound**, not a promise that other processes cannot exhaust device
memory. Initial budgeted support requires an enforceable allocation domain or conservatively
bounded sharing; otherwise reject the budgeted configuration. Compiler EP compilation or EP setup
may allocate before this hook: any such allocations must already be admitted under the existing
resource owner, or that configuration cannot claim this hard-bound mode.
The same rule applies to allocation-bearing DSO static initialization or registration: prohibit
device staging there in the supported package contract, or admit/charge it before loading.

Narrow adapters to existing recipes are allowed follow-on work. The general planner in
[#33372](https://github.com/microsoft/onnxruntime/issues/33372) and P's reconciliation are future
dependencies, not current guarantees or prerequisites for baseline fusion. Memory determines
feasibility, not default speed preference. If a later authorized placement workflow splits a fused
region, invalidate/rebuild affected fusion/resource data or reject that workflow; automatic defusion
search is deferred.

The post-partition hook deliberately runs after the partitioner's accountant objects have been
used to produce reservations. Budgeted implementation therefore needs a planning-safe target
estimator/resource-owner adapter; invoking the existing source consolidation callback cannot stand
in for it. For this fixed-placement path, do not move matching into a per-EP capability loop to
regain access to those objects: partitioning may revisit graphs/providers. This missing adapter
blocks budgeted mode, not baseline.

Baseline/budgeted are resource modes, not fixed/planning placement modes. Budgeted admission here
validates the chosen assignment; it does not search alternative ownership or guarantee discovery of
a GPU placement made feasible by fusion. Future planning needs the nonmutating recipe queries in
section 3 before final placement acceptance, followed by reconciliation of the selected graph and
memory plan. That integration is proposed, not a guarantee supplied by current ORT main.

## 9. Qualification plan, not execution evidence

### 9.1 Generic mechanism and API

Create the following tests after contract approval; the `RuntimeFusion*` suites do **not** exist
at R. A skip/missing binary, missing fixture, unavailable CUDA EP, or lack of independently built
DSO is a blocked prerequisite, never a pass.

| Suite / target | Required assertions |
|---|---|
| `RuntimeFusionTests.*` / `onnxruntime_test_all` | Absent/disabled/Basic preserve behavior; required-at-Basic error; trace partition, extension once, L2, casts, copies, including repeated optimization loops and nested graphs. |
| `RuntimeFusionTests.*` | EP-split, opaque compiler regions, deterministic overlaps, stale IDs, all escaping outputs/captures, missing schema, invalid combined DAG, failed Resolve, callback failure, proposal/byte limits. Verify graph unchanged on precommit non-match and session discarded on commit failure. |
| `RuntimeFusionApiTest.*` / `onnxruntime_shared_lib_test` | Static and separately built DSO; null discovery/version mismatch/short descriptors; copied descriptors; cloned options after original release; concurrent sessions and no callback reentrancy; partial registration failure; retained-state/destructor/unload order. |
| `RuntimeFusionApiTest.SessionConfigBounds` / `onnxruntime_shared_lib_test` | `Match` reads the validated effective config, including nested bounds, through a callback-scoped read-only borrow. Clones configured with different bounds and concurrent sessions see their own snapshots; releasing/reconfiguring options cannot alter another clone or an existing session. Invalid configuration fails before Match. |
| `RuntimeFusionApiTest.*` | Build DSO using only public C headers, no private ORT includes/link symbols; load into separately built ORT. Test no-native-loader build using the same static registration API. |
| `RuntimeFusionTests.WholeScan*` | CUDA parent/body positive; CPU parent/CUDA body negative; CUDA parent/mixed body negative; capture/state order; unsupported nested control flow; recursive overlap; descendant resource retirement before graph destruction; no loop after acceptance. |
| `RuntimeFusionTests.Budget*` | Unknown estimate, target larger than source, partial CSV, shared initializer, initialization peak, stale body IDs, later-transform invalidation, final graph failure even for optional fusion, unchanged EP budget, out-of-bound Run rejected before state mutation. |
| `RuntimeFusionCudaTest.*` / `onnxruntime_provider_test` | Pruned/overridden kernels and actual selected memory/alias metadata; dense GQA, convolution, recurrence mapping and numerical gates; source/replacement ownership trace. |
| Existing regression targets | `InferenceSessionTests.TestRegisterTransformers`, `ExperimentalCApiTest.*`, relevant existing resource-accountant, attention and stateful tests, legacy contrib models and disabled-extension behavior. |

Provider/contrib tests belong to `onnxruntime_provider_test`, not `onnxruntime_test_all`
([R CMake path split](https://github.com/microsoft/onnxruntime/blob/1988019b6e23ac4def4628cddd603c1bfb504fe8/cmake/onnxruntime_unittests.cmake#L318-L360)).
CUDA-positive tests must assert actual parent/body and replacement assignments and the expected
kernel registration, not infer CUDA execution from a provider list. CPU shape scaffolding may
remain only outside consumed regions and must be explicitly allowlisted in the trace.

### 9.2 Exact small graph fixtures

Use ONNX opset 24 (Attention resolves its applicable standard schema), IR version 10, dense tensors,
no external downloads, no quantization, and no external custom ops in the canonical files.
Store fixture construction in the eventual test suites, not a model-specific runtime facility.
Use NumPy `Generator(PCG64(20261015))` for reference data, with recorded generation order
Q, K, V, weights, biases, states; keep checked-in deterministic small golden inputs when porting
to native fixtures. Reset the generator per case. Draw normal(0,0.1) activations and
normal(0,0.02) weights/biases in FP32, then cast at canonical graph boundaries. Nonzero state
uses `0.01 * (1 + ((3*b + 5*h + 7*k + 11*v) % 97))`; convolution history uses the analogous
`0.01 * (1 + ((3*b + 5*c + 7*t) % 97))`. These patterns are deliberately nonsymmetric.

| Fixture family | Exact dimensions / variants |
|---|---|
| GQA dense | `B={1,2}, Hq=4,Hkv=2,D=64, P={0,3,17}, S={1,2,7,65}`, scale `1/8`; only `P=0` or `S=1` or `B=1` combinations are positive. Causal, nonpaged, dynamic cache. Compare Y, present K, present V. |
| GQA negative | FP32, unequal K/V head widths, observed diagnostic output, nondivisible heads, noncausal, arbitrary/runtime additive mask, INT64 length overflow, static cache, any padding, all-masked row, `B=2,P=3,S=2`. Right-prefix ragged totals `[P+S,P+S-1]` must remain unfused in V1. |
| Convolution | `B={1,2}, C=12,K=4,S={1,2,3,7,65}`; group 12, stride/dilation 1, valid Conv over explicit history, Swish; zero/nonzero h; bias present/absent; exact output and trailing raw history. Negative stride 2/dilation 2/channels-last/state-window forms remain unfused. |
| Recurrence | `B={1,2}, H_key=2,H_value=4,Dk=32,Dv=64,S={1,2,7,65}`; repeat Q/K to 4 heads in the same order as M, then first target has `H=4`. Also equal `Dk=Dv=128` control. K-major FP32 state and FP32 Q/K/V/g/beta at the recurrence boundary; FP16 projection/output casts stay outside. |
| Recurrence arithmetic | Q/K divide by `sqrt(sum(x*x)+1e-6)`; beta is sigmoid of sampled input; log decay is `-exp(A_log)*softplus(a+dt_bias)`, with synthetic `A_log=dt_bias=0` for the boundary fixture. Scale `1/sqrt(Dk)`. Keep these ops outside; test zero/nonzero state. |
| Chunk/state progression | For total 65 tokens, compare one prefill, chunks `[2,1,7,55]`, and 65 single-token calls, feeding each output state to the next call. Add prefix 7 then 16 generated-token-feedback steps for reduced decoder tests. |

The initial recurrence source fixture must use the same in-body cast schedule as M's standard
lowering, not a mathematically similar hand-reassociated Scan. Its independent reference implements
the displayed recurrence separately in FP64, applying the canonical round/cast points explicitly.
If the registered kernel cannot preserve that schedule within the predeclared gates, decline this
pattern; do not silently select a different state ABI.

For the configuration-visibility regression, use one saved canonical GQA graph with symbolic
`B/P/S`, `Hq=4,Hkv=2,D=64`, and the same package registration shared by cloned options.
Configure session A with `B=1,P=[1,17],S=[2,7]`, and session B with
`B=2,P=[1,17],S=[2,7]`. Each `Match` must observe its own bounds through `GetSessionConfig`;
A is eligible if the remaining contract checks pass, while B must decline batched continuation
chunks. Initialize the sessions concurrently and verify that reconfiguring one options clone
cannot change the other's snapshot. Add a third envelope whose `P_max + S_max > INT32_MAX`:
decline GQA during matching without allocating tensors at those maximum dimensions.

### Reduced decoder configuration and export recipe

Use the pinned M config fixture, not a remote model identifier, through
`transformers.Qwen3_5Config.from_dict`. For an exact small **text-only** test configuration,
override the following `text_config` fields, retaining all other values from that fixture:

```json
{
  "hidden_size": 256,
  "intermediate_size": 512,
  "num_hidden_layers": 2,
  "layer_types": ["linear_attention", "full_attention"],
  "full_attention_interval": 2,
  "num_attention_heads": 2,
  "num_key_value_heads": 1,
  "head_dim": 256,
  "linear_num_key_heads": 2,
  "linear_num_value_heads": 4,
  "linear_key_head_dim": 32,
  "linear_value_head_dim": 64,
  "linear_conv_kernel_dim": 4,
  "mamba_ssm_dtype": "float32",
  "vocab_size": 256,
  "bos_token_id": 1,
  "eos_token_id": 2,
  "max_position_embeddings": 128
}
```

Keep full-attention head dimension 256 so the retained partial rotary width is 64, exactly
`2*sum([11,11,10])`; do not shrink the head without also redesigning the MRoPE fixture.
Preserve RMS epsilon, offset normalization, attention output gating and dense MLP.
For local configuration validation also remap top-level image/video/start/end token IDs to
`240/241/242/243`, but never feed multimodal tokens or run a vision graph. Save with
`config.save_pretrained` to an absolute local config directory. The eventual reduced test must
build the text path only; a whole-VLM graph export is not text-only qualification.

Use PyTorch seed `20261015` to construct random FP32 parameters with the pinned test's initializer
policy, then apply the FP16 export policy while preserving FP32 recurrence. Initialize RMS offset
weights according to the model, not arbitrary Gaussian scales. Use prompts `[1,17,23,41,5,8,9]`
and `[1,7,11,19,31,6,13]` for B=2, no padding; positions are three identical text-only aranges.
Compare extension-off/on with 7-token prefill, chunks `[2,1,4]`, then 16 greedy feedback tokens,
all convolution and recurrent states, and both GQA caches. Run chunked GQA positives at B=1.
For B=2 use separate prefill/decode envelopes, or require GQA to remain unfused in a general
dynamic/chunked session; do not expect an unsupported batched chunk to execute GQA.

Public prior art for local config construction and random-weight binding is
[`qwen38_standard_test.py:27-50,333-371`](https://github.com/onnxruntime/mobius/blob/b1e9c8e9dec1dfa206183f957e77d07cfb734df2/src/mobius/models/qwen38_standard_test.py#L333-L371).
Its `_from_config`, preprocessing and binding calls are **pinned test internals**, not a stable
public random-weight CLI. Reuse their construction in the authorized qualification fixture and
record Transformers/ONNX/NumPy/PyTorch dependency pins from M's approved test environment.
Missing dependencies require separate authorization, not installation by this design task.
The new `RuntimeFusionCudaTest.TinyQwen35Generation` target must provide this bound-weight fixture;
a `--no-weights` graph by itself cannot provide numerical evidence.

An existing public weight-free structural route is
[`python -m mobius build --config ... --ep onnx-standard --dtype f16 --no-weights`](https://github.com/onnxruntime/mobius/blob/b1e9c8e9dec1dfa206183f957e77d07cfb734df2/src/mobius/integrations/ort_genai/qwen38_standard_test.py#L73-L111).
The text-only feature must be selected for the text campaign. It does not provide extension
registration or qualify native GenAI loading. Inspect the resulting standard graph, inline
functions through the standard-only export path, and verify no contrib nodes remain.

### 9.3 Numerical acceptance gates

Three comparisons are required: canonical extension-off versus on; each versus an independently
constructed semantic reference; and, where applicable, the pinned ModelBuilder contrib path.
Two graphs calling the same kernel are **not** independent proof. Check intermediate tensors,
every state at every chunk/token boundary, and full logits, not just top-1 text.
Use elementwise `abs(actual-reference) <= atol + rtol*abs(reference)` and require all values finite.
Structural, length, mask decisions and copied cache/history values must match exactly.

| Boundary dtype | Intermediate/output atol, rtol | Recurrent state atol, rtol | Reduced full logits atol, rtol | Status |
|---|---|---|---|---|
| FP32 | `1e-5,1e-4` | `1e-5,1e-4` | `1e-4,1e-4` | Canonical/reference control; GQA remains a non-match. |
| FP16 | `2e-2,2e-2` | `2e-2,2e-2` | `5e-2,2e-2` | First qualified activation type; state gates apply to actual stored state dtype, including FP32 state. |
| BF16 | `5e-2,5e-2` | `5e-2,5e-2` | `1e-1,5e-2` | Reserved gate for separate future qualification, not V1 support. |

These are proposed gates to ratify **before** running, not measured accuracies. Do not loosen
them after inspecting failures without an explicit reviewed contract change. Additionally report
max absolute/relative error, RMS error, first divergence and logit margin. Greedy feedback tests
use lowest-token-ID tie breaking and feed each path's generated token back to that path; require
identical 16-token sequences for the tiny fixture as well as the tensor gates. A small logit margin
does not excuse a changed token. Full official-model HF numerical limitations are separate known
qualification concerns, not erased by a tiny pass or by loose FP16 bit-identity claims.

### 9.4 Proposed commands and artifact update experiment

Commands below require separately authorized execution and **already-built**, absolute-path
targets. New suites and their fixtures must first be implemented. No build/install command is
part of this assignment. Fixture tests must fail if required CUDA/DSO prerequisites are absent,
and record the full runtime/EP/package/device pins and test count.

```bash
# Planned only. Set ORT_BUILD_DIR to an absolute directory containing authorized existing binaries.
timeout 120s "${ORT_BUILD_DIR:?}/onnxruntime_test_all" \
  --gtest_filter='RuntimeFusionTests.*:InferenceSessionTests.TestRegisterTransformers'
timeout 120s "${ORT_BUILD_DIR:?}/onnxruntime_shared_lib_test" \
  --gtest_filter='RuntimeFusionApiTest.*:ExperimentalCApiTest.*'
timeout 300s "${ORT_BUILD_DIR:?}/onnxruntime_provider_test" \
  --gtest_filter='RuntimeFusionCudaTest.*'
# Planned structural export only, using M's existing CLI in an authorized pinned environment.
python -m mobius build --config /tmp/ort-runtime-fusion/tiny-config \
  /tmp/ort-runtime-fusion/tiny-standard --ep onnx-standard --dtype f16 \
  --features text-only --no-weights
```

Run the smallest relevant fixture first using an exact test filter, then the family. Existing
attention/stateful/resource test names must be enumerated from the chosen binary before scheduling;
do not interpret an empty GoogleTest filter result as success. Sanitizer-enabled lifetime tests,
CUDA memory checks, and CPU/WebGPU implementation campaigns require their own authorized builds.

Package-update proof: serialize the tiny canonical graphs once, hash the model and every external
data file with SHA256, then initialize with no package, explicit disable, package revision A and B.
A recognizes one approved post-L1 topology. B additionally recognizes a proven redundant
reshape/identity adapter form already in the **same saved artifact**, without changing the target
op or numerical envelope. Record increased eligible coverage, graph trace and the same hashes.
Ensure L1 actually preserves this distinguishing form; if not, the topology-survival probe blocks
this experiment until an appropriate form is chosen. Do not re-export between A and B.
Unavailable optional package leaves ordinary execution; required-package policy is an application
load error. Reverting B to A must not require editing the canonical files.

### 9.5 Performance envelopes

Before an authorized run, record runtime/EP/package/compiler/device/driver pins, clocks/power policy,
CPU affinity, streams, warm-up, repetitions and synchronization. No per-session microbenchmark
chooses a fusion. Offline package qualification decides whether to ship the eligible contract.

For tiny fixtures, require bounded matching storage (at most configured proposal/descriptor caps),
and record time per graph/node rather than claiming a meaningful throughput speedup.
For the reduced decoder and later text-only 27B campaign, pre-register these proposed envelopes:
batch `{1,2}`, prompt `{128,512,2048}`, decode `128` generated tokens, dense FP16, nonpaged state,
no speculative decoding, one session/stream initially; `max_total_length=2176`.
Use a B=1 dynamic session or separate batched prefill/decode sessions where required by GQA.
The reduced fixture uses its own context limit 128 and 16 generated tokens, not the full
27B lengths listed here.
Run extension-off/on with identical weights, inputs, generation settings and hardware.

| Metric | Proposed release gate, ratified before collection |
|---|---|
| Matching/validation wall time | At most `max(50 ms, 5% of extension-off initialization)` on the same machine; separately report graph copying/Resolve. |
| Initialization and PrePack | Report cold/warm init and persistent/scratch bytes; no hidden exclusion of load, upload or packing costs. More than 10% total-init regression needs explicit acceptance. |
| TTFT | No greater than 5% regression in any qualified envelope, including initialization-inclusive and steady-state figures separately. |
| Decode | Target at least 10% throughput gain in the declared pilot workload; publish per-case results and confidence intervals, not a universal gain claim. |
| Transfers and peaks | Record H2D/D2H/D2D volume, host/pinned/device live and retained peaks, cache/state growth and initialization overlap. Baseline permits no hard-budget claim; budgeted mode must stay within the exact user-admitted limits. |

Use 3 warm-ups and 10 measured repetitions per case, report median and p95 and all raw run identifiers.
If the speed target is not met, report feasibility without claiming a performance success; it does
not authorize changing ONNX semantics, runtime speed competition, or the user's memory budget.

### 9.6 Official checkpoint campaign admission

M5 requires a fresh export, not an assumed cached ONNX. Before authorization, approve a local
read-only HF snapshot/weight revision and file manifest, verify the official raw config hash in
section 6.4, pin M and its complete exporter dependency environment, and document the deliberate
BF16-to-FP16 conversion policy. Export text-only, standard-only, dynamic/nonpaged cache, FP32
recurrent state, no static-cache feature, no speculative state window. Use the pinned public CLI's
local source route; a proposed invocation after authorization is:

```bash
# Planned only; approved-hf-snapshot must already have independently authorized local assets.
python -m mobius build --model /tmp/ort-runtime-fusion/approved-hf-snapshot \
  /tmp/ort-runtime-fusion/official-text-standard --ep onnx-standard --dtype f16 \
  --features text-only
```

Check no contrib nodes or unexpanded local functions remain. Pin the ModelBuilder comparator at G
with `linear_attn_op=linear_attention`, nonpaged mode and `state_window=0`, recording its actual
exporter options, output state layout and all file hashes; do not invent an unverified GenAI CLI
switch or claim canonical/contrib equivalence before comparison. Compare to HF independently,
retaining historical numerical differences as separate failures/limitations.

The authorization must name GPU/device count, CUDA/ORT build, host/device peak and storage caps,
transfer allowance, maximum run time, number of sessions, checkpoint/artifact retention period,
and who deletes intermediates. Reject the campaign until those concrete limits are approved;
this document allocates nothing. Retain small digests/configs/logs and required numerical evidence,
not redundant weight/intermediate caches. Vision/full-VLM needs separate configs, budgets and
correctness gates and is not covered by a text-only success.

## 10. Approval ledger and smallest blocked probes

**No decision below is approved.** The named proposed approval coordinator for D1-D6 is
**@titaiwangms**, the issue author/assignee. This is a routing proposal, not a claim of delegated
authority or consent. The coordinator must confirm an individually named responsible ORT maintainer
for each row, with subsystem/API co-review as listed. Source inspection supplies no authoritative
individual delegation. Until ratified, **named delegated approval ownership is an attributed
governance blocker**, and this draft does not satisfy maintainer acceptance. Do not invent reviewers
or mark an issue-author acknowledgment as public-API approval.

| ID | Concrete proposed decision and rationale | Named routing owner; required approval | Status / smallest evidence gate |
|---|---|---|---|
| D1 | Explicit opt-in, L2-gated, once-only stage after partition/accounting callbacks and before outer L2. Avoid repeated discovery and preserve ownership. | @titaiwangms to confirm session/optimizer maintainer. | Proposed, unapproved; P1 ordering/ownership trace blocks M1 hook acceptance. |
| D2 | Exact-version experimental C table via existing named experimental lookup; static + explicit V2 custom-op loader; staged registration and shared library leases. Proves independent deployment without stable ABI overclaim. | @titaiwangms to confirm C API maintainer; request `@microsoft/onnxruntime-api` review. | Proposed, unapproved; P2 blocks M1 public ABI/loader delivery. |
| D3 | Declarative combined-DAG validation, preserved boundary IDs, host-only mutation, fail/discard on commit error. Avoid unsafe live rollback. | @titaiwangms to confirm graph/session and kernel-registry maintainers. | Proposed, unapproved; P3 blocks M1 transaction acceptance and whole-Scan M3. |
| D4 | FP16 dense GQA + narrow convolution; bounded whole-Scan to K-major LinearAttention separately gated. No implicit GatedDeltaNet/precision expansion. | @titaiwangms to confirm CUDA attention/control-flow and stateful-kernel maintainers, with Mobius/GenAI export co-review. | Proposed, unapproved; P1/P4 block M2; P5 blocks M3 recurrence implementation. |
| D5 | Baseline first; budgeted candidates require new target coverage and final pre-initialization admission, no automatic retry. Source sums and late declarations are insufficient. | @titaiwangms to confirm resource-accountant/session/CUDA recipe maintainers. | Proposed, unapproved; P6 blocks budgeted portion of M4, not baseline M2. |
| D6 | ORT owns generic API/hook/validation/test DSO; performance-package owner owns matcher/adapter/release matrix; no competing EP/new kernel. Proposed distribution below. | @titaiwangms to confirm ORT ownership and Mobius package/release owner. | Proposed, unapproved; owner/support nomination plus P2/P7 blocks independently supported M4 package. |

The coordinator must replace routing placeholders with confirmed named approvers/status and linked
review decisions before calling the design accepted. An accepted design can retain an explicitly
owned runtime blocker; it cannot pretend the blocked implementation contract was approved.

### Separately authorized probes

None has run. Public source inspection alone cannot resolve these questions. Use existing binaries
and tiny local random data after authorization; if missing, request a bounded build scope separately.

| Probe | Smallest work / observable result | Exact milestone blocked |
|---|---|---|
| P1: topology/partition | One Attention+cache fixture, one convolution fixture, one tiny Qwen35 text graph. Capture post-L1/post-partition node IDs, EP assignments and trace before L2; include a second optimization iteration. No official weights. | M1 exact hook ordering; M2 survival/ownership and package-update topology choice. |
| P2: independent ABI | One independently built fusion-only DSO against public headers and a separately built runtime; cloned options, incompatible API, failing registration, concurrent two-session init and unload trace. | M1 external registration and lifetime; M4 distribution support. |
| P3: transaction closure | Two overlapping flat proposals and one parent Scan/body conflict with an escaping captured value; inject Resolve failure. Assert unchanged precommit graph or discarded session, no dangling graph/reservation IDs. | M1 rewrite contract; M3 recursive destruction. |
| P4: semantics | GQA positives `(B,P,S)=(2,0,2),(2,3,1),(1,3,2)`, D=64; negative `(2,3,2)` and ragged/runtime masks; convolution `K=4,S=1,2`, nonzero history. Compare every output with independent reference at declared gates. | M2 first CUDA mapping acceptance. |
| P5: recurrence | One standard-only lowered Scan `B=1,Hk=2,Hv=4,Dk=32,Dv=64,S=7`, nonsymmetric FP32 state and FP32 recurrence inputs; trace CUDA parent/body and exact cast/head order; compare whole/chunks/single-token references. | M3 recurrence mapping **before implementation delivery**; mixed dtype/ownership is a blocker, not permission to change casts/reassign. |
| P6: preventive budget | Synthetic bounded fixture with target greater than source, missing/partial CSV, initialization scratch and a later transform. Verify target estimates exist before commit and admission precedes initializer upload; inject over-budget failure. Add actual target allocation/declaration comparison before CUDA budgeted support. | M4 budgeted integration; each uncovered kernel recipe remains blocked independently. |
| P7: unchanged artifact | Fixed tiny canonical digest manifest, A/B matcher revisions and explicit disable/unavailable cases, using the exact same files. Record accepted regions plus numerical and load-time behavior. | M4 independent-package value proof. |

No actual full-model export or cache package is assumed available. If P1 shows canonical shape
scaffolding is CPU-owned inside a proposed region, shrink to a valid boundary or report non-match;
do not add pre-partition ownership negotiation to solve the probe.

### D6 proposed ownership and support matrix

Place only generic core code and conformance fixtures in ORT (`core/session`, `core/optimizer` or a
generic graph-rewrite helper, public experimental headers and framework/shared-library tests).
Keep operator patterns out of core. Propose the optional, separately versioned performance
distribution **`mobius-ort-cuda-fusion`** under the Mobius project's ownership; this is a name/owner
proposal, not an existing package, new repository, or published artifact. It must depend only on
the public fusion headers/ABI at runtime, not require the exporter or model weights in-process.
Existing contrib schemas/kernels stay owned by their existing ORT maintainers. Package owners must
qualify their supported ORT/EP/build/device tuple and respond to topology drift.

Initial candidate platform scope: full ONNX-enabled Linux x64 and Windows x64 CUDA builds, matching
native pointer width/calling convention, explicit static or DSO registration, inference only.
The first implementation may claim only the tuples actually qualified by P2/P4/P5; this proposal
does not certify either platform. Reduced/pruned builds decline missing kernels, and builds without
native loading use static registration. Web/WASM cannot load native DSOs; a future native WebGPU or
CPU adapter needs its own registered kernels, numerical/resource recipes and conformance campaign.
No automatic package discovery and no universal cross-provider support claim.

## 11. Roadmap and design acceptance

These are planned milestones, **not created implementation children or assigned tasks**. Nothing
is approved to implement by this draft. Create separate children only after their relevant
decisions are accepted; attach actual predecessor PRs, approved API/contract revision, bounded
affected-file scope, acceptance gates and exact verification targets. Dependencies are not schedules.

| Milestone | Scope / acceptance | Predecessor and blockers |
|---|---|---|
| M1 generic mechanism | Session hook, experimental table/static+DSO registration, lifetime/validation/diagnostics, no model pattern in core. `RuntimeFusionTests.*`, `RuntimeFusionApiTest.*`, legacy registration regressions. | D1-D3 and generic part of D6 accepted; P1 ordering, P2, P3. No predecessors exist yet. |
| M2 flat CUDA pilot | Package GQA mapping, convolution contract, actual registry checks, coexistence with existing contrib fusions. Tiny fixtures, negative masks/dtypes, all outputs/states and unchanged-artifact topology probe. | Actual M1 PR and approved D4 flat-region contract; P1/P4. No recurrence approval implied. |
| M3 stateful/whole-Scan | Recursive read-only inspection, core-derived closure, body accounting retirement, narrow LinearAttention sequence replacement and asymmetric-state tests. Convolution state qualification extended here. | Actual M1/M2 PRs, D3/D4 whole-Scan acceptance and P5 before sequence implementation; no wider GatedDeltaNet approval. |
| M4 reduced generation/package/resources | Random-weight text decoder, generated-token feedback, A/B immutable-artifact proof, distribution matrix and optional budgeted recipes. | Actual M1-M3 PRs, D6 release owner, P7. Budgeted portion separately blocked on D5/P6 and each target recipe. Baseline qualification can proceed without general planner #33372. |
| M5 official 27B campaign | New pinned text-only export, independent numerical comparison, coverage/performance/resource report with retained provenance. Vision/full-VLM is a separate qualification. | Actual M4 PR/report; explicit weight/export/device/build/retention budget authorization. Historical Mobius #758 is context, not a substitute. |

Future planner integration is outside M1-M5 and is not a prerequisite for the fixed-placement
pilot. After the relevant fusion and placement contracts are accepted, a separate adapter milestone
must qualify query isolation, tentative-assignment/revision invalidation, and a case where the
unfused estimate would trigger offload but an eligible fused recipe fits. It must verify the
selected rewrite and reconciled memory plan before final ownership commitment and device
materialization; no runtime planning or qualification is authorized by this roadmap.

### Acceptance record

- This PR supplies a reviewable design, not a qualification receipt or a maintainer approval.
- Required for design acceptance: responsible ORT maintainer review, confirmed named D1-D6 approval
  routing/status with rationale, and explicitly attributed blockers/probes tied to milestones.
- Runtime tests are **not** required to close the design issue, but their absence must remain
  explicit. A later implementation cannot substitute a planned command, skip, source registration
  or historical report for executed evidence.
- Design acceptance does not authorize unresolved API/lifetime, whole-Scan transaction,
  resource-owner contracts, production coding, new implementation issues or full-model execution.
