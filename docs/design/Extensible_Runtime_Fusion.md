# EP-registered runtime fusion for canonical ONNX

Status: **engineering design for review, not approved for implementation** (revised October 26, 2026).
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

### Revised integration direction

Fusion patterns register on the EP side. Each EP owns their priority order and exposes eligible
execution candidates through the capability path. ORT chooses compatible candidates in that order,
subject to ownership, semantic, kernel, and, when enabled, memory-feasibility checks. There is no
per-session speed competition and no separate post-partition package-selection pipeline.

Separate candidate description from application so memory estimation can be connected without
rerunning a mutating matcher. Reuse `GetCapability`, `ComputeCapability`, and the capability
optimization application model first; define only the missing adapters. Independent package delivery
remains a goal, not a reason to freeze a second graph-inspection ABI or require a competing EP.
These are the proposed design direction, not a claim of maintainer approval or existing support.

## 2. Evidence ledger and baseline discipline

All runtime claims below refer to **R**, unless a difference is explicitly noted. Links are immutable
research references, not claims that proposed interfaces exist.

| Key | Exact source |
|---|---|
| R | [ORT main `1988019b6e23ac4def4628cddd603c1bfb504fe8`](https://github.com/microsoft/onnxruntime/tree/1988019b6e23ac4def4628cddd603c1bfb504fe8) |
| M | [Mobius main `b1e9c8e9dec1dfa206183f957e77d07cfb734df2`](https://github.com/onnxruntime/mobius/tree/b1e9c8e9dec1dfa206183f957e77d07cfb734df2) |
| G | [GenAI main `991e3560e63b8a26806d4ec7d702c04e75fb6e51`](https://github.com/microsoft/onnxruntime-genai/tree/991e3560e63b8a26806d4ec7d702c04e75fb6e51) |
| F | [Fusion proposal `587ef1bf3467935b083740a52ac8fcba3785b2a6`](https://github.com/microsoft/onnxruntime/blob/587ef1bf3467935b083740a52ac8fcba3785b2a6/docs/design/Extensible_Runtime_Fusion.md) |
| P | [Placement proposal `2f96ad081fdbe1d76a2a731262a998b3cd4c9c0d`](https://github.com/microsoft/onnxruntime/blob/2f96ad081fdbe1d76a2a731262a998b3cd4c9c0d/docs/annotated_partitioning/PartitioningWithAnnotationsAndMemoryConstraints.md), refreshed October 26 for the proposed recipe/planning integration only |
| W | [Composable rewrite rules, gramalingam/onnxruntime#16 at `6d707086e04fe287f162b9dc2baf58b935d8e2bc`](https://github.com/gramalingam/onnxruntime/blob/6d707086e04fe287f162b9dc2baf58b935d8e2bc/docs/Pattern_Rewriter.md); optional matcher implementation, not an ORT-main dependency |
| V | [OpenVINO plugin `7463f66e48d09fe5150af8c76520fbf75e08c490`](https://github.com/onnxruntime/onnxruntime-ep-openvino/tree/7463f66e48d09fe5150af8c76520fbf75e08c490); compiler-EP comparison only |

The PR base is default-branch commit `327d11ab6c22e387acf793dd9de8caf406055ca9`.
Source comparison with R found no changes in the cited session hook, public API, registry selection,
Scan registration, or GQA/LinearAttention/GatedDeltaNet kernel files. Relevant differences are the
accountant's new `INT32_MAX` validation for the CUDA GQA total-sequence bound and updates in the GQA
workspace bounds/estimate helpers. These do not supply target re-estimation for arbitrary rewrites.
Do not combine P's proposed reconciliation with R and describe the result as implemented.
Any implementation PR must repeat the comparison against its actual base and publish its build pin.
The October 26 revision also inspected the PR head's `execution_provider.h:152-179`,
`compute_capability.h:27-44`, and `graph_partitioner.cc:1145-1181` for the capability integration below.
Refreshing P and W does not refresh R or turn either proposal into shipped runtime functionality.
V's `CreateEpFactories`/`ReleaseEpFactory` lifetime patterns are useful references. Its
GetCapability/Compile and AddNodesToFuse path owns compiled regions; it is not a public extension
that borrows existing CUDA kernels. Creating a competing EP solely to reach those kernels is rejected.

### Current behavior versus missing mechanism

| Evidence at R | Implication |
|---|---|
| [`inference_session.cc:1590-1602,1896-1952,2015-2087`](https://github.com/microsoft/onnxruntime/blob/1988019b6e23ac4def4628cddd603c1bfb504fe8/onnxruntime/core/session/inference_session.cc#L1896-L1952) | L1 precedes partition; mutation accounting is installed after partition; L2+ and casts repeat; copies follow. |
| [`graph_transformer_utils.cc:413-426,464-467`](https://github.com/microsoft/onnxruntime/blob/1988019b6e23ac4def4628cddd603c1bfb504fe8/onnxruntime/core/optimizer/graph_transformer_utils.cc#L413-L426), [`graph_transformer_mgr.cc:38-59`](https://github.com/microsoft/onnxruntime/blob/1988019b6e23ac4def4628cddd603c1bfb504fe8/onnxruntime/core/optimizer/graph_transformer_mgr.cc#L38-L59) | Repeated optimization loops must not independently reselect a capability's implementation. |
| [`execution_provider.h:152-179`](https://github.com/microsoft/onnxruntime/blob/3207aa12931ac9de054fdf0de183f7f036c01269/include/onnxruntime/core/framework/execution_provider.h#L152-L179) | `GetCapability` can return overlapping source-node sets; ORT decides assignment. The signature includes optimizer-registry and resource-accountant inputs. |
| [`compute_capability.h:27-44`](https://github.com/microsoft/onnxruntime/blob/3207aa12931ac9de054fdf0de183f7f036c01269/onnxruntime/core/framework/compute_capability.h#L27-L44), [`graph_partitioner.cc:1145-1181`](https://github.com/microsoft/onnxruntime/blob/3207aa12931ac9de054fdf0de183f7f036c01269/onnxruntime/core/framework/graph_partitioner.cc#L1145-L1181) | Child optimization callbacks can update an available parent capability before placement/fusion. This is an application hook, not a complete alternative-recipe planner. |
| [`group_query_attention_fusion.cc:413-440`](https://github.com/microsoft/onnxruntime/blob/1988019b6e23ac4def4628cddd603c1bfb504fe8/onnxruntime/core/optimizer/group_query_attention_fusion.cc#L413-L440) | Existing GQA fusion starts from contrib GQA, not canonical Attention. |
| [`attention_fusion.cc:1030-1046`](https://github.com/microsoft/onnxruntime/blob/1988019b6e23ac4def4628cddd603c1bfb504fe8/onnxruntime/core/optimizer/attention_fusion.cc#L1030-L1046), [`session_state.cc:242-310`](https://github.com/microsoft/onnxruntime/blob/1988019b6e23ac4def4628cddd603c1bfb504fe8/onnxruntime/core/framework/session_state.cc#L242-L310) | Rewrites can retain EP assignment; finalization later binds registered kernels. |
| [`inference_session.cc:1113-1129`](https://github.com/microsoft/onnxruntime/blob/1988019b6e23ac4def4628cddd603c1bfb504fe8/onnxruntime/core/session/inference_session.cc#L1113-L1129) | Private C++ registration is not an external binary contract. Preserve its existing enable behavior. |
| [`onnxruntime_c_api.h:6130-6653,8210-8244`](https://github.com/microsoft/onnxruntime/blob/1988019b6e23ac4def4628cddd603c1bfb504fe8/include/onnxruntime/core/session/onnxruntime_c_api.h#L6130-L6653) | Reuse public `OrtGraph`/`OrtNode`/`OrtValueInfo` inspection where possible; Model Editor alone does not supply the capability recipe/application contract. |
| [`onnxruntime_experimental_c_api.inc:21-37`](https://github.com/microsoft/onnxruntime/blob/1988019b6e23ac4def4628cddd603c1bfb504fe8/include/onnxruntime/core/session/onnxruntime_experimental_c_api.inc#L21-L37) | Experimental discovery exists, but no independently deployed EP-pattern registration interface is established here. |
| [`kernel_registry_manager.cc:60-104`](https://github.com/microsoft/onnxruntime/blob/1988019b6e23ac4def4628cddd603c1bfb504fe8/onnxruntime/core/framework/kernel_registry_manager.cc#L60-L104) | Custom registries have precedence; an operator name alone proves neither implementation nor eligibility. |

## 3. Integration and graph lifecycle (D1)

### EP-owned registration and capability discovery

Each EP owns a registry of enabled fusion patterns and an explicit ordered pattern list. A pattern
can be EP-bundled or supplied by an optional independently updated package, but both enter the same
registry and selection path. EP options express permission and priority, not proof of eligibility.
Keep existing cross-EP priority; this design does not let an external package claim nodes on behalf
of an unrelated provider.

During `GetCapability`, the EP matches the normalized graph and describes executable candidates.
A candidate identifies a source-node set and a prepared implementation recipe, not a second
complete model that ORT must execute to compare performance. Different patterns may cover the same
source nodes or emit different replacement DAGs. Standalone execution remains a candidate where
supported; a region executable only after fusion need not first have standalone target-EP kernels.

Use the existing `ComputeCapability` application model, including `nodes_to_optimize` and an
ORT-controlled `optimization_func` adapter, rather than add an unrelated post-partition callback.
Existing registry optimizers can be shared through `GraphOptimizerRegistry`. Neither `Graph&`
nor `std::function` crosses a future external ABI.

**This is reuse, not an assertion that the current API is sufficient unchanged.** The current
partitioner checks availability, runs child optimizations, and places/fuses the updated parent.
Explicit candidate ordering, nonmutating recipe descriptions, arbitrary replacement-DAG capability
updates/splitting, graph-revision validation, and external registration still need supported adapters.
The first structural probe must verify that emitted adapters and existing contrib nodes receive
normal kernel registration/finalization without incorrectly becoming one compiled synthetic node.

### Proposed lifecycle and ordering

The logical lifecycle is:

```text
canonical load -> shared preprocessing / L0-L1
 -> EP GetCapability: describe candidates from EP-ordered patterns
 -> ORT checks availability, eligibility, and ordered conflicts
 -> [budgeted integration: query recipe requirements before final selection]
 -> provisionally select compatible candidates
 -> apply selected recipes through capability optimization adapters
 -> reconcile graph, capabilities, kernels, and resource information
 -> finalize placement -> kernel creation / PrePack -> Run
```

Within an EP, sort candidates first by the EP's pattern-list ordinal, then by stable graph traversal,
anchor node ID, and candidate ID. Package registration/load order must not override this list.
Process valid candidates greedily in that order; an accepted candidate excludes overlapping ones,
including parent whole-Scan versus descendant rewrites. A declined candidate does not claim nodes
or consume reservations, so a lower-priority feasible candidate can still be considered.
Priority never overrides schemas, semantics, ownership, kernel availability, or memory feasibility.
There is no original-versus-fused speed tournament, online benchmark, or optimal-combination search.

Pattern selection requires explicit enablement and optimization level at least L2. Absent, disabled,
and Basic-only configurations preserve ordinary behavior. Required-pattern policy at Basic fails
configuration validation. Initial scope is ordinary ONNX inference initialization; ORT-format
loading, compile-only sessions, training, online repartition, and optimized-cache replay need
separate support. Optional use on those paths is diagnosed as not applicable; required use fails.

`GetCapability` can be called again as graphs/providers/layouts change. Discovery must therefore be
read-only and repeatable, not globally once-only. Key prepared candidates by graph revision,
ownership view, EP instance, effective configuration and workload bounds. Do not reuse stale matches
or double-charge resources on another pass. Apply only a selected candidate valid for that revision;
its application callback must execute the chosen recipe, not rerun matching to select another one.
Install the necessary graph/accounting update support before application; do not depend solely on
callbacks currently installed after partition.

Publish positive and near-match fixtures at capability discovery, not only untouched exporter
topology. Preserve unrelated private transformer behavior. Later transforms may further optimize
emitted nodes, but must not silently invalidate selected kernel/state/resource contracts.

### Memory estimation and placement planning

Baseline delivery can first exercise ordered capability selection without claiming a memory bound.
The next integration connects target-specific estimators to those same recipes, not a separate
post-partition fusion pipeline. A budgeted selector evaluates memory feasibility before accepting
a candidate, then considers the next EP-ordered candidate if the higher-priority one is infeasible.
The baseline order-only selector does not need a full CPU/GPU planner to demonstrate fusion.

For the future planner in [#33372](https://github.com/microsoft/onnxruntime/pull/33372), supply tentative
EP ownership without final assignment. Describe logical graph effects and allocation requirements
before selecting placement; the fused implementation may fit where unfused source estimates do not.
Queries must not mutate the live graph, construct kernels, upload weights, or materialize trial
device resources. Apply a provisional winner on a candidate graph, reconcile transfers/casts and the
execution/backing plan, verify, then commit final placement before materialization.
Placement-sensitive L2+ passes must participate as selected recipes or be gated in this mode.
This planner is proposed follow-on work, not current ORT behavior or a prerequisite for baseline.

## 4. Recipe contracts and independent delivery (D2)

### Logical descriptors, not a frozen ABI

Define the capability integration before fixing external function tables. The previous draft's
parallel `OrtRF*` inspection/registration table is not the selected interface for this revision.
Reuse `OrtGraph`, `OrtNode`, and `OrtValueInfo` for external read-only inspection where applicable,
adding only missing revision, EP-instance, workload, and bounded-constant-access contracts.
No new public API names, binary layouts, experimental factory, or loading entry point are approved
by this document. The first prototype can use internal adapters, but cannot claim independent
binary deployment on that basis.

The following describes required information, not new C/C++ structs:

| Descriptor | Required information |
|---|---|
| EP pattern registration | Pattern/contract ID and revision, implementation/package identity and revision, enablement, explicit EP-owned priority ordinal, retained callback state. |
| Candidate context | Graph/revision, source-node IDs, anchor, target EP instance and current/tentative ownership view, immutable effective session configuration and workload bounds. |
| Prepared recipe | Candidate ID, complete consumption closure, boundary inputs/outputs/captures, logical replacement DAG or compiler-private implementation description, attributes/types/opsets/constants, dependencies, and retained application state. |
| Requirement query | Selected recipe/kernel/build/device, dtype and workload/alias/concurrency bounds; target-owned requirements with an explicit unknown result, not package-asserted byte counts. |
| Selection/application | Exact candidate/recipe identity and revision, selected contracts, validated boundary mapping, and an application handle that cannot substitute another implementation. |

ORT must bound candidate count, descriptor storage, and constant reads; recommended initial caps
remain 8,192 candidates, 64 MiB descriptor storage, and 1 MiB new/read constant payload.
Metadata inspection must not materialize 27B weights. Existing initializers can be referenced without
copying payloads; new constants become ORT-owned. Preserve shared storage and every observable
boundary value identity. A consumed CPU compute node is not made eligible by a CPU-resident input.

Freeze effective configuration and the ordered registry for each session; another session may use
a different order or workload envelope without sharing mutable candidate state. Reject conflicting
duplicate pattern identities rather than let package loading silently replace priority.
Required-pattern policy means at least one accepted candidate of that qualified pattern, not a
promise that every occurrence fuses. A candidate discovered or retained is not yet selected.

### Delivery, ownership, and lifetime requirements

EP-bundled patterns and optional independent packages must obey the same recipe/selection model.
An external package registers into the target EP's registry; it does not become a competing EP,
claim nodes directly, or establish priority by load order. No exporter or model weights are required
in the runtime package. EP-owned registration does not require matcher updates to ship with the EP.

Before external delivery, specify and qualify a version-negotiated registration/recipe adapter,
including which existing C/EP APIs it reuses and which minimal additions are necessary. Do not expose
`Graph&`, STL, protobuf objects, provider pointers, CUDA streams, or kernel factories across that
boundary. Compiler-private lowering and host-visible rewrites must normalize to the same selected
recipe contract; compiler EPs need not adopt the pilot to continue working.

Retain the following requirements when defining that adapter:

- Application-controlled static registration and explicit library loading; no directory scanning or
  executable paths taken from model metadata. Native packages are trusted application code.
- Atomic registration and cleanup on failure; copied descriptors and explicit retain/release rules
  for package, candidate, and recipe state. Graph inspection handles are callback-scoped borrows;
  retained recipes must not contain raw pointers that survive a graph revision.
- SessionOptions cloning/configuration isolation and immutable per-session registry snapshots.
  Release candidate/session state before package callbacks or libraries can be unloaded.
- Serialized, non-reentrant callbacks within a session and safe concurrent initialization across
  sessions. No asynchronous callback outlives its call; shared immutable registrations are thread-safe.
- No exceptions crossing a C boundary, explicit `OrtStatus` ownership, unchanged out-parameters on
  failure, and release callbacks that cannot fail.
- Explicit errors for incompatible versions, failed loads/registration, malformed recipes, and
  callback errors. Optional absence is an application choice, not permission to swallow errors.
  Missing optional patterns leave ordinary execution; required unavailability fails.

Independent DSO conformance and a supported ORT/EP/build/device/package matrix are delivery gates,
not prerequisites to choosing the capability architecture. Exact public signatures remain a
follow-on approval item; this revision does not promise indefinite cross-release ABI stability.

### Workload eligibility

Workload bounds are application-declared in both resource modes. Validate `0 <= min <= max`,
input names, uniqueness, and ranks before discovery. Matchers and requirement queries see the same
immutable validated session view, not mutable options or another session's shared package state.
The eventual implementation must reject out-of-envelope feeds/bindings before any kernel launches.
Graph symbolic dimensions alone are not bounds and missing bounds remain unknown.

Eligibility must hold for the entire fixed-shape or bounded envelope. No package Run callback,
dynamic match/defuse, or data-dependent fallback is added. For example, a session allowing
`B=2,P>0,S>1` cannot receive GQA just because its first call is prefill. Use B=1, or separate
B=2 prefill (`P=0`) and decode (`S=1`) envelopes with the same canonical file.
Mask contents and arbitrary length-tensor values cannot be certified by shape bounds.

### Matcher choice is a demo-stage decision

W's composable pattern engine is an optional implementation, not the system interface. Its
standalone `Pattern::MatchAt`/`VisitMatches` can inspect matches without applying a rule; standalone
matching does not enforce the rule engine's EP/opset policy. `Replacement::Instantiate` constructs
a fragment, not a complete boundary-preserving replacement transaction. Those components may be
adapted behind the EP registry, but do not supply memory admission or an external ABI.

Do not make adopting that DSL or merging W a system-design prerequisite. During the Qwen demo,
first use small exported Attention/convolution/Scan fixtures to measure pattern expressiveness,
adapter effort, topology survival, and matching overhead; then validate the full model.
A different matcher must produce the same recipe/selection contract. Demo results decide practical
adoption, not whether the architecture needs a second optimizer pipeline.

## 5. Host validation and commitment (D3)

1. Freeze the session's EP registry/order and configuration. Discover bounded candidate descriptions
   without graph mutation or resource charges; retain them only for their graph/ownership revision.
   Enforce count/byte caps and unique candidate identities within that revision.
2. Validate IDs, current revision, graph scope, acyclicity, schemas/import versions, type/shape
   compatibility, constants, and every use crossing the region boundary. Preserve all escaping
   outputs, including graph outputs and descendant captures. Missing observable state is not dead
   just because it has no ordinary local consumer. Do not raise an existing domain import and
   reinterpret surviving nodes; reject incompatible requests. Adding an otherwise absent target
   domain import is allowed only after validating all emitted schemas.
3. Require the complete consumed executable region to be available for the target EP instance.
   Unassigned nodes can be claimed if the proposed implementation supports them; final assignment
   of every source node is not a discovery prerequisite. A tentative ownership view can establish
   same-EP eligibility. Never steal work committed to another EP, enter compiler-owned opaque
   regions, or use host-resident constants as permission to consume CPU-assigned compute.
4. Resolve replacement kernels through the **actual session registry precedence**, without kernel
   construction. Validate schema/type constraints, build pruning, host-input requirements, device
   capabilities, and package-qualified operator contract. V1 declines any custom-registry override
   of an emitted kernel, even one with the same op name; it does not bypass normal lookup. Core
   records the selected built-in registration privately and checks it again at finalization.
5. In budgeted mode, obtain target-specific coverage before selection/commit (section 8).
6. Preserve ordinary EP priority, then process candidates in the EP-owned pattern order and stable
   tie-break order from section 3. Choose compatible, non-overlapping feasible candidates. Consumption
   closure, including Scan descendants, defines overlap. Rejected candidates do not reserve nodes
   or bytes. Lower-priority alternatives are available before application, not as recovery from
   an already failed mutation. No per-session benchmark or cost tournament is added.
7. Validate the accepted **combined** replacement against a disposable graph representation before
   touching the live graph. This catches cycles and boundaries introduced by interactions between
   individually valid recipes. Through a validated capability adapter, ORT materializes selected
   nodes, updates/splits the parent capability as needed, and preserves boundary values. New compute
   nodes must have target kernels; host adapters are explicit and checked, not hidden callbacks.
8. Apply only the prepared recipes, update capabilities/identities/accounting, and Resolve. Reconcile
   the actual implementations before final placement/resource commitment. Unexpected application or
   Resolve failure aborts baseline initialization; the pilot promises no live-graph rollback or Run
   retry. A future planner may reject an isolated candidate graph under P's separate contract, but
   cannot continue with a partially mutated live graph or recurrent state.

Unsupported kernels, masks, ownership, or resource coverage detected before commitment leave the
original region intact with a reason, unless a required pattern cannot be satisfied. If neither a
selected recipe nor ordinary execution covers the region, initialization fails; leaving it unchanged
does not manufacture missing kernel support.
Distinguish `NO_MATCH`, `EP_MISMATCH`, `OPAQUE_REGION`, `UNSUPPORTED_MASK/STATE/TYPE`,
`KERNEL_MISSING/OVERRIDDEN`, `RESOURCE_UNKNOWN/INSUFFICIENT`, `OVERLAP`, `STALE_SNAPSHOT`,
`INVALID_PROPOSAL`, and `COMMIT_FAILED`. Stale/malformed proposals and callback failures are errors,
not silent non-matches.

Diagnostics identify artifact digest, package/revision, pattern contract, graph/revision, consumed
and emitted nodes, EP pattern priority, target instance diagnostic ID, selected registration,
rejection reason, and later transformations. Default logging includes no tensor payload, library
search path, or weight data. An opt-in trace records timing and resource provenance.

## 6. Pilot source-to-target contracts (D4)

The mappings below are package pattern contracts, not model names in the core API. They require
structural and semantic qualification; registration of a kernel establishes feasibility only.
Canonical ONNX inputs/outputs remain unchanged, including all cache and recurrent state values.
Adapters are ordinary existing operations in the replacement DAG, not hidden package execution.
These are demo/implementation qualification gates, not prerequisites to selecting the capability
architecture or a particular matcher DSL.

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
host during discovery or infer runtime mask contents from a symbolic shape. A graph input mask that
happens to contain ones in a test is not structurally proven redundant. V1 declines arbitrary
runtime mask inputs; supporting a checked runtime predicate before any state mutation would require
a separately approved adapter/error contract. A workload promise alone must not erase mask semantics.

For padding, map a standard prefix-valid mask to `L_b` only under a **separate qualified contract**
that proves query alignment, cache padding, output values for padded queries, and per-row lengths.
V1 rejects different per-row valid lengths, left/interior padding, arbitrary additive bias,
noncausal mode, unknown mask provenance, and fully masked rows. Finite large-negative values are
not automatically equivalent to `-infinity`. Mask conversion must preserve original error/bounds
semantics; `Cast(INT64->INT32)` alone is not range validation. Unproven cases remain unfused.

The smallest blocker is a tiny capability-discovery Attention fixture with `P>0,S>1`, plus a ragged
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

Add only a bounded whole-Scan recipe: the candidate consumes the entire Scan node in its enclosing graph,
plus explicitly listed flat adapters if eligible. Core, not the package, derives its recursive
executable consumption closure. No general nested-graph editing API or replacement graph attribute
is introduced. Inspect the body read-only and require:

- Forward scan input/output directions, proven scan axes, matching carried-state count/order,
  no early termination, and exactly the supported recurrence.
- Complete outer captures and invariant weights/gates; no mutation or unaccounted side effects.
- Availability for CUDA of the parent Scan and every consumed executable descendant, same instance.
  Unassigned source work may qualify through the whole recipe. Reject committed CPU ownership,
  mixed tentative ownership, compiler-owned body, and nested control flow. Verify final target
  assignment rather than treating a discovery-time provider name as proof.
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
   check the combined plan. Unknown/underreserved candidates are declined without charges; consider
   lower-priority alternatives in EP order, including standalone execution where supported and
   covered. An unmet required pattern fails. Do not alter the user's EP budget.
2. After **all** L2+/cast/copy transformations, rebuild or revalidate coverage for the final graph,
   including unaffected source nodes and every inserted adapter/copy. If coverage is invalidated
   or no longer fits, do not commit placement/materialize resources. Baseline application has no
   automatic unfused retry after mutation; a future planner can discard an isolated candidate under
   its own bounded search contract.
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
may allocate before final admission: any such allocations must already be admitted under the
existing resource owner, or that configuration cannot claim this hard-bound mode.
The same rule applies to allocation-bearing DSO static initialization or registration: prohibit
device staging there in the supported package contract, or admit/charge it before loading.

Narrow adapters to existing estimators are allowed follow-on work. The general planner in
[#33372](https://github.com/microsoft/onnxruntime/issues/33372) and P's reconciliation are future
dependencies, not current guarantees or prerequisites for baseline fusion. Memory determines
feasibility, not default speed preference. If a later authorized placement workflow splits a fused
region, invalidate/rebuild affected fusion/resource data or reject that workflow; automatic defusion
search is deferred.

`GetCapability` already accepts a resource-accountant input, but that does not establish target
estimates or noncharging exploration for every recipe. Add a query-only estimator/resource-owner
adapter and separate speculative queries from accepted reservations. Repeated capability discovery
must not double-charge source or target nodes; source consolidation alone is not target estimation.
This adapter blocks budgeted use, not baseline ordered capability selection.

Baseline/budgeted are resource modes, not separate optimization-selection pipelines. Baseline
selects eligible candidates by EP order without a hard memory promise. Budgeted integration adds
target feasibility to that same selection before final acceptance. General CPU/GPU boundary search
is P's follow-on responsibility; this design does not claim an optimal placement or current
ORT-main support for the proposed planner.

## 9. Qualification plan, not execution evidence

### 9.1 Generic mechanism and API

Create the following tests after contract approval; the `RuntimeFusion*` suites do **not** exist
at R. These are logical coverage targets, not a commitment to the removed C table. A skip/missing
binary, fixture, CUDA EP, or DSO required by the selected delivery milestone is a blocked
prerequisite, never a pass.

| Suite / target | Required assertions |
|---|---|
| `RuntimeFusionTests.*` / `onnxruntime_test_all` | EP registry/order, absent/disabled/Basic behavior, required-at-Basic error. Repeat discovery after graph/layout changes; assert no mutation/reservation during queries and no duplicate application. Preserve ordinary EP priority. |
| `RuntimeFusionTests.OrderedCandidates*` | Overlapping high/low-priority patterns, deterministic same-pattern anchors, package load-order independence, rejected candidate leaving lower-priority candidates available, and emitted multi-node DAG/capability updates. No benchmark selection. |
| `RuntimeFusionTests.*` | EP-split, opaque regions, stale revision/ownership views, escaping outputs/captures, missing schema, invalid combined DAG, failed Resolve, callback errors and candidate/byte limits. Assert prepared recipe is applied without reselecting it; baseline mutation failure discards initialization. |
| `RuntimeFusionTests.SessionConfigBounds` / `onnxruntime_test_all` | Discovery sees validated per-session bounds and EP order. Cloned options/concurrent sessions cannot alter another session's registry or workload snapshot. Invalid bounds fail before discovery. |
| `RuntimeFusionApiTest.*` / `onnxruntime_shared_lib_test`, external-delivery milestone | Once the adapter is approved: static/DSO registration, version/descriptor errors, copied/retained state, clone isolation, partial registration cleanup, concurrent sessions and destructor/unload order. Build DSO with only approved public headers, not private symbols. |
| `RuntimeFusionTests.WholeScan*` | Unassigned or same-tentative-CUDA parent/body positive; committed CPU or mixed ownership negative; capture/state order, nested control flow, recursive overlap, descendant resource retirement, no Scan after acceptance. |
| `RuntimeFusionTests.Budget*` | Unknown/insufficient high-priority recipe allows a covered lower-priority candidate without stale charges. Cover target larger than source, partial CSV, shared initializer, initialization peak, repeat discovery, later-transform invalidation, unchanged budget, and out-of-bound Run rejection before mutation. |
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
`B=2,P=[1,17],S=[2,7]`. Each discovery callback must observe its own validated session bounds;
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

The October 26 design direction is EP-side registration and EP-owned pattern order. This is not
maintainer approval of the adapters or their implementation. The named proposed approval coordinator
for D1-D6 is
**@titaiwangms**, the issue author/assignee. This is a routing proposal, not a claim of delegated
authority or consent. The coordinator must confirm an individually named responsible ORT maintainer
for each row, with subsystem/API co-review as listed. Source inspection supplies no authoritative
individual delegation. Until ratified, **named delegated approval ownership is an attributed
governance blocker**, and this draft does not satisfy maintainer acceptance. Do not invent reviewers
or mark an issue-author acknowledgment as public-API approval.

| ID | Concrete proposed decision and rationale | Named routing owner; required approval | Status / smallest evidence gate |
|---|---|---|---|
| D1 | Register/order patterns on the EP side; discover through `GetCapability` and reuse capability application. ORT resolves conflicts by EP order, not package load order or speed competition. | @titaiwangms to confirm partitioner/EP maintainer. | Revised direction; P1 must validate ordering, repeated discovery and multi-node replacement support before M1 acceptance. |
| D2 | Reuse existing graph inspection; define only missing capability/recipe registration adapters. Do not freeze the former parallel C table. | @titaiwangms to confirm C/EP API maintainer; request `@microsoft/onnxruntime-api` review before any public additions. | Public ABI deferred; P2 gates external delivery, not an internal M1 prototype. |
| D3 | Describe/select/apply separately; validate combined effects and boundaries; update capabilities/accounting and reconcile before final commitment. No baseline live rollback. | @titaiwangms to confirm graph/partitioner and kernel-registry maintainers. | Proposed, unapproved; P3 blocks M1 application acceptance and whole-Scan M3. |
| D4 | Qualify FP16 GQA/convolution and separately whole-Scan LinearAttention in the model demo. Grama's engine is optional, not a system-design dependency. | @titaiwangms to confirm CUDA/stateful-kernel maintainers, with Mobius/GenAI co-review. | Demo-stage gates P4/P5; matcher fitness does not block the architecture decision. |
| D5 | Baseline ordered selection first; target estimates add feasibility before selection in budgeted mode. Unknown is not zero; admit before materialization. | @titaiwangms to confirm resource-accountant/session/CUDA recipe maintainers. | Proposed, unapproved; P6 gates budgeted M4, not baseline M1/M2. |
| D6 | EP owns registry/order; ORT owns selection/validated application; optional package owns matcher/adapter release. One recipe model for bundled and external patterns. | @titaiwangms to confirm ORT/EP ownership and Mobius package/release owner. | Proposed, unapproved; P2/P7 plus support ownership gate independent M4 delivery. |

The coordinator must replace routing placeholders with confirmed named approvers/status and linked
review decisions before calling the design accepted. An accepted design can retain an explicitly
owned runtime blocker; it cannot pretend the blocked implementation contract was approved.

### Separately authorized probes

None has run. Public source inspection alone cannot resolve these questions. Use existing binaries
and tiny local random data after authorization; if missing, request a bounded build scope separately.

| Probe | Smallest work / observable result | Exact milestone blocked |
|---|---|---|
| P1: capability integration | Tiny flat fixture with two EP-ordered overlapping patterns and a multi-node replacement; repeat discovery after a revision/layout change. Trace returned/selected capabilities, registered target kernels, no duplicate application or speculative charges. Use Attention/convolution exports for M2, no official weights. | M1 ordering/application adapters; M2 topology survival and package-update choice. |
| P2: independent ABI | After adapter approval, independently built fusion-only DSO and runtime; EP-side registration, order unchanged by load order, clones, version mismatch, failing registration, concurrent init and unload trace. | M4 external delivery; not a gate for an internal M1 prototype. |
| P3: application closure | Two overlapping flat recipes and parent Scan/body conflict with an escaping capture; assert selected identity, boundary preservation and parent capability updates. Inject stale revision and Resolve failure; assert no dangling graph/reservation IDs or continuation of partial initialization. | M1 application contract; M3 recursive destruction. |
| P4: semantics | GQA positives `(B,P,S)=(2,0,2),(2,3,1),(1,3,2)`, D=64; negative `(2,3,2)` and ragged/runtime masks; convolution `K=4,S=1,2`, nonzero history. Compare every output with independent reference at declared gates. | M2 first CUDA mapping acceptance. |
| P5: recurrence | One standard-only lowered Scan `B=1,Hk=2,Hv=4,Dk=32,Dv=64,S=7`, nonsymmetric FP32 state and FP32 recurrence inputs; trace CUDA parent/body and exact cast/head order; compare whole/chunks/single-token references. | M3 recurrence mapping **before implementation delivery**; mixed dtype/ownership is a blocker, not permission to change casts/reassign. |
| P6: preventive budget | High-priority recipe exceeding budget and lower-priority fitting recipe; unknown estimate, repeat queries, partial CSV, shared storage and initialization scratch. Verify selection/charges, final reconciliation and admission before upload; compare target declarations/allocations. | M4 budgeted integration; each uncovered target recipe remains blocked. |
| P7: unchanged artifact | Fixed tiny canonical digest manifest, EP-bundled and external A/B matcher registration with the same explicit EP order, disable/unavailable cases. Record coverage, numerics and load behavior without model changes. | M4 independent-package value proof; W suitability evaluated here, not required. |

No actual full-model export or cache package is assumed available. If P1 shows shape scaffolding
unavailable to the target EP, shrink the boundary or report non-match; do not consume CPU-committed
compute to force eligibility. An unassigned node and a node already committed to another EP are
different cases.

### D6 proposed ownership and support matrix

Place generic selection/application adapters and conformance fixtures in ORT (`core/framework`,
`core/optimizer`, and framework/shared-library tests). EPs own registration, priority and support
validation; patterns may reuse shared optimizer implementations without duplicating machinery.
Keep model-specific pattern knowledge out of the generic planner. Propose the optional, separately
versioned performance
distribution **`mobius-ort-cuda-fusion`** under the Mobius project's ownership; this is a name/owner
proposal, not an existing package, new repository, or published artifact. External delivery must use
the approved EP registration/recipe boundary, not private runtime symbols or exporter/model weights.
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
| M1 capability mechanism | EP-owned ordered registry, repeatable discovery, prepared recipes and validated capability application adapters. Internal prototype permitted; no frozen external ABI or model-specific matcher dependency. `RuntimeFusionTests.*` and existing capability regressions. | D1/D3 and generic D6 approval; P1/P3. D2 external API and P2 are not internal prototype gates. |
| M2 flat CUDA demo | Register GQA/convolution patterns in the EP; verify kernels, topology, negative masks/dtypes and all outputs/state. Evaluate W on small exported fixtures without making adoption mandatory. | Actual M1 PR and D4 flat-region contract; P1/P4. No recurrence approval implied. |
| M3 stateful/whole-Scan | Recursive read-only inspection, core-derived closure, body accounting retirement, narrow LinearAttention sequence replacement and asymmetric-state tests. Convolution state qualification extended here. | Actual M1/M2 PRs, D3/D4 whole-Scan acceptance and P5 before sequence implementation; no wider GatedDeltaNet approval. |
| M4 reduced generation/package/resources | Random-weight text decoder, feedback generation, independent EP-side package registration and A/B artifact proof. Approve/qualify minimal external adapter; separately connect recipe estimators and budget feasibility. | Actual M1-M3 PRs; D2/D6 and P2/P7 for external release. Budgeted part needs D5/P6 and each target recipe. No general planner prerequisite. |
| M5 official 27B campaign | New pinned text-only export, independent numerical comparison, coverage/performance/resource report with retained provenance. Vision/full-VLM is a separate qualification. | Actual M4 PR/report; explicit weight/export/device/build/retention budget authorization. Historical Mobius #758 is context, not a substitute. |

General placement search remains outside M1-M5. After the shared recipe contracts are accepted,
P's planner can consume the same EP-ordered candidates, not a second fusion mechanism. Qualify
query isolation, tentative-ownership/revision invalidation, and an unfused-offloads/fused-fits case;
verify the selected graph and execution/backing plan before final commitment/materialization.
Neither that planner nor model execution is authorized by this documentation revision.

### Acceptance record

- This PR supplies a reviewable design, not a qualification receipt or a maintainer approval.
- Required for design acceptance: responsible ORT maintainer review, confirmed named D1-D6 approval
  routing/status with rationale, and explicitly attributed blockers/probes tied to milestones.
- Runtime tests are **not** required to close the design issue, but their absence must remain
  explicit. A later implementation cannot substitute a planned command, skip, source registration
  or historical report for executed evidence.
- Design acceptance does not authorize unresolved API/lifetime, whole-Scan transaction,
  resource-owner contracts, production coding, new implementation issues or full-model execution.
