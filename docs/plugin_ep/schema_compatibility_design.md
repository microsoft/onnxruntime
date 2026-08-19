# Plugin EP Operator Schema Compatibility

## Status

Proposed design.

This document describes how ONNX Runtime (ORT) core and a separately built plugin
Execution Provider (EP) negotiate operator-schema compatibility. The immediate
motivation is the CUDA plugin EP, but the contract is intended for all plugin EPs.

## Summary

`ORT_API_VERSION` protects the layout of C API function tables. It does not protect
the meaning of an operator node. Today, all `com.microsoft` operators are registered
in opset 1, even when their input list, attributes, or semantics evolve. Plugin
kernels are also commonly registered with an open-ended `SinceVersion(1)` range.
Consequently, a newer plugin can claim and execute a node resolved by an older core
against a different schema.

The proposed solution has four parts:

1. Version `com.microsoft` schemas. Freeze opset 1 and introduce opset 2 for the
   first post-v1 contract change. Every later contract change receives a new schema
   version.
2. Add a factory-level plugin ABI callback that returns the operator-set schema
   catalogs against which the plugin was built. A catalog is identified by domain,
   opset version, and a generated ABI digest. ORT and the plugin use only exact
   catalog matches.
3. Require bounded version ranges for plugin kernels in ORT-owned mutable domains.
   Kernel lookup must match the node's resolved schema `since_version`, not merely
   the domain or operator name.
4. Gate every EP-specific contrib-op fusion on both the negotiated schema catalog
   and an exact target-kernel lookup. If support is unknown, the fusion does not run.

The negotiated opset answers "do core and plugin agree on this operator contract?"
It does **not** mean that the EP implements every operator in the opset. The kernel
registry (or a future compile-EP support query) remains the source of truth for
implementation support.

## Goals

- Never execute a plugin kernel against a node whose input/output/attribute contract
  the plugin may interpret differently from ORT core.
- Allow a new plugin to run with an older core when the plugin contains kernels for
  the older, negotiated schemas.
- Allow a new core to use an older plugin without creating newer fused operators for
  that plugin.
- Preserve normal fallback: an unsupported version is left for another EP instead of
  being invoked with an incompatible kernel.
- Make schema-version mistakes visible in code review and CI.

## Non-goals

- This contract does not replace `ORT_API_VERSION` or the existing minimum-runtime
  version check. Those protect the C ABI and availability of API functions.
- A schema match does not guarantee support for a particular shape, data type,
  attribute value, or device. `GetCapability()` and kernel type constraints still
  make those decisions.
- This design does not version private compiled artifacts. EPContext compatibility
  remains covered by `ValidateCompiledModelCompatibilityInfo()`.
- This design does not allow a plugin to replace ORT's built-in definition of an
  ORT-owned domain. Plugin-specific custom domains continue to use
  `OrtCustomOpDomain`.

## Current behavior and failure

There are currently three independent version mechanisms:

| Contract | Existing mechanism | What it protects |
|---|---|---|
| C API layout | `ORT_API_VERSION`, `ort_version_supported`, and `ApiInit()` | Function-table and struct fields |
| Model schema | Model `opset_import` and `Node::SinceVersion()` | The schema selected by core |
| Kernel implementation | `KernelDef::SinceVersion(start, end)` | The versions a kernel claims to implement |

The first mechanism works for plugin ABI evolution. The latter two are ineffective
for contrib ops when the domain remains permanently at version 1 and the kernel's
end version is unbounded.

For example, suppose a new `GroupQueryAttention` contract appends inputs 14 and 15:

1. An old ORT core resolves `com.microsoft::GroupQueryAttention` as schema version 1
   with only the old inputs.
2. A new CUDA plugin registers its newly compiled kernel as `SinceVersion(1)`.
3. `EpGraphSupportInfo_LookUpKernel()` finds that kernel, so `GetCapability()` claims
   the node.
4. The kernel may read inputs or attributes according to the new v1 definition even
   though core validated and transformed the node according to the old v1 definition.

The process has a valid C ABI but an invalid operator ABI.

## Compatibility invariants

The implementation must maintain these invariants:

1. A published `(domain, op_type, since_version)` contract is immutable.
2. A plugin kernel is eligible only when core and plugin have an exact schema-catalog
   match covering the node's resolved schema version.
3. A plugin kernel for an ORT-owned mutable domain has an explicit inclusive end
   version. An earlier kernel must never match a later schema accidentally.
4. A graph rewrite that introduces or upgrades an EP-specific operator is allowed
   only when the assigned EP has the exact target schema and kernel support.
5. Unknown compatibility is treated as unsupported, not as compatible.
6. The plugin independently validates the actual node in `GetCapability()`; core's
   negotiation is an additional guard, not a substitute for validation.

## Contrib operator versioning

### Domain version

Change the registered range for `com.microsoft` from `[1, 1]` to `[1, 2]` and add
`OpSet_Microsoft_ver2`. Opset 1 remains registered and unchanged.

An operator that has not changed does not need a duplicate v2 schema. As with ONNX
opsets, its latest schema with `since_version <= 2` remains active. An operator whose
contract changes has both its historical schema and a new schema, for example:

```cpp
ONNX_MS_OPERATOR_SET_SCHEMA(GroupQueryAttention, 1, LegacyGqaSchema());
ONNX_MS_OPERATOR_SET_SCHEMA(GroupQueryAttention, 2, GqaWithQkNormInputsSchema());
```

The following changes require a new schema version:

- adding, removing, reordering, or changing the option of an input or output;
- adding a model-visible attribute, changing an attribute type or default, or changing
  the meaning of an attribute value;
- changing allowed data types in a way a kernel needs to distinguish;
- changing shape or semantic rules in a way that can alter valid execution;
- changing aliasing or mutation behavior visible to execution.

Documentation-only corrections and shape-inference fixes that do not change the
accepted contract may remain in the same version. When uncertain, create a new
version.

### Opset imports created by optimizers

An ONNX opset import applies to an entire domain, not to one node. A transformer
cannot silently create a v2 node in a graph that imports `com.microsoft` opset 1.

Use these rules:

- If the graph already imports a sufficient version, create the target node.
- If the graph had no nodes or explicit import for the domain and ORT supplied its
  normal default import, the current negotiated version may be used.
- If the model explicitly imports an older version, skip the rewrite unless a
  contrib-opset converter has first proven and performed a semantics-preserving
  upgrade for all affected nodes in the graph.
- Saving an optimized model must serialize the actual domain opset selected by the
  graph.

The current model loader already adds the latest registered version for a missing
domain and preserves an explicitly imported version. The fusion rules should retain
that distinction instead of treating an automatically supplied import and an explicit
older model contract as interchangeable.

New schema versions should preserve the behavior of an old serialized node whenever
practical (for example, by adding trailing optional inputs with defaults). A truly
breaking change requires an explicit version converter or a new operator name.

## Plugin ABI addition

### Schema catalog descriptor

Append a descriptor type to `onnxruntime_ep_c_api.h`:

```c
typedef struct OrtEpOperatorSetCompatibilityInfo {
  uint32_t struct_version;
  const char* domain;
  int32_t opset_version;
  uint8_t schema_abi_digest[32];
} OrtEpOperatorSetCompatibilityInfo;
```

`struct_version` is initially 1. `domain` is UTF-8 and owned by the plugin. The digest
is SHA-256 over a canonical representation of the schema catalog at that opset.

Append this optional callback to `OrtEpFactory` in the next plugin ABI version:

```c
ORT_API2_STATUS(
    GetOperatorSetCompatibilityInfo,
    _In_ const OrtEpFactory* this_ptr,
    _Outptr_ const OrtEpOperatorSetCompatibilityInfo** entries,
    _Out_ size_t* num_entries);
```

The returned array and strings are owned by the factory and remain valid until
`ReleaseEpFactory()`. ORT copies the entries during factory registration. On failure,
the callback leaves both outputs unchanged, following the C API convention.

ORT calls the appended field only when `ort_version_supported` reaches the ABI version
that introduced it and the function pointer is non-null, following the existing
plugin-EP struct-extension rules.

This callback belongs on `OrtEpFactory`, rather than `OrtEp`, because the schema
catalog is a build property and ORT needs it before constructing the optimizer set.
Per-session options may affect whether a kernel is enabled, but not the schema against
which that kernel was compiled.

The callback reports catalogs understood by the plugin, not implemented ops. A CUDA
plugin built with contrib opsets 1 and 2 normally reports two entries:

```text
com.microsoft, 1, sha256:<v1 canonical catalog>
com.microsoft, 2, sha256:<v2 canonical catalog>
```

### Why a digest is needed

An integer protects compatibility only if published schemas are never edited in
place. The digest provides a fail-closed guard against accidental edits, release
branch cherry-picks, or private builds that reuse an opset number for a different
contract.

The canonical digest includes only execution-relevant schema data:

- domain, operator name, and `since_version`;
- ordered formal inputs and outputs, including option and arity;
- type-parameter bindings and sorted allowed type sets;
- attributes, required/optional state, type, and canonical default value;
- schema-visible differentiators such as deprecation state.

Documentation, source location, and function pointer addresses are excluded. Shape
inference code cannot be hashed directly; changes to its execution contract are
handled by the versioning policy.

The digest generator must produce checked-in historical manifests. CI regenerates
them and fails if an existing `(domain, opset_version)` digest changes. A new digest
may only be added under a new opset version.

The first checked-in v1 digest is the freeze-point baseline. It does not retroactively
certify older ORT releases, because those releases already shipped different schemas
under the same v1 label. This is why the first manifest-aware CUDA plugin must set its
minimum ORT version to the release that introduced negotiation.

### Negotiation

For every plugin factory, ORT computes the exact intersection of its built-in catalog
entries and the plugin entries. A domain/opset pair is compatible only when both the
version and digest match. ORT stores the result on the internal plugin EP factory and
copies it to each `PluginExecutionProvider` instance.

Negotiation is not simply `min(core_max, plugin_max)`. The maximum common exact match
is convenient for logging, but every accepted catalog entry must match independently.

Example:

| Core catalogs | Plugin catalogs | Negotiated result |
|---|---|---|
| ms/1=A, ms/2=B | ms/1=A, ms/2=B | 1 and 2 |
| ms/1=A | ms/1=A, ms/2=B | 1 |
| ms/1=A, ms/2=B | ms/1=A | 1 |
| ms/1=A | ms/1=X | none; digest mismatch |

The existing graph APIs already give the plugin the model's imports and resolved node
version (`Graph_GetOperatorSets` and `Node_GetSinceVersion`). No second core-to-plugin
opset setter is required. A plugin must use those values when doing its own
`GetCapability()` analysis.

## Kernel registration and lookup

### Explicit ranges

For ORT-owned mutable domains such as `com.microsoft`, plugin registration must reject
an open-ended kernel range. The CUDA plugin adapter should make non-versioned contrib
kernel macros exact by default:

```cpp
// Plugin build for kMSDomain only:
SinceVersion(1, 1)
```

When an implementation supports multiple schema versions, register the intended
ranges explicitly:

```cpp
ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_EX(
    GroupQueryAttention, kMSDomain, 1, 1, /* type and builder */, LegacyGqaKernel);

ONNX_OPERATOR_TYPED_KERNEL_EX(
    GroupQueryAttention, kMSDomain, 2, /* type and builder */, GqaV2Kernel);
```

The second registration must also be bounded to `[2, 2]` by the plugin adapter. If one
kernel class safely handles both contracts, it may be registered twice or explicitly
for `[1, 2]`; the declaration is intentional either way.

This rule does not change normal ONNX-domain kernel conventions. Standard ONNX kernel
ranges already follow published ONNX schema versions and often deliberately span
multiple opsets.

The initial implementation should cover `com.microsoft` and
`com.ms.internal.nhwc`. Other ORT-owned domains can opt in as soon as independently
released plugins consume them.

### Core validation

When importing a plugin kernel registry, ORT validates each kernel in an ORT-owned
mutable domain:

1. `start_version <= end_version` and the end is not the open-ended sentinel.
2. A core schema exists for `(domain, op_type, start_version)`.
3. Every distinct schema `since_version` covered by the range is present in an exactly
   negotiated catalog.

An incompatible kernel is excluded from the effective registry and logged once with
the EP name, operator, requested range, and negotiation result. Other compatible
kernels from the plugin remain usable. A malformed range is a plugin registration
error and should fail session initialization.

`EpGraphSupportInfo_LookUpKernel()` then remains the authoritative path used by the
CUDA plugin's `GetCapability()`: it can only return a kernel that passed catalog and
version validation.

## Fusion safety

### Required query

Add an internal helper to `KernelRegistryManager` (name illustrative):

```cpp
Status HasImplementationForSchema(
    ProviderType provider,
    std::string_view domain,
    std::string_view op_type,
    int schema_since_version,
    const KernelRegistry::TypeConstraintMap& type_constraints,
    bool& supported) const;
```

For a plugin EP, the helper first checks the negotiated catalog and then performs an
exact kernel lookup. For an in-tree EP, it performs the existing exact kernel lookup.
Unknown or ambiguous support returns `supported = false`.

Each EP-specific transformer declares the exact operator it will produce. A match is
rewritten only if:

1. the graph's domain import selects the intended schema;
2. the node's assigned/target EP is one of the transformer's allowed EPs;
3. that EP negotiated the target schema catalog; and
4. its effective kernel registry contains a kernel for the target schema and types.

Checking only the EP name (for example, `CUDAExecutionProvider`) is no longer
sufficient. The static CUDA EP and the separately released CUDA plugin can have
different kernel/schema coverage while exposing the same logical provider name.

The check should occur at the matched rewrite, where the target types are known, not
only when the transformer list is constructed. If no eligible EP supports the exact
target, ORT preserves the original unfused subgraph.

### Compile-based plugin EPs

A compile-based EP without a kernel registry cannot safely authorize a destructive
core fusion through a domain-wide opset claim. Initially, ORT must not run such an
EP-specific contrib fusion unless another exact support mechanism already exists.

A later ABI can add a query over a fully described candidate node. That query must
include inputs, outputs, attributes, and resolved schema; a `(domain, op_type,
version)` callback alone would overstate configuration support. This is separate from
the schema-catalog negotiation proposed here.

## `GroupQueryAttention` example

Assume the q/k normalization inputs become the first versioned change:

- `GroupQueryAttention-1` is restored to and frozen at its historical contract.
- `GroupQueryAttention-2` contains `q_norm_weight`, `k_norm_weight`, and
  `qk_norm_epsilon`.
- The `com.microsoft` domain range becomes `[1, 2]`.
- CUDA registers a legacy v1 kernel if it still supports the old contract and a
  separately bounded v2 kernel for the new contract.
- CPU, WebGPU, and JS registrations are independently versioned according to the
  contracts they actually implement.
- `GroupQueryAttentionPreNormFusion` matches and produces version 2 only. It does not
  mutate a v1 GQA node while retaining `since_version == 1`.
- `GroupQueryAttentionFusion`, when targeting CUDA, checks exact GQA v1 or v2 kernel
  support based on the graph's selected `com.microsoft` opset before replacing the
  original attention subgraph.

Expected behavior:

| Core | CUDA plugin | Result |
|---|---|---|
| v1 catalog | v1 + v2 catalogs/kernels | v1 kernel may run; v2 is invisible |
| v1 + v2 catalogs | v1-only plugin | v2 fusion is skipped; v1 nodes may run |
| v1 + v2 catalogs | v2-only plugin | v1 nodes fall back; v2 nodes may run |
| v1 digest A | plugin v1 digest X | contrib kernels are quarantined |

At no point does the v2 kernel execute a node that core resolved as the incompatible
v1 contract.

## Legacy compatibility and rollout

Old cores do not know the new factory callback, and old plugins do not implement it.
This must be handled explicitly.

### Immediate containment before the ABI ships

Until catalog negotiation is available, a CUDA plugin containing a changed contrib
contract must raise `MIN_ONNXRUNTIME_VERSION` to the first core release with the same
schema, bound the affected kernel to the exact schema version, and disable the related
fusion against older imports. This temporarily reduces the compatibility matrix but
closes the unsafe execution path immediately.

### New plugin on an old core

The first CUDA plugin release that relies on versioned contrib schemas should raise
`plugin-ep-cuda/MIN_ONNXRUNTIME_VERSION` to the first core release that implements
schema negotiation. This is the only reliable fail-closed choice for arbitrary old
or privately patched cores.

If supporting an older core is a hard product requirement, the plugin must carry a
reviewed compatibility table for released core builds and register only the exact
legacy kernels proven compatible with those builds. Comparing only the ORT release
number is not sufficient for development or private builds, so unknown builds still
fail closed.

### Old plugin on a new core

If the callback is absent, a new core treats compatibility for ORT-owned mutable
domains as unknown:

- do not expose that plugin's contrib kernels in the effective registry;
- do not run EP-specific contrib fusions for it;
- continue allowing compatible standard ONNX-domain kernels.

During one transition window, ORT may provide an explicitly named opt-in session
configuration that restores legacy behavior, with a warning that schema safety is not
guaranteed. It must not be the default and should have a removal release.

### Suggested rollout phases

1. **Infrastructure:** add catalog generation, the factory callback, negotiation,
   diagnostics, and strict kernel-registry filtering. Keep contrib opset 1 unchanged.
2. **Plugin adoption:** make the example plugin and CUDA plugin publish the v1 catalog;
   add old/new core-plugin matrix tests.
3. **Kernel hardening:** make CUDA plugin contrib registrations bounded and add CI that
   rejects open-ended ranges in mutable ORT domains.
4. **First version bump:** add `com.microsoft` opset 2 and migrate
   `GroupQueryAttention` plus its kernels and transformers.
5. **Fusion hardening:** convert all EP-specific contrib fusions from provider-name
   checks to exact target-schema/kernel checks.
6. **Enforcement:** switch missing-manifest handling to strict-by-default after the
   transition window and remove the temporary opt-in later.

## Diagnostics

At INFO level, log one summary per plugin factory:

```text
CUDAExecutionProvider schema negotiation: com.microsoft common={1,2}, highest=2
```

At WARNING level, log actionable incompatibilities:

```text
Ignoring CUDAExecutionProvider kernel com.microsoft::GroupQueryAttention [2,2]:
plugin catalog digest does not match ORT for com.microsoft opset 2.
```

When a fusion is skipped for this reason, VERBOSE logging should identify the
transformer, target operator version, assigned EP, and missing catalog or kernel.
No per-node warning should be emitted during normal fallback to avoid log spam.

## Test plan

### Schema tests

- Assert the registered `com.microsoft` range and lookup of GQA v1/v2.
- Load explicit opset-1 and opset-2 GQA models and verify their resolved
  `Node::SinceVersion()` and formal input counts.
- Regenerate catalog manifests in CI and fail if a historical digest changes.
- Verify optimized-model serialization retains the correct contrib opset import.

### ABI negotiation tests

Extend the example plugin with selectable manifests and kernels:

- core v1 / plugin v1;
- core v2 / plugin v1;
- core v1 / plugin v1+v2;
- matching version with mismatched digest;
- missing callback;
- duplicate, malformed, and unsupported catalog entries.

Verify compatible standard ONNX kernels remain available when contrib kernels are
quarantined.

### Kernel tests

- Reject or filter an open-ended plugin kernel in `com.microsoft`.
- Verify a `[1,1]` kernel does not match a GQA-2 node.
- Verify a `[2,2]` kernel does not match a GQA-1 node.
- Verify explicitly registered compatible v1 and v2 kernels execute the correct
  implementation.
- Verify kernel type constraints still filter nodes after catalog negotiation.

### Fusion tests

- New core + old plugin: GQA v2 fusion is not applied.
- Old-compatible graph + new plugin: only the v1 fusion/kernel path is used.
- Mismatched digest: the original unfused graph is preserved.
- Multiple EPs: fusion follows the support of the node's actual target EP, not another
  EP with the same op available.
- Explicit `com.microsoft` opset 1: no implicit upgrade to v2.
- Saving and reloading an optimized model produces identical assignments.

### Release matrix

For every plugin release, CI should run at least:

- newest plugin against its minimum supported ORT core;
- newest plugin against current ORT core;
- previous supported plugin against current ORT core;
- a deliberately mismatched schema-catalog build that must fail closed.

## Implementation map

The main expected code areas are:

- `onnxruntime/core/graph/contrib_ops/bert_defs.cc`: immutable GQA v1 and new v2
  schema;
- `onnxruntime/core/graph/contrib_ops/ms_opset.h` and
  `onnxruntime/core/session/environment.cc`: opset registration and domain range;
- `include/onnxruntime/core/session/onnxruntime_ep_c_api.h`: factory ABI descriptor
  and callback;
- `onnxruntime/core/session/plugin_ep/`: callback validation, catalog negotiation,
  and effective kernel filtering;
- `onnxruntime/core/providers/cuda/plugin/cuda_kernel_adapter.h`: bounded contrib
  kernel registrations;
- `onnxruntime/core/framework/kernel_registry*` and optimizer utilities: exact target
  support query;
- GQA CUDA/CPU/WebGPU/JS registrations and GQA transformers: explicit version use;
- `onnxruntime/test/autoep/`: ABI and cross-version matrix coverage.

## Alternatives considered

### Use only `ORT_API_VERSION`

Rejected. A stable C function table says nothing about operator input positions,
attributes, defaults, or semantics.

### Exchange only the maximum `com.microsoft` opset

Insufficient by itself. It cannot detect two builds that assign different contracts
to the same opset number, and it does not say which operators or configurations the EP
implements. The proposed catalog digest and exact kernel lookup address both gaps.

### Let `GetCapability()` handle everything

Insufficient for core fusions, which may destructively replace a portable subgraph
before the plugin sees the resulting node. It also relies on the plugin recognizing a
schema mismatch that core has not exposed as a distinct version.

### Treat the plugin's schemas as authoritative

Rejected for ORT-owned domains. Core performs model validation, shape inference,
optimization, memory planning, and fallback using its own schemas. Two authoritative
definitions in one process would make those stages inconsistent.

## Decision checklist for future contrib changes

Before merging a contrib-op contract change:

1. Does the change affect inputs, outputs, attributes, types, shape rules, aliasing, or
   semantics? If yes, create a new schema version.
2. Is the old schema definition still present and byte-for-byte catalog compatible?
3. Are every EP's old and new kernel ranges explicit and truthful?
4. Does every optimizer that creates the op request the exact new version and verify
   target-EP support?
5. Can a graph with an explicit older domain import remain unchanged, or is a version
   converter required?
6. Do the old-core/new-plugin and new-core/old-plugin tests fail closed?
