# Plugin EP Operator Schema Compatibility

## Status

Implementation in progress.

The API-30 implementation now includes the fixed operator descriptor, the optional
factory callback, canonical per-schema digest computation, negotiation during plugin
EP creation, effective kernel-registry filtering, post-`GetCapability()` validation,
the API-30 `com.microsoft` freeze-point manifest, CUDA manifest publication, and
central current/last-released contrib opset constants. A synthetic v1/v2 matrix now
exercises model loading, negotiation, and kernel filtering. The two GQA transformers
are the first optimizer gates: they query the session's effective kernel registry
before changing an assigned GQA node. Converting the remaining EP-specific contrib
optimizers, automatic manifest generation, true historical-binary matrix coverage,
and strict handling of plugins without the callback remain follow-up work.

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

1. Version `com.microsoft` schemas. API 30 establishes the current opset-1 contracts
   as the first immutable baseline. The first contract change after that release
   opens opset 2; later changes in the same release train share that open opset.
2. Add a factory-level plugin ABI callback that returns **per-operator** schema
   digests for the contracts the plugin was built against. An entry is identified by
   domain, operator name, opset version, and a generated ABI digest. ORT and the
   plugin agree on an operator only when that entry matches exactly.
3. Preserve kernel lookup's exact-start semantics for open-ended registrations and
   require an explicit bounded range when one kernel intentionally supports multiple
   schema versions. Lookup must match the node's resolved schema `since_version`, not
   merely the domain or operator name.
4. Gate every EP-specific contrib-op fusion on both the negotiated operator digest
   and an exact target-kernel lookup. If support is unknown, the fusion does not run.

A matching digest answers "do core and plugin agree on this operator contract?" It
does **not** mean that the EP implements the operator. The kernel registry (or a
future compile-EP support query) remains the source of truth for implementation
support.

Negotiation is per operator, not per domain. A single disagreeing operator quarantines
only that operator, so routine operator additions and unrelated contract changes do
not disable an otherwise compatible plugin.

## Goals

- Never execute a plugin kernel against a node whose input/output/attribute contract
  the plugin may interpret differently from ORT core.
- Allow a new plugin to run with an older core when the plugin contains kernels for
  the older, negotiated schemas.
- Allow a new core to use an older plugin without creating newer fused operators for
  that plugin.
- Preserve normal fallback: an unsupported version is left for another EP instead of
  being invoked with an incompatible kernel.
- Keep the compatible matrix wide: routine operator additions and unrelated contract
  changes must not disable an otherwise compatible plugin.
- Keep existing models and existing producers working. A graph that does not need a
  new contract must not gain an import that older cores reject.
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
2. A plugin kernel is eligible only when core and plugin have an exact digest match
   for that operator at the node's resolved schema version.
3. Once a `com.microsoft` operator has more than one schema, every kernel for it has
   an explicit inclusive end version. An earlier kernel must never match a later
   schema accidentally. This applies to in-tree EP kernels as well as plugin kernels.
4. A graph rewrite that introduces or upgrades an EP-specific operator is allowed
   only when the assigned EP has the exact target schema and kernel support.
5. Once strict enforcement is enabled, unknown compatibility is treated as
   unsupported, not as compatible. During the API-30 transition, a plugin that
   predates the callback remains enabled with a warning as described below.
6. The plugin independently validates the actual node in `GetCapability()`; core's
   negotiation is an additional guard, not a substitute for validation.
7. Quarantine granularity matches negotiation granularity. A digest mismatch on one
   operator must not disable unrelated operators or the whole plugin.

## Contrib operator versioning

### Domain version

Maintain two central constants for `com.microsoft`: the highest opset used in the
source tree and the highest opset published in a stable release. Immediately after a
release they are equal. The first model-visible contract change in the next release
train increments the current version and opens that opset; subsequent contrib-op
changes in the same train reuse it. At release time the open version becomes the new
last-released version.

```cpp
constexpr int kMSDomainOpsetVersion = 1;
constexpr int kMSDomainOpsetVersionLastReleased = 1;
```

Only one development opset may be open:

```cpp
static_assert(kMSDomainOpsetVersion == kMSDomainOpsetVersionLastReleased ||
              kMSDomainOpsetVersion == kMSDomainOpsetVersionLastReleased + 1);
```

The lock advances for every stable ONNX Runtime release that promises serialized-model
compatibility, not only for a major-number release. Schemas at or below the
last-released version are immutable. Models produced by an early nightly with an open
opset are not guaranteed to remain compatible with later nightlies; the per-operator
digest still guarantees that mismatched core and plugin binaries fail closed for the
affected operator.

The existing registration in `environment.cc` is guarded so that it is skipped
entirely when another component already added the domain:

```cpp
if (map.find(kMSDomain) == map.end()) {
  // External shared providers may have already added kMSDomain
  domain_to_version_range.AddDomainToVersion(kMSDomain, 1, 1);
}
```

Editing the literal to `2` inside this guard would silently do nothing when a shared
provider registered `[1, 1]` first, and core would then reject every opset-2 import
with a generic unresolved-schema error. The bump must instead raise the recorded
maximum and normalize the last-released value unconditionally to the built-in values.
An older maximum is accepted and upgraded; a different minimum or a maximum newer
than the core understands is a configuration error.

The same check should reject an `OrtCustomOpDomain` or a shared provider that
registers an ORT-owned domain name such as `com.microsoft` with its own range. Custom
op registration currently uses `AddDomainToVersion(domain, 1, 1000)` for an arbitrary
domain string, which would otherwise let a custom op silently redefine the negotiated
domain.

An operator that has not changed does not need a schema in the open opset. As with
ONNX opsets, its latest schema with `since_version <=` the model's import remains
active. An operator whose contract changes after the API-30 baseline keeps its
historical schema and adds a schema in the open opset, for example:

```cpp
ONNX_MS_OPERATOR_SET_SCHEMA(GroupQueryAttention, 1, FrozenApi30GqaSchema());
ONNX_MS_OPERATOR_SET_SCHEMA(GroupQueryAttention, 2, GqaWithNextContractChange());
```

Q/K normalization, `causal`, and sliding-cache support are already part of the API-30
GQA v1 freeze-point contract. They are not retroactively moved to v2.

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

### Adding a new operator

A new contrib operator is introduced at the currently open `kMSDomainOpsetVersion`,
the same rule ONNX and every other domain already follow. Adding an operator to an
already-released opset would mutate the meaning of that opset: a core built before the
addition resolves `com.microsoft` opset N to a smaller catalog than a core built after
it, even though both claim to implement opset N. So if the domain is closed
(`kMSDomainOpsetVersion == kMSDomainOpsetVersionLastReleased`), adding an operator
opens the next version exactly like any other model-visible change; if the domain is
already open, the new operator reuses the open version.

Adding an operator is still not a contract change to any *existing* operator. Existing
schemas keep their `since_version`, existing plugins keep matching them, and only the
new operator is unavailable to a plugin that was built before it existed.

This granularity is what makes the rule cheap. A whole-domain catalog digest would
change every time an operator is added to the open opset, which happens on most
releases, and would quarantine every contrib kernel of a plugin built one commit
earlier. See [Digest granularity](#digest-granularity).

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

### Serialized model compatibility

Raising the domain maximum changes what ORT can emit, not only what it can accept.
Three rules keep existing artifacts and existing producers working:

- A model serialized with a `com.microsoft` opset-2 import cannot be loaded by any
  previously released ORT. Core must reject such a load with an explicit message that
  names the domain, the requested version, and the maximum this build supports,
  instead of a generic unresolved-schema error.
- ORT must not raise a graph's `com.microsoft` import above the minimum version the
  graph actually needs. If no node in the saved graph resolves to a schema newer
  than 1, the serialized import stays at 1 and the artifact remains loadable by older
  cores. This is the difference between "ORT supports opset 2" and "this graph
  requires opset 2".
- First-party producers emit `com.microsoft` opset 1 today: the ONNX Runtime GenAI
  model builder, Olive, and `onnxruntime/python/tools/transformers`. Under the rules
  above they keep working unchanged, but they will not receive any opset-2 fusion
  until they are updated. Each must be updated in the same release train as the first
  version bump, and each must write the emitted version explicitly rather than
  inheriting a default that shifts with the ORT version.

## Plugin ABI addition

### Operator compatibility descriptor

Append a descriptor type to `onnxruntime_ep_c_api.h`:

```c
typedef struct OrtEpOperatorCompatibilityInfo {
  const char* domain;
  const char* op_type;
  int32_t since_version;
  uint8_t schema_abi_digest[32];
} OrtEpOperatorCompatibilityInfo;
```

`domain` and `op_type` are UTF-8 and owned by the plugin. `since_version` is the
schema version of the contract, not the domain import version; the two differ for any
operator that has not changed since an earlier opset. The digest is SHA-256 over the
canonical representation of that single operator schema.

This struct is deliberately fixed-size and must never grow. ORT receives a contiguous
array whose stride is fixed by the plugin's compiled `sizeof(...)`, so a struct that
grows in a later ABI version would be unindexable by an older core, and a per-element
`struct_version` field could not be read safely enough to discover that. A future
extension adds a new descriptor type and a new callback rather than extending this
one. (The alternative, an explicit `entry_size` out-parameter, was rejected because it
has no precedent in this header.)

Append this optional callback to `OrtEpFactory` in the next plugin ABI version,
following the header's existing doxygen conventions:

```c
  /** \brief Get the operator schema contracts this factory was built against.
   *
   * ...
   * \since Version 1.30.
   */
  ORT_API2_STATUS(GetOperatorCompatibilityInfo, _In_ OrtEpFactory* this_ptr,
                  _Outptr_ const OrtEpOperatorCompatibilityInfo** entries,
                  _Out_ size_t* num_entries);
```

The returned array and strings are owned by the factory and remain valid until
`ReleaseEpFactory()`. ORT copies the entries during plugin provider creation. On
failure, the callback leaves both outputs unchanged, following the C API convention.
Duplicate `(domain, op_type, since_version)` entries are a plugin bug; ORT treats the
operator as incompatible and logs once.

ORT calls the appended field only when `ort_version_supported` reaches the ABI version
that introduced it and the function pointer is non-null, matching the existing
plugin-EP struct-extension rules used elsewhere in `OrtEpFactory` and `OrtEp`.

This callback belongs on `OrtEpFactory`, rather than `OrtEp`, because the schema set
is a build property and ORT needs it before constructing the optimizer set.
Per-session options may affect whether a kernel is enabled, but not the schema against
which that kernel was compiled.

The callback reports contracts understood by the plugin, not implemented ops. A CUDA
plugin built after the first version bump reports one entry per contrib operator it
was compiled against, for example:

```text
com.microsoft, GroupQueryAttention, 1, sha256:<...>
com.microsoft, GroupQueryAttention, 2, sha256:<...>
com.microsoft, SkipLayerNormalization, 1, sha256:<...>
...
```

A plugin that only cares about a handful of contrib operators may report only those.
An operator absent from the list is simply not negotiated, and its kernels are
filtered out.

### Why a digest is needed

`since_version` is the contract. The digest never replaces it and never widens what a
plugin may claim; it only verifies that both sides mean the same thing by a given
version number.

An integer alone protects compatibility only if published schemas are never edited in
place, which is exactly the property `com.microsoft` has never enforced. Today every
contrib schema sits at `since_version 1` and has been edited in place for years, so
there is no historical evidence that the discipline holds on its own. The digest turns
that convention into a checked invariant: `SchemaAbiDigestTest.MSManifestMatchesSchemas`
fails the build when a published schema changes without a version bump, which is where
the mistake is cheapest to fix.

It also covers the cases a version number cannot:

- **Open (unreleased) opsets.** Within a development train the open version is mutable
  by design, so the same `since_version` legitimately denotes different contracts at
  different commits. Nightly and CI plugin/core pairings are the combinations most
  likely to be mixed, and they are precisely the ones `since_version` cannot separate.
- **Release-branch cherry-picks and private builds** that patch a schema without
  taking the version bump.
- **Two independent shared-provider or plugin binaries** that each registered
  `com.microsoft` from a different source snapshot.

The cost is one 32-byte constant per manifest entry and one `memcmp` per operator at
factory load. A mismatch is not fatal: it quarantines that single operator with a
warning, and the rest of the plugin runs. Without it, the same situation is a silent
contract violation that surfaces as a wrong result or a crash inside a kernel.

### Digest granularity

The digest covers **one operator schema**, not the whole operator set at an opset.

A whole-catalog digest was the first design and does not work in practice. The
catalog at opset N is the resolved schema set for every operator with
`since_version <= N`, so it changes whenever *any* contrib operator is added or
changed. New contrib operators land in the open opset on most releases, which would
change that opset's catalog digest on most releases and quarantine every contrib
kernel of a plugin built one commit earlier. The compatibility matrix would collapse
to "core and plugin must be the same commit", defeating the goal of running a new
plugin on an older core.

With per-operator digests, adding `NewOp@1` does not affect the digest of
`GroupQueryAttention@1`, and a disagreement on one operator quarantines only that
operator's kernels. A rolled-up hash over all matched entries may still be logged as a
convenience identifier, but it must never be the unit of comparison.

### Canonical form

The digest is SHA-256 over a canonical byte encoding of a single schema. The encoding
must be specified precisely enough that two independent implementations agree. It
includes only execution-relevant schema data:

- domain, operator name, and `since_version`;
- ordered formal inputs and outputs, including formal parameter name, option
  (single/optional/variadic), homogeneity, variadic minimum arity, differentiation
  category, and the declared type-parameter name;
- the schema's global minimum and maximum input and output arity;
- type-parameter bindings, with allowed type strings sorted by byte value;
- attributes sorted by name, with required/optional state, attribute type, and the
  canonical serialization of the default value;
- schema-visible differentiators such as deprecation state.

Canonical encoding version 1 starts with the length-prefixed byte string
`ort.schema_abi.v1`. Every following string is encoded as an unsigned 64-bit
little-endian byte length followed by its bytes. Integers and booleans are unsigned
64-bit little-endian values. Lists start with their element count and then encode
elements in the order stated above. Attribute defaults are deterministic protobuf
wire serializations of `AttributeProto` after clearing `doc_string`; their byte string
is length-prefixed like every other string. Sorting is by raw byte value, never
locale-sensitive.

Formal and allowed type strings are normalized by removing ASCII whitespace before
encoding, so that a cosmetic formatting change in the ONNX submodule does not flip
every digest. Documentation, source location, and function pointer addresses are
excluded. Shape inference code cannot be hashed directly; changes to its execution
contract are handled by the versioning policy.

### Where each side's digest comes from

This must be stated explicitly, because the plausible options have opposite failure
modes. A checked-in constant on both sides would only detect "different manifest file
revision" and would give no protection against the in-place-edit case the digest
exists to catch. A naive runtime hash of the entire live registry would differ for
any build that registers a different set of schemas, and would fail closed on builds
that are in fact compatible.

The rule is:

- The **plugin** embeds the digests generated at its build commit, taken from the
  checked-in manifest.
- **Core** computes its digests at runtime from its live schema registry, restricted
  to the operators the plugin actually reported. Core never hashes operators nobody
  asked about, so an operator that a reduced build did not register simply has no core
  entry and is reported as unavailable rather than as a mismatch.
- CI asserts that the runtime-computed digests of the default inference build equal
  the checked-in inference-plugin manifest, so an in-place schema edit fails the build
  unless the manifest is regenerated, and regenerating an existing entry fails the
  historical-digest check. Feature builds may register additional private schemas;
  every published manifest entry must still match, but unreported feature-specific
  schemas do not make the inference-plugin manifest invalid.

Manifest generation and validation must use the finalized schemas from the live
registry. Hashing the raw objects returned by an opset's `ForEachSchema()` is not
equivalent: ONNX schema registration finalizes those objects before runtime lookup.
The manifest must also cover `com.microsoft` schemas registered outside the central
`OpSet_Microsoft_ver*` classes.

Consequences for non-default builds:

- `DISABLE_CONTRIB_OPS` and reduced/selective-op builds: absent operators produce no
  core entry, so the plugin's contrib kernels for those operators are filtered out.
  This is correct and matches what the build actually supports.
- `ORT_MINIMAL_BUILD`: there is no schema registry and no plugin kernel registry path,
  so negotiation does not apply.
- SHA-256 is required in core only where negotiation runs, that is, non-minimal builds
  that support plugin EPs.

The digest generator must produce checked-in historical manifests. CI regenerates them
and fails if an entry at or below `kMSDomainOpsetVersionLastReleased` changes. A new
contract for an existing operator may only be added in the currently open opset (or
under a new operator name). During development, entries in the one open opset may be
regenerated; cross-nightly core/plugin mismatches remain safe because negotiation is
based on the exact per-operator digest.

The first checked-in v1 digests are the freeze-point baseline. They do not
retroactively certify older ORT releases, because those releases already shipped
different schemas under the same v1 label. This is why the first manifest-aware CUDA
plugin must set its minimum ORT version to the release that introduced negotiation.

### Negotiation

For every plugin factory, ORT resolves each reported entry independently. An entry is
compatible only when core has a schema for `(domain, op_type, since_version)` and the
digests are equal. During `PluginExecutionProvider` creation, ORT snapshots the
resulting set of accepted `(domain, op_type, since_version)` triples and shares that
immutable snapshot with the provider and its effective kernel registry.

Negotiation is not `min(core_max, plugin_max)` and is not a per-domain decision. Every
accepted entry matches independently.

Example, for a single operator `Foo`:

| Core schemas | Plugin entries | Negotiated result |
|---|---|---|
| Foo@1=A, Foo@2=B | Foo@1=A, Foo@2=B | Foo@1 and Foo@2 |
| Foo@1=A | Foo@1=A, Foo@2=B | Foo@1 only |
| Foo@1=A, Foo@2=B | Foo@1=A | Foo@1 only |
| Foo@1=A | Foo@1=X | none for `Foo`; other operators are unaffected |

The last row quarantines only `Foo`'s kernels. Kernels for other `com.microsoft`
operators whose digests matched, and all standard ONNX-domain kernels, remain
available.

Negotiation is snapshotted when a provider is created. The ONNX schema registry is
process-global and can still be mutated afterwards by custom-op registration or by a
late-loaded shared provider. To keep each snapshot honest, ORT rejects any later
attempt to register schemas into an ORT-owned domain (see
[Domain version](#domain-version)); mutation of unrelated custom domains does not
affect negotiation.

The existing graph APIs already give the plugin the model's imports and resolved node
version (`Graph_GetOperatorSets` and `Node_GetSinceVersion`). No second core-to-plugin
opset setter is required. A plugin must use those values when doing its own
`GetCapability()` analysis.

## Kernel registration and lookup

### Explicit ranges

`KernelRegistry` treats an open-ended `SinceVersion(v)` registration as an exact match
for `v`, because a future schema version may change the contract. That form remains
safe when a `com.microsoft` operator gains another schema. New CUDA plugin contrib
registrations should continue to be exact by default:

```cpp
// Both forms match only schema version 1.
SinceVersion(1)
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

If one kernel class safely handles both contracts, it may be registered twice or
explicitly for `[1, 2]`; the multi-version declaration is intentional either way.

### Scope of the version-range rule

The rule applies to `com.microsoft` only. It does not change normal ONNX-domain kernel
conventions, where ranges already follow published ONNX schema versions and often
deliberately span multiple opsets.

`com.ms.internal.nhwc` is explicitly **out of scope**, despite being ORT-owned. It is
registered as `AddDomainToVersion(kMSInternalNHWCDomain, 1, onnx_version)` precisely
because it mirrors ONNX operators, and its kernels intentionally span opsets the same
way their ONNX counterparts do. Its schemas track ONNX's published versions rather
than a frozen ORT-private contract, so the freeze-and-digest argument does not apply
and forcing exact ends there would break the layout transformer. Other ORT-owned
domains can opt in individually once independently released plugins consume them and
only if their schemas are ORT-private.

### In-tree EP kernels

The same version discipline applies inside a single static build. An in-tree contrib
kernel registered as open-ended `SinceVersion(1)` remains an exact v1 match after a v2
schema is added; it does not silently claim v2. Each EP must add a v2 registration
only if its kernel implements the new contract.

Therefore, versioning an operator requires adding registrations for every new schema
version that an in-tree EP implements, across CPU, CUDA, ROCm, DML, WebGPU, JS, and
QNN. Existing open-ended registrations remain exact matches for their start versions.
A kernel that intentionally supports several schema versions must declare a bounded
range covering those versions. The
[decision checklist](#decision-checklist-for-future-contrib-changes) enforces this at
review time.

### Core validation

When importing a plugin kernel registry, ORT validates each kernel in `com.microsoft`:

1. `start_version <= end_version`. An open-ended range is validated as the exact
   start version, matching `KernelRegistry` lookup semantics.
2. A range wholly newer than the core's domain maximum is ignored, not treated as a
   plugin error. For a range that overlaps the core's supported versions, only that
   core-visible intersection is validated.
3. A core schema exists for `(domain, op_type, start_version)` in that intersection,
   and every distinct schema `since_version` it covers was negotiated exactly for
   that operator.

An incompatible kernel is excluded from the effective registry and logged once with
the EP name, operator, requested range, and negotiation result. Other compatible
kernels from the plugin remain usable, including other operators in the same domain. A
malformed range is a plugin registration error and should fail session initialization.

`EpGraphSupportInfo_LookUpKernel()` then remains the authoritative path used by the
CUDA plugin's `GetCapability()`: it can only return a kernel that passed digest and
version validation.

## Fusion safety

### Required query

`KernelRegistryManager` already exposes the right entry point for a fully resolved
node:

```cpp
static bool HasImplementationOf(const KernelRegistryManager& registry_manager,
                                const Node& node,
                                const std::string& provider_type,
                                const logging::Logger& logger);
```

It uses the node's resolved schema and actual type constraints. For a plugin EP,
`RegisterKernels()` has already received the effective registry produced by schema
negotiation: kernels whose per-operator digest did not match have been removed.
Consequently the existing lookup answers both questions without introducing a
parallel support mechanism. For an in-tree EP it remains the normal kernel lookup.
Unknown or ambiguous support is reported as unsupported.

Keeping a single lookup path matters: two subtly different definitions of "supported"
between the fusion gate and `EpGraphSupportInfo_LookUpKernel()` would reintroduce the
case where a fusion is authorized but the resulting node cannot be assigned.

Each EP-specific transformer declares the exact operator it will produce. A match is
rewritten only if:

1. the graph's domain import selects the intended schema;
2. the node's assigned/target EP is one of the transformer's allowed EPs;
3. that EP negotiated the target operator schema; and
4. its effective kernel registry contains a kernel for the target schema and types.

Checking only the EP name (for example, `CUDAExecutionProvider`) is no longer
sufficient. The static CUDA EP and the separately released CUDA plugin can have
different kernel/schema coverage while exposing the same logical provider name.

The check occurs at the matched rewrite, where the target types are known, not only
when the transformer list is constructed. Session-created transformers receive a
checker backed by `KernelRegistryManager`; standalone/offline transformer use may
omit the checker because it has no session registry. A transformer that creates a
new operator or changes its resolved schema must construct a fully described target
candidate before querying. The current GQA transforms retain the existing GQA schema
and types, so they query the assigned GQA node immediately before mutating it. If the
lookup fails, ORT preserves the original unfused subgraph.

### Compile-based plugin EPs

A compile-based EP without a kernel registry cannot safely authorize a destructive
core fusion through a domain-wide opset claim. Initially, ORT must not run such an
EP-specific contrib fusion unless another exact support mechanism already exists.

A later ABI can add a query over a fully described candidate node. That query must
include inputs, outputs, attributes, and resolved schema; an
`(domain, op_type, version)` callback alone would overstate configuration support.
This is separate from the operator digest negotiation proposed here.

## `GroupQueryAttention` example

The q/k normalization inputs and `qk_norm_epsilon` are already part of the API-30
freeze point, so they remain in `GroupQueryAttention-1`. The current
`GroupQueryAttentionPreNormFusion` still produces version 1 and is safe only when the
assigned EP's GQA-1 kernel survived negotiation.

Assume instead that a future breaking GQA contract change is the first change after
the API-30 release:

- `GroupQueryAttention-1`, including q/k normalization, remains frozen.
- `GroupQueryAttention-2` contains only that future contract change.
- The `com.microsoft` domain range becomes `[1, 2]`.
- CUDA registers a legacy v1 kernel if it still supports the old contract and a
  separately bounded v2 kernel for the new contract.
- CPU, WebGPU, and JS registrations are independently versioned according to the
  contracts they actually implement.
- `GroupQueryAttentionPreNormFusion` continues to produce version 1 unless it is
  deliberately updated to use the future v2 contract and the graph imports opset 2.
- `GroupQueryAttentionFusion`, when targeting CUDA, checks exact GQA v1 or v2 kernel
  support based on the graph's selected `com.microsoft` opset before replacing the
  original attention subgraph.

Expected behavior:

| Core | CUDA plugin | Result |
|---|---|---|
| GQA v1 schema only | GQA v1 + v2 entries/kernels | v1 kernel may run; v2 is invisible |
| GQA v1 + v2 schemas | v1-only plugin | v2 fusion is skipped; v1 nodes may run |
| GQA v1 + v2 schemas | v2-only plugin | v1 nodes fall back; v2 nodes may run |
| GQA v1 digest A | plugin GQA v1 digest X | GQA kernels are quarantined; other contrib operators are unaffected |

At no point does the v2 kernel execute a node that core resolved as the incompatible
v1 contract.

## Legacy compatibility and rollout

Old cores do not know the new factory callback, and old plugins do not implement it.
This must be handled explicitly.

### Immediate containment before the ABI ships

Until digest negotiation is available, a CUDA plugin containing a changed contrib
contract must raise `plugin-ep-cuda/MIN_ONNXRUNTIME_VERSION` to the first core release
with the same schema, bound the affected kernel to the exact schema version, and
disable the related fusion against older imports. That file is already the single
source of truth consumed by the packaging and CI scripts, so this is actionable today
and does not depend on any of the work below. It temporarily reduces the compatibility
matrix but closes the unsafe execution path immediately.

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

If the callback is absent, a new core cannot know which contracts the plugin was built
against. The eventual strict behavior for `com.microsoft` is:

- do not expose that plugin's contrib kernels in the effective registry;
- do not run EP-specific contrib fusions for it;
- continue allowing compatible standard ONNX-domain kernels.

This cannot be the behavior on day one. Contrib operators are most of what the shipped
CUDA and WebGPU plugins provide, so switching to strict immediately would turn every
already-released plugin into an ONNX-only plugin against a new core. During the
transition window the permissive behavior stays the default, accompanied by a warning
naming the plugin and the risk, and an explicitly named session configuration opts in
to strict early.

The default flips to strict only when all of the following hold:

1. every first-party plugin (CUDA, WebGPU, and the example plugin) publishes operator
   digests;
2. at least one full ORT release has shipped in which both a digest-publishing plugin
   and a permissive core were available, so users have an upgrade path that never
   requires running strict core against a pre-digest plugin;
3. the release notes for that window state the flip date and the opt-out.

After the flip, the opt-out configuration that restores legacy behavior survives one
further release and then is removed. Both configurations carry an explicit removal
release from the day they are introduced.

### Suggested rollout phases

1. **Infrastructure:** add digest generation, the factory callback, negotiation,
   diagnostics, and kernel-registry filtering. Keep contrib opset 1 unchanged and keep
   missing-callback handling permissive.
2. **Plugin adoption:** make the example plugin and CUDA plugin publish digests; add
   old/new core-plugin matrix tests.
3. **Kernel hardening:** verify CUDA plugin contrib registrations follow exact-start
   lookup semantics and add CI that requires an explicit bounded range whenever one
   kernel claims multiple `com.microsoft` schema versions, for in-tree EPs as well as
   plugins.
4. **Version-bump plumbing:** centralize the current and last-released
   `com.microsoft` versions (see [Domain version](#domain-version)), raise the domain
   maximum safely when the first post-release contract change opens a new opset, add
   the load-time error for an unsupported import, and update first-party producers to
   write the contrib opset explicitly.
5. **First version bump:** when the first post-API-30 contrib contract change lands,
   add its new schema in opset 2 and migrate its kernels and transformers across every
   EP that registers it.
6. **Fusion hardening:** convert all EP-specific contrib fusions from provider-name
   checks to exact target-schema/kernel checks.
7. **Enforcement:** flip missing-digest handling to strict once the gates in
   [Old plugin on a new core](#old-plugin-on-a-new-core) are met, then remove the
   temporary configuration one release later.

## Diagnostics

At INFO level, log one summary per plugin factory:

```text
CUDAExecutionProvider schema negotiation: com.microsoft 184/186 operators matched,
2 quarantined (GroupQueryAttention@2, SparseAttention@1)
```

At WARNING level, log actionable incompatibilities:

```text
Ignoring CUDAExecutionProvider kernel com.microsoft::GroupQueryAttention [2,2]:
plugin schema digest does not match ORT for GroupQueryAttention since_version 2.
```

When a fusion is skipped for this reason, VERBOSE logging should identify the
transformer, target operator version, assigned EP, and missing digest or kernel.
No per-node warning should be emitted during normal fallback to avoid log spam.

Because field failures are reported by users running at default severity, the
negotiation result must also be retrievable without re-running at INFO. Expose the
matched/quarantined counts and the quarantined operator list through EP metadata on
the session, and include the counts in telemetry so a regression is visible in
aggregate before it is reported.

## Test plan

### Schema tests

- Assert the registered `com.microsoft` current and last-released versions.
- When an opset is open, assert lookup of both the frozen and new schema contracts.
- Assert the domain maximum is still raised when another component pre-registered
  `com.microsoft`, and that a lower pre-existing maximum is reported as an error.
- Assert that registering an `OrtCustomOpDomain` named `com.microsoft` is rejected.
- After the first versioned contract is added, load models importing the frozen and
  open opsets and verify their resolved `Node::SinceVersion()` and formal contracts.
- Load an opset-2 model on a build registering only `[1, 1]` and assert the error
  message names the domain, requested version, and supported maximum.
- Regenerate digest manifests in CI and fail if a historical digest changes.
- Assert digests are unchanged by adding a new operator at `since_version 1`.
- Assert digests are unchanged across an ONNX submodule bump that only reformats type
  strings.
- Verify optimized-model serialization retains the correct contrib opset import, and
  that a graph needing only v1 schemas is not written with a v2 import.

### ABI negotiation tests

Extend the example plugin with selectable digest sets and kernels:

- core and plugin agree on every operator;
- core has only v1 of an operator, plugin reports v1 and v2;
- core has v1 and v2, plugin reports only v1;
- matching version with mismatched digest on one operator, asserting other operators
  stay available;
- missing callback, under both permissive and strict settings;
- duplicate `(domain, op_type, since_version)` entries with conflicting digests;
- malformed and unknown-domain entries;
- a `DISABLE_CONTRIB_OPS` core, asserting contrib kernels are filtered without a
  spurious digest-mismatch warning.

Verify compatible standard ONNX kernels remain available when contrib kernels are
quarantined.

### Kernel tests

- Preserve an open-ended v1 plugin kernel as an exact v1 match after the operator
  gains a v2 schema, without making it eligible for v2.
- Verify a `[1,1]` kernel does not match a GQA-2 node.
- Verify a `[2,2]` kernel does not match a GQA-1 node.
- Verify the same holds for in-tree CPU and CUDA GQA kernels in a static build.
- Verify explicitly registered compatible v1 and v2 kernels execute the correct
  implementation.
- Verify kernel type constraints still filter nodes after digest negotiation.
- Verify `com.ms.internal.nhwc` multi-opset kernel ranges are unaffected.

### Fusion tests

- New core + old plugin: a fusion targeting a schema from the open opset is not applied.
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
- a deliberately mismatched-digest build that must fail closed for the affected
  operator only.

### Synthetic v1/v2 experiment

`PluginEpSchemaCompatibilityTest.VersionedContribSchemaCompatibilityMatrix` exercises
the matrix against the real model loader, schema negotiation, effective plugin kernel
registry, and kernel lookup. It registers a test `com.microsoft` operator with these
contracts:

- OP1: one `tensor(float)` input and one output;
- OP2: a second required input, a second output, a required `axis` attribute, and
  `tensor(int32)` added to the type constraint.

CUDA0 has no API-30 callback and an open-ended v1 kernel registration. CUDA1 publishes
the OP1 digest and also has an open-ended exact-v1 kernel. CUDA2 publishes both OP1 and OP2
digests and retains bounded `[1,1]` and `[2,2]` kernels. "Fallback" below means the
CUDA plugin does not have an eligible kernel; another EP may execute the node if it
has an implementation.

| Core/model | CUDA0 | CUDA1 | CUDA2 |
|---|---|---|---|
| ORT0 + OP1 | CUDA v1 | CUDA v1 | CUDA v1 |
| ORT0 + OP2 | model rejected | model rejected | model rejected |
| ORT1 + OP1 | CUDA v1 | CUDA v1 | CUDA v1 |
| ORT1 + OP2 | model rejected | model rejected | model rejected |
| ORT2 + OP1 | CUDA v1 | CUDA v1 | CUDA v1 |
| ORT2 + OP2 | fallback | fallback | CUDA v2 |

The ORT0 rows emulate a pre-API-30 core by making the appended callback invisible;
they verify the ABI behavior after a plugin has loaded. They do not override the CUDA
package compatibility floor. The current CUDA plugin has
`MIN_ONNXRUNTIME_VERSION=1.30.0`, so an actual CUDA1 or CUDA2 package rejects ORT0 at
plugin initialization before reaching the behavior shown in those rows. This is the
intended deployment result described in [New plugin on an old core](#new-plugin-on-an-old-core).

The experiment establishes the following:

1. An old core rejects an OP2 model before EP assignment, regardless of how new the
   plugin is.
2. A new core can use CUDA1 for OP1 but does not reinterpret its open-ended or bounded
   v1 kernel as an OP2 implementation. OP2 falls back.
3. CUDA2 remains backward compatible only because it carries both historical and new
   manifest entries and kernels. Its future OP2 kernel is excluded from ORT1's
   effective registry, while its OP1 kernel remains usable.
4. Proper schema versioning protects ORT2 from CUDA0 even under the temporary
   permissive missing-callback policy: `KernelRegistry` treats an unbounded v1
   registration as an exact v1 match once a v2 node is resolved.

A companion control, `SameVersionContractChangeIsQuarantined`, labels the changed
contract as v1 instead. API-30 negotiation detects the different digest and removes
the v1 plugin kernel. The callback-free CUDA0 path cannot detect that mistake and
leaves the kernel enabled under the temporary legacy policy. This confirms both why
the API-30 digest is necessary and why every post-freeze breaking change must use a
new schema version; the permissive policy is not a general defense for arbitrary
pre-freeze or privately patched binaries.

## Implementation map

The main expected code areas are:

- the applicable contrib schema definition: preserve every last-released schema and
  add the changed contract in the open opset;
- `onnxruntime/core/graph/contrib_ops/ms_opset.h` and
  `onnxruntime/core/session/environment.cc`: opset registration and domain range,
  including the unconditional maximum raise;
- `onnxruntime/core/graph/schema_registry.cc`: domain-version resolution and the
  load-time error for an unsupported contrib import;
- `onnxruntime/core/session/custom_ops.cc` and
  `onnxruntime/core/session/provider_bridge_ort.cc`: reject registration of ORT-owned
  domain names;
- `include/onnxruntime/core/session/onnxruntime_ep_c_api.h`: factory ABI descriptor
  and callback;
- `onnxruntime/core/graph/schema_abi_digest.*`: canonical encoding and runtime digest
  computation;
- `onnxruntime/core/session/plugin_ep/`: callback validation, digest negotiation,
  and effective kernel filtering;
- `onnxruntime/core/providers/cuda/plugin/cuda_kernel_adapter.h`: bounded contrib
  kernel registrations;
- `onnxruntime/core/framework/kernel_registry*` and optimizer utilities: wrapper over
  the existing `TryFindKernel` overload;
- GQA CUDA/CPU/WebGPU/JS registrations and GQA transformers: explicit version use;
- `docs/ContribOperators.md` and its generator: regenerate for the v2 schema;
- `plugin-ep-cuda/MIN_ONNXRUNTIME_VERSION`: immediate containment and the first
  digest-aware release;
- `onnxruntime/test/autoep/`: ABI and cross-version matrix coverage.

## Alternatives considered

### Use only `ORT_API_VERSION`

Rejected. A stable C function table says nothing about operator input positions,
attributes, defaults, or semantics.

### Exchange only the maximum `com.microsoft` opset

Insufficient by itself. It cannot detect two builds that assign different contracts
to the same opset number, and it does not say which operators or configurations the EP
implements. The proposed operator digest and exact kernel lookup address both gaps.

### Hash the whole operator set at each opset

Rejected. The catalog at an opset changes whenever any operator is added, and new
contrib operators are added at `since_version 1` on most releases. A whole-catalog
digest would therefore change on most releases and quarantine every contrib kernel of
a plugin built one commit earlier, reducing the supported matrix to "same commit".
Requiring new operators to enter at the current maximum opset would avoid that, but at
the cost of bumping the domain version on nearly every release and making the
resulting models unloadable by older cores. Per-operator digests avoid both.

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
   semantics? If yes, open the next `com.microsoft` opset if current equals
   last-released, then create the new schema in the open opset. Adding a new operator
   is not a change to an existing contract and does not by itself require a bump.
2. Is the old schema definition still present and digest-identical to the checked-in
   manifest entry?
3. Are every EP's old and new kernel ranges truthful, for in-tree EPs as well as
   plugins? An open-ended registration is an exact start-version match; a kernel that
   intentionally supports multiple schema versions must use an explicit bounded range.
4. Does every optimizer that creates the op request the exact new version and verify
   target-EP support?
5. Can a graph with an explicit older domain import remain unchanged, or is a version
   converter required?
6. Does the serialized import stay at the lowest version the graph actually needs, so
   artifacts remain loadable by older cores when possible?
7. Do the old-core/new-plugin and new-core/old-plugin tests fail closed?
