# Explicit categories for native WebGPU shader inputs

Status: proposed refinement of PR #33153, reviewed against `d3dd8e0a9f` on
2026-10-11. The API examples below describe proposed interfaces, not implemented
APIs. The existing implementation is described in [shader cache identity](webgpu_shader_cache.md).

PR #33242 merged as `8fc9fd597453baae1ad73379c313f0586d812867`. Its five fixes,
BiasSplitGelu FP16 correction, and regression tests form the correctness baseline.
This proposal addresses authoring and maintainability; it does not reintroduce
those fixes or another copy of their GPU regressions.

## Decision

Keep the existing exact key and ready/pending cache. Make the origin and handling
of every value crossing the shader-generation boundary explicit. Independently
supplied generation values must be keyed. Exemptions must use a representation
that explains why the value is already covered or cannot affect generation.

Use six categories, but only two per-program stored-value declarations:
specialization and uniform payload. Common metadata has a central declaration;
derived values are computations, constants belong to immutable code/scope, and
host bookkeeping stays outside the generator. Do not introduce a runtime
dependency graph, per-value provenance tracking, or an `IgnoreKey` annotation.

This is a stronger authoring contract, not a proof of arbitrary C++ purity.
The precise guarantee and its limits are described below.

## What must the key identify?

For a fixed immutable cache environment, equal keys must select compatible WGSL,
pipeline settings, binding layout, and uniform interpretation. Output tensor
values do not need to be equal: changing tensor data or a uniform is supposed to
reuse the same executable shader.

Conceptually:

```text
artifact = Generate(program identity, specialization, common metadata, environment)
key      = Encode(program identity, specialization, common metadata)
result   = Execute(artifact, uniform payload, tensor data, dispatch arguments)
```

Environment is omitted from each entry only when the cache's lifetime and
ownership already isolate it. Diagnostic labels are not computational identity.
Do not embed a per-invocation diagnostic label into generated source; preserve
invocation names separately in tracing, even if a pipeline retains its first label.

## Complete classification

Classify a value by its use, not its name, C++ type, or whether the ONNX attribute
is constant for one node. One host variable can produce several classified
representations, such as a runtime dimension and a vector-width selector.

| Category | Examples | Identity treatment | Authoring representation |
| --- | --- | --- | --- |
| **Specialization** | algorithm mode, Boolean broadcast read mode, epsilon embedded in WGSL | Exact value is automatically encoded | Generated `Specialization` field |
| **Common metadata** | shader types, component counts, ranks, binding structure, uniform schema, pipeline overrides | Framework owns the encoding; no duplicate operator field | Read-only metadata accessors backed by the central schema |
| **Derived value** | `has_bias = input_count == 3`, a type alias computed from input type | No additional field; its parents already determine it | Local expression or stateless function evaluated from covered inputs |
| **Runtime payload** | tensor elements, uniform epsilon, dynamic sizes/strides, dynamic buffer offsets | Values excluded; their structural interpretation is common metadata | Uniform payload or resource binding; generator sees an expression, not a host value |
| **Immutable code or cache-scoped state** | fixed constants, selected immutable WGSL fragment, device features | Covered by program/code identity or immutable cache environment | `constexpr`, existing `ShaderLiteral`, or framework-owned environment |
| **Host-only value** | log label, allocation bookkeeping, scheduling state | Excluded and unavailable to generation | Ordinary host code outside the generator inputs |

There is no seventh category for an independently supplied, unkeyed value that
the generator can read. If a value does not fit, keep it specialized until its
dependency or runtime representation is established.

Local variables do not each need a schema entry. A C++ generator local derived
from covered generation inputs inherits that coverage. A WGSL local derived from
runtime data is runtime data. The expression that emits its instructions is
still covered by program identity and specialization. A local initialized by a
hidden external read is an undeclared input, not a legitimate derived value.

## When can B be omitted?

The required condition is **equal covered inputs imply equal B**, for all valid
calls in the same cache environment. If A is the only parent:

```text
A(call1) == A(call2)  =>  B(call1) == B(call2)
```

"B changes whenever A changes" is the opposite implication and is insufficient.
For example, let A be dtype and B be `(dtype, vector_width)`. A change forces B
to change, but B can also change at the same dtype. Keying A alone loses width.

The default implementation of a valid dependency is to remove B from stored
configuration and compute `B = f(A)` in generation. Do not accept both A and B
from the host and merely document that callers keep them consistent. That would
create another obligation whose violation can silently corrupt the cache.

Additional rules:

- Parents must themselves be covered by identity or immutable scope. A value
  derived from an unkeyed uniform is not automatically safe for C++ generation.
- The whole parent set matters: `B = f(A, C)` cannot be exempted using A alone.
- A device-dependent function also depends on the cache environment.
- Choose one authoritative representation; cyclic declarations prove nothing.
- Prefer readable dependencies to speculative compression. Independently
  encoding an extra Boolean is safe when eliminating it would couple unrelated
  algorithm heuristics. The target is completeness, not the shortest possible key.

For example, host `channels` can be a uniform while `components =
GetMaxComponents(channels)` affects generated types. Its component count must
reach common metadata; it cannot be omitted merely because it was calculated
from channels. The generator subsequently derives components from the keyed
shader type, rather than rereading the uniform value.

## A small authoring API

Extend the field-schema approach already in #33153. Generate complete closed
types rather than permitting additional fields or handwritten encoders inside
an author-defined `Config` class. Move custom initialization into host factories
that construct those generated types.

Illustrative syntax:

```cpp
#define EXAMPLE_INPUTS(SPECIALIZE, UNIFORM) \
  SPECIALIZE(bool, use_fast_path)         \
  UNIFORM(uint32_t, element_count)         \
  UNIFORM(float, epsilon)

WEBGPU_DECLARE_INPUTS(ExampleInputs, EXAMPLE_INPUTS);

struct ExampleShader {
  using Inputs = ExampleInputs;
  static Status GenerateShaderCode(const Inputs::Specialization& values,
                                  ConfiguredShaderHelper& shader) {
    const auto& input = shader.AddInput("x");
    const auto components = input.NumComponents();  // Covered by common metadata.
    auto epsilon = shader.Uniform<&Inputs::UniformValues::epsilon>();
    // epsilon.Expr() emits "uniforms.epsilon"; it is not a float.
    // if (epsilon == 0.0f) is a type error.
    return EmitBody(values.use_fast_path, components, epsilon.Expr(), shader);
  }
};
```

The declaration generates specialization storage/encoding, named uniform slots,
and the uniform payload description. Specialization encoding visits only the
specialization fields; it must not scan a combined list and skip uniform values
on every lookup. The existing dispatch uniform buffer remains the runtime
storage destination, without a second payload copy introduced by the schema.

The generator signature exposes specialization and restricted helper capabilities.
It receives neither `UniformValues` nor a general host parameter bundle.
`UniformRef<T>` only describes an expression and its type; it offers no numeric
conversion or accessor to the payload. It is created during generation, not as
per-value tracking on every warm lookup. This is a small typed reference to
existing emission, not a replacement language for all WGSL.

Retain the existing uniform declarations while migrating if needed, but the final
authoring API must not require two lists of the same uniform names/types. Scalar,
array, optional, and packed FP16 uniform representations need the same boundary:
payload is runtime; declared type, active length, and layout are structural.

Common properties use a separate, framework-owned schema. A newly exposed
metadata property must either be encoded by that schema or be implemented as a
derived accessor. Do not maintain one unrelated getter list and encoder list.
Common getters use authoritative program state, not an operator-supplied copy.
Expose effective workgroup sizes, resolving default zero to the actual default;
the current helper returns raw overrides, so this needs explicit handling when
removing duplicated workgroup fields.

Derived values need no generic `Derived<T>` container. Ordinary stateless
functions and locals are sufficient. Similarly, host-only variables need no
`HostOnly<T>` wrappers: they are excluded by the generator's interface.

## Concrete classification of existing programs

These are authoring changes relative to #33153, not additional fixes to #33242.

### LayerNorm

| Current value | Proposed treatment |
| --- | --- |
| `has_bias` | Derive from `InputCount() == 3` under the existing x/scale/optional-bias interface |
| `simplified` | Specialization |
| `has_mean_output` | Specialization; distinguishes the two one-statistic cases |
| `has_inv_std_dev_output` | Derive from output count and mean presence |
| `split_norm_dim` | Specialization: semantic selection, not an assumption about a coincidentally related workgroup size |
| `fp32_normalization` | Specialization; retain the explicit semantic override |
| input/output dtype, rank, vector width | Common metadata |
| `components`, `normalize_in_f32`, type aliases | Derive from common types and specialization |
| `epsilon`, `norm_count`, `norm_size`, `norm_size_vectorized` | Uniform payload; existing host derivation and packing retained |
| dispatch counts | Runtime dispatch; any source-changing dispatch mode must be represented separately |
| WGSL mean/variance temporaries | Runtime shader locals |

For valid LayerNorm output configurations:

```cpp
const bool has_bias = shader.InputCount() == 3;
const bool has_inv_std_dev = shader.OutputCount() == 3 ||
                            (shader.OutputCount() == 2 && !values.has_mean_output);
```

The operator interface already determines which combinations are valid. Do not
permit an arbitrary role flag/count combination through a second authoring path.
This reduces stored specialization Booleans from six to four. It does not
require a runtime dependency check or an additional GPU branch.

### InstanceNorm's channel scale/shift program

Its current configuration has `components`, `epsilon`, and `workgroup_size`.
Keep embedded epsilon as specialization, and derive components and effective
workgroup size from common metadata. Three stored fields become one.

Moving epsilon into a uniform is a separate numerical/performance choice. It is
not required for this refinement; preserve #33242's corrected behavior and the
existing literal formatting. Exact float encoding may conservatively distinguish
values that format to identical WGSL, which is safe.

### Binary elementwise and activation

- Keep expression identity (`ShaderLiteral`) and implementation selector covered;
  several operators share the same C++ generator type.
- `is_int64_input`/`is_int64_output` derive from common tensor element types.
- Boolean vector-versus-lane read modes remain specialization: `Boolx4` alone
  does not determine them. The merged regression remains the counterexample.
- Keep broadcast/scalar/algorithm choices specialized unless their complete
  dependency follows from the effective metadata actually exposed. Flattened
  rank is not necessarily the original ONNX rank.
- `program_name` is host diagnostics, provided computational selection continues
  to be covered by expression/implementation identity. Keep per-invocation names
  correct when cached objects retain the first label.
- Activation alpha is normally uniform payload. QuickGelu's `alpha == 1`
  optimization creates an additional specialization projection. The generator
  reads that selector, not the host alpha value. This explicitly represents
  the two roles of one host variable.

### Pool and Einsum

Pool's `is_float16` can derive from common element type. Kernel rank must remain
specialized until it is derived from an authoritative keyed dimension/array
length; do not assume tensor rank always supplies it. Kernel sizes, strides,
pads, and dilations remain runtime payload where the existing generator reads
uniforms. A reduction-strategy choice computed from shape is still specialization.

Einsum's indexing recipe is specialization; concrete extents remain runtime.
Derived symbol sets and loop-emission decisions stay local to generation.
Keep the existing borrowed immutable recipe when its lifetime spans key encoding
and generation. Do not replace it with a newly allocated dependency graph or
deep copy. If generation becomes deferred beyond that lifetime, ownership must
be changed explicitly; `const` access alone does not freeze aliased storage.

## Other situations requiring explicit handling

| Situation | Required rule |
| --- | --- |
| Field matters only in one algorithm branch | Use a host factory to canonicalize inactive fields, or a tagged alternative with only active fields; generation consumes that same canonical representation |
| Dynamic shape versus static specialization | Dynamic extents are payload. `Shape` specialization records effective extents in common metadata. A mode computed from dynamic shape is separately covered |
| Uniform array or optional binding | Payload excluded; length/type/presence/role that changes source or layout remains covered |
| Buffer handles, views and offsets | Resource contents/handles are runtime; binding ownership, segments, atomic/access mode are structural. An embedded offset is keyed; a uniform offset's value is not |
| Workgroup counts versus size | Counts are runtime when read via builtins/indirect data; size and pipeline override values are common metadata. Any count-dependent source specialization needs a covered selector |
| Device, features, limits, backend options | Exempt only when immutable for this exact cache. Mutable compilation policy requires keyed state or cache invalidation |
| Static text versus a runtime string | Fixed text is code identity; choosing between fixed fragments is still identity. Runtime text needs exact content encoding |
| Template constants | Covered by distinct generator identity for distinct template instantiations; runtime-selected code cannot silently share that identity |
| Host-only data later used for generation | Promote the necessary projection to specialization/common metadata; do not grant the generator access to the host object |

Do not introduce user-provided equivalence predicates, arbitrary key overrides,
or annotations asserting that an independently supplied value is "already keyed."
Canonicalization must be deterministic and shared by key construction and source
generation. Retain exact equality after hashing; a digest alone is not identity.

## Enforcement and its limits

Use the compiler plus the existing source-policy job, not an operator variant
matrix as the mechanism for completeness.

1. Generated closed types make every supported stored field specialization or
   uniform payload. Unsupported field types and handwritten encoders are rejected.
   Replace the current open `WEBGPU_CONFIG_MEMBERS` escape with generated types
   and host factories; merely adding more regexes around it is insufficient.
2. The static generator cannot capture an operator instance. Uniform/resource
   payloads and host bookkeeping have no accessor through its interface. Common
   getters are generated/derived from the central schema. Enforce the boundary
   through transitively returned helper objects as well as top-level methods.
3. Derived values are computed from covered parents, not independently stored
   exclusions. Keep small pure helpers explicit. A wrong formula is an operator
   bug, but equal covered inputs still produce the same artifact.
4. Extend the existing source policy to reject legacy construction, unclassified
   schema declarations, mutable generator members, and direct context/tensor or
   environment access in generator code. Reuse its existing CI invocation.
5. Review shared helpers and immutable cache scope as trusted framework code.
   A new exposed property or pipeline descriptor field needs a category and one
   focused framework check before it is accepted.

The **mechanical guarantee** covers fields admitted through these interfaces:
there is no supported independently supplied generator field that silently opts
out of identity. It does not prove that all arbitrary C++ free-function calls are
pure. A helper can hide global state, I/O, aliased mutation, or environment reads;
the current regex-based checker cannot reliably discover those transitively.
Pointers/member casts and deliberate framework bypasses are also outside this
contract. Do not describe a field-count check as a whole-program proof.

A strict guarantee for unrestricted C++ would require much stronger whole-program
analysis, or restricting generation to a fully controlled language. Neither is
recommended for this PR. The proposed tradeoff is closed inputs, no opt-out path,
a small policy check, and a narrow reviewed framework boundary. Residual purity
limitations must remain explicit in the PR description.

### Optional stronger gate: a restricted C++ helper subset

If rejecting hidden-state reads is a requirement, a conservative Clang-based
check can restrict the C++ accepted for generator functions and their reachable
helpers. This is not a compiler `pure` attribute: an annotation selects functions
for checking, rather than trusting the author's assertion of purity.

Allow reads of explicit generation inputs, local values and verified immutable
constants; allow mutation of owned locals and the designated emission builder.
Reject mutable global/static/thread-local reads, environment/I/O/time/random
access, unclassified pointer access, casts that bypass the boundary, and calls
whose effects are unknown. Every call must either have a checked definition or
be an explicitly trusted framework/library operation. Follow nested calls,
constructors/destructors, overloaded operators and lambdas, not only direct
function-call syntax. Reject unresolved indirect/virtual dispatch. A constant
pointer to mutable storage is not an immutable value.

Unknown code fails closed. This avoids having to decide the behavior of every
possible C++ program, at the cost of rejecting some safe programs. Template
instantiations, external definitions and supported conditional configurations
must be covered before claiming enforcement. Standard formatting facilities can
read locale or other state, so trusting all standard-library functions would
defeat the check. Prefer narrowly approved deterministic emission primitives.

Such a gate adds no inference-time tracking and needs no GPU variant matrix.
It does add a Clang tooling dependency, parsing/build time, checker fixtures and
maintenance of the trusted boundary. The cost and compatibility with existing
generators need a pilot before adoption. The existing regex source checker is
not sufficient to implement this guarantee.

The guarantee is conditional on the checked subset, immutable input lifetime,
and trusted primitives; it is not a security sandbox for hostile native C++.
For a smaller trusted authoring language, generators could instead use a closed
template language with approved builtins and no native-code escape. Current WGSL
templates allow raw/expression code to become C++, so using those templates as
they stand does not close this gap. A restricted language is a larger migration.

## Runtime and test cost

Categories are authoring/build-time concepts; do not store category tags on each
value or traverse dependencies at runtime. Keep one existing ready/pending cache
and the current exact binary encoding.

| Operation | Expected change relative to current #33153 |
| --- | --- |
| Warm lookup | Encode common metadata and specialization only, then perform the existing lookup |
| Derived metadata | No duplicate config storage or encoding; compute cheap locals on cache misses |
| Uniform handling | Same payload upload and layout; generated named slots replace author duplication, not add another pass |
| Constant WGSL | Retain immutable code identity; do not hash/copy static text on hits |
| Large recipes | Keep existing immutable borrowing; no mandatory new allocation |
| Cache miss | A few ordinary derived expressions, followed by existing source generation/compilation |

LayerNorm removes two Boolean encodings; channel scale/shift removes two integer
encodings. Those are concrete reductions in work, not an end-to-end speedup claim.
The key still allocates a string, common metadata can be conservatively redundant,
and optional canonicalization has host cost. Avoid expensive normalization merely
to save a few key bytes. Do not introduce source regeneration/comparison on hits.

Previous A/B results measured `d3dd8e0a9f`, not this proposal or a build based on
merged #33242. They cannot validate the refinement. At implementation time compare
the refined branch with current upstream including #33242 using the existing
local harness: small warm operators, LayerNorm, an allocation-sensitive Einsum
case, and representative inference. Report key cost, allocations, pipeline count,
end-to-end time, and uncertainty separately. Do not add performance runs to CI.

Keep the permanent checks bounded:

- Reuse #33242's GPU regressions once; remove the duplicates from
  `configured_program_test.cc` when updating #33153 to the merged baseline.
- Use tiny synthetic programs for framework identity tests, rather than assuming
  each old operator-config flag must always produce a distinct key.
- Test common metadata versus uniform payload, derived equality, active/inactive
  specialization, static/dynamic shape, and exact encoding in small CPU cases.
- Reuse the existing Python policy job and compiler targets. Compile-time checks
  should reject payload-to-host numeric conversion and unsupported schema types;
  use a few negative snippets, not one compiler invocation per operator/field.
- Run affected existing GPU suites for implementation changes. No new Cartesian
  product of operator types, shapes and attributes is required.

## Comparison and implementation order

| Aspect | Current #33153 | Proposed refinement |
| --- | --- | --- |
| Stored config | All declared fields keyed; custom structs can mix construction and schema | Closed generated specialization/uniform types; initialization outside them |
| Common information | Captured centrally, but some operators also store copies | One authoritative accessor/schema; copies become derived locals |
| Derived information | May remain redundantly stored/keyed | Computed from covered parents; no unkeyed stored derivative |
| Runtime information | Uniforms separate from config by convention/API | Explicit payload declaration and expression-only access in generation |
| Exemption explanation | Mostly comments and review | Distinct API placement for each category; no generic ignore flag |
| Completeness limit | Helper purity not formally established | Same honest limit, smaller permitted input surface |
| Lookup machinery | Exact binary key; existing cache | Retained |

Recommended implementation sequence:

1. Update the working baseline to include #33242 while preserving its numerical
   fixes and tests. Remove duplicate bug-fix claims and duplicate GPU tests from
   #33153. Reconcile newer upstream code rather than restoring an older snapshot.
2. Add the closed declarations and central metadata contract, preserving existing
   encoding/storage paths. Start with LayerNorm and channel scale/shift to verify
   both dependency removal and unchanged uniform behavior.
3. Migrate every generator currently present on that updated baseline, including
   shared/template/vendor generators. The old count of 187 is historical, not an
   acceptance threshold after upstream adds programs.
4. For each old config member, record its new category and, for derived values,
   its authoritative parents. For each uniform record value versus structural
   metadata. Audit free generator inputs and every shared-helper accessor as well.
   Any unresolved value remains specialized or blocks that generator's migration.
5. Have the policy checker report schema/program coverage; keep detailed migration
   evidence with the PR. Do not add a second hand-maintained production key list
   or a runtime registry just to produce the report. Mechanically counted fields
   and reviewed helper purity are separate evidence.
6. Run the small framework checks, existing operator tests and local performance
   comparison, then update the PR description around this final authoring contract.

This document completes the design proposal. The refined API, complete migration
inventory, compiler rejection checks and new performance measurements are not yet
implemented. No production behavior changes or new performance results are
claimed by this document.
