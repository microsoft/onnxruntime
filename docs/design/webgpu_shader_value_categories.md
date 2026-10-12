# Native WebGPU shader value categories

Native generators declare specialization fields and uniform slots explicitly.
Every other value used by generation must be common metadata, derived from
covered inputs, or immutable code/cache state. Host-only values remain outside
that interface. This refines [shader cache identity](webgpu_shader_cache.md).

PR #33242 supplies the merged correctness fixes and GPU regressions. This PR
changes the authoring contract and does not add another copy of those regressions.

## Categories

| Category | Examples | Handling |
| --- | --- | --- |
| Specialization | algorithm selector, Boolean broadcast read mode, an embedded epsilon | `WEBGPU_DECLARE_SPECIALIZATION` generates the complete stored type and encoder |
| Common metadata | types, ranks, vector widths, binding structure, workgroup sizes, pipeline overrides, uniform layout | Framework owns the encoding; generation uses restricted accessors |
| Derived | bias presence from input count, scalar type decisions, emitted type aliases | Compute from covered inputs; no separately supplied unkeyed field |
| Runtime payload | tensor elements, uniform values, dynamic extents, dispatch counts | Upload/bind normally; generator sees expressions, not host payloads |
| Immutable code/cache scope | literals, generator code, immutable device features and limits | Program/code identity or the owning device cache covers them |
| Host-only | diagnostic label, allocation and scheduling bookkeeping | Keep outside the specialization passed to generation |

The category belongs to a *use* of a value. One host variable can have multiple
roles. A channel count can be a runtime uniform while its chosen vector width
is structural metadata. QuickGelu alpha is a runtime uniform, but `alpha == 1`
is a specialized selector when it changes generated code.

C++ locals computed from covered inputs inherit that coverage. WGSL locals
computed from runtime tensor values remain runtime. Neither needs another
stored schema field. Initializing a local by reading mutable external state is
an undeclared input, not a valid derivation.

## When B can be omitted

Equal covered parents must imply equal B for every valid call in the same cache
scope. Merely observing that B changes whenever A changes proves the opposite
implication and is insufficient. For example, A=dtype and B=(dtype,width): changing
A changes B, but at one dtype B can still vary with width.

Implement a valid derivation as `B = f(A, ...)` during generation rather than
accepting both A and B and promising callers keep them consistent. Parents must
be covered by identity or immutable scope. A value calculated from a runtime
uniform is not automatically safe to use as a C++ code-generation selector.

The API intentionally has no `IgnoreKey` or user-defined equivalence predicate.
When independence is uncertain, retaining a redundant specialized field is safe.
Do not couple algorithm heuristics simply to save one key byte.

## Examples in this tree

| Program/value | Classification |
| --- | --- |
| LayerNorm `has_bias` | Derived from input count (x, scale, optional bias) |
| LayerNorm inverse-statistic presence | Derived from output count plus specialized mean presence |
| LayerNorm simplified/split/FP32 arithmetic selection | Specialization |
| LayerNorm epsilon and normalization extents | Uniform payload; its epsilon expression can be obtained from `Uniforms::epsilon` |
| Channel scale/shift components and workgroup size | Common metadata accessed through the helper |
| Channel scale/shift embedded epsilon | Specialization; existing literal formatting retained |
| Pool FP16 selection | Derived from the common input element type |
| Pool kernel rank and reduction strategy | Specialization; concrete pad/stride/kernel values remain uniforms |
| Binary int64 input/output decisions | Derived from common element types |
| Boolean vector/lane broadcast reads | Specialization; Boolx4 storage does not determine the read mode |
| Unary/binary expression text | Immutable `ShaderLiteral` code identity; selection remains keyed |
| Program name | Host-only label returned alongside specialization in `ShaderProgramSetup` |
| Einsum indexing recipe | Specialization with immutable borrowing; extents remain runtime |
| Einsum symbol sets and emitted loops | Derived on misses from the keyed recipe |

LayerNorm stores four flags instead of six. Channel scale/shift stores epsilon
instead of epsilon plus component/workgroup copies. Input/output counts still
participate in common identity. Output-role derivation relies on the operator's
valid optional-output combinations, not arbitrary combinations of flags.

## Edge cases

- Conditional specialization: canonicalize inactive fields in a host factory or
  retain their exact values conservatively. Both key and generator consume the
  resulting same configuration. No independent key-only normalization.
- Dynamic shapes: extents are uniforms unless explicitly marked `Shape`. The
  shape-specialization flag and effective extents are common metadata.
- Optional bindings and arrays: payload excluded, presence/type/active length
  covered when they affect code or layout.
- Views and offsets: view ownership/segments are structural; uniform offsets are
  payload. Offsets embedded into generated code remain keyed.
- Workgroup counts are runtime when read through builtins/indirect buffers;
  effective workgroup sizes and pipeline overrides are structural. A new source
  variant chosen from dispatch counts needs its own covered selector.
- Device capabilities/limits are exempt only while immutable for the owning
  cache. Mutable compilation policy requires identity or cache invalidation.
- Selecting among static fragments still affects identity. Runtime strings need
  content encoding; arbitrary addresses are not shader identity.
- Borrowed spans and string views must stay alive and immutable from key
  construction through source generation. Const access does not prevent another
  alias from mutating their storage.
- Cached objects may retain the first creation label; per-invocation profiling
  uses the current program's label. Labels never select generated arithmetic.

## Enforcement and cost

The compiler enforces final generated specialization types, static generator
signatures, supported encoding types and the restricted helper interface.
Factories sit outside stored configuration. Uniform slot references expose only
an expression; they do not expose values or convert to C++ numeric/Boolean types.
The Python policy rejects old open-member schemas, manual encoders/hints, and
legacy program construction. `--report` inventories the declarations and fields
from the production schemas; it is not a second key list.

Common scalar/tensor accessors and their encoding share declarations in
`shader_metadata.h`; conditional shape/view/atomic handling and pipeline/uniform
layout stay in the framework. Each new common property requires a matching
framework test and review of its source/layout effect.

Categories add no runtime tags or dependency graph. Warm lookups encode common
metadata and specialization, then use the existing ready/pending cache. Derived
locals and uniform-expression objects are evaluated during source generation on
misses. Uniform buffers retain their existing packing and upload path. Named
slots and layout come from one declaration; no duplicate UniformValues storage
or second upload pass is introduced. Positional host uniform packing is retained.

A key still allocates a string; variable-sized specialization still has encoding
cost. Conservative metadata may create equivalent shader variants. Removal of
redundant fields is a work reduction, not a universal end-to-end speedup claim.
Use local measurements against the merged upstream baseline; no permanent
performance benchmark or operator Cartesian-product matrix is added to CI.

The Clang helper/purity checker is explicitly deferred. Arbitrary C++ helpers,
mutable globals, I/O and environment reads cannot be ruled out by the current
source checker. Such reads violate the generator contract. The guarantee is
that supported declared inputs receive explicit handling, not a whole-program
purity proof or a security sandbox. Trusted framework functions and cache scope
remain review responsibilities.
