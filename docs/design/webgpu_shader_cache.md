# Native WebGPU shader cache identity

Native programs use `ConfiguredProgram<Spec>`. A static generator receives a
closed specialization type and `ConfiguredShaderHelper`. The framework encodes
specialization and common structural metadata before querying the existing
device-local ready/pending pipeline cache.

Within an immutable device environment, equal keys must describe compatible WGSL,
pipeline settings and binding/uniform layout. Runtime tensor values and uniform
payloads can change while the executable shader remains reusable. The
[value-category contract](webgpu_shader_value_categories.md) explains each
category, dependency rule and exception.

## Declare inputs

```cpp
#define EXAMPLE_SPECIALIZATION(F) F(bool, use_fast_path)
#define EXAMPLE_UNIFORMS(U) \
  U(Uint32, element_count)  \
  U(Float32, epsilon)

struct ExampleShader {
  WEBGPU_DECLARE_SPECIALIZATION(Config, EXAMPLE_SPECIALIZATION);
  WEBGPU_DECLARE_UNIFORMS(EXAMPLE_UNIFORMS);
  static constexpr std::string_view name = "Example";
  static Status GenerateShaderCode(const Config& config,
                                  ConfiguredShaderHelper& shader);
};
#undef EXAMPLE_SPECIALIZATION
#undef EXAMPLE_UNIFORMS
using ExampleProgram = ConfiguredProgram<ExampleShader>;
```

Each specialization field automatically participates in exact encoding. Generated
specialization structs are final aggregates and cannot be extended with extra
members. Use a static `Spec::MakeConfig(...)` host factory when initialization
requires conversion, validation or derivation. It returns the generated Config.
Do not write an encoder, schema marker or open configuration class.

Uniform declarations generate the existing layout metadata and named slots.
`shader.Uniform(Uniforms::epsilon).Expr()` emits `uniforms.epsilon`, without
exposing its payload to C++. Host code still supplies the values through the
existing `AddUniformVariables`/`AppendUniforms` packing path. Uniform types and
active lengths participate in common identity; their values do not.

For dynamic diagnostic names, a factory returns `ShaderProgramSetup<Config>`
with `{specialization, name}`. The wrapper copies the name to its host-side
ProgramBase and passes only specialization to the generator. Source selection
must be represented by the specialization, code identity or common metadata,
not by the diagnostic label.

## Common, derived and runtime values

The framework captures shader/element types, ranks, counts, static-shape mode,
segments, buffer-view ownership, embedded offsets and output atomic status.
Shader types include vector width. It also records effective workgroup sizes,
subgroup size, indirect-dispatch presence, override values, index ranks and
uniform schema lengths/types. Shared declarations in `shader_metadata.h` generate
common property accessors and encoding; conditional layout handling stays central.

A generator obtains type/shape expressions through its restricted helper. It has
no live Tensor, ComputeContext, ProgramBase or uniform-payload accessor. Helper
objects returned by that interface must preserve the same boundary.

A derived value is computed from already-covered parents. For example, LayerNorm
bias presence follows from input count; inverse-statistic presence follows from
output count plus mean presence. Do not independently store a derived value and
exempt it from encoding. When the dependency is unclear, keep it specialized.

`ProgramTensorMetadataDependency::None` uses dynamic extents but still keys types
and ranks. `Shape` specializes on effective dimensions and keys them centrally.
Only capture the shape information generation needs: rank, an indexing recipe,
or explicit static extents. Runtime dispatch counts, buffer handles, tensor data
and uniform offsets remain outside identity unless they also select a source or
pipeline variant.

Static generator metadata and literal constants belong to code identity. Device
features and limits are scoped by the device cache. Mutable policy must be keyed
or invalidate that cache. Hidden globals/environment reads violate this contract;
arbitrary C++ helper purity is not mechanically proven. The proposed Clang checker
is deferred.

## Storage and lookup

Supported scalars, enums, strings, sequences/maps and nested closed schemas have
exact encodings. Sequence/string lengths delimit variable-size fields; float bits
are preserved. Unsupported objects and arbitrary pointers are rejected. The cache
compares the complete binary key after hashing, so a hash collision cannot alias
unequal keys.

Use `ShaderLiteral` for immutable static WGSL. Its immediate constructor accepts
constant static character arrays, allowing code identity instead of copying and
hashing their contents on every hit. Runtime strings retain content encoding.
Borrowed spans/string views must remain alive and immutable through encoding and
generation. Einsum uses this to avoid an extra recipe allocation on warm calls.

A warm lookup constructs one key and queries the existing cache. It does not
regenerate WGSL, run a dependency graph, or query a second pipeline cache.
Min/Max/Pow helper text and derived Einsum symbols are generated only on misses.
Key construction still allocates a string, and conservative specialization may
produce distinct keys for equivalent shaders. This is not source deduplication.

Keys use process-local generator/code identity and are not persistent disk-cache
identifiers. Use `ProgramManager::KeyForLogging` for dumps and profiling. It assigns opaque
`shader:<id>` labels without exposing addresses. Its synchronized diagnostic table
is populated only when logging, dumping, profiling or an error requests a label,
and is released with the ProgramManager. No diagnostic lookup runs on quiet,
unprofiled cache hits; diagnostic mode uses extra memory proportional to its
distinct observed keys.

## Validation and maintenance

`tools/python/webgpu/check_shader_config.py` rejects legacy program construction,
manual hints/encoders, open-member schemas and legacy uniform declarations. Shader
and pipeline creation remain owned by ProgramManager. `--report <path>` writes a
field inventory derived from the declarations. The policy is part of the existing
WGSL test job and incremental build; it does not require Clang.

Small CPU framework cases cover specialization, common metadata, excluded runtime
payloads, effective defaults, uniform expression types, host labels, canonical
inactive fields, exact equality and code identity. The numerical regressions
merged in #33242 remain the single permanent GPU coverage for those collisions.
Existing operator suites validate generator migrations; no exhaustive variant
matrix is required merely to test automatic field encoding.

When adding a helper-visible property, classify it and update common identity as
needed. Review transitive helper behavior, borrowed lifetimes and cache scope.
A field inventory accounts for declared inputs; it does not prove absence of all
hidden state in arbitrary C++.
