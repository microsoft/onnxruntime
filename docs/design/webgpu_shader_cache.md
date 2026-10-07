# Native WebGPU shader cache identity

Native WebGPU programs use `ConfiguredProgram<Spec>`. Each generator receives one
declared configuration and a restricted `ConfiguredShaderHelper`. The framework
encodes the configuration and the metadata exposed by that helper before looking
up the existing device-local pipeline cache. Operators do not supply cache hints
or select whether tensor types and ranks participate in identity.

The invariant is: within one device environment, equal keys describe the same
generated shader, pipeline settings and cached binding/uniform layout. This
depends on generators and their helpers reading only declared inputs and immutable
code. Mutable globals, environment reads and other hidden generation state violate
the contract.

## Declare a program

```cpp
#define WEBGPU_EXAMPLE_CONFIG(F) \
  F(bool, has_optional_output)  \
  F(float, embedded_epsilon)

struct ExampleShader {
  WEBGPU_DECLARE_CONFIG(Config, WEBGPU_EXAMPLE_CONFIG);
  static constexpr std::string_view name = "Example";
  static Status GenerateShaderCode(const Config& config, ConfiguredShaderHelper& shader);

  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES({"size", ProgramUniformVariableDataType::Uint32});
};
#undef WEBGPU_EXAMPLE_CONFIG

using ExampleProgram = ConfiguredProgram<ExampleShader>;
```

The field list declares the data and its encoder together. Adding a declared
field automatically changes identity when that value changes. A configuration
requiring a constructor can use `WEBGPU_CONFIG_MEMBERS(fields)` inside its final
`Config` struct. Do not add data members outside the macro or write an encoder.

The program owns its configuration and exposes it only as a const reference.
The static generator has no receiver carrying additional instance state. The
wrapper checks the generator signature, final configuration type and absence of
instance fields on `Spec`. Direct construction through the old `Program<T>` base
and manual `CacheHint` calls are unavailable.

Scalar values, strings, supported sequences/maps, and nested schema objects have
exact encodings. Pointers and arbitrary object representations are rejected.
Convert host objects to the values that generation actually needs: for example,
a tensor-presence Boolean, a dtype enum, an owned permutation, or the activation
kind and its source-selection flags. Do not copy numerical activation parameters
into configuration when the shader reads those parameters from uniforms.

String views and spans must refer to immutable values that remain alive throughout
program execution and source generation. Their contents, rather than their
addresses, participate in identity. Einsum borrows its invocation's const indexing
recipe this way, avoiding additional recipe allocations on cache hits.

## Runtime data and shape specialization

The framework always captures input/output shader types (including vector width),
element types, ranks, counts, segment counts, buffer-view ownership and output
atomic status. Non-uniform buffer offsets are keyed; dynamic view offsets are
uniform values. It also captures index ranks, uniform types and lengths, override
constant values, workgroup/subgroup settings and indirect-dispatch presence.
Static generator metadata is identified by `Spec`; device capabilities and limits
are scoped by the existing device-local cache.

Ordinary uniform payloads, buffer handles and dispatch counts do not enter the
key. Changing runtime data should reuse a shader with the same structure.

`ProgramTensorMetadataDependency::None` uses dynamic dimensions. Types and ranks
are still keyed automatically. The restricted helper adds shape uniforms when
needed. `ProgramTensorMetadataDependency::Shape` explicitly specializes on the
effective dimensions; those dimensions are included in the key. This preserves
existing static-shape optimizations without exposing a live tensor to generators.

Avoid putting an entire host shape into configuration when generation only uses
its rank. Similarly, use an indexing recipe rather than concrete extents when
the shader reads extents from uniforms. Pool, GatherBlockQuantized and Einsum
illustrate these distinctions.

## Lookup and diagnostics

The compact binary string contains exact values, with length delimiters for
variable-sized fields and bit-preserving floating-point encoding. The existing
map checks full string equality after hashing; a hash collision cannot alias
different keys. Ready-cache and pending-build reuse use the same identity.

A warm lookup constructs a key and queries that cache. It does not regenerate
WGSL, compile a pipeline or query an additional pipeline cache. Key construction
still allocates a string, and variable-sized configuration contributes copying
and hashing work. Conservatively distinct configurations may compile identical
source; this implementation does not deduplicate them by source.

Generator identity uses a process-local token. Keys are not stable disk-cache
identities. Use `ProgramCacheKeyForLogging` for diagnostics, shader dumps and
profiling; binary keys can contain NUL bytes.

## Validation and maintenance

The build runs `tools/python/webgpu/check_shader_config.py` after source changes.
The existing WGSL template test job also runs this check and its small unit suite.
It rejects legacy program inheritance, manual hints/encoders and ordinary extra
configuration member declarations outside the schema. Shader and pipeline creation
must remain in `ProgramManager`. These checks complement
the C++ API restrictions; they are not a proof of arbitrary C++ helper purity.

`ConfiguredProgramTest` covers automatic fields, type/vector metadata, excluded
uniform values, exact equality under forced hash collisions, length delimiters,
printable diagnostics, and a small set of known GPU collisions in both orders.
Existing operator suites validate the migrated generation logic. No Cartesian
product of every operator's variants is required to test field encoding.

When extending the helper or pipeline descriptor, update central identity for
every newly observable property and add a focused framework test. Review new
generation helpers for hidden mutable state. Retain a numerical regression when
a concrete cache collision is found: two individually correct cold variants can
still expose a warm-cache defect when executed together.

Key fields need not be mathematically independent. For example, one BiasAdd
input's shader type determines the common dtype/vector width, and LayerNorm's
output count plus one role bit distinguishes its optional statistics outputs.
The framework may conservatively capture redundant structural metadata to avoid
requiring every operator author to maintain such implications. An absent field
is a defect only if the complete key fails to distinguish incompatible artifacts.
