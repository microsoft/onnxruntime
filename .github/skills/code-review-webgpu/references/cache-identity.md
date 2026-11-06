# Cache identity

**Owns:** Whether a cached program can safely serve multiple calls and sessions.
**Update when:** Cache composition, specialization, or context-sharing contracts
change. Keep execution bounds in dispatch-and-wgsl and validation commands in validation.

## Audit the complete identity

This check has the highest value at review time. Nightly shader-key validation
occurs after merge; it does not replace reviewing the key against the generator.

1. List every value the shader text depends on: members and attributes, data
   types, rank and component counts, shape-derived vectorization, feature or
   vendor branches, provider options, and `.wgsl.template` parameters.
2. For each value, find what keys it: `CacheHint`, an input/output
   `ProgramTensorMetadataDependency` (`Type` keys element type, `Rank` the rank,
   `Shape` dimensions), component count, or workgroup/subgroup size. Confirm
   against `CalculateProgramCacheKey` in
   `onnxruntime/core/providers/webgpu/program_cache_key.cc` rather than assuming.
   Inspect overridable pipeline constants too: setting an override is not proof
   its value is automatically in the key; trace its identity through the cache.
3. Report any unkeyed specialization as blocking. Give two calls differing only
   in that value that share a key and therefore reuse the wrong pipeline.

The cache belongs to a `WebGpuContext`. Device-fixed features and limits need
no additional key; options varying between sessions sharing that context do.

## Common misses

- Concrete dimensions in WGSL loop bounds, unrolled loops, or `const` arrays
  with dependency `TypeAndRank` or `None`.
- Component-count branches or shape-derived fast paths without matching keys.
- A new member in `GenerateShaderCode` or template argument without a key.
- Floats keyed by `std::to_string`, which keeps six decimals and loses identity.
  Encode keyed values exactly.

Pass per-run sizes, strides, offsets, and scalar attributes as uniforms using
`WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES` and `AddUniformVariables`. Formatting
them into WGSL compiles a new pipeline per distinct value. Conversely, an extra
key component that does not change the WGSL only duplicates pipelines; report
it when it causes measurable compile churn.

See [validation](validation.md#shader-generation-and-cache-keys) for shader dumps
and the key validator. A dump must exercise differing specializations to test reuse.
