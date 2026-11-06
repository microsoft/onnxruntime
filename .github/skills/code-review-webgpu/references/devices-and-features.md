# Devices and features

**Owns:** Provider options, context identity, adapter selection, enabled features,
and hardware eligibility; plugin ABI lives in [runtime](runtime.md#plugin-contracts).
**Update when:** Parsers/defaults, enumeration, negotiation, context reuse,
device-free modes, or fast-path predicates change.

- Trace public options to effective behavior in every affected binding/build.
  Reject malformed/trailing/out-of-range values and unsupported combinations;
  preserve absent/explicit values and defaults. JS types do not prove Node or
  browser/WASM forwarding: require documented-API session-creation coverage.
- Adapter indices depend on enumeration configuration. Compare the explicit
  (`adapterIndex`, `powerPreference`, `dawnBackendType`) tuple on context reuse.
  Nonzero `deviceId` may select a custom instance/device pair, not a physical
  adapter index. Power preference proves neither GPU class nor hardware execution.
- Gate `shader-f16`, subgroups, subgroup matrix/IDs, and subgroup-size-control
  on **enabled device** features, including supplied devices. Reconcile
  enable/disable lists, dependencies, limits, and conflicts; adapter support alone
  is insufficient.
- Separate algorithm constraints from vendor restrictions; retain/test generic
  and ineligible fallbacks. Measure actual model-layer eligibility. Native-only
  configuration tables can break WASM despite shared shader sources.
- Device-free compile-only/virtual modes must avoid forbidden enumeration or
  allocation and reject virtual-device inference clearly. Software-adapter
  advertisement does not prove hardware execution.
- Parser/compile-only tests may bypass device creation/reuse. Add guarded runtime
  coverage for affected custom devices, unsupported features, selectors, and
  reuse rejection.

Inspect `webgpu_provider_factory.cc`, `webgpu_context.cc`,
`webgpu_provider_options.h`, and `ep/factory.cc`. Session-varying codegen options
also need [cache identity](shaders.md#cache-identity).
