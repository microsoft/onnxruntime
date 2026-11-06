# Devices and features

**Owns:** Provider configuration, context identity, adapter selection, optional
capabilities, and hardware eligibility. Plugin API contracts live separately.
**Update when:** Options/defaults, enumeration, feature negotiation, context
reuse, device-free modes, or hardware fast-path predicates change.

- Trace each option from public parser to effective context/kernel behavior.
  Reject malformed/trailing/out-of-range values and unsupported combinations;
  preserve absent versus explicit values and defaults in supported build modes.
  Public JS option types do not prove Node and browser/WASM parsers accept and
  forward the option. Check each affected binding's validation/conversion and
  add a session-creation regression through the documented public API.
- Adapter index is relative to enumeration configuration. Compare the explicit
  selector tuple (`adapterIndex`, `powerPreference`, `dawnBackendType`) during
  context reuse. A nonzero `deviceId` can identify a custom context requiring
  an instance/device pair; it is not automatically a physical adapter index.
  Power preference is a hint, not proof of GPU class or hardware execution.
- Gate optional `shader-f16`, subgroups, subgroup matrix, subgroup IDs, and
  subgroup-size-control on actual enabled device support. Adapter availability
  alone is insufficient for a supplied device. Reconcile enable/disable lists,
  required dependencies, limits, and conflicting requests.
- Separate necessary algorithm constraints from incidental vendor restrictions.
  Retain generic fallbacks, test ineligible cases, and inspect how many target
  model layers actually reach a new optimization. Configuration tables populated
  only in native code can break WASM despite shared shader sources.
- Device-free compile-only/virtual-device modes must not enumerate or allocate
  when the contract forbids it; reject inference on a virtual device clearly.
  Software adapter advertisement and real GPU execution are different evidence.
- Parser/compile-only tests can exit before enumeration, device creation, and
  context reuse. Add guarded runtime tests for those behaviors; test custom
  devices, unsupported features, selectors, and reuse rejection as applicable.

Source landmarks: `webgpu_provider_factory.cc`, `webgpu_context.cc`,
`webgpu_provider_options.h`, and `ep/factory.cc`.
Session-varying codegen options also need [cache identity](cache-identity.md).
