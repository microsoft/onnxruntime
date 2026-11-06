# Plugin contracts

**Owns:** Plugin API/ABI boundaries, host negotiation, capability parity,
factory/device/allocator contracts, and static/shared plugin integration.
**Update when:** EP API use, minimum runtime, exported symbols, adapter helpers,
memory-device semantics, or native/plugin support predicates change.

- Distinguish plugin release version, build SDK/API-table bounds, minimum host
  version, and runtime negotiation. Check API plus patch-release compatibility
  as encoded by `ep/runtime_compatibility.*`; a fix in one release line does not
  imply support in an intervening line. Preserve conservative compatibility and
  explicit override behavior. Mocks do not establish multi-runtime GPU support.
- Trace API initialization before using wrappers or exception-to-status macros.
  A failing negotiation still needs a supported way to report status. Audit
  `noexcept` C callbacks, including transitive allocation/Free/destruction;
  contain exceptions using the boundary's conventions and preserve cleanup.
- Compare native `GetCapability` and plugin `ep/ep.cc` predicates for every
  affected operator, attribute, dtype, feature, and layout. Common kernel source
  does not prove equal acceptance. Verify graph support and kernel execution
  together, including rejected nodes and supported CPU partition boundaries.
- Keep `OrtMemType`, `OrtDeviceMemoryType`, and memory-device identity distinct.
  Verify vendor/device IDs, default versus read-only allocators, session versus
  environment ownership, streams/notifications, and release order against the
  host ABI. Follow [runtime ownership](runtime-and-portability.md) for ordering.
- Shared plugins need exact exported names and usable runtime dependencies;
  static plugins need prefixed entry points and correct API initialization.
  Check factory registration/unregistration and context/logger/library teardown.
- New ORT-internal dependencies must be available in the standalone/plugin
  source set (`plugin-ep-webgpu/paths.txt`) and EP adapters. Inspect shared
  adapter consumers such as CUDA before changing supposedly WebGPU-only code.
  Review native, shared plugin, static plugin, and WASM integration as affected.

Source landmarks: `ep/api.cc`, `ep/ep.cc`, `ep/factory.cc`, `ep/allocator.cc`,
`ep/sync_stream.cc`, public EP API/adapters, and session `ep_factory_webgpu.*`.
See [build/deployment checks](build-and-dependencies.md).
