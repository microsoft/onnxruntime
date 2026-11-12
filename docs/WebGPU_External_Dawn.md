# External Dawn API packages

Native WebGPU hosts can supply their own Dawn implementation while ONNX Runtime
compiles only the Dawn API dispatch layer. Set `onnxruntime_DAWN_PREBUILT_DIR` to
a directory containing matching public/generated headers and standalone proc
sources:

```text
dawn-api/
  include/
    dawn/          # Dawn public and generated headers
    webgpu/        # WebGPU C and C++ headers
  src/
    dawn_proc.cpp
    dawn_thread_dispatch_proc.cpp
```

The proc sources must compile using only these headers and the C++ standard
library. In particular, `dawn_proc.cpp` must not require private Dawn build
headers. Retain its proc-table version check. Compile the proc sources with
ONNX Runtime's toolchain rather than supplying a precompiled proc library.

```text
python tools/ci_build/build.py --config Release --use_webgpu --use_external_dawn \
  --build_shared_lib --skip_tests --target onnxruntime \
  --cmake_extra_defines onnxruntime_DAWN_PREBUILT_DIR=/absolute/path/to/dawn-api
```

This option requires `onnxruntime_USE_WEBGPU` and
`onnxruntime_USE_EXTERNAL_DAWN`. It cannot be combined with a custom Dawn source
path, shared Dawn, Emscripten, PIX, or the Agility SDK. It skips Dawn source
configuration and does not build or package native Dawn or DXC binaries.
The host owns its Dawn backend's runtime dependencies.

Pass the host's matching `DawnProcTable` address through the WebGPU
`dawnProcTable` provider option before creating a session. Headers, proc sources,
and the host implementation must come from the same Dawn version; mismatched
proc tables are not ABI-compatible.

Windows WebGPU CI exports a matching API package from the source-based external
Dawn build, then builds a separate package-backed ONNX Runtime DLL. It runs the
same native host against that DLL, verifies Abs inference with CPU fallback
disabled, and checks missing proc-table and invalid-package failures.

## Host-owned devices and caches

### Cache callbacks on an ORT-owned device

For native WebGPU, include `webgpu_provider_factory.h` and initialize an
`OrtDawnCacheInterface` with `version = ORT_DAWN_CACHE_INTERFACE_VERSION`,
`load`, `store`, `userdata`, and an optional `isolation_key`. Supply its decimal
address as the `dawnCacheInterface` provider option (session configuration key
`ep.webgpuexecutionprovider.dawnCacheInterface`). This works with both bundled
and external Dawn. External Dawn still requires a matching `dawnProcTable`.

The interface and its referenced state must remain alive and unchanged until
the device is released. Callbacks may be invoked concurrently and must not
throw. A load with a null destination is a size probe; return zero for a miss.
For a copy, require an exact destination size and return zero if the blob
changed between the probe and copy.

Omitting the option inherits an existing context's cache configuration. If no
context exists, omission leaves caching disabled. Passing `"0"` explicitly
requests disabled caching. Configure the cache before the first session or
shared allocation creates the default device.
Shared allocator registration alone does not create the device; its first
allocation retains an existing session device or creates an uncached default
device if no session exists. Acquisition and creation share the same lock, so
an allocator cannot accidentally request default configuration after a
concurrent session creates the configured context. Sessions reusing a device
may omit the option or use the same interface address. A different interface
address, explicit `"0"` against a cached context, or a nonzero address against
an uncached context is rejected. Unsupported interface versions,
missing callbacks, and malformed addresses are rejected using ORT's normal
configuration-error policy (fail-fast in no-exceptions builds).

Applications creating an allocator-initialization session before their model
session must forward this option to that first session as well. ONNX Runtime
GenAI forwards `dawnCacheInterface` with its other global WebGPU options to
its persistent initialization session. This requires a GenAI build containing
that forwarding change; updating ONNX Runtime alone is insufficient.

These callbacks cache Dawn compiled shader/pipeline blobs, not model weights.
They do not change ORT's internally created device or accelerated D3D12
weight-loading path. Do not pass this option for a host-owned device: attach
its Dawn cache descriptor when creating the device instead.

When native Dawn is built for testing (bundled or source-based external Dawn),
`ctest -C RelWithDebInfo -R "^onnxruntime_webgpu_owned_cache_test$"` (using the
built configuration) runs Abs inference with CPU
fallback disabled on three newly created ORT-owned devices. It verifies
callback userdata, cold stores, warm cache hits, misses with a different
isolation key, and auxiliary sessions omitting the cache option after the
configuring session is released. The plugin allocator tests additionally cover registering a
shared allocator and making zero-size/null calls before supplying cache options,
then retaining and using the configured device after session destruction.

### Host-owned device configuration

A host can create a native Dawn instance and device, including its own
`DawnCacheDeviceDescriptor` load/store callbacks and isolation key, before
creating the ORT session. Pass the handles as decimal pointer values through
the existing WebGPU provider options:

- `dawnProcTable`: the matching host `DawnProcTable`.
- `webgpuInstance`: the host `WGPUInstance`.
- `webgpuDevice`: the host `WGPUDevice`.
- `deviceId`: a positive ORT context ID. ID zero is reserved for ORT's default
  context. This is also the allocator and tensor memory-device ID and must fit
  in `OrtDevice::DeviceId` (1 through 32767). Reusing an ID requires the same
  instance and device. Copies between different context IDs are not supported.
- `preserveDevice`: `"1"` to retain the custom context across session releases.

An `OrtMemoryInfo` with device ID `N` identifies an external memory domain; it
does not register the matching Dawn proc table, instance, or device. The
Env/global allocator path currently supports only context zero. Streamless
`Env::CopyTensor` for an external context requires that context to remain
registered by an active provider factory/session or through `preserveDevice=1`.
Pre-session global allocation/copy for an external Dawn context requires a
separate environment-level registration API that also exposes its memory
device and shared allocator; a nonzero memory-device ID alone is insufficient.

The device must request `ImplicitDeviceSynchronization` in
`DeviceDescriptor.requiredFeatures`, plus the features and limits required by
the model. Supplying a device does not let ORT change how that device was
created. The host must not destroy it while ORT can still use it.

Keep the Dawn implementation, instance/device handles, cache callbacks, and
callback userdata alive until all ORT contexts and Dawn work using them have
been released. Cache callbacks must be safe for concurrent calls and must not
throw across the Dawn callback boundary. Use distinct isolation keys for cache
domains that must not share compiled data.

The native host test's `--host_device` mode runs GPU inference with CPU fallback
disabled on three fresh devices: a cold cache, a warm cache with the same
isolation key, and a separate isolation key. It checks actual cache reads and
writes and device operations after session/environment teardown. It also keeps
default and host-owned sessions alive together, checks distinct allocator/output
IDs, exercises CPU/GPU and same-context GPU/GPU copies, and rejects cross-context
GPU copies. The
`--no_implicit_sync` negative mode verifies that omitting the required feature
is rejected. The `--cache_callback_failure` mode injects allocation failures in
both callbacks. The callbacks contain the exceptions as a cache miss/no-op and
record a failure flag, which normal test code diagnoses after inference rather
than allowing exceptions to escape or silently treating the failure as success.
Windows CI exercises these modes for source-based, package-backed,
and plugin builds; other native platforms have not been validated locally.
