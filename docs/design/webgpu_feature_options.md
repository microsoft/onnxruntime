# WebGPU execution provider feature options

`enableFeatures` and `disableFeatures` control named ORT capabilities. JavaScript
bindings accept arrays of strings:

```ts
executionProviders: [{ name: "webgpu", disableFeatures: ["subgroupMatrix"] }];
```

Native provider options accept comma-separated names under `enableFeatures` and
`disableFeatures`. Session configuration keys use the prefix
`ep.webgpuexecutionprovider.`. Empty lists preserve defaults. Whitespace around
names is trimmed. Names are case sensitive; unknown names, duplicates, and names in both lists are rejected.

Enabling permits use when the device supports a feature. It does not require
hardware support or guarantee a particular kernel will execute. Disabling
prevents ORT from using the feature, including on externally supplied devices;
it does not change the external device itself.

| Feature          | Default                | Scope                        | Behavior when disabled                                     |
| ---------------- | ---------------------- | ---------------------------- | ---------------------------------------------------------- |
| `subgroupMatrix` | Enabled when supported | First context initialization | Existing fallback kernels replace subgroup matrix kernels. |

An explicit request that conflicts with an initialized shared context is ignored
with a warning. Omitted requests inherit that context's settings, with a warning
if they differ from the defaults. Use a separate context or process for
different settings.
The initialized setting records permission to use a capability, independently
of whether hardware supports it.

The native `kWebGpuFeatures` registry defines supported names and defaults.
It is independent of WebGPU device features; the context maps each ORT feature to
the device capabilities it gates. `WebGpuContext::IsDeviceFeatureAllowed`
checks the effective permission from explicit enable/disable requests or the
feature default when requesting device features and caching the features
available to kernels. For an externally supplied device, filtering the cache
ensures kernels use their fallbacks while preserving the device's enabled features.
The bindings forward lists without maintaining a second registry.
Only context-scoped device capabilities are registered today;
future per-session or device-creation-only features need their corresponding
lifecycle handling and documented scope. Numeric and enum settings remain
explicit typed options.
