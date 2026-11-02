# WebGPU execution provider feature options

`enableFeatures` and `disableFeatures` control named ORT capabilities. JavaScript
bindings accept arrays of strings:

```ts
executionProviders: [{ name: "webgpu", disableFeatures: ["subgroupMatrix"] }];
```

Native provider options accept comma-separated names under `enableFeatures` and
`disableFeatures`. Session configuration keys use the prefix
`ep.webgpuexecutionprovider.`. Empty lists use defaults. Whitespace around
names is trimmed. Names are case sensitive; unknown names, duplicates, and names in both lists are rejected.

Enabling permits use when the device supports a feature. It does not require
hardware support or guarantee a particular kernel will execute. Disabling
prevents this session from using the feature, including on externally supplied
devices; it does not change the device itself.

| Feature          | Default                | Scope                        | Behavior when disabled                                     |
| ---------------- | ---------------------- | ---------------------------- | ---------------------------------------------------------- |
| `subgroupMatrix` | Enabled when supported | Session                      | Existing fallback kernels replace subgroup matrix kernels. |

Each session applies its own feature settings, independently of creation order.
Omitted features use their defaults, including when sharing a context with a
session that disables them. Settings remain fixed for the session's lifetime.

The native `kWebGpuFeatures` registry defines supported names and defaults.
It is independent of WebGPU device features. The shared context requests and
caches supported device capabilities regardless of session feature settings.
`ComputeContextBase::HasFeature` uses `WebGpuExecutionProvider::DeviceHasFeature`
to filter those capabilities by the session's permissions when selecting kernels.
Shader generation still uses the device's actual capabilities, so fallback
shaders may contain an unused subgroup matrix extension declaration. Existing
program keys distinguish matrix kernels from their fallbacks.
The bindings forward lists without maintaining a second registry.
Only capabilities that can be gated per session are registered today;
future device-creation-only features need separate lifecycle handling and
documented scope. Numeric and enum settings remain explicit typed options.
