# ORT Model Package Integration

This directory implements ONNX Runtime's consumer-side glue for the
standalone [`model_package` library](../../../../model_package/README.md):
loading packages, selecting variants against the runtime's execution
providers, and creating an `OrtSession` for the chosen variant.

The package format, manifest schema, shared-asset rules, and the C
authoring/inspection API all live in `model_package/`. **This directory
adds three things on top**:

1. The `executor_info["ort"]` payload schema (this is ORT's slot in the
   variant body).
2. The variant selection algorithm considers execution providers in priority
   order. For the first EP with matching variants, it selects the variant
   with the highest compatibility score.
3. The stable `OrtModelPackageApi` C API table that wraps the library and
   exposes session creation through `OrtApi::GetModelPackageApi`.

ORT links the `model_package` library as a static archive; the library
itself never links against ORT.

---

## Files

| File                                  | Responsibility |
| ------------------------------------- | -------------- |
| `model_package_context.h/.cc`         | Translates the `model_package` library's C info tree into ORT-internal C++ structs (`ModelPackageInfo`, `ComponentInfo`, `VariantInfo`, `VariantModelInfo`). Parses the `executor_info["ort"]` payload. Owns `ModelPackageContext` (package-level) and `ModelPackageComponentContext` (per-component, with the selected variant, the captured EPs, and their provider factories saved for session creation). |
| `model_package_options.h/.cc`         | `ModelPackageOptions` snapshots EP intent (factories, devices, EP-name list) from an `OrtSessionOptions` when `OrtModelPackageApi::CreateModelPackageOptionsFromSessionOptions` is called. Drives variant selection and provider construction. |
| `model_package_variant_selector.h/.cc`| `VariantSelector::SelectVariant` picks the best variant from a component given the EP list. Uses `OrtEpFactory::ValidateCompiledModelCompatibilityInfo`. |

The C entry points themselves live in
`onnxruntime/core/session/model_package_api.cc` under
`namespace OrtModelPackageAPI`.

---

## `executor_info["ort"]` schema

ORT's slot in `variant.executor_info` is a JSON object. All fields are
optional, but in practice `model_file` is required to load a session.

```jsonc
{
  "model_file":       "model.onnx",
  "session_options":  {
    "session.intra_op_num_threads": "4",
    "session.model_external_initializers_file_folder_path": "weights"
  },
  "provider_options": { "device_id": "0" }
}
```

| Field              | Type   | Required | Notes |
| ------------------ | ------ | -------- | ----- |
| `model_file`       | string | yes (for session) | Path to the model file inside the variant. Resolved via `ModelPackage_ResolveStringRef`, anchored at the variant directory. Accepts relative paths, absolute paths or `..` segments (installed layout only), and `sha256:<hex>[/sub/path]` for shared-asset content. |
| `session_options`  | object | no       | Map of `string -> string`. Merged on top of a fresh `OrtSessionOptions` when the caller passes `session_options == NULL` to `CreateSession`. Values of path-valued keys (see `IsModelPackagePathSessionOption`, e.g. `session.model_external_initializers_file_folder_path`, `ep.context_file_path`) are resolved with the same rules as `model_file` at parse time. Those path-valued keys are also applied on the advanced path if the caller did not set them (see below). Output file options (`session.debug_layout_transformation`, `session.collect_node_memory_stats_to_file`, `session.enable_profiling`, and `session.optimized_model_filepath`) are forbidden in a package and cause package parsing to fail; only caller-supplied `OrtSessionOptions` may set those keys. |
| `provider_options` | object | no       | Map of `string -> string`. Overrides the selected device's default EP options on the default path. Requires the selected EP to expose `OrtEpDevice` metadata; otherwise a non-empty map is rejected. Ignored when the caller supplies their own `OrtSessionOptions`. |

#### Inline vs external

The slot follows the standard `executor_info` shape: the value may be either

- a **string**, a path to a JSON file containing the body above (commonly
  `ort_info.json` next to `model.onnx`), or
- an **object**, the body inlined into `component.json` /
  `manifest.json`.

Inline form keeps the package single-file. External form (the common case)
keeps the variant directory self-describing and survives `executor_info`
schema evolution without rewriting the manifest.

The key under `executor_info` is the **executor namespace name** (`"ort"`),
not the EP. Other consumers use their own namespace key, so a single
variant can carry per-consumer payloads side by side.

---

## Variant selection

`ModelPackageOptions(env, session_options)` records the EPs in priority
order and saves their device information and metadata. When a policy
selects the EPs, ORT records the selected devices and the factory used
to create each EP. For explicitly added EPs, device information comes
from the provider's `GetEpDevices()`, which currently returns devices
only for plugin EPs. ORT saves the provider factories from the session
options so it can create legacy EPs again after those options are
released.

`VariantSelector::SelectVariant` selects the best matching variant for
the component and identifies its target EP:

1. Consider the captured EPs in priority order, as determined by the
   selection policy or the order in which they were added to the session
   options. Select the first EP with a matching variant.
2. For each variant, require `variant.ep == ep_info.ep_name`.
3. If `variant.device` is set (`"cpu"` / `"gpu"` / `"npu"`), require it to
   match at least one of the EP's `OrtHardwareDevice` entries.
   The built-in CPU EP matches `"device": "cpu"` even without captured
   devices, but does not match other device types.
   If captured devices exist but none has the required type, the variant
   does not match.
   If no devices were captured for an EP other than the built-in CPU EP,
   ORT cannot check the requirement. If no other variant matches that EP,
   selection returns an error instead of considering EPs with lower priority.
   A default memory device type is not enough to determine the target
   hardware.
4. If both pass, call `OrtEpFactory::ValidateCompiledModelCompatibilityInfo`
   with `variant.compatibility_string`. The EP returns an
   `OrtCompiledModelCompatibility` enum which maps to a score:

   | Enum                                                               | Score |
   | ------------------------------------------------------------------ | ----- |
   | `EP_SUPPORTED_OPTIMAL`                                             | 100   |
   | `EP_SUPPORTED_PREFER_RECOMPILATION`                                |  50   |
   | `EP_NOT_APPLICABLE` (or EP too old / no ABI / no factory captured) |   0   |
   | `EP_UNSUPPORTED`                                                   | rejected |

   `EP_UNSUPPORTED` means the check succeeded but the variant is not
   compatible. If the callback itself returns an error, compatibility
   remains unknown. If no other variant matches that EP, selection returns
   an error instead of considering EPs with lower priority.

5. Select the matching variant with the highest compatibility score for
   that EP. If several variants have the same score, select the first one
   in the manifest.

If no variant matches, `SelectComponent` fails with "No suitable model
variant found for the configured execution providers." If selection stops at
an EP as described in steps 3 and 4, the error also names that EP.

ORT does **not** parse `compatibility_string`. The EP owns the format and
may encode multiple sub-targets (SoC ids, ISA flags, etc.) into the single
string internally; ORT only round-trips it through the EP callback.

---

## Session creation contract

`OrtModelPackageApi::CreateSession(env, component_ctx, session_options, &session)`.

The `component_ctx` already knows which variant won selection and which
EP was selected. Two paths:

- **`session_options == NULL` (default).** ORT starts from a fresh
  `OrtSessionOptions` and merges the variant's `session_options` /
  `provider_options` from `executor_info["ort"]` on top. The selected EP's
  default device options are included even when the package does not provide
  overrides. Its custom-op domains are registered before the model is loaded.
  A retained legacy factory keeps its own EP-specific configuration.

- **`session_options != NULL` (advanced).** ORT copies the caller-supplied
  `OrtSessionOptions`, preserving explicit provider factories and their order,
  including fallback providers. The manifest's `session_options` and
  `provider_options` are **not** merged, with one exception: path-valued
  session options (see `IsModelPackagePathSessionOption`) are carried over
  from the variant for keys the caller did not set, so a model that needs its
  external-initializers folder still loads. Use this path when you need custom
  EP setup that does not round-trip through string options (shared CUDA
  streams, shared QNN EP contexts, custom allocators, ...). The
  `OrtSessionOptions` passed earlier to
  `CreateModelPackageOptionsFromSessionOptions` only drives variant
  selection / EP discovery; its non-EP session configuration is not
  implicitly inherited.

When the effective session options contain no EP factories, ORT adds
only the EP targeted by the selected variant, as in
`SessionOptionsAppendExecutionProvider_V2`. ORT uses the devices captured
for that EP. If the variant specifies a device type, ORT uses only
devices of that type. Otherwise, it uses all captured devices for that EP.

Device defaults are used only for EP options that have not already been
set, including options from the variant's `session_options`. Custom
operator domains provided by the EP are registered before loading the
model. On the default path, the variant's `provider_options` are passed
to the EP as in `SessionOptionsAppendExecutionProvider_V2`. For example,
CoreML rejects an `MLComputeUnits` value that enables an accelerator
outside the device selection.

If no `OrtEpDevice` instances were captured for the EP, ORT uses the
provider factory saved from the session options used to create the
model package options. The built-in CPU EP does not require a saved
factory. On the default path, non-empty package `provider_options`
cause an error if no devices were captured.

No other EP is added, apart from the usual ORT CPU EP fallback. This
prevents an EP without a matching variant from taking nodes of a model
built for another EP. `CreateSession` uses the EP chosen for the
variant, even if the session options specify a selection policy.

If the caller supplies EP factories, ORT uses that list. The caller must
ensure compatibility with the selected variant. If the selected EP is not
the built-in CPU EP, at least one factory in the list must create it.
The CPU registration APIs behave differently:

- Adding the built-in ORT CPU EP as "CPU" through
  `SessionOptionsAppendExecutionProvider` does not add a provider factory
  to the options.
- `OrtSessionOptionsAppendExecutionProvider_CPU` adds a CPU EP factory.

ORT automatically adds the built-in CPU EP if it has not already
been registered. If any other selected EP cannot be created or
registered, session creation fails. After successful registration,
unsupported nodes may still use the usual ORT CPU EP fallback unless
it is disabled.

Each session creates its own EP instances through the normal ORT factory
registration path. A component context can create multiple sessions sequentially;
creating a session does not consume its captured factory.

A variant points ORT at external-initializer weights by setting
`session.model_external_initializers_file_folder_path` in its
`session_options` to a folder (relative, absolute, or `sha256:<hex>` shared
asset). The value is resolved at parse time and overrides the model's own
directory, so the model file can reference weights stored next to (or shared
by) the package.

---

## C API surface

The model package API is a stable companion table returned by
`OrtApi::GetModelPackageApi`. Its opaque handles
(`OrtModelPackageOptions`, `OrtModelPackageContext`, and
`OrtModelPackageComponentContext`) and function table are declared in
`onnxruntime_c_api.h`. `onnxruntime_cxx_api.h` provides RAII wrappers.

API entries:

| Function                                              | Notes |
| ----------------------------------------------------- | ----- |
| `CreateModelPackageOptionsFromSessionOptions`         | Snapshots EP intent. |
| `ReleaseModelPackageOptions`                          |       |
| `CreateModelPackageContext`                           | Parses the manifest. |
| `ReleaseModelPackageContext`                          |       |
| `ModelPackage_GetSchemaVersion`                       | Returns the schema major version. |
| `ModelPackage_GetComponentCount`                      |       |
| `ModelPackage_GetComponentNames`                      |       |
| `ModelPackage_GetVariantCount`                        |       |
| `ModelPackage_GetVariantNames`                        |       |
| `ModelPackage_GetVariantEpName`                       |       |
| `ModelPackage_ResolveStringRef`                       | Resolves UTF-8 path references. |
| `SelectComponent`                                     | Resolves the best-matching variant. |
| `ReleaseModelPackageComponentContext`                 |       |
| `ModelPackageComponent_GetSelectedVariantName`        |       |
| `ModelPackageComponent_GetSelectedVariantFolderPath`  |       |
| `CreateSession`                                       |       |

Typical flow:

```cpp
#include "onnxruntime_c_api.h"

const OrtApi* ort = OrtGetApiBase()->GetApi(ORT_API_VERSION);
const OrtModelPackageApi* package_api = ort->GetModelPackageApi();

OrtSessionOptions* so = nullptr;
ort->CreateSessionOptions(&so);
ort->SessionOptionsSetEpSelectionPolicy(so, OrtExecutionProviderDevicePolicy_PREFER_GPU);

OrtModelPackageOptions* mp_opts = nullptr;
package_api->CreateModelPackageOptionsFromSessionOptions(env, so, &mp_opts);

OrtModelPackageContext* ctx = nullptr;
package_api->CreateModelPackageContext(ORT_TSTR("/path/to/pkg"), &ctx);

OrtModelPackageComponentContext* comp_ctx = nullptr;
package_api->SelectComponent(ctx, "decoder", mp_opts, &comp_ctx);

OrtSession* session = nullptr;
package_api->CreateSession(env, comp_ctx, nullptr, &session);

ort->ReleaseSession(session);
package_api->ReleaseModelPackageComponentContext(comp_ctx);
package_api->ReleaseModelPackageContext(ctx);
package_api->ReleaseModelPackageOptions(mp_opts);
ort->ReleaseSessionOptions(so);
```

Borrowed names, arrays, and selected folder paths remain valid until their context
is released, including across repeated queries. `ModelPackage_ResolveStringRef`
is the exception: its result is valid only until the next resolver call on the
same package context. Do not free borrowed pointers. Failed calls leave output
parameters unchanged.

Strings and path references represented as `char*` are UTF-8. Native filesystem
paths explicitly represented as `ORTCHAR_T*` use UTF-16 on Windows and UTF-8 on
other platforms. The integration performs explicit conversions at the standalone
library boundary.

The environment and registered EP libraries must outlive the model package
options, component contexts, and sessions that use them. A selected component
does not depend on the lifetime of its original package or options handles.
Calls sharing a package or component context require external synchronization.

---

## See also

- [`model_package/README.md`](../../../../model_package/README.md): package
  format, manifest/component schema, shared assets, path resolution, the
  authoring C API, and the `executor_info` extension point.
- `include/onnxruntime/core/session/onnxruntime_c_api.h`: the stable
  `OrtModelPackageApi` declaration.
