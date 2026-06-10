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
2. The variant selection algorithm, which queries each execution provider
   factory and picks the highest-scoring variant.
3. The public `OrtModelPackageApi` (C) and `onnxruntime.ModelPackageContext`
   (Python) surface that wraps the library and exposes session creation.

ORT links the `model_package` library as a static archive; the library
itself never links against ORT.

---

## Files

| File                                  | Responsibility |
| ------------------------------------- | -------------- |
| `model_package_context.h/.cc`         | Translates the `model_package` library's C info tree into ORT-internal C++ structs (`ModelPackageInfo`, `ComponentInfo`, `VariantInfo`, `VariantModelInfo`). Parses the `executor_info["ort"]` payload. Owns `ModelPackageContext` (package-level) and `ModelPackageComponentContext` (per-component, with selected variant and provider list). |
| `model_package_options.h/.cc`         | `ModelPackageOptions` snapshots EP intent (factories, devices, EP-name list) from an `OrtSessionOptions` at the moment `CreateModelPackageOptionsFromSessionOptions` is called. Drives variant selection and provider construction. |
| `model_package_variant_selector.h/.cc`| `VariantSelector::SelectVariant` — picks the best variant from a component given the EP list. Uses `OrtEpFactory::ValidateCompiledModelCompatibilityInfo`. |

---

## `executor_info["ort"]` schema

ORT's slot in `variant.executor_info` is a JSON object. All fields are
optional, but in practice `model_file` is required to load a session.

```jsonc
{
  "model_file":       "model.onnx",       // path to the ONNX file
  "external_data":    "weights",          // path to the external-initializers folder (or sha256: URI)
  "session_options":  { "session.intra_op_thread_count": "4" },
  "provider_options": { "device_id": "0" }
}
```

| Field              | Type   | Required | Notes |
| ------------------ | ------ | -------- | ----- |
| `model_file`       | string | yes (for session) | Path to the model file inside the variant. Resolved via `ModelPackage_ResolveStringRef`, anchored at the variant directory. Accepts relative paths, absolute paths or `..` segments (installed layout only), and `sha256:<hex>[/sub/path]` for shared-asset content. |
| `external_data`    | string | no       | Folder containing the model's external-initializers blobs. Wired into the session as ORT's external-initializers folder hint. Same resolution rules as `model_file`. |
| `session_options`  | object | no       | Map of `string → string`. Merged on top of a fresh `OrtSessionOptions` when the caller passes `session_options == NULL` to `CreateSession`. Ignored when the caller supplies their own `OrtSessionOptions`. |
| `provider_options` | object | no       | Map of `string → string`. Merged into the variant's EP provider options on the default path. Ignored when the caller supplies their own `OrtSessionOptions`. |

#### Inline vs external

The slot follows the standard `executor_info` shape: the value may be either

- a **string** — a path to a JSON file containing the body above (commonly
  `ort_info.json` next to `model.onnx`), or
- an **object** — the body inlined into `component.json` /
  `manifest.json`.

Inline form keeps the package single-file. External form (the common case)
keeps the variant directory self-describing and survives `executor_info`
schema evolution without rewriting the manifest.

Example variant declaration with the external form:

```jsonc
// component.json
{
  "variants": {
    "cpu": {
      "variant_directory": "cpu",
      "ep":     "CPUExecutionProvider",
      "device": "cpu",
      "executor_info": {
        "ort": "ort_info.json"          // → <variant_dir>/ort_info.json
      }
    }
  }
}
```

```jsonc
// cpu/ort_info.json
{ "model_file": "model.onnx" }
```

The key under `executor_info` is the **executor namespace name** (`"ort"`),
not the EP. Other consumers (e.g. GenAI) use their own namespace key
(`"genai"`), so a single variant can carry per-consumer payloads side by
side.

---

## Variant selection

`ModelPackageOptions(env, session_options)` captures the **EP intent**: the
ordered list of execution providers registered on the session options, plus
their associated `OrtEpDevice` / `OrtHardwareDevice` / metadata.

`VariantSelector::SelectVariant(component, ep_infos, &selected)` then walks
the component's variants and picks the best match:

1. Use only the **first** EP from the captured list. (A policy may rank
   several EPs; callers that need a specific EP should put it first.
   Ranking across the full EP list is on the TODO list.)
2. For each variant, require `variant.ep == ep_info.ep_name`.
3. If `variant.device` is set (`"cpu"` / `"gpu"` / `"npu"`), require it to
   match at least one of the EP's `OrtHardwareDevice` entries.
4. If both pass, call `OrtEpFactory::ValidateCompiledModelCompatibilityInfo`
   with `variant.compatibility_string`. The EP returns an
   `OrtCompiledModelCompatibility` enum which maps to a score:

   | Enum                                         | Score |
   | -------------------------------------------- | ----- |
   | `EP_SUPPORTED_OPTIMAL`                       | 100   |
   | `EP_SUPPORTED_PREFER_RECOMPILATION`          |  50   |
   | `EP_NOT_APPLICABLE` (or EP too old / no ABI) |   0   |
   | `EP_UNSUPPORTED`                             | rejected |

5. Pick the highest-scoring matching variant. Manifest declaration order
   breaks ties.

If no variant matches, `SelectComponent` fails with "No suitable model
variant found for the configured execution providers."

ORT does **not** parse `compatibility_string`. The EP owns the format and
may encode multiple sub-targets (SoC ids, ISA flags, etc.) into the single
string internally; ORT only round-trips it through the EP callback.

---

## Session creation contract

```c
OrtModelPackageApi::CreateSession(env, component_ctx, session_options, &session);
```

The `component_ctx` already knows which variant won selection and which
provider list it should use. Two paths:

- **`session_options == NULL` (default).** ORT starts from a fresh
  `OrtSessionOptions` and merges the variant's `session_options` /
  `provider_options` from `executor_info["ort"]` on top. EPs declared in the
  manifest are constructed and registered. This is what nearly all callers
  want.

- **`session_options != NULL` (advanced).** ORT uses the caller-supplied
  `OrtSessionOptions` as-is. The manifest's `session_options` and
  `provider_options` are **not** merged. Use this when you need custom EP
  setup that doesn't round-trip through string options (shared CUDA streams,
  shared QNN EP contexts, custom allocators, …). The `OrtSessionOptions`
  passed earlier to `CreateModelPackageOptionsFromSessionOptions` only drives
  variant selection / EP discovery; it's never silently re-applied here.

In both modes, `external_data` from `executor_info["ort"]` is wired in as
ORT's external-initializers folder hint, so the model file can reference
weights stored next to (or shared by) the package.

---

## C API surface

The public ORT C API for model packages is defined in
`include/onnxruntime/core/session/onnxruntime_c_api.h` under
`struct OrtModelPackageApi`. The function table is reached through
`OrtApi::GetModelPackageApi()`. Available since ORT 1.27.

Typical flow:

```c
const OrtApi* ort = OrtGetApiBase()->GetApi(ORT_API_VERSION);
const OrtModelPackageApi* mpkg = ort->GetModelPackageApi();

// 1. Capture EP intent from a session options.
OrtSessionOptions* so = NULL;
ort->CreateSessionOptions(&so);
ort->SessionOptionsAppendExecutionProvider(so, "CUDAExecutionProvider", NULL, NULL, 0);

OrtModelPackageOptions* mp_opts = NULL;
mpkg->CreateModelPackageOptionsFromSessionOptions(env, so, &mp_opts);

// 2. Open the package.
OrtModelPackageContext* ctx = NULL;
mpkg->CreateModelPackageContext(ORT_TSTR("/path/to/pkg"), &ctx);

// 3. Inspect (optional).
const char* const* names = NULL;
size_t n = 0;
mpkg->ModelPackage_GetComponentNames(ctx, &names, &n);

// 4. Select a component / variant.
OrtModelPackageComponentContext* comp_ctx = NULL;
mpkg->SelectComponent(ctx, "decoder", mp_opts, &comp_ctx);

const char* variant_name = NULL;
mpkg->ModelPackageComponent_GetSelectedVariantName(comp_ctx, &variant_name);

// 5. Create the session.
OrtSession* session = NULL;
mpkg->CreateSession(env, comp_ctx, /*session_options=*/NULL, &session);

// Release in reverse order.
ort->ReleaseSession(session);
mpkg->ReleaseModelPackageComponentContext(comp_ctx);
mpkg->ReleaseModelPackageContext(ctx);
mpkg->ReleaseModelPackageOptions(mp_opts);
ort->ReleaseSessionOptions(so);
```

All `const char*` / `const ORTCHAR_T*` / array pointers returned by the API
are owned by the context that produced them and remain valid until the
context is released.

---

## Python API surface

The Python bindings mirror the C API:

```python
import onnxruntime as ort

ctx = ort.ModelPackageContext("/path/to/pkg.ortpackage")
print(ctx.get_component_names())
for v in ctx.get_variant_names("decoder"):
    print(v, ctx.get_variant_ep_name("decoder", v))

# Capture EP intent (this snapshot drives variant selection).
so = ort.SessionOptions()
so.add_provider("CUDAExecutionProvider", {})
opts = ort.ModelPackageOptions(so)

# Select the best variant for the captured EPs.
comp = ctx.select_component("decoder", opts)
print(comp.get_selected_variant_name())
print(comp.get_selected_variant_folder_path())

# Default path: variant's session/provider options are merged automatically.
session = comp.create_session()

# Advanced path: caller controls SessionOptions; manifest-side options are NOT merged.
custom_so = ort.SessionOptions()
custom_so.intra_op_num_threads = 4
session = comp.create_session(custom_so)
```

---

## Internal data flow

```
manifest.json ─► model_package (C)
                  │
                  │ ModelPackage_Info() / FindExecutorInfo("ort")
                  ▼
        model_package_context.cc
          (translate C info tree into ORT C++ structs;
           parse executor_info["ort"] → VariantModelInfo)
                  │
                  ▼
        ModelPackageContext  ◄── public API: traversal, EP inspection
                  │
                  │ SelectComponent(name, ModelPackageOptions)
                  ▼
        ModelPackageComponentContext
                  │
                  │ VariantSelector::SelectVariant(ep_infos)
                  ▼
          selected variant
                  │
                  │ CreateSession(env, session_options_or_null)
                  ▼
              OrtSession
```

`ModelPackageOptions` is independent of any single component context: it
holds the captured EP intent and is passed to `SelectComponent` for every
component you select from the same package.

---

## See also

- [`model_package/README.md`](../../../../model_package/README.md) — package
  format, manifest/component schema, shared assets, path resolution, the
  authoring C API, and the `executor_info` extension point.
- `onnxruntime/core/session/onnxruntime_c_api.h`,
  `struct OrtModelPackageApi` — the canonical C API reference (Doxygen
  comments).
- The GenAI repo (`onnxruntime-genai`) — consumer of the same packages
  through the `executor_info["genai"]` slot; uses this ORT API under the
  hood to create sessions.
