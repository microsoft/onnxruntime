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
3. The stable `OrtModelPackageApi` C API table that wraps the library and
   exposes session creation through `OrtApi::GetModelPackageApi`.

ORT links the `model_package` library as a static archive; the library
itself never links against ORT.

Schema 1.1 adds JSON-only named chain metadata to the standalone library.
This consumer interprets chain/step `executor_info.ort` and constructs owned
executable chains through `OrtModelPackageApi::CreateSessionChain`. Existing
per-component Session creation remains supported and does not interpret unrelated
chains' execution policies.

## Executable named chains

Select a chain explicitly, including when the package contains only one:

```cpp
Ort::ModelPackageContext package(package_path);
Ort::SessionOptions options;
OrtModelPackageChainConfig config{};
config.struct_size = sizeof(config);
config.chain_name = "generate";
config.default_session_options = options;
Ort::SessionChain chain = package.CreateSessionChain(env, config);
```

The returned chain uses the existing `CreateExecution`, `Step`, `Run`, `Reset`,
`Cancel` and immutable result/value APIs. Inputs are ordinary
`OrtStatefulInputs`: external inputs, finite named sequences and initial state.
For example, a `prompt_batches` sequence supplies one CPU int64 `[1,S]` tensor
to a prefill step whose token input is bound to that sequence.

Package-built chains own their initialized per-step Sessions. Executions retain
the native definition, so package context, caller option handles and the public
chain handle may be released after construction/execution creation. The Env and
externally owned buffers, allocators, streams and custom-op resources retain
their ordinary Session lifetime obligations. Caller Sessions used to construct
programmatic chains are **not** adopted.

`default_session_options == nullptr` uses the established package default
creation path and CPU selection defaults. Otherwise existing advanced-path
precedence applies, including missing package path-valued options. Optional
`OrtModelPackageStepSessionOptions` entries replace the complete default options
snapshot for named step IDs; clone defaults first if partial changes are desired.
Duplicate/unknown IDs or NULL per-step options fail. Model EP/variant selection
is independent of cache/sampler provider policies.

### Execution payloads

The common standalone schema still owns `chains[name].steps`, IDs and component
references. The ORT namespace is strict: unknown fields/policies, missing exits,
invalid references and incompatible model ports fail before publication of a
chain. Other consumer namespaces remain opaque. Future common-schema minors do
not authorize execution of unknown ORT policies.

| Scope | Fields |
| --- | --- |
| Chain ORT payload | Optional named `kv_caches` specifications; required nonempty `outputs` map. |
| Session step ORT payload | `kind: "session"`, optional `inputs`, `state_outputs`, `kv_caches`, `cache_provider`. |
| Loop step ORT payload | `kind: "session_loop"`, optional `inputs`/`state_outputs`, required `loop`. |
| Loop configuration | `finite_input_names`, `nonempty_input`, `feedback`, explicit `exit_conditions`, `generated_token_output`, `generation`, `sampler`, `cache_provider`, `kv_caches`. |

Input sources are `external_input`/`external_sequence`/`state` with `name`, or
`model_output` with an earlier `step` and `output`. `state_outputs` maps state
names to model-output names; feedback entries contain `input` and `output`.
Final output sources are `model_output`, or `generated_token_ids` with a
generation `step`.

Exit entries are `end_of_inputs`, `max_iterations`/`max_generated_tokens` with
`limit`, `eos` with `token_ids`, or `boolean_output` with `output`. Conditions use
the native OR semantics, type/range checks and deterministic exit reasons.
Cache specs contain `element_type` (`float32`, `float16`, `bfloat16`), `shape`,
`sequence_axis` and `initial_capacity`; bindings contain `cache`, `input`,
`output` and `update` (`share`, `copy`, `append`).

Native generation declares `phase`, `input_ids`, `logits_output`, optional
`attention_mask`, `position_ids`, `logits_position` and, for explicit decode
bootstrap, `initial_logits`. It requires an explicit greedy `sampler`; a
sampler without generation is invalid. Provider objects use `mode` (`cpu`,
`prefer_provider`, `require_provider`); non-CPU modes also specify
`provider_name`, `device_type` and `device_id`. Omitted cache provider or omitted
provider inside an explicit sampler selects the documented CPU policy.

### Sampling-free prefill

Prefill can be an ordinary Session or a generic SessionLoop that consumes the
full prompt and updates caches without any sampler/generation policy. Decode
owns token selection and references prefill's final logits:

```json
{
  "phase": "decode",
  "input_ids": "token",
  "logits_output": "logits",
  "initial_logits": {
    "kind": "model_output",
    "step": "prefill",
    "output": "logits"
  }
}
```

The prefill loop's explicit `nonempty_input` identifies its prompt binding by
default; `initial_logits.prompt_input` can name it explicitly and is required
for a plain Session producer. `initial_logits.position` optionally selects the
initial logits row, independently of the later decoder `logits_position`.
Prompt length comes from the actual original CPU token tensor, not logits shape.
An external prompt sequence must contain one full batch; chunking is not added.
Prefill masks/positions, if required by its model, use explicit normal bindings.

The first decode Step is **sample-only**: `executed == false`, a generated token
is present, decode iterations remain0 and model invocation count does not change.
Do not treat `executed == false` as completion; inspect `completed`. Run handles
these events internally. The next model Step consumes the pending token.
First-token EOS or a one-token budget can finish with no decode-model invocation.
The initial token counts/emits once; sample-only events do not republish model
outputs or update caches.

In this opt-in mode, a zero decode budget still permits the declared prefill
work/cache population, but decode invokes neither sampler nor model. Existing
programmatic/package prefill-sampling mode retains its prior zero-budget
behavior and matching prefill/decode budgets. Manifest generation policies are
not overridden per execution.

Explicit GPU policies require compatible producers and initialized providers;
there is no hidden full-logits migration/host argmax. Native GPU cache/greedy
implementations remain CUDA-plugin-only. Construction failures preserve output
parameters and release partial Sessions. Stepping failures/cancellation after
possible advancement require Reset, as with programmatic chains.

---

## Files

| File                                  | Responsibility |
| ------------------------------------- | -------------- |
| `model_package_context.h/.cc`         | Translates the `model_package` library's C info tree into ORT-internal C++ structs (`ModelPackageInfo`, `ComponentInfo`, `VariantInfo`, `VariantModelInfo`). Parses the `executor_info["ort"]` payload. Owns `ModelPackageContext` (package-level) and `ModelPackageComponentContext` (per-component, with selected variant and captured provider factory). |
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

`ModelPackageOptions(env, session_options)` captures the **EP intent**: the
first execution provider selected from the session options or EP policy, plus
its associated `OrtEpDevice` / `OrtHardwareDevice` metadata. Explicit provider
factories are retained so legacy providers can also be recreated after the
original options are released.

`VariantSelector::SelectVariant(component, ep_infos, &selected)` then walks
the component's variants and picks the best match:

1. Use only the **first** EP. A policy may rank several EPs; callers that
   need a specific EP should put it first. Selection does not fall through
   to other EPs when that EP has no matching variant.
2. For each variant, require `variant.ep == ep_info.ep_name`.
3. If `variant.device` is set (`"cpu"` / `"gpu"` / `"npu"`), require it to
   match at least one of the EP's `OrtHardwareDevice` entries. The built-in
   CPU EP also matches `"device": "cpu"` without device metadata. Other EPs
   without hardware metadata can only match variants with no device constraint;
   a default memory-device type is not sufficient to infer the target hardware.
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

When no explicit provider factories are supplied, either path uses the captured
EP, fills missing device-default options, and registers its custom-op domains.
The selection policy is not re-run. When explicit factories are supplied, the
caller must keep them compatible with the selected variant. Failure to recreate
the selected EP is an error, not an implicit switch to CPU.

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
ort->SessionOptionsAppendExecutionProvider(so, "CUDAExecutionProvider", nullptr, nullptr, 0);

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
