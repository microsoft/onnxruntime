# Weightless mode

Status: implemented
Last updated: 2026-10-06

## Motivation

A compiled (EPContext) model normally contains a copy of every constant initializer it uses, often inside the
EP's opaque `ep_cache_context` binary. Each hardware-specific variant of the model then carries its own copy
of the weights. For large models this means:

- compiled models are about as large as the source model, and
- an application that ships several variants (e.g., for different hardware generations) ships the weights
  several times.

In weightless mode the EP does not embed or copy constant initializers. The weights are stored once, outside
the EP binary, and are provided to the EP when a session is created. Several compiled variants can share one
copy of the weights.

Not every EP can do this for every initializer. Some hardware has to transform weights at compile time (for
example, reorder or quantize them), and that is only possible for initializers the EP is allowed to copy. For
this reason weightless mode comes in several *modes*, and each EP reports which modes it supports on each
device.

## Design contract

| Item | Decision |
|---|---|
| Mode type | `OrtWeightlessSupport` enum (`onnxruntime_c_api.h`), a sequential enum. Values that cover several modes, such as `OrtWeightlessSupport_ALL_OR_EXTERNAL_ONLY`, describe EP capabilities only. Future versions may add values. |
| EP discovery | `"weightless_supported_modes"` EP metadata entry (`kOrtEpDevice_EpMetadataKey_WeightlessSupportedModes`) on each `OrtEpDevice`, with `"weightless_support"` (`kOrtEpDevice_EpMetadataKey_WeightlessSupport`) as the fallback for EPs built for earlier versions. |
| EP enforcement | `OrtEp::GetWeightlessSupport(const OrtEp*, OrtWeightlessSupport* support)` returns the same value. ORT calls it during `Compile()`. |
| Application request | Exactly **one** mode, via `OrtCompileApi::ModelCompilationOptions_SetWeightlessMode()` or the `"ep.enable_weightless_mode"` session option (`kOrtSessionOptionEpEnableWeightlessMode`). |
| Validation | When the session is initialized, ORT rejects unknown values. Before creating each plugin EP, ORT checks the requested mode against the EP metadata and sets the deprecated `"ep.enable_weightless"` option to match. During `Compile()`, ORT checks the mode against `GetWeightlessSupport()`. |
| Compiled model | ORT records the mode in the compiled model's metadata under `"weightless_mode"` (`kOrtModelMetadata_WeightlessMode`). |
| Deprecated | `ModelCompilationOptions_SetWeightlessEnabled(bool)`, `"ep.enable_weightless"` (since 1.31) and `"ep.enable_weightless_ep_context_nodes"` (since 1.29). |

### Modes

| Value | Name | EP metadata value | Meaning |
|---|---|---|---|
| `0` | `OrtWeightlessSupport_NONE` | `"none"` | EP: weightless mode not supported. Application: weightless mode disabled (default). |
| `1` | `OrtWeightlessSupport_EXTERNAL_ONLY` | `"external_only"` | Weightless for initializers stored **outside** the ONNX file (external data). Initializers stored inside the ONNX file are still copied by the EP during compilation. |
| `2` | `OrtWeightlessSupport_ALL` | `"all"` | Weightless for **all** initializers, internal and external. The source model must be available at runtime (see [section 4](#4-runtime-and-packaging-requirements)). |
| `3` | `OrtWeightlessSupport_ALL_OR_EXTERNAL_ONLY` | `"all_or_external_only"` | EP capability only: the EP supports both `EXTERNAL_ONLY` and `ALL`. Applications can't request it. |

An application selects `NONE`, `EXTERNAL_ONLY` or `ALL`. Future versions may add values, both modes and
capabilities. Applications must expect values they don't recognize, in the EP metadata or elsewhere, and ignore
them, i.e., not select a weightless mode based on them.

## 1. EP reports its supported modes

An EP reports its weightless modes in two places. The two must agree (see [1.4](#1-ep-reports-its-supported-modes)).

**1.1 EP metadata (discovery, before a session exists).** In `OrtEpFactory::GetSupportedDevices()` the EP adds
weightless entries to the metadata passed to `OrtEpApi::CreateEpDevice()`. The values may differ per device, for
example when older hardware or drivers only support `EXTERNAL_ONLY`.

| Key | Since | Values |
|---|---|---|
| `"weightless_supported_modes"` | 1.31 | `"none"`, `"external_only"`, `"all"`, `"all_or_external_only"`, and values added later. |
| `"weightless_support"` | 1.29 | `"none"`, `"external_only"`, `"all"`. |

An EP reports `"weightless_supported_modes"` and keeps reporting `"weightless_support"` with the value it reported
before, so that applications written for earlier versions keep working. For example, an EP that reported
`"weightless_support"` = `"all"` and now also supports `EXTERNAL_ONLY`:

```cpp
ort_api.AddKeyValuePair(ep_metadata, kOrtEpDevice_EpMetadataKey_WeightlessSupport, "all");
ort_api.AddKeyValuePair(ep_metadata, kOrtEpDevice_EpMetadataKey_WeightlessSupportedModes, "all_or_external_only");
```

ORT uses `"weightless_supported_modes"` when present, and `"weightless_support"` otherwise. If neither is present,
the EP does not support weightless mode on the device.

**1.2 `OrtEp::GetWeightlessSupport()` (enforcement, during compilation).** When the application requests a
weightless mode, ORT calls this function from `PluginExecutionProvider::Compile()`:

```cpp
OrtStatus* ORT_API_CALL MyEp::GetWeightlessSupportImpl(const OrtEp* this_ptr, OrtWeightlessSupport* support) noexcept {
  *support = OrtWeightlessSupport_ALL_OR_EXTERNAL_ONLY;
  return nullptr;
}
```

The callback signature is unchanged since 1.29. ORT versions before 1.31 only check that the value isn't
`OrtWeightlessSupport_NONE`, so an EP can report `OrtWeightlessSupport_ALL_OR_EXTERNAL_ONLY` to them as well.

**1.3 How the EP gets the weights.** The EP reads the mode the application chose from the
`"ep.enable_weightless_mode"` session config entry (`OrtEpApi::GetSessionConfigEntry`). An EP that supports
several modes must honor that entry. When it isn't set but the deprecated `"ep.enable_weightless"` entry is `"1"`,
the EP keeps the behavior it had before 1.31. When the EP later creates a session from the compiled model, it can
read the recorded mode from the model metadata (`OrtApi::Graph_GetModelMetadata()`, key `"weightless_mode"`).

Weightless mode does not prescribe how the EP obtains the initializer data. There are two strategies, and an EP may
pick either one:

| | ORT-provided initializers | EP-managed initializers |
|---|---|---|
| `drop_constant_initializers` in `OrtNodeFusionOptions` | `false` | `true` |
| What the EP keeps at compile time | Nothing; the initializers stay inputs of the fused/EPContext node. | References to the initializers (e.g., the file, offset and length of external data) in its EPContext data. |
| Where the weights are in the compiled model | ORT copies them into the compiled model (see [4.1](#41-where-the-weights-are-stored)). | Not in the compiled model. They stay in the source model and its external data files. |
| How the EP gets the data at runtime | ORT passes them to `Compute()` through `KernelContext_GetInput()`. | The EP loads the data itself when the session is created, from the locations in [4.2](#42-requirements-per-mode). |

The EP-managed strategy lets the EP prepare the weights once at session creation instead of receiving them
through the kernel context. The ORT-provided strategy needs no file handling in the EP.

**1.4 Consistency check.** Applications choose a mode from the EP metadata, while ORT validates the request
against `GetWeightlessSupport()` during compilation. After calling `GetWeightlessSupport()`, ORT compares its
result with the weightless metadata of every `OrtEpDevice` the EP was created for (`"weightless_supported_modes"`,
or `"weightless_support"` if not present; no entry counts as `"none"`). If a device's value differs:

- ORT logs a warning naming the device and both values. A mismatch is an EP bug, but the requested mode may
  still be supported, so compilation continues.
- If the requested mode is also unsupported, the `ORT_EP_FAIL` error includes the same description. That is the
  case where an application chose a mode the metadata advertised and `GetWeightlessSupport()` rejected it.

An EP that only reports `"weightless_support"` but returns `OrtWeightlessSupport_ALL_OR_EXTERNAL_ONLY` gets this
warning: it should also report `"weightless_supported_modes"`.

## 2. Application checks the supported modes

The application reads the metadata of the `OrtEpDevice` it plans to use. It prefers `"weightless_supported_modes"`,
falls back to `"weightless_support"`, and ignores values it doesn't recognize:

```cpp
Ort::ConstEpDevice ep_device = /* selected from env.GetEpDevices() */;
Ort::ConstKeyValuePairs metadata = ep_device.EpMetadata();

const char* value = metadata.GetValue(kOrtEpDevice_EpMetadataKey_WeightlessSupportedModes);
if (value == nullptr) {
  value = metadata.GetValue(kOrtEpDevice_EpMetadataKey_WeightlessSupport);
}

const std::string supported = value != nullptr ? value : "none";
// Unrecognized values (from later versions) support neither mode.
const bool supports_external_only = supported == "external_only" || supported == "all_or_external_only";
const bool supports_all = supported == "all" || supported == "all_or_external_only";
```

## 3. Application chooses a mode and requests it

The application picks the mode that fits its deployment, or none at all. The main trade-off is what has to be
shipped and available at runtime (see [section 4](#4-runtime-and-packaging-requirements)):

- `EXTERNAL_ONLY`: the weights in the source model's external data file are not embedded in the EP binary.
  Small internal initializers are still embedded. The source model file itself is **not** needed at runtime.
- `ALL`: no initializer is embedded in the EP binary. The source model **is** needed at runtime.
- `NONE`: the compiled model is self-contained. No weight sharing between variants.

The request is made at compile time, with the compile API:

```cpp
Ort::SessionOptions session_options;
session_options.AppendExecutionProvider_V2(env, {ep_device}, ep_options);

Ort::ModelCompilationOptions compile_options(env, session_options);
compile_options.SetInputModelPath(ORT_TSTR("model.onnx"));
compile_options.SetOutputModelPath(ORT_TSTR("model_ctx.onnx"));
compile_options.SetOutputModelExternalInitializersFile(ORT_TSTR("model_ctx.onnx.data"), 0);
compile_options.SetWeightlessMode(OrtWeightlessSupport_EXTERNAL_ONLY);

Ort::CompileModel(env, compile_options);
```

or with the equivalent session options (for example when compiling through `"ep.context_enable"`):

```cpp
session_options.AddConfigEntry(kOrtSessionOptionEpEnableWeightlessMode, "1");  // OrtWeightlessSupport_EXTERNAL_ONLY
session_options.AddConfigEntry(kOrtSessionOptionsEpContextModelExternalInitializersFileName, "model_ctx.onnx.data");
```

The same option also works in the JIT flow (no EPContext model): the EP runs without copying the initializers
covered by the mode.

The mode is a compile-time choice. ORT writes it to the compiled model's metadata as `"weightless_mode"` =
`"1"` or `"2"`. Nothing is written for `NONE` or for the deprecated `"ep.enable_weightless"` option, which does
not select a mode. The application does not need to set `"ep.enable_weightless_mode"` again when it creates a
session from the compiled model. It can read the recorded mode from the model metadata to find out what the
model needs at runtime.

### Compatibility with EPs built for earlier versions

EPs built for 1.29 or 1.30 don't know `"ep.enable_weightless_mode"`; they only read `"ep.enable_weightless"`.
When `"ep.enable_weightless_mode"` is set, ORT therefore prepares the session options each plugin EP receives in
`OrtEpFactory::CreateEp()`:

| `"ep.enable_weightless_mode"` | EP metadata (`"weightless_supported_modes"`, else `"weightless_support"`) | Result |
|---|---|---|
| Not set | Any | No change: `"ep.enable_weightless"` keeps its previous behavior. |
| `"0"` (`NONE`) | Any | The EP sees `"ep.enable_weightless"` = `"0"`. |
| `"1"` (`EXTERNAL_ONLY`) | `"external_only"` or `"all_or_external_only"` | The EP sees `"ep.enable_weightless"` = `"1"`. |
| `"2"` (`ALL`) | `"all"` or `"all_or_external_only"` | The EP sees `"ep.enable_weightless"` = `"1"`. |
| `"1"` or `"2"` | Another value, an unrecognized value, or no entry | `ORT_EP_FAIL` before the EP is created. |
| Any other value, including `"3"`, `"0x2"` and `""` | Any | `ORT_INVALID_ARGUMENT`. An explicitly empty value is not treated as "not set". |

The EP gets a copy of the session options; the application's session options are not modified. An EP built for an
earlier version that reports `"weightless_support"` = `"all"` therefore works with `ALL`, and is rejected with
`EXTERNAL_ONLY` instead of silently running in a mode the application didn't ask for.

### Validation and errors

| Condition | Result |
|---|---|
| `ModelCompilationOptions_SetWeightlessMode()` called with a value other than 0, 1 or 2 (e.g. `3`) | `ORT_INVALID_ARGUMENT`, returned immediately. |
| `"ep.enable_weightless_mode"` set to anything other than exactly `"0"`, `"1"` or `"2"`, including other spellings such as `"0x2"` and an explicitly empty value | `ORT_INVALID_ARGUMENT` when the session is initialized. This applies to every API (C, C++, Python) and even if the session uses no plugin EP or compiles no nodes. |
| Mode is `NONE` | No weightless checks. |
| The EP metadata of a device doesn't include the requested mode | `ORT_EP_FAIL` before the EP is created. |
| EP built against API 29 or later does not implement `GetWeightlessSupport` | `ORT_NOT_IMPLEMENTED`. |
| `GetWeightlessSupport()` returns a value that doesn't include the requested mode, including `NONE` and unrecognized values | `ORT_EP_FAIL`. The message names the value and any mismatch with the EP metadata. |
| `GetWeightlessSupport()` differs from the weightless EP metadata of a device | Warning (see [1.4](#1-ep-reports-its-supported-modes)). |
| EP built against an API older than 29 | No `GetWeightlessSupport` check. ORT logs an INFO message and lets the EP handle the request. |
| Session created from a model with `"weightless_mode"` = `"2"` without `"ep.context_source_model_path"` or a source model buffer | Warning. The EP may still find the source model through `"onnx_model_filename"` (see [4.2](#42-requirements-per-mode)). |

The deprecated `"ep.enable_weightless"` = `"1"` (and `SetWeightlessEnabled(true)`, which sets it) still works when
`"ep.enable_weightless_mode"` isn't set. It does not select a mode, so ORT accepts any mode the EP supports
(any value of `GetWeightlessSupport()` other than `NONE`).

## 4. Runtime and packaging requirements

Weightless mode moves the weights out of the EP binary. The application is responsible for keeping them
available and telling ORT where they are when the session is created.

### 4.1 Where the weights are stored

This depends on the EP's strategy ([1.3](#1-ep-reports-its-supported-modes)).

**ORT-provided initializers.** The initializers are inputs of the EPContext nodes, and ORT copies every
initializer the compiled graph references into the compiled model:

- If an external initializers file is set, through `ModelCompilationOptions_SetOutputModelExternalInitializersFile()`
  or the `"ep.context_model_external_initializers_file_name"` session option
  (`kOrtSessionOptionsEpContextModelExternalInitializersFileName`), the initializers are written to that file.
  The session option places **all** initializers in the file (size threshold 0). The compile API applies the
  given size threshold.
- Otherwise the initializers are embedded in the compiled `.onnx` file. The model is then not weightless in
  any useful sense: it is as large as before and the weights can't be shared between variants.

**EP-managed initializers.** The initializers are dropped from the compiled graph, so ORT does not copy them.
The EP's references point to the source model's data:

- `EXTERNAL_ONLY`: the source model's external data files.
- `ALL`: the source model's external data files and the source model itself (for the initializers stored inside
  the `.onnx`).

An application generally can't tell which strategy an EP uses, so it should follow the requirements in 4.2,
which cover both.

### 4.2 Requirements per mode

| | `NONE` | `EXTERNAL_ONLY` | `ALL` |
|---|---|---|---|
| **Compile time:** external initializers file (`"ep.context_model_external_initializers_file_name"` or `SetOutputModelExternalInitializersFile`) | Optional | **Required** | **Required** |
| **Runtime:** compiled model (`*_ctx.onnx`) and EP context binary (if `ep_cache_context` is stored in a separate file, `embed_mode = 0`) | Required | Required | Required |
| **Runtime:** compiled model's external initializers file | Only if one was generated | **Required** (ORT-provided) | **Required** (ORT-provided) |
| **Runtime:** source model's external data files | Not used | **Required** (EP-managed) | **Required** (EP-managed) |
| **Runtime:** source model `.onnx` (`"ep.context_source_model_path"`, source model buffer, or `"onnx_model_filename"`) | Not used | Not used | **Required** |

The external initializers file is only strictly needed by EPs that use ORT-provided initializers. It is
required in the table because the application can't rely on knowing the EP's strategy. For EP-managed
initializers the file holds only the initializers of nodes that weren't compiled.

**Locating external data files.** Both the compiled model's external initializers file and the source model's
external data files are looked up relative to the directory of the model that references them. When they are
elsewhere, for example in one folder shared by several variants, or when the model is loaded from memory, set
`"session.model_external_initializers_file_folder_path"`
(`kOrtSessionOptionsModelExternalInitializersFileFolderPath`). All external data files must then be in that
folder.

**Locating the source model (`ALL`).** The EP looks for the source model in this order:

1. The buffer set with `OrtApi::SessionOptionsSetWeightlessSourceModelBuffer()`, for when the source model is
   not a file (e.g., loaded from a package or downloaded). The buffer must stay valid for the lifetime of the
   session.
2. The path in `"ep.context_source_model_path"` (`kOrtSessionOptionEpContextSourceModelPath`).
3. The `"onnx_model_filename"` attribute of the EPContext node, which records the source model file name
   at compile time. A relative value is resolved against the directory of the compiled model: the directory of
   the model path, or of `"ep.context_file_path"` (`kOrtSessionOptionEpContextFilePath`) when the compiled
   model is loaded from memory. This works when the source model is deployed next to the compiled model, or at
   the same relative location as at compile time.

`"ep.context_source_model_path"` is only needed in `ALL` mode, and only when the source model isn't at the
location given by `"onnx_model_filename"`. ORT logs a warning if a model recorded as `ALL` is loaded without a
path or buffer, because the attribute fallback is then the only way left.

```cpp
// Creating a session from a model compiled with OrtWeightlessSupport_ALL.
Ort::SessionOptions session_options;
session_options.AppendExecutionProvider_V2(env, {ep_device}, ep_options);
session_options.AddConfigEntry(kOrtSessionOptionEpContextSourceModelPath, "C:/models/model.onnx");
session_options.AddConfigEntry(kOrtSessionOptionsModelExternalInitializersFileFolderPath, "C:/models/weights");

Ort::Session session(env, ORT_TSTR("C:/models/model_ctx.onnx"), session_options);
```

When the EP context binary is stored in a separate file (`embed_mode = 0`) and the compiled model is loaded
from memory, or its location is overridden, also set `"ep.context_file_path"` so the binary can be found.

### 4.3 Packaging

What an application ships for each mode:

| Artifact | `NONE` | `EXTERNAL_ONLY` | `ALL` |
|---|---|---|---|
| Compiled model `*_ctx.onnx` (one per variant) | Yes | Yes | Yes |
| EP context binary (one per variant, if not embedded) | Yes, includes weights | Yes, includes internal initializers only | Yes, no weights |
| Compiled model external initializers file | Only if generated | Yes | Yes |
| Source model external data files | No | Yes, if the EP manages initializers | Yes, if the EP manages initializers |
| Source model `.onnx` | No | No | Yes |

Weight files can be shared by all variants compiled from the same source model.

**Model packages.** A variant sets its session options in `executor_info["ort"].session_options` (see
`onnxruntime/core/session/model_package/README.md`). `"ep.context_source_model_path"`,
`"session.model_external_initializers_file_folder_path"` and `"ep.context_file_path"` are path-valued options:
ORT resolves their values with the same rules as `model_file`, relative to the variant directory, or as a
`sha256:<hex>` reference to a shared asset. A package can therefore store the source model and its weights once,
as shared assets, and point every `ALL` variant at them:

```jsonc
"session_options": {
  "ep.context_source_model_path": "sha256:<hex>/model.onnx",
  "session.model_external_initializers_file_folder_path": "sha256:<hex>"
}
```

If a variant does not set `"ep.context_source_model_path"`, the EP falls back to `"onnx_model_filename"`,
resolved against the directory of the variant's compiled model. That only works if the source model is stored
inside the variant directory under that name.

## 5. Open issues

1. **The external initializers file is not enforced.** With ORT-provided initializers, compiling in weightless
   mode without an external initializers file embeds the weights in the compiled `.onnx`. ORT can't require
   the file in general, because EPs that manage their initializers don't need it. ORT could warn when a
   weightless EPContext node has initializer inputs and no external initializers file is set.
2. **The source model is not enforced for `ALL`.** ORT warns, but doesn't fail, when a model recorded as `ALL`
   is loaded without a source model path or buffer, because the EP may still find the source model through
   `"onnx_model_filename"`.
3. **Metadata and `GetWeightlessSupport` consistency is only checked when weightless mode is requested.**
   `GetWeightlessSupport()` is only called then, so an inconsistent EP goes unnoticed until an application
   requests a mode.
