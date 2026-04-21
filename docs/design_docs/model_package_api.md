# Model Package API — Proposal

## Table of Contents

- [Problem Statement](#problem-statement)
- [Design Principles](#design-principles)
- [Proposed API](#proposed-api)
  - [1. Package Context (Parse, Filter & Discover)](#1-package-context-parse-filter--discover)
  - [2. Query APIs](#2-query-apis)
  - [3. Session Creation](#3-session-creation)
  - [API Summary](#api-summary)
- [Variant Metadata Schema](#variant-metadata-schema)
  - [Schema Field Reference](#schema-field-reference)
  - [Variant Selection Algorithm](#variant-selection-algorithm)
- [Package Layout](#package-layout)
- [Usage Examples](#usage-examples)
  - [Example 1: Simple Consumer (Single-File Component)](#example-1-simple-consumer-single-file-component)
  - [Example 2: Multi-Component Package with QNN](#example-2-multi-component-package-with-qnn)
  - [Example 3: Advanced Consumer (Custom Session Options)](#example-3-advanced-consumer-custom-session-options)
  - [Example 4: Querying Package Contents](#example-4-querying-package-contents)
- [Key Design Decisions](#key-design-decisions)
  - [EP Criteria at Context Creation](#ep-criteria-at-context-creation)
  - [Session Creation Stays with `OrtSession`](#session-creation-stays-with-ortsession)
  - [ORT Applies Metadata Options (Not the Consumer)](#ort-applies-metadata-options-not-the-consumer)
  - [Session Options Precedence](#session-options-precedence)
  - [Two Parameters for Component + File (Not a Dotted String)](#two-parameters-for-component--file-not-a-dotted-string)
  - [No OrtEnv for Context Creation](#no-ortenv-for-context-creation)
  - [Partial Match Is Not an Error](#partial-match-is-not-an-error)
  - [Per-File EP Compatibility List](#per-file-ep-compatibility-list)
  - [`ep: null` for Neutral/CPU Files](#ep-null-for-neutralcpu-files)
  - [Variant Score Is the File-Score Average](#variant-score-is-the-file-score-average)
  - [Provider Options Are Flat](#provider-options-are-flat)
- [Implementation Plan](#implementation-plan)
  - [Phase 1: ORT](#phase-1-ort)
  - [Phase 2: ORT-GenAI](#phase-2-ort-genai)
- [Open Questions](#open-questions)
  - [1. Different EPs for Different Components](#1-different-eps-for-different-components)
  - [2. ORT-Managed EP Registration](#2-ort-managed-ep-registration)
  - [3. Known Session Options Setters](#3-known-session-options-setters)
  - [4. Shared Weights / External Data Files](#4-shared-weights--external-data-files)
  - [5. Cross-Component Consistency](#5-cross-component-consistency)
  - [6. JIT Compilation Caching](#6-jit-compilation-caching)
- [Appendix A: ORT-GenAI Integration](#appendix-a-ort-genai-integration)
  - [GenAI Session Creation Flow](#genai-session-creation-flow)
  - [What Changes in GenAI](#what-changes-in-genai)
  - [GenAI Config in Model Package World](#genai-config-in-model-package-world)
  - [Backward Compatibility](#backward-compatibility)
- [Appendix B: Session Options Reference](#appendix-b-session-options-reference)
  - [Known Session Options (Require Dedicated Setters)](#known-session-options-require-dedicated-setters)

## Problem Statement

The current ORT model package API bundles discovery, variant selection, and session creation into a single `Session(env, path, session_options)` call. This only works for single-component, single-ONNX-file packages.

Real-world models require:
- **Multi-component packages** — decoder + vision + embedding + speech
- **Multi-file variants** — e.g., QNN decoder split into 4 ONNX files (embeddings, context, iterator, lm_head)
- **Per-file session/provider options** — each file has the EP options it needs to load correctly
- **Per-variant consumer metadata** — opaque to ORT, consumed by higher-level frameworks
- **Consumer-managed session lifecycle** — destroy/reload individual sessions for memory management
- **Mixed EP within a variant** — some files forced to CPU (e.g., embeddings in a QNN pipeline)

## Design Principles

1. **Parse once, query freely** — create a package context filtered by EP criteria, then query it for components, files, identifiers, and metadata. No intermediate handles.
2. **Session creation stays with `OrtSession`** — an overload of the existing session creation API accepts a package context plus a component/file identifier in place of a model path. Model package parsing and session creation remain independent concerns, and consumers use the same `OrtSession` surface they already know.
3. **ORT applies required options** — session and provider options live in variant metadata and are applied by ORT during session creation. The consumer does not read and forward them.
4. **Consumer metadata is opaque to ORT** — ORT stores and returns it; never interprets it.
5. **EP selection is the consumer's responsibility** — the consumer decides the EP, then passes that decision to the package API as selection criteria. The package API does not discover or instantiate EPs.
6. **Keep ORT generic** — ORT handles package parsing, variant matching, and file path resolution. Consumer semantics (pipeline structure, execution phases, KV cache) stay in consumer config.

---

## Proposed API

### 1. Package Context (Parse, Filter & Discover)

```c
// EP selection criteria for variant filtering.
// Passed at context creation to filter variants across all components.
struct OrtModelPackageSelectionCriteria {
    const char* ep_name;         // e.g., "QNNExecutionProvider", NULL = CPU/no preference
    const char* device_type;     // e.g., "npu", "gpu", "cpu", NULL = no preference
    // Future: compatibility_info for device-specific checks
};

// Parse a model package directory and select variants matching the criteria.
//
// Reads manifest.json, discovers components and their metadata.json files,
// and for each component selects the best variant matching the criteria.
// The context is immutable after creation.
//
// Behavior:
//   - Single-variant components: criteria may be NULL — the only variant is selected.
//   - Multi-variant components: criteria must have at least ep_name.
//   - If a component has no matching variant, it is excluded from the context
//     (not returned by GetComponentNames). This is not an error — the package
//     may contain components for EPs the consumer is not using.
//
ORT_API_STATUS(CreateModelPackageContext,
    _In_ const ORTCHAR_T* package_path,
    _In_opt_ const OrtModelPackageSelectionCriteria* criteria,
    _Outptr_ OrtModelPackageContext** context);

ORT_API(void, ReleaseModelPackageContext, _Frees_ptr_ OrtModelPackageContext* context);
```

Context creation is pure parsing — read manifest, read metadata, match variants. It does not need `OrtEnv` because no runtime state (EP registry, logging, allocators) is involved. `OrtEnv` is only needed at session creation time.

### 2. Query APIs

```c
// --- Components ---

// List component names that have a matching variant for the given criteria.
ORT_API_STATUS(ModelPackageGetComponentNames,
    _In_ const OrtModelPackageContext* context,
    _Out_ size_t* num_components,
    _Outptr_ const char* const** component_names);

// --- Selected Variant ---

// Get the name of the selected variant for a component (for debugging/logging).
// e.g., "qnn-npu", "cpu", "cuda"
ORT_API_STATUS(ModelPackageGetSelectedVariantName,
    _In_ const OrtModelPackageContext* context,
    _In_ const char* component_name,
    _Outptr_ const char** variant_name);

// --- Files ---

// Get the number of ONNX model files in the selected variant for a component.
// Single-file components return 1. Multi-file (e.g., QNN pipeline) return N.
ORT_API_STATUS(ModelPackageGetFileCount,
    _In_ const OrtModelPackageContext* context,
    _In_ const char* component_name,
    _Out_ size_t* count);

// Get the identifiers for all files in a component's selected variant.
// Identifiers are logical names (e.g., "embeddings", "context", "iterator", "lm_head").
// For single-file components, the identifier may be the component name itself.
ORT_API_STATUS(ModelPackageGetFileIdentifiers,
    _In_ const OrtModelPackageContext* context,
    _In_ const char* component_name,
    _Out_ size_t* num_identifiers,
    _Outptr_ const char* const** identifiers);

// --- Per-File Metadata ---

// Get the resolved file path for a specific file in a component.
ORT_API_STATUS(ModelPackageGetFilePath,
    _In_ const OrtModelPackageContext* context,
    _In_ const char* component_name,
    _In_opt_ const char* file_identifier,   // NULL for single-file components
    _Outptr_ const ORTCHAR_T** file_path);

// Get the EP that was selected for a specific file during variant resolution.
// NULL means CPU (either the file had empty `ep_compatibility`, or no criteria
// was provided).
ORT_API_STATUS(ModelPackageGetFileEp,
    _In_ const OrtModelPackageContext* context,
    _In_ const char* component_name,
    _In_opt_ const char* file_identifier,
    _Outptr_opt_ const char** ep_name);

// Get the device type declared on the selected EP-compatibility entry for this
// file (e.g., "cpu", "gpu", "npu"). NULL if the entry did not declare one.
ORT_API_STATUS(ModelPackageGetFileDeviceType,
    _In_ const OrtModelPackageContext* context,
    _In_ const char* component_name,
    _In_opt_ const char* file_identifier,
    _Outptr_opt_ const char** device_type);

// Get session options for a specific file (JSON string) for the selected EP.
// NULL if none specified.
ORT_API_STATUS(ModelPackageGetFileSessionOptions,
    _In_ const OrtModelPackageContext* context,
    _In_ const char* component_name,
    _In_opt_ const char* file_identifier,
    _Outptr_opt_ const char** session_options_json);

// Get provider options for a specific file (JSON string) for the selected EP.
// Flat key-value — no EP-name nesting (EP is known from GetFileEp).
// NULL if none specified.
ORT_API_STATUS(ModelPackageGetFileProviderOptions,
    _In_ const OrtModelPackageContext* context,
    _In_ const char* component_name,
    _In_opt_ const char* file_identifier,
    _Outptr_opt_ const char** provider_options_json);

// --- Consumer Metadata (Opaque to ORT) ---

// Get consumer metadata for a component's selected variant.
// Raw JSON string. ORT does not parse or validate it. NULL if absent.
ORT_API_STATUS(ModelPackageGetConsumerMetadata,
    _In_ const OrtModelPackageContext* context,
    _In_ const char* component_name,
    _Outptr_opt_ const char** json_string);
```

### 3. Session Creation

Session creation uses a new overload of the existing `CreateSession` API. Instead of a file path, it takes a package context plus a component/file identifier. No new API surface is introduced for session management — consumers use the same `OrtSession` they already know.

```c
// Create a session from a file in a model package.
// Overload of CreateSession that takes a package context + component/file
// identifier instead of a model path.
//
// Session options precedence:
//   1. If session_options is NULL:
//      ORT creates a default OrtSessionOptions and applies:
//      - Session options from the selected EP's entry in `ep_compatibility`
//        (known keys via dedicated setters, unknown keys via AddConfigEntry)
//      - Provider options from the selected EP's entry (via AppendExecutionProvider)
//      - The selected EP itself (or CPU if the file's `ep_compatibility` is empty)
//      This is the simple path — "just load it with what the package says."
//
//   2. If session_options is provided:
//      ORT uses the caller's session options as-is. The caller is responsible
//      for EP registration and all configuration. Variant metadata session
//      and provider options are NOT applied. This is the advanced path for
//      consumers that need custom EP setup (shared CUDA streams, QNN shared
//      EP contexts, custom allocators, etc.).
//
// The component_name is required. file_identifier is optional:
//   - NULL: valid only for single-file components (returns error if multi-file)
//   - non-NULL: selects a specific file by its identifier
//
ORT_API_STATUS(CreateSession,                    // overload
    _In_ const OrtEnv* env,
    _In_ const OrtModelPackageContext* context,
    _In_ const char* component_name,
    _In_opt_ const char* file_identifier,
    _In_opt_ const OrtSessionOptions* session_options,
    _Outptr_ OrtSession** session);
```

The existing path-based `CreateSession(env, model_path, options, &session)` is unchanged. Language bindings can expose the overload naturally (e.g., `Ort::Session(env, package_ctx, "decoder", "iterator", opts)` in C++).

### API Summary

| API | Purpose | When to Use |
|---|---|---|
| `CreateModelPackageContext` | Parse package, filter by EP, discover components | Once per package |
| `ModelPackageGetComponentNames` | List components with matching variants | After context creation |
| `ModelPackageGetSelectedVariantName` | Get selected variant name (debugging) | After context creation |
| `ModelPackageGetFileCount` | Number of ONNX files in a component | Before creating sessions |
| `ModelPackageGetFileIdentifiers` | Logical names for files in a component | Before creating sessions |
| `ModelPackageGetFile*` | Per-file metadata (path, EP, options) | Optional — for advanced consumers |
| `ModelPackageGetConsumerMetadata` | Opaque consumer data (e.g., GenAI config overlay) | Framework-specific setup |
| `CreateSession` (package overload) | Create an ORT session for a specific component file | Per file that needs a session |

---

## Variant Metadata Schema

```json
{
    "component_model_name": "decoder",
    "model_variants": {
        "cpu": {
            "model_info": [
                {
                    "model_file": "model.onnx",
                    "identifier": "decoder",
                    "ep_compatibility": [
                        {
                            "ep": "CPUExecutionProvider",
                            "device_type": "cpu",
                            "session_options": {
                                "intra_op_num_threads": 4,
                                "graph_optimization_level": "ORT_ENABLE_ALL"
                            }
                        }
                    ]
                }
            ],
            "consumer_metadata": {
                "genai_config_overlay": { "model_type": "decoder" }
            }
        },
        "gpu": {
            "model_info": [
                {
                    "model_file": "model.onnx",
                    "identifier": "decoder",
                    "ep_compatibility": [
                        {
                            "ep": "CUDAExecutionProvider",
                            "device_type": "gpu",
                            "session_options": { "intra_op_num_threads": 1 },
                            "provider_options": { "enable_skip_layer_norm_strict_mode": "1" }
                        },
                        {
                            "ep": "WebGpuExecutionProvider",
                            "device_type": "gpu",
                            "session_options": { "intra_op_num_threads": 1 }
                        }
                    ]
                }
            ],
            "consumer_metadata": { "genai_config_overlay": { "model_type": "decoder" } }
        },
        "qnn-npu": {
            "model_info": [
                {
                    "model_file": "embeddings.onnx",
                    "identifier": "embeddings",
                    "ep_compatibility": [
                        {
                            "ep": null,
                            "device_type": "cpu",
                            "session_options": { "intra_op_num_threads": 2 }
                        }
                    ]
                },
                {
                    "model_file": "context.onnx",
                    "identifier": "context",
                    "ep_compatibility": [
                        {
                            "ep": "QNNExecutionProvider",
                            "device_type": "npu",
                            "compatibility_info": "abc123",
                            "session_options": { "intra_op_num_threads": 4 },
                            "provider_options": {
                                "htp_performance_mode": "burst",
                                "htp_graph_finalization_optimization_mode": "3",
                                "soc_model": "60"
                            }
                        }
                    ]
                },
                {
                    "model_file": "iterator.onnx",
                    "identifier": "iterator",
                    "ep_compatibility": [
                        {
                            "ep": "QNNExecutionProvider",
                            "device_type": "npu",
                            "compatibility_info": "def456",
                            "session_options": { "intra_op_num_threads": 4 },
                            "provider_options": {
                                "htp_performance_mode": "burst",
                                "htp_graph_finalization_optimization_mode": "3",
                                "soc_model": "60"
                            }
                        }
                    ]
                },
                {
                    "model_file": "lm_head.onnx",
                    "identifier": "lm_head",
                    "ep_compatibility": [
                        {
                            "ep": null,
                            "device_type": "cpu",
                            "session_options": { "intra_op_num_threads": 2 }
                        }
                    ]
                }
            ],
            "consumer_metadata": {
                "genai_config_overlay": {
                    "model_type": "decoder-pipeline",
                    "model": {
                        "decoder": {
                            "pipeline": [
                                { "model_id": "embeddings", "run_on_prompt": true, "run_on_token_gen": true },
                                { "model_id": "context", "run_on_prompt": true, "run_on_token_gen": false },
                                { "model_id": "iterator", "run_on_prompt": false, "run_on_token_gen": true, "reset_session_idx": 1 },
                                { "model_id": "lm_head", "run_on_prompt": true, "run_on_token_gen": true }
                            ]
                        }
                    }
                }
            }
        }
    }
}
```

### Schema Field Reference

**Per-file** (entries in `model_info`):

| Field | Required | Description |
|---|---|---|
| `model_file` | yes | ONNX model filename |
| `identifier` | yes | Logical name for consumer mapping |
| `ep_compatibility` | yes | List of EPs this file can run on. Always a list, even for single-EP files. Must have at least one entry. |

**Per `ep_compatibility` entry:**

| Field | Required | Description |
|---|---|---|
| `ep` | yes (may be `null`) | EP name (e.g., `"QNNExecutionProvider"`, `"CUDAExecutionProvider"`), or `null` to declare a neutral/CPU fallback entry that matches any criteria. |
| `device_type` | no | Hardware class this entry targets: `"cpu"`, `"gpu"`, `"npu"`, etc. Used as an optional secondary filter when the consumer provides `criteria.device_type`. If omitted, the entry matches any device type. |
| `compatibility_info` | no | Hash for device-specific compatibility checking (e.g., Qualcomm SoC model). |
| `session_options` | no | ORT session options to apply when running this file on this EP (flat key-value). Applied by ORT when session created without explicit session options. |
| `provider_options` | no | Provider-specific options for this EP (flat key-value, no EP-name nesting). Applied by ORT when session created without explicit session options. Only meaningful for non-null `ep`. |

**Per-variant:**

| Field | Required | Description |
|---|---|---|
| `consumer_metadata` | no | Opaque JSON blob for consumer frameworks. ORT stores and returns it; never interprets it. |

Notes:
- The previous per-variant `constraints` field is removed. Variant-level EP compatibility is now derived from the per-file `ep_compatibility` lists — a variant is compatible with EP X if every file either lists X or has a null (neutral) entry.
- The previous `force_cpu` flag is removed. An `{"ep": null}` entry naturally means "this file runs on CPU as a neutral fallback and does not constrain EP matching."
- `ep_compatibility` is always a list. For a file that only supports one EP, the list has one entry. This keeps the schema uniform for consumers and tooling.
- `device_type` is an optional free-form string (`"cpu"`, `"gpu"`, `"npu"`, ...) that lets producers disambiguate same-EP variants targeting different hardware classes (e.g., CUDA on discrete vs. integrated GPU). Consumers ignore it unless they set `criteria.device_type`.

### Variant Selection Algorithm

Selection happens at context creation based on `OrtModelPackageSelectionCriteria`. For each component, every variant is scored and the highest-scoring compatible variant wins.

**Per-file scoring** (given criteria `C`):

For each file, the selector walks the `ep_compatibility` list and picks the best matching entry:

| Entry condition | Entry score | Selected EP for file |
|---|---|---|
| `entry.ep == C.ep_name` **and** device filter passes | positive (see below) | `entry.ep` |
| `entry.ep == null` **and** device filter passes | `0` — neutral | `null` (CPU) |
| No entries pass and no null entry passes | — | **variant rejected** |

The **device filter** passes when: `C.device_type` is NULL, or `entry.device_type` is absent, or they match. In other words, `device_type` acts as an optional constraint on the criteria side and an optional declaration on the entry side — entries without a declared device type are wildcards.

If multiple entries match (e.g., a file has both a matching EP entry and a null fallback), the selector prefers the positive-scoring entry.

A matching entry's score is computed from:
- Base: `+1` for EP-name match.
- `+0.5` if `C.device_type` is provided and equals `entry.device_type`. (An entry without a declared device type passes the filter but does not earn this bonus — so a producer that wants to disambiguate "gpu CUDA" from "cpu-compiled CUDA" variants should set `device_type` on both.)
- `+0.5` if `C` provides device-specific compatibility info and it matches `entry.compatibility_info` (future work).

(Exact numerics are placeholders — the important properties are: null entry = 0, EP match > 0, additional signals monotonically increase the score.)

**Per-variant scoring:**
- If any file in the variant has no matching entry and no null fallback, the variant is rejected.
- Otherwise, variant score = arithmetic mean of file scores.
- Neutral (null-EP) files contribute `0` to the average, so a pure EP-matched variant will outscore a mixed variant (e.g., `avg(1,1,1,1) = 1.0` beats `avg(0,1,1,0) = 0.5`).
- Among variants with the same score, a deterministic tie-breaker (e.g., insertion order in `model_variants`) is used.

**Criteria edge cases:**

| Criteria | Behavior |
|---|---|
| NULL | Every file must have a null entry. All such variants score 0 and tie-break deterministically. Variants that require a specific EP are rejected. |
| `ep_name` = "CPUExecutionProvider" | Variants where every file has either a `CPUExecutionProvider` entry or a null entry are candidates. |
| `ep_name` = any other EP | Variants where every file has either a matching entry or a null entry are candidates. |

**Worked example** — package with three variants (`cpu`, `gpu`, `qnn-npu`) from the schema above:

- Criteria `{ ep_name: "CUDAExecutionProvider" }`:
  - `cpu`: `decoder` file's `ep_compatibility` lists only CPU (no null entry) → rejected.
  - `gpu`: `decoder` file's `ep_compatibility` includes CUDA → matched, score 1. Variant score = 1.0. ✅
  - `qnn-npu`: `context` lists only QNN → rejected.
  - Winner: `gpu`.
- Criteria `{ ep_name: "WebGpuExecutionProvider" }`:
  - `gpu`: `decoder` file also lists WebGPU → matched, score 1. Variant score = 1.0. ✅
  - Others: rejected.
  - Winner: `gpu`. Selected EP for the `decoder` file = `WebGpuExecutionProvider`; session/provider options come from the WebGPU entry.
- Criteria `{ ep_name: "QNNExecutionProvider" }`:
  - `qnn-npu`: `embeddings` null (0), `context` QNN (1), `iterator` QNN (1), `lm_head` null (0). Variant score = 0.5. ✅
  - `cpu`, `gpu`: rejected.
  - Winner: `qnn-npu`. Selected EP per file: `null` (CPU), QNN, QNN, `null` (CPU). The null-EP files still contribute their own session options to their CPU sessions.
- Criteria NULL:
  - `cpu`: `decoder` requires CPU EP (not null) → rejected.
  - `gpu`: requires CUDA/WebGPU → rejected.
  - `qnn-npu`: requires QNN for `context`/`iterator` → rejected.
  - Result: component excluded from context.

Filtering is at the variant level, not the file level. Once a variant is selected, **all** files in that variant are visible via the query APIs — including files running on CPU via a null entry. The consumer sees the complete set of files needed to run the component.

---

## Package Layout

```
phi-4mm.ortpackage/
├── manifest.json
├── configs/                          # Consumer files (convention, not ORT-managed)
│   ├── genai_config.json
│   ├── tokenizer.json
│   └── processor_config.json
└── models/
    ├── decoder/
    │   ├── metadata.json
    │   ├── cpu/
    │   │   └── model.onnx
    │   └── qnn-npu/
    │       ├── embeddings.onnx
    │       ├── context.onnx
    │       ├── iterator.onnx
    │       └── lm_head.onnx
    ├── vision/
    │   ├── metadata.json
    │   ├── cpu/
    │   │   └── vision.onnx
    │   └── qnn-npu/
    │       ├── patch_embed.onnx
    │       ├── vision_attn.onnx
    │       └── patch_merger.onnx
    └── embedding/
        ├── metadata.json
        └── cpu/
            └── embedding.onnx
```

---

## Usage Examples

### Example 1: Simple Consumer (Single-File Component)

```c
// Open package — no EP preference (works for single-variant packages)
OrtModelPackageContext* ctx = NULL;
CreateModelPackageContext(L"my-model.ortpackage", NULL, &ctx);

// Create session — ORT applies metadata options automatically
OrtSession* session = NULL;
CreateSession(env, ctx, "decoder", NULL, NULL, &session);
//                 ^ctx ^component ^single-file ^no custom opts

// Use session...
ReleaseSession(session);
ReleaseModelPackageContext(ctx);
```

### Example 2: Multi-Component Package with QNN

```c
// Open package filtered for QNN
OrtModelPackageSelectionCriteria criteria = {
    .ep_name = "QNNExecutionProvider",
    .device_type = "npu"
};
OrtModelPackageContext* ctx = NULL;
CreateModelPackageContext(L"phi-4mm.ortpackage", &criteria, &ctx);

// See what's available for QNN
size_t num_components = 0;
const char* const* names = NULL;
ModelPackageGetComponentNames(ctx, &num_components, &names);
// names = ["decoder", "vision"] — "embedding" excluded (CPU-only, no QNN variant)

// Query decoder files
size_t num_files = 0;
const char* const* ids = NULL;
ModelPackageGetFileIdentifiers(ctx, "decoder", &num_files, &ids);
// num_files = 4, ids = ["embeddings", "context", "iterator", "lm_head"]

// Create sessions for each file — ORT handles session/provider options from metadata
OrtSession* sessions[4];
for (size_t i = 0; i < num_files; i++) {
    CreateSession(env, ctx, "decoder", ids[i], NULL, &sessions[i]);
}

// Create vision session
OrtSession* vision_session = NULL;
CreateSession(env, ctx, "vision", NULL, NULL, &vision_session);
// If vision has multiple files, NULL file_identifier returns error — must specify

// Cleanup
for (size_t i = 0; i < num_files; i++) ReleaseSession(sessions[i]);
ReleaseSession(vision_session);
ReleaseModelPackageContext(ctx);
```

### Example 3: Advanced Consumer (Custom Session Options)

For consumers like GenAI that need custom EP setup (shared CUDA streams, QNN shared contexts):

```c
OrtModelPackageSelectionCriteria criteria = { .ep_name = "QNNExecutionProvider" };
OrtModelPackageContext* ctx = NULL;
CreateModelPackageContext(L"phi-4mm.ortpackage", &criteria, &ctx);

// Read consumer metadata for GenAI-specific config
const char* overlay_json = NULL;
ModelPackageGetConsumerMetadata(ctx, "decoder", &overlay_json);
// GenAI parses overlay for pipeline config, model_type, etc.

// Query files
size_t num_files = 0;
const char* const* ids = NULL;
ModelPackageGetFileIdentifiers(ctx, "decoder", &num_files, &ids);

for (size_t i = 0; i < num_files; i++) {
    // Read per-file EP to decide how to configure
    const char* ep = NULL;
    ModelPackageGetFileEp(ctx, "decoder", ids[i], &ep);

    // Build custom session options with GenAI's EP-specific setup
    OrtSessionOptions* opts = NULL;
    CreateSessionOptions(&opts);
    // ... GenAI applies shared CUDA stream, QNN shared EP context, etc. ...
    // ... GenAI appends EP with custom provider options ...

    OrtSession* session = NULL;
    CreateSession(env, ctx, "decoder", ids[i], opts, &session);
    // ORT does NOT apply metadata options — caller provided explicit opts

    ReleaseSessionOptions(opts);
    // Store session...
}

ReleaseModelPackageContext(ctx);
```

### Example 4: Querying Package Contents

```c
OrtModelPackageContext* ctx = NULL;
CreateModelPackageContext(L"phi-4mm.ortpackage", NULL, &ctx);

size_t num_components = 0;
const char* const* names = NULL;
ModelPackageGetComponentNames(ctx, &num_components, &names);

for (size_t c = 0; c < num_components; c++) {
    const char* variant = NULL;
    ModelPackageGetSelectedVariantName(ctx, names[c], &variant);

    size_t num_files = 0;
    ModelPackageGetFileCount(ctx, names[c], &num_files);

    printf("Component: %s (variant: %s, %zu files)\n", names[c], variant, num_files);

    const char* const* ids = NULL;
    size_t num_ids = 0;
    ModelPackageGetFileIdentifiers(ctx, names[c], &num_ids, &ids);

    for (size_t f = 0; f < num_files; f++) {
        const char* ep = NULL;
        ModelPackageGetFileEp(ctx, names[c], ids[f], &ep);
        printf("  File: %s (EP: %s)\n", ids[f], ep ? ep : "CPU");
    }
}

ReleaseModelPackageContext(ctx);
```

---

## Key Design Decisions

### EP Criteria at Context Creation

The EP decision is always made by the consumer before touching the package API — whether via auto-discovery, user config, or explicit choice. Folding EP criteria into context creation means the consumer asks the package a single question: "for my EP, what is available?" The package answers with the filtered set of components, files, and metadata. No per-component selection calls, no intermediate variant handles.

### Session Creation Stays with `OrtSession`

The model package is a file format. Session creation is an ORT runtime operation. Keeping them separate means the package API stays small and focused on discovery and metadata, while session creation stays with the existing `OrtSession` surface that consumers already understand. The package-aware form is just an overload of `CreateSession` that takes a package context plus a component/file identifier instead of a file path — no new "session from package" API is introduced.

This avoids creating a parallel surface for session management inside the package API. If the package itself created sessions, there would be two places where sessions get created and lifecycle-managed, which adds confusion for no real benefit — the caller already has an `OrtEnv` and `OrtSessionOptions` at session-creation time.

### ORT Applies Metadata Options (Not the Consumer)

The producer of the model knows what session and provider options the model needs to run correctly. Those options belong with the model, in its variant metadata. When the consumer asks ORT to load a file, ORT reads the metadata and applies the options.

Having the consumer read options from metadata and forward them to ORT during session creation adds a pass-through layer with no value — the consumer isn't making any decisions about those options. Keeping the application in ORT means the consumer code is simpler and the options stay close to the model they apply to.

Only options that are *required* to run the component belong in variant metadata. Anything overridable or policy-like (thread tuning, profiling, debug flags) belongs in the consumer's own config.

### Session Options Precedence

Two modes, no merging:

| `session_options` param | Behavior |
|---|---|
| **NULL** | ORT creates default options and applies ALL metadata (session options, provider options, EP). Simple path. |
| **Provided** | ORT uses caller's options as-is. Metadata options are NOT applied. Advanced path. |

All-or-nothing avoids complex precedence rules and subtle bugs from conflicting options. If a consumer needs something in between (e.g., apply metadata but tweak one field), they can query the metadata via the Get* APIs and build their own `OrtSessionOptions`.

### Two Parameters for Component + File (Not a Dotted String)

The `CreateSession` overload takes `component_name` and `file_identifier` as separate parameters rather than a single `"component.file"` string. Dotted strings have escaping pitfalls (what if an identifier contains a dot?), weaker validation, and awkward language bindings. Two parameters also match how the rest of the query APIs already address files.

### No OrtEnv for Context Creation

Context creation is pure parsing — manifest, metadata, variant matching. There's no need for runtime state until a session is actually created. Dropping `OrtEnv` from the context creation signature removes coupling between package discovery and ORT runtime setup.

### Partial Match Is Not an Error

If a package contains components that don't have a matching variant for the consumer's EP criteria, those components are silently excluded from the context. The consumer sees only what's usable. This matches the mental model of the API: "tell me what's available for my EP."

Components that *are* available can still be loaded. If a consumer expects a specific component and it's missing, that shows up at session creation time with a clear error.

### Per-File EP Compatibility List

A file declares its compatible EPs as a list (`ep_compatibility`), always — even when the file only supports one EP, the list has a single entry. Each entry carries its own `ep` name (or `null` for CPU/neutral), `compatibility_info`, `session_options`, and `provider_options`. The variant selector matches the consumer's EP criteria against each file's list and picks the best-scoring variant where every file either matches or has a null fallback.

Keeping a uniform list shape — rather than a flat "single-EP" shorthand — means consumers, tooling, and the selection algorithm always see one schema. The cost is one extra level of nesting for single-EP files, which is negligible given metadata.json is typically generated by tools rather than hand-written.

Storing per-EP options alongside each compatible EP keeps the options close to the execution target they apply to — when the selector picks CUDA for a file, it uses the CUDA-specific options attached to that entry. The null entry lets neutral/CPU files carry their own options too.

### `ep: null` for Neutral/CPU Files

An `ep_compatibility` entry with `ep: null` runs on CPU and matches any criteria as a neutral fallback. This serves two use cases:

1. **Base models meant for JIT compilation** — the package ships a single generic ONNX file that any EP can consume at runtime via JIT. The file has a single null entry, so it matches any criteria and scores neutrally.
2. **Mixed-EP variants** — a QNN decoder pipeline where embeddings and `lm_head` run on CPU while the rest run on QNN. The CPU-resident files have a null entry; they ride along with the variant without affecting its match decision against non-CPU criteria.

Null entries can still carry session options (and those are applied by ORT when the file is loaded on CPU), so a CPU file with specific threading or optimization settings can have them declared alongside any EP-specific entries.

Scoring null entries as `0` (neutral) means a purely EP-matched variant will outscore a mixed variant when both are viable, which is the right preference: if the producer ships a full-CUDA variant alongside a CUDA-decoder-with-CPU-embeddings variant, the full-CUDA one wins.

This subsumes the old `force_cpu` flag — a null entry expresses the same intent without an extra field, and it naturally allows session options on neutral files.

### Variant Score Is the File-Score Average

A multi-file variant is either fully compatible with the criteria (every file has a matching EP entry or a null entry) or rejected outright. When compatible, the variant's score is the mean of its file scores. Averaging gives a simple, monotonic score that:

- Prefers variants with more EP-matched files over variants with more null/neutral files
- Has a natural interpretation — "how EP-specific is this variant, on average?"
- Scales cleanly when additional signals (device match, compatibility info) are added later

More sophisticated aggregation (weighted means, per-file importance) can be layered in later without changing the schema.

### Provider Options Are Flat

Since each `ep_compatibility` entry already names its EP, provider options inside that entry don't need EP-name nesting. Flat key-value pairs, passed directly to `AppendExecutionProvider` for the selected EP.

---

## Implementation Plan

The work splits into two phases. Phase 1 lands everything in ORT so a C/C++/Python consumer can load a model package end-to-end. Phase 2 wires ORT-GenAI on top of the new APIs.

Within Phase 1, the workstreams are designed so that, after a short shared foundation, multiple developers can work in parallel on disjoint parts of the codebase.

> **Tracking note:** Check off items as they land. Link the PR/issue next to each bullet when one is filed (e.g., `— #1234`). A workstream is "done" only when all its items are checked and tests are green in CI.

### Phase 1: ORT

#### Shared foundation (must land first)

**WS0. Schema types & internal data model**
- [ ] Internal C++ structs mirroring the metadata schema: `ModelPackage`, `Component`, `Variant`, `FileEntry`, `EpCompatibilityEntry`.
- [ ] Public opaque handle `OrtModelPackageContext`.
- [ ] Public `OrtModelPackageSelectionCriteria` struct (including `device_type`).
- [ ] Error codes and status messages for malformed packages.
- [ ] Headers published so other workstreams can compile against them.

Everyone else depends on WS0. Once its header lands, the remaining workstreams below can proceed in parallel on separate PRs.

#### Parallel workstreams (A / B / C / D can all proceed independently after WS0)

**WS-A. Package loader & metadata parser** *(Dev 1)*
- [ ] Parse `manifest.json` (list of components, paths to per-component `metadata.json`).
- [ ] Parse `metadata.json` (component name, variants, per-file `ep_compatibility`, consumer metadata blob).
- [ ] Validation: required fields, at least one `ep_compatibility` entry per file, file paths resolve, identifiers unique within a variant.
- [ ] Populate `ModelPackage` structs from WS0.
- [ ] Unit tests with fixture packages (valid, malformed, missing fields).

**WS-B. Variant selection algorithm** *(Dev 2)*
- [ ] Per-file scoring per [Variant Selection Algorithm](#variant-selection-algorithm): EP match, `ep: null` neutral entries, device filter + bonus.
- [ ] Variant-level rejection when any file has no matching entry and no null fallback.
- [ ] Variant score = arithmetic mean of file scores.
- [ ] NULL criteria path (all-null-files variants only).
- [ ] Deterministic tie-break (insertion order).
- [ ] Pure function over WS0 structs — no I/O, no ORT runtime deps.
- [ ] Unit tests: single-EP variants, multi-EP files, mixed neutral+EP variants, all-rejected case, NULL criteria, ties, device-type filter.

**WS-C. Public C API — context & queries** *(Dev 3)*
- [ ] `CreateModelPackageContext` (calls WS-A loader + WS-B selector; returns opaque handle).
- [ ] `ReleaseModelPackageContext`.
- [ ] `GetComponentNames`, `GetSelectedVariantName`.
- [ ] `GetFileCount`, `GetFileIdentifiers`, `GetFilePath`.
- [ ] `GetFileEp`, `GetFileDeviceType`.
- [ ] `GetFileSessionOptions`, `GetFileProviderOptions`.
- [ ] `GetConsumerMetadata`.
- [ ] Memory ownership, release semantics, thread-safety notes documented in the headers.
- [ ] C API smoke tests.

**WS-D. Session creation integration** *(Dev 4)*
- [ ] `CreateSession` overload taking `OrtModelPackageContext*`, `component_name`, `file_identifier`, optional `OrtSessionOptions*`.
- [ ] Session options precedence: NULL caller options → ORT builds options from metadata; non-NULL → caller's options used as-is.
- [ ] ORT-side applier for per-file `session_options` (generic passthrough + dispatch for known setters — see [Appendix B](#appendix-b-session-options-reference)).
- [ ] ORT-side applier for per-file `provider_options` — wire `AppendExecutionProvider` for the selected EP.
- [ ] Integration tests that go from a sample package on disk to a runnable session.
- [ ] This workstream depends on A/B/C landing; can develop against stubs in parallel.

#### Post-integration (serial, once A–D are merged)

**WS-E. Language bindings**
- [ ] C++ wrappers (`Ort::ModelPackageContext`, `Ort::SelectionCriteria`, `Ort::Session` overload).
- [ ] Python bindings (`onnxruntime.ModelPackageContext`, `InferenceSession` from context).
- [ ] Binding-level tests (C++ and Python).

**WS-F. Sample packages & end-to-end tests**
- [ ] Fixture package: single-file CPU.
- [ ] Fixture package: multi-EP single-file (CUDA + WebGPU).
- [ ] Fixture package: multi-file QNN pipeline with CPU-resident files.
- [ ] Fixture package: no matching variant (negative case).
- [ ] E2E tests driving the public API across all fixtures.
- [ ] Some of this can start in parallel with WS-A/B using hand-written fixtures; the bulk lands after WS-D.

#### Dependency graph (Phase 1)

```
WS0 ──┬──► WS-A ──┐
      ├──► WS-B ──┼──► WS-D ──► WS-E
      └──► WS-C ──┘         └─► WS-F
```

Four developers can productively work in parallel on WS-A, WS-B, WS-C, WS-D immediately after WS0 lands, using stubs or mocks for cross-workstream dependencies until they merge.

### Phase 2: ORT-GenAI

Depends on Phase 1 being available in a released (or internal) ORT build.

**WS-G. Package detection & context creation**
- [ ] Detect "flat directory" vs "model package" at GenAI `Model::Create` time (presence of `manifest.json`).
- [ ] For package path: build `OrtModelPackageSelectionCriteria` from GenAI's existing EP config.
- [ ] Call `CreateModelPackageContext`; hold it on the GenAI `Model`.
- [ ] Flat-directory code path left unchanged (regression guard).

**WS-H. Session creation via package context**
- [ ] Replace GenAI's per-component session creation with the `CreateSession(env, ctx, component, file_id, opts)` overload.
- [ ] Continue owning EP-specific runtime plumbing (CUDA streams, QNN shared context handles) and pass via session options (relying on "consumer's options win" precedence).
- [ ] Multi-file components: iterate file identifiers from the package context, create one session per file.
- [ ] Derive `p_device_` from `GetFileEp` / `GetFileDeviceType` instead of `is_primary_session_options`.

**WS-I. `genai_config` overlay consumption**
- [ ] Merge per-variant `consumer_metadata.genai_config_overlay` on top of GenAI's defaults / legacy `genai_config.json`.
- [ ] Drive pipeline construction (e.g., QNN `decoder-pipeline` entries) from the overlay.
- [ ] Fall back cleanly when a package provides no overlay.

**WS-J. Tests & sample models**
- [ ] Single-file package test.
- [ ] Multi-EP package test.
- [ ] QNN multi-file pipeline package test.
- [ ] Legacy flat-directory regression test.
- [ ] At least one canonical sample package checked in or referenced from the test fixtures.

**WS-K. Documentation**
- [ ] Update GenAI docs to describe the model package path alongside the flat-directory path.
- [ ] Migration notes for existing consumers.

#### Dependency graph (Phase 2)

```
Phase 1 released ──► WS-G ──► WS-H ──┬──► WS-J
                            │        │
                            └► WS-I ─┘
                                     │
                                     └──► WS-K
```

WS-G is strictly first in Phase 2; after that, WS-H and WS-I can be developed in parallel, with WS-J and WS-K wrapping up.

---

## Open Questions

### 1. Different EPs for Different Components

The current design uses one `SelectionCriteria` for the entire context. What if a consumer needs decoder on QNN but embedding on CPU?

Options:
- Create two contexts, one per EP.
- Extend `OrtModelPackageSelectionCriteria` to support per-component overrides.

Most real deployments use one EP for the whole pipeline, so this is an edge case. We can start with one criteria and extend later if needed.

### 2. ORT-Managed EP Registration

When `session_options` is NULL, ORT applies the EP from metadata via `AppendExecutionProvider`. Open questions:
- Does the EP need to be already registered in the ORT build?
- Should ORT attempt to load the EP plugin if missing?

Proposed: require the EP to be available. If not, return an error with a clear message.

### 3. Known Session Options Setters

ORT's "known" session options (`intra_op_num_threads`, etc.) require dedicated setter APIs — `AddConfigEntry` silently does nothing for them. For the metadata-driven path, ORT needs a fixed mapping of ~9 known key names to their setters. See Appendix B for the list. Tractable.

### 4. Shared Weights / External Data Files

Deferred. When tackled, the package-aware `CreateSession` overload abstracts external data path resolution — this is a key benefit of having a package-aware session creation path.

### 5. Cross-Component Consistency

When selecting variants for multiple components independently, how do we ensure they're compatible? Package-level profiles can be added later if needed. For now, leave coordination to the consumer.

### 6. JIT Compilation Caching

Deferred. After compilation, ORT saves artifacts into the package and updates variant metadata.

---

## Appendix A: ORT-GenAI Integration

This section details how ORT-GenAI, a sophisticated consumer, integrates with the API.

### GenAI Session Creation Flow

```cpp
ComponentSessions Model::CreateComponentSessions(
    const std::string& component_name) {

    ComponentSessions result;

    if (model_package_ctx_) {
        // --- Model Package Path ---

        // 1. Read consumer metadata for GenAI-specific config
        const char* overlay_json = nullptr;
        ModelPackageGetConsumerMetadata(model_package_ctx_.get(),
            component_name.c_str(), &overlay_json);
        if (overlay_json) {
            config_->ApplyOverlay(component_name, overlay_json);
        }

        // 2. Get file list
        size_t num_files = 0;
        const char* const* identifiers = nullptr;
        ModelPackageGetFileIdentifiers(model_package_ctx_.get(),
            component_name.c_str(), &num_files, &identifiers);

        // 3. Create sessions per file
        for (size_t i = 0; i < num_files; i++) {
            const char* ep = nullptr;
            ModelPackageGetFileEp(model_package_ctx_.get(),
                component_name.c_str(), identifiers[i], &ep);

            result.identifiers.push_back(identifiers[i]);
            result.file_eps.push_back(ep ? ep : "");

            // GenAI builds custom session options with EP-specific runtime state
            auto session_opts = OrtSessionOptions::Create();
            SetupEpSpecificOptions(ep, *session_opts);
            // ^ Injects shared CUDA stream, QNN shared EP context, etc.

            // Use the package-aware CreateSession overload
            result.sessions.push_back(OrtSession::Create(
                GetOrtEnv(), model_package_ctx_.get(),
                component_name.c_str(), identifiers[i],
                session_opts.get()));
        }

        // 4. Derive device from per-file EPs
        p_device_ = GetDeviceInterface(DeriveDeviceFromFileEps(result));

    } else {
        // --- Legacy Flat Directory Path (unchanged) ---
        auto opts = OrtSessionOptions::Create();
        CreateSessionOptionsFromConfig(
            config_->model.decoder.session_options, *opts, true);
        result.sessions.push_back(
            OrtSession::Create(GetOrtEnv(), config_path_ / filename, *opts));
        result.identifiers.push_back(component_name);
    }
    return result;
}
```

### What Changes in GenAI

| Current Code | With Model Packages | Status |
|---|---|---|
| EP-specific session handlers (CUDA, QNN, DML) | Still needed — GenAI passes custom session options | **Stays** |
| EP dispatch (`session_options.cpp`) | Still needed — GenAI registers EPs | **Stays** |
| `CreateSessionOptionsFromConfig()` | For model package path: replaced by ORT reading metadata directly. GenAI only adds EP-specific runtime state. | **Simplified** |
| File path resolution (`config_path_ / filename`) | Replaced by package-aware `CreateSession` overload | **Removed** |
| `is_primary_session_options` parameter | No longer needed — `p_device_` derived from per-file EPs | **Removed** |
| `SetProviderSessionOptions()` | EP from `GetFileEp()`, GenAI only adds runtime state | **Adapted** |

### GenAI Config in Model Package World

**Base genai_config.json** (in `<package>/configs/`):
```json
{
    "model_type": "decoder",
    "search": { "max_length": 2048 },
    "model": {
        "decoder": { "component": "decoder" },
        "vision": { "component": "vision" },
        "embedding": { "component": "embedding" }
    }
}
```

`"component": "decoder"` replaces `"filename": "model.onnx"`. GenAI uses the component name with the model package API instead of a file path.

### Backward Compatibility

| Scenario | Detection | Session Creation | EP Discovery |
|---|---|---|---|
| Flat directory (current) | No manifest.json | Legacy GenAI flow | `SetProviderSessionOptions()` → `p_device_` |
| Model package | manifest.json present | Package-aware `CreateSession` overload | Per-file EPs → `DeriveDeviceFromFileEps()` |

Legacy path is preserved unchanged.

---

## Appendix B: Session Options Reference

### Known Session Options (Require Dedicated Setters)

| Field | Type | ORT Setter |
|---|---|---|
| `intra_op_num_threads` | int | `SetIntraOpNumThreads` |
| `inter_op_num_threads` | int | `SetInterOpNumThreads` |
| `enable_cpu_mem_arena` | bool | `EnableCpuMemArena` / `DisableCpuMemArena` |
| `enable_mem_pattern` | bool | `EnableMemPattern` / `DisableMemPattern` |
| `log_id` | string | `SetSessionLogId` |
| `log_severity_level` | int | `SetSessionLogSeverityLevel` |
| `log_verbosity_level` | int | `SetSessionLogVerbosityLevel` |
| `enable_profiling` | string | `EnableProfiling` |
| `graph_optimization_level` | string enum | `SetSessionGraphOptimizationLevel` |

All other keys are passed through via `AddSessionConfigEntry`. This is a fixed, tractable list.
