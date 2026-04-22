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
  - [Example 1: Simple Consumer (Single-File Component, Flat-Directory Style)](#example-1-simple-consumer-single-file-component-flat-directory-style)
  - [Example 2: Multi-Component Package with QNN](#example-2-multi-component-package-with-qnn)
  - [Example 3: Advanced Consumer (Custom Session Options)](#example-3-advanced-consumer-custom-session-options)
  - [Example 4: Querying Selected Variants](#example-4-querying-selected-variants)
- [Key Design Decisions](#key-design-decisions)
  - [EP Selection Is Captured from `OrtSessionOptions`](#ep-selection-is-captured-from-ortsessionoptions)
  - [NULL Options Means "Single-Variant Package"](#null-options-means-single-variant-package)
  - [Multi-EP Matching Commits One EP Globally](#multi-ep-matching-commits-one-ep-globally)
  - [Captured Options Are a Catalog at Session Creation](#captured-options-are-a-catalog-at-session-creation)
  - [Session Creation Stays with `OrtSession`](#session-creation-stays-with-ortsession)
  - [ORT Applies Metadata Options (Not the Consumer)](#ort-applies-metadata-options-not-the-consumer)
  - [Session Options Precedence](#session-options-precedence)
  - [Two Parameters for Component + File (Not a Dotted String)](#two-parameters-for-component--file-not-a-dotted-string)
  - [`OrtEnv` Only Needed at Options Creation](#ortenv-only-needed-at-options-creation)
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
  - [7. `GetSelectedVariantPath` with NULL Options and EP-Declaring Variants](#7-getselectedvariantpath-with-null-options-and-ep-declaring-variants)
  - [8. Metadata-Aware Policy Delegates (Deferred)](#8-metadata-aware-policy-delegates-deferred)
  - [9. Package Inspection Mode (Deferred)](#9-package-inspection-mode-deferred)
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

1. **Parse once, query freely** — create a package context filtered by the captured EP selection, then query it for components, files, identifiers, and metadata. No intermediate handles.
2. **Session creation stays with `OrtSession`** — an overload of the existing session creation API accepts a package context plus a component/file identifier in place of a model path. Model package parsing and session creation remain independent concerns, and consumers use the same `OrtSession` surface they already know.
3. **ORT applies required options** — session and provider options live in variant metadata and are applied by ORT during session creation. The consumer does not read and forward them.
4. **Consumer metadata is opaque to ORT** — ORT stores and returns it; never interprets it.
5. **EP selection is the consumer's responsibility** — the consumer decides the EP through the same `OrtSessionOptions` surface they already use for session creation. The model package API captures that decision into an `OrtModelPackageOptions` handle and consults it during variant selection. The package API does not discover or instantiate EPs.
6. **Keep ORT generic** — ORT handles package parsing, variant matching, and file path resolution. Consumer semantics (pipeline structure, execution phases, KV cache) stay in consumer config.

---

## Proposed API

The model package surface is exposed through a small sub-API, `OrtModelPackageApi`, obtained from the main `OrtApi`. This mirrors the shape of `OrtCompileApi` and keeps the package-specific functions grouped. Ownership and release semantics for the two new handles (`OrtModelPackageOptions`, `OrtModelPackageContext`) are standard ORT opaque-struct patterns.

### 1. Package Options & Context (Parse, Filter & Discover)

EP selection is **captured from the consumer's existing `OrtSessionOptions`**. Rather than introducing a new selection DSL, the model package API reuses the mechanisms ORT already has for declaring EP intent — `SessionOptionsAppendExecutionProvider_V2` and `SessionOptionsSetEpSelectionPolicy`/`SessionOptionsSetEpSelectionPolicyDelegate`. ORT then captures the resolved `OrtEpDevice` list (plus any non-EP session-level settings) onto an `OrtModelPackageOptions` handle that drives both variant selection and, later, session creation.

```c
// Create a snapshot of EP selection + session-level settings from a session_options.
//
// Captures (by copy):
//   - If SessionOptionsAppendExecutionProvider_V2 was called: the appended
//     OrtEpDevices and their EP options are captured directly.
//   - Else if SessionOptionsSetEpSelectionPolicy / *Delegate was set: the policy
//     is resolved against env's currently registered OrtEpDevices and the
//     resulting OrtEpDevices are captured.
//   - Non-EP session-level settings (threading, optimization level, log severity,
//     free-dim overrides, custom ops, config entries) are also captured so they
//     can be replayed on sessions created from this context.
//
// After this call returns, session_options may be released by the caller.
//
// The captured OrtEpDevice list acts as both:
//   - the ordered candidate-EP list for variant selection at CreateModelPackageContext time, and
//   - the EP *catalog* that session creation looks up when the selected variant's
//     file declares its EP (so the exact OrtEpDevice identity is reused — factory,
//     hardware device, ep_metadata).
ORT_API2_STATUS(CreateModelPackageOptionsFromSessionOptions,
    _In_ const OrtEnv* env,
    _In_ const OrtSessionOptions* session_options,
    _Outptr_ OrtModelPackageOptions** out);

// Standard release.
ORT_CLASS_RELEASE(ModelPackageOptions);

// Parse a model package directory and resolve variants.
//
// `options` drives variant selection:
//   - NULL options: NULL-options path. Each component is resolved by variant
//     count — a single variant per component is selected as-is, regardless of
//     its declared EPs. If any component has >1 variants, returns an ambiguity
//     error. Intended for flat-directory migration / inspection scenarios where
//     the consumer handles EP setup elsewhere (or not at all).
//   - Non-NULL options with no EPs captured: match only variants whose files
//     either declare no EP (neutral/CPU) or are covered by null `ep_compatibility`
//     entries. Effective "CPU / unconstrained only" filter.
//   - Non-NULL options with a captured EP list: run the two-pass walk described
//     in the Variant Selection Algorithm.
//
// If a component has no matching variant, it is excluded from the context (not
// returned by GetComponentNames). This is not an error — the package may contain
// components for EPs the consumer is not using.
//
// `options` may be released by the caller after this call; the context deep-copies
// what it needs.
ORT_API2_STATUS(CreateModelPackageContext,
    _In_ const ORTCHAR_T* package_path,
    _In_opt_ const OrtModelPackageOptions* options,
    _Outptr_ OrtModelPackageContext** out);

ORT_CLASS_RELEASE(ModelPackageContext);
```

Context creation is pure parsing + matching — it does not instantiate EP factories, create sessions, or touch the env. Policy resolution (if any) has already happened inside `CreateModelPackageOptionsFromSessionOptions`; the captured `OrtEpDevice` list on `options` is what the two-pass matcher walks. `OrtEnv` is supplied later, at `CreateSession` time, on the main `OrtApi::CreateSession` signature.

### 2. Query APIs

```c
// --- Components ---

// List component names that have a matching variant for the captured EP selection.
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
// NULL means CPU (either the file had empty `ep_compatibility`, the selected
// entry had `ep: null`, or the context was created with NULL options and the
// selected single variant declared no EP for this file).
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
// The component_name is required. file_identifier is optional:
//   - NULL: valid only for single-file components (returns error if multi-file)
//   - non-NULL: selects a specific file by its identifier
//
// Session options precedence (depends on what `context` was created with, and
// whether `session_options` is provided):
//
// +-------------------+----------------------+------------------------------------------------+
// | context options   | session_options arg  | What ORT does                                  |
// +-------------------+----------------------+------------------------------------------------+
// | NULL              | NULL                 | Built entirely from the selected variant's     |
// |                   |                      | file metadata: EP, session_options, and        |
// |                   |                      | provider_options from the chosen               |
// |                   |                      | `ep_compatibility` entry. CPU for `ep: null`.  |
// |                   |                      | (Equivalent to "the package tells us what      |
// |                   |                      | to do.")                                       |
// +-------------------+----------------------+------------------------------------------------+
// | NULL              | provided             | Caller's session_options used as-is. Caller    |
// |                   |                      | is responsible for EP registration and all     |
// |                   |                      | configuration. Metadata is NOT applied.        |
// +-------------------+----------------------+------------------------------------------------+
// | non-NULL          | NULL                 | Non-EP settings from the captured session      |
// |                   |                      | options + the single OrtEpDevice from the      |
// |                   |                      | captured catalog that matches this file's      |
// |                   |                      | selected EP (or no EP for `ep: null`) +        |
// |                   |                      | per-file metadata session_options and          |
// |                   |                      | provider_options merged on top.                |
// +-------------------+----------------------+------------------------------------------------+
// | non-NULL          | provided             | Caller's session_options used as-is. Neither   |
// |                   |                      | captured options nor variant metadata is       |
// |                   |                      | applied. Escape hatch for consumers that need  |
// |                   |                      | custom EP setup (shared CUDA streams, QNN      |
// |                   |                      | shared EP contexts, custom allocators, etc.).  |
// +-------------------+----------------------+------------------------------------------------+
//
// Note on the "non-NULL context + NULL session_options" row:
// The captured OrtEpDevice list is NOT replayed wholesale. It is used as a
// catalog — ORT looks up the specific device corresponding to this file's
// selected EP and registers only that one. Files with `ep: null` do not have
// any EP registered (they run on default CPU). This avoids registering every
// candidate EP on every session.
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
| `CreateModelPackageOptionsFromSessionOptions` | Snapshot EP selection + session settings from a `OrtSessionOptions` | Before opening a package (optional — pass NULL options for the NULL-options path) |
| `ReleaseModelPackageOptions` | Release the options handle | After context creation; options are deep-copied into the context |
| `CreateModelPackageContext` | Parse package, filter by captured EP selection, discover components | Once per package |
| `ReleaseModelPackageContext` | Release the context handle | When done with the package |
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
| `ep` | yes (may be `null`) | EP name (e.g., `"QNNExecutionProvider"`, `"CUDAExecutionProvider"`), or `null` to declare a neutral/CPU fallback entry that matches any captured EP. |
| `device_type` | no | Hardware class this entry targets: `"cpu"`, `"gpu"`, `"npu"`, etc. Used as an optional secondary filter against the `OrtHardwareDevice->type` of the captured `OrtEpDevice` being considered. If omitted, the entry matches any device type. |
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
- `device_type` is an optional free-form string (`"cpu"`, `"gpu"`, `"npu"`, ...) that lets producers disambiguate same-EP variants targeting different hardware classes (e.g., CUDA on discrete vs. integrated GPU). Consumers influence it indirectly via the `OrtHardwareDevice` attached to each captured `OrtEpDevice`.

### Variant Selection Algorithm

Selection happens at context creation based on the **captured EP list** on `OrtModelPackageOptions` (or its absence, for the NULL-options path). For each component, every variant is scored and the highest-scoring compatible variant wins.

**Algorithm input**

Variant selection consumes an ordered list of `(ep_name, device_type)` pairs derived from the captured `OrtEpDevice`s on the options handle. The ordering is insertion order for explicit `Append_V2` calls, or policy-preference order for policy-resolved selection. If `options` is NULL, the algorithm short-circuits to the NULL-options path described below.

**Per-file scoring** (given the candidate EP currently being considered, call it `C = (ep_name, device_type)`):

For each file, the selector walks the `ep_compatibility` list and picks the best matching entry:

| Entry condition | Entry score | Selected EP for file |
|---|---|---|
| `entry.ep == C.ep_name` **and** device filter passes | positive (see below) | `entry.ep` |
| `entry.ep == null` **and** device filter passes | `0` — neutral | `null` (CPU) |
| No entries pass and no null entry passes | — | **variant rejected** |

The **device filter** passes when: `C.device_type` is unspecified, or `entry.device_type` is absent, or they match. `device_type` is a secondary filter on the entry side and an optional declaration from the captured `OrtEpDevice`'s hardware side — entries without a declared device type are wildcards.

If multiple entries match (e.g., a file has both a matching EP entry and a null fallback), the selector prefers the positive-scoring entry.

A matching entry's score is computed from:
- Base: `+1` for EP-name match.
- `+0.5` if `C.device_type` is present and equals `entry.device_type`. (An entry without a declared device type passes the filter but does not earn this bonus — so a producer that wants to disambiguate "gpu CUDA" from "cpu-compiled CUDA" variants should set `device_type` on both.)
- `+0.5` if the captured `OrtEpDevice` provides device-specific compatibility info that matches `entry.compatibility_info` (future work).

(Exact numerics are placeholders — the important properties are: null entry = 0, EP match > 0, additional signals monotonically increase the score.)

**Per-variant scoring:**
- If any file in the variant has no matching entry and no null fallback, the variant is rejected.
- Otherwise, variant score = arithmetic mean of file scores.
- Neutral (null-EP) files contribute `0` to the average, so a pure EP-matched variant will outscore a mixed variant (e.g., `avg(1,1,1,1) = 1.0` beats `avg(0,1,1,0) = 0.5`).
- Among variants with the same score, a deterministic tie-breaker (e.g., insertion order in `model_variants`) is used.

**Options edge cases:**

| Options input | Behavior |
|---|---|
| `options` = NULL | NULL-options path — each component is resolved by variant count. A single variant is selected as-is regardless of its declared EPs; >1 variants for any component → ambiguity error at `CreateModelPackageContext`. Intended for flat-directory migration and inspection scenarios where EP setup happens elsewhere. |
| Captured EP list is **empty** (no `Append_V2`, no policy result) | Only variants whose files either declare no EP requirement or are fully covered by null entries are candidates. Effectively "CPU / unconstrained only." |
| Captured EP list has **one** EP | Standard single-EP matching (see rules above). |
| Captured EP list has **multiple** EPs | Run the two-pass walk below — strict first (prefer EP-specific matches), permissive second (fall back to neutral entries). |

Note: `ep: null` entries in a file's `ep_compatibility` list (CPU-resident files in a mixed variant) are independent of the NULL *options* path. The former is a per-file declaration in metadata; the latter is a per-call statement from the consumer.

**Worked example (single captured EP)** — session_options had `Append_V2` of a single CUDA device, options captured it; package has three variants (`cpu`, `gpu`, `qnn-npu`) from the schema above:

- Captured list: `[CUDAExecutionProvider]`.
- `cpu`: `decoder` file's `ep_compatibility` lists only CPU (no null entry) → rejected.
- `gpu`: `decoder` file's `ep_compatibility` includes CUDA → matched, score 1. Variant score = 1.0. ✅
- `qnn-npu`: `context` lists only QNN → rejected.
- Winner: `gpu`.

Same schema with a single WebGPU device appended:

- Captured list: `[WebGpuExecutionProvider]`.
- `gpu`: `decoder` file also lists WebGPU → matched, score 1. Variant score = 1.0. ✅
- Others: rejected.
- Winner: `gpu`. Selected EP for the `decoder` file = `WebGpuExecutionProvider`; session/provider options come from the WebGPU entry.

Same schema with a single QNN device appended:

- Captured list: `[QNNExecutionProvider]`.
- `qnn-npu`: `embeddings` null (0), `context` QNN (1), `iterator` QNN (1), `lm_head` null (0). Variant score = 0.5. ✅
- `cpu`, `gpu`: rejected.
- Winner: `qnn-npu`. Selected EP per file: `null` (CPU), QNN, QNN, `null` (CPU). The null-EP files still contribute their own session options to their CPU sessions.

#### Multi-EP Walk (Two Passes)

When the captured EP list has more than one entry — from multiple `Append_V2` calls, from a policy that returned multiple candidates, or a mix — ORT walks the list in two passes and commits to a single EP globally before running the normal per-component matching. The steps:

1. **Input.** An ordered list of candidate EPs. For `Append_V2`-derived lists, the order is insertion order; for policy-derived lists, it is policy-preference order.
2. **Pass 1 — strict walk.** For each EP in order:
   - Attempt to match each component's variants against this EP, but **exclude variants that rely solely on null/neutral entries** (a variant qualifies only if at least one file has a non-null `ep_compatibility` entry for this EP; other files in the variant may still resolve via null entries).
   - Stop at the first EP where **at least one component has a qualifying variant**. That EP is committed globally.
3. **Pass 2 — permissive fallback.** If Pass 1 yielded no match across all candidate EPs, walk the list again with null entries allowed as standalone matches. Stop at the first EP that produces a match.
4. **Commit phase.** Once an EP is chosen, match every component against it using the normal per-variant algorithm (null entries fully allowed). Components with no matching variant are excluded as usual.

**Why two passes?** Consider a captured list `[TRT, CUDA]` and a package with a CUDA-optimized variant *plus* a neutral-only JIT variant. Without the strict pass, the neutral variant would match TRT (score 0), TRT would win, and the consumer would silently run the generic JIT file on TRT instead of the purpose-built CUDA variant. The strict pass prevents null entries from hijacking the top-ranked EP when a real EP-specific variant exists further down the list.

**"At least one component" criterion.** The walk commits to an EP as soon as one component has a qualifying variant. Other components may end up excluded if they have no matching variant for that EP — same partial-match semantics as the single-EP path. The consumer's ordering is authoritative; ORT does not re-score across EPs to maximize component coverage. A consumer who wants "best coverage" semantics can supply a custom policy delegate that factors coverage into its ranking.

**Worked example (captured list from policy)** — policy resolved to `[CUDAExecutionProvider, WebGpuExecutionProvider, CPUExecutionProvider]` over the three-variant schema above:

- Pass 1 CUDA: `gpu` qualifies (real CUDA entry). CUDA wins. Commit phase selects `gpu`. ✅
- Pass 1 WebGpu / CPU: not reached.

Same walk over a package with only a QNN variant and a neutral JIT variant:

- Pass 1 CUDA / WebGpu / CPU: no qualifying strict matches anywhere.
- Pass 2 CUDA: neutral variant's null entry matches → CUDA wins → neutral variant selected.

Same walk over a package with QNN-specific *and* CUDA-specific variants:

- Pass 1 CUDA: CUDA variant qualifies → CUDA wins. The QNN variant is ignored — consistent with the policy's `CUDA > QNN` preference even though both would run.

**Worked example (captured list from explicit appends)** — consumer called `Append_V2` with CUDA, then with WebGpu (in that order), then created options. Captured list: `[CUDA, WebGpu]`:

- The two-pass walk runs identically. CUDA is tried first, WebGpu second. A package that has WebGpu-only variants would lose to any CUDA match; this reflects the consumer's intent (CUDA appended first = CUDA preferred).

#### NULL-Options Path

When `options` is NULL, the algorithm short-circuits before any scoring:

- Each component is resolved by variant count, not by scoring.
- If a component has a single variant, it is selected regardless of the EPs declared in its files. This is the "flat directory" case — the consumer trusts what's on disk and handles EP setup at session creation time (either by passing their own `session_options` or by relying entirely on the variant metadata).
- If any component has more than one variant, `CreateModelPackageContext` fails with an ambiguity error. The consumer must retry with a non-NULL options handle derived from a `session_options` that expresses an EP preference (or accept that it must, by configuring session_options with an explicit EP or policy).

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

### Example 1: Simple Consumer (Single-File Component, Flat-Directory Style)

```c
// NULL options path: no EP preference needed — works when every component
// has exactly one variant (e.g., a flat ORT-GenAI directory repackaged as
// a .ortpackage with a single variant per component).
OrtModelPackageContext* ctx = NULL;
pkg->CreateModelPackageContext(L"my-model.ortpackage",
                               /*options*/ NULL, &ctx);

// Create session — ORT applies file metadata (EP, session_opts, provider_opts)
OrtSession* session = NULL;
api->CreateSession(env, ctx, "decoder", /*file_identifier*/ NULL,
                   /*session_options*/ NULL, &session);
//                          ^ctx     ^component            ^no caller opts

// Use session...
api->ReleaseSession(session);
pkg->ReleaseModelPackageContext(ctx);
```

### Example 2: Multi-Component Package with QNN

```c
// Consumer expresses EP intent through normal session_options surface.
OrtSessionOptions* so = NULL;
api->CreateSessionOptions(&so);
api->SessionOptionsAppendExecutionProvider_V2(so, env, /*ep_devices*/ qnn_npu_devices,
                                              /*num*/ 1, qnn_keys, qnn_vals, qnn_n);

// Capture that intent for the package API.
OrtModelPackageOptions* options = NULL;
pkg->CreateModelPackageOptionsFromSessionOptions(env, so, &options);

OrtModelPackageContext* ctx = NULL;
pkg->CreateModelPackageContext(L"phi-4mm.ortpackage", options, &ctx);

// See what's available for QNN
size_t num_components = 0;
const char* const* names = NULL;
pkg->ModelPackageGetComponentNames(ctx, &num_components, &names);
// names = ["decoder", "vision"] — "embedding" excluded (CPU-only, no QNN variant)

// Query decoder files
size_t num_files = 0;
const char* const* ids = NULL;
pkg->ModelPackageGetFileIdentifiers(ctx, "decoder", &num_files, &ids);
// num_files = 4, ids = ["embeddings", "context", "iterator", "lm_head"]

// Create sessions for each file — ORT uses the captured catalog to register
// the single EP for each file, and applies per-file variant metadata on top.
OrtSession* sessions[4];
for (size_t i = 0; i < num_files; i++) {
    api->CreateSession(env, ctx, "decoder", ids[i], /*session_options*/ NULL,
                       &sessions[i]);
}

// Create vision session
OrtSession* vision_session = NULL;
api->CreateSession(env, ctx, "vision", NULL, NULL, &vision_session);
// If vision has multiple files, NULL file_identifier returns an error — must specify.

// Cleanup
for (size_t i = 0; i < num_files; i++) api->ReleaseSession(sessions[i]);
api->ReleaseSession(vision_session);
pkg->ReleaseModelPackageContext(ctx);
pkg->ReleaseModelPackageOptions(options);
api->ReleaseSessionOptions(so);
```

### Example 3: Advanced Consumer (Custom Session Options)

For consumers like GenAI that need custom EP setup (shared CUDA streams, QNN shared contexts):

```c
// Same capture pattern as Example 2, just to drive variant selection.
OrtSessionOptions* so = NULL;
api->CreateSessionOptions(&so);
api->SessionOptionsAppendExecutionProvider_V2(so, env, qnn_devices, 1,
                                              NULL, NULL, 0);

OrtModelPackageOptions* options = NULL;
pkg->CreateModelPackageOptionsFromSessionOptions(env, so, &options);

OrtModelPackageContext* ctx = NULL;
pkg->CreateModelPackageContext(L"phi-4mm.ortpackage", options, &ctx);

// Read consumer metadata for GenAI-specific config
const char* overlay_json = NULL;
pkg->ModelPackageGetConsumerMetadata(ctx, "decoder", &overlay_json);
// GenAI parses overlay for pipeline config, model_type, etc.

// Query files
size_t num_files = 0;
const char* const* ids = NULL;
pkg->ModelPackageGetFileIdentifiers(ctx, "decoder", &num_files, &ids);

for (size_t i = 0; i < num_files; i++) {
    // Read per-file EP to decide how to configure
    const char* ep = NULL;
    pkg->ModelPackageGetFileEp(ctx, "decoder", ids[i], &ep);

    // Build custom session options with GenAI's EP-specific setup
    OrtSessionOptions* caller_opts = NULL;
    api->CreateSessionOptions(&caller_opts);
    // ... GenAI applies shared CUDA stream, QNN shared EP context, etc. ...
    // ... GenAI appends EP with custom provider options ...

    OrtSession* session = NULL;
    api->CreateSession(env, ctx, "decoder", ids[i], caller_opts, &session);
    // Caller-provided session_options: ORT does NOT replay captured catalog
    // or apply variant metadata — caller is fully in charge.

    api->ReleaseSessionOptions(caller_opts);
    // Store session...
}

pkg->ReleaseModelPackageContext(ctx);
pkg->ReleaseModelPackageOptions(options);
api->ReleaseSessionOptions(so);
```

### Example 4: Querying Selected Variants

The query APIs surface the *selected* variant per component — i.e., what the consumer will actually run for the captured EP selection (or the single variant, under NULL options). Full package inspection (all components × all variants × all files, ignoring selection) is not supported by these APIs; tooling that needs that should parse `manifest.json` and per-component `metadata.json` directly, or wait for the dedicated inspection entry point tracked in [Open Question #9](#9-package-inspection-mode-deferred).

```c
// Inspection doesn't need an EP preference — NULL options is fine as long as
// every component has a single variant. For multi-variant packages, the consumer
// must supply a non-NULL options handle; the query APIs surface only the
// *selected* variant per component.
OrtModelPackageContext* ctx = NULL;
pkg->CreateModelPackageContext(L"phi-4mm.ortpackage",
                               /*options*/ NULL, &ctx);

size_t num_components = 0;
const char* const* names = NULL;
pkg->ModelPackageGetComponentNames(ctx, &num_components, &names);

for (size_t c = 0; c < num_components; c++) {
    const char* variant = NULL;
    pkg->ModelPackageGetSelectedVariantName(ctx, names[c], &variant);

    size_t num_files = 0;
    pkg->ModelPackageGetFileCount(ctx, names[c], &num_files);

    printf("Component: %s (variant: %s, %zu files)\n", names[c], variant, num_files);

    const char* const* ids = NULL;
    size_t num_ids = 0;
    pkg->ModelPackageGetFileIdentifiers(ctx, names[c], &num_ids, &ids);

    for (size_t f = 0; f < num_files; f++) {
        const char* ep = NULL;
        pkg->ModelPackageGetFileEp(ctx, names[c], ids[f], &ep);
        printf("  File: %s (EP: %s)\n", ids[f], ep ? ep : "CPU");
    }
}

pkg->ReleaseModelPackageContext(ctx);
```

---

## Key Design Decisions

### EP Selection Is Captured from `OrtSessionOptions`

The consumer expresses EP intent through the mechanisms ORT already has — `SessionOptionsAppendExecutionProvider_V2` for explicit EPs, `SessionOptionsSetEpSelectionPolicy`/`SessionOptionsSetEpSelectionPolicyDelegate` for auto selection. `CreateModelPackageOptionsFromSessionOptions` takes a snapshot of that intent (resolving the policy against env's `OrtEpDevice`s if needed) onto an `OrtModelPackageOptions` handle, which then drives variant selection and is reused as a catalog at session-creation time.

Three wins from this shape:

- **No new EP-selection DSL.** Consumers don't learn a bespoke selection struct; the surface they already use for session creation is the same one they use here. `OrtCompileApi` follows the same pattern, so there is precedent in ORT.
- **Single source of truth.** The captured `OrtEpDevice` list is what both variant matching and session creation consult. They cannot drift because they look at the same bytes.
- **Natural generalization to multiple EPs.** Consumers may call `Append_V2` more than once, or use a policy that returns multiple devices. The captured list holds the ordered set; the two-pass walk (see the Algorithm) handles it without any extra orchestration surface.

### NULL Options Means "Single-Variant Package"

When the consumer passes NULL for `options` on `CreateModelPackageContext`, the context resolves variants purely by count: a single variant per component is selected as-is, and any multi-variant component causes an ambiguity error. This preserves the flat-directory migration ergonomic where the consumer handles EP setup elsewhere (e.g., a legacy ORT-GenAI directory) and doesn't need to convey EP intent through the package API at all.

NULL options is a *package-level* precondition: if even one component is multi-variant, the consumer must provide a non-NULL options handle (from a session_options that carries their EP preference).

Sessions created from a NULL-options context fall back to the variant metadata for EP and per-file options, which is always sufficient because each file's `ep_compatibility` carries what ORT needs. The consumer may alternatively pass their own `session_options` at session creation to take full control. See the precedence table in *Session Creation*.

### Multi-EP Matching Commits One EP Globally

When the captured EP list has more than one entry — whether from multiple `Append_V2` calls or a policy that returned several `OrtEpDevice`s — variant matching walks the list in two passes (strict first, permissive fallback) and commits to a single EP globally before per-component matching runs. Three choices deserve calling out:

- **EP selection is a list, not an unordered set.** Insertion order (for explicit appends) or policy-preference order is authoritative. The walk tries EPs in that order and stops at the first match, so the consumer's ranking decides ties.
- **The two-pass strict/permissive walk.** A single-pass walk with null entries always allowed would let a generic JIT variant hijack the top-ranked EP (e.g., a captured list `[TRT, CUDA]` with only a neutral variant present would match TRT via the null entry, ignoring a real CUDA variant that might exist further down). The strict pass prevents this: neutral variants only win when nothing EP-specific matches anywhere in the list.
- **No cross-EP coordination.** Once an EP is committed, ORT does *not* revisit the decision to maximize component coverage. If component A matches CUDA but component B doesn't, the consumer gets A and loses B — same partial-match semantics as the single-EP path. This keeps the algorithm predictable and sidesteps hard ranking questions ("is three components on QNN better than two on CUDA?"). Consumers who want coverage-aware behavior can supply a custom `EpSelectionDelegate` that factors coverage into its policy output.

### Captured Options Are a Catalog at Session Creation

The captured `OrtEpDevice` list is not replayed wholesale at session creation. Blindly registering every candidate on every session would be wrong: for an `ep: null` file that should run on default CPU, registering CUDA or TRT would force it onto an accelerator. Instead, the captured list acts as a **catalog** — ORT looks up the specific `OrtEpDevice` corresponding to this file's selected EP and registers only that one. Non-EP session-level settings from the captured options (threading, optimization level, log severity, custom ops, config entries) *are* replayed on every session in the package, since those are consumer-wide configuration rather than EP-specific.

This ensures the EP chosen for a file at variant-selection time is the same EP used to run that file at session-creation time — no drift — while still honoring the consumer's session-level configuration.

### Session Creation Stays with `OrtSession`

The model package is a file format. Session creation is an ORT runtime operation. Keeping them separate means the package API stays small and focused on discovery and metadata, while session creation stays with the existing `OrtSession` surface that consumers already understand. The package-aware form is just an overload of `CreateSession` that takes a package context plus a component/file identifier instead of a file path — no new "session from package" API is introduced.

This avoids creating a parallel surface for session management inside the package API. If the package itself created sessions, there would be two places where sessions get created and lifecycle-managed, which adds confusion for no real benefit — the caller already has an `OrtEnv` and `OrtSessionOptions` at session-creation time.

### ORT Applies Metadata Options (Not the Consumer)

The producer of the model knows what session and provider options the model needs to run correctly. Those options belong with the model, in its variant metadata. When the consumer asks ORT to load a file, ORT reads the metadata and applies the options.

Having the consumer read options from metadata and forward them to ORT during session creation adds a pass-through layer with no value — the consumer isn't making any decisions about those options. Keeping the application in ORT means the consumer code is simpler and the options stay close to the model they apply to.

Only options that are *required* to run the component belong in variant metadata. Anything overridable or policy-like (thread tuning, profiling, debug flags) belongs in the consumer's own config.

### Session Options Precedence

Four cases, no merging at session-options level:

| Package context `options` | `session_options` at `CreateSession` | Behavior |
|---|---|---|
| NULL | NULL | ORT builds the session entirely from the variant's file metadata — EP + session_options + provider_options come from the selected `ep_compatibility` entry. The "trust what's on disk" path. |
| NULL | Provided | ORT uses the caller's `session_options` as-is. Metadata options are NOT applied. The caller takes full responsibility for EP setup and session configuration. |
| Non-NULL | NULL | ORT replays the non-EP settings from the captured `OrtSessionOptions` (threading, optimization level, custom ops, config entries), registers the single captured `OrtEpDevice` matching this file's chosen EP (catalog lookup), and then applies variant metadata (provider options and additional session settings) on top. |
| Non-NULL | Provided | ORT uses the caller's `session_options` as-is. Neither the captured catalog nor variant metadata is applied — the advanced path for callers who need shared CUDA streams, shared QNN EP contexts, custom allocators, etc. |

All-or-nothing at session-options level avoids complex precedence rules and subtle bugs from conflicting options. The two "Provided" rows behave identically (caller is authoritative); the two "NULL" rows differ only in where the EP comes from (captured catalog vs. file metadata).

### Two Parameters for Component + File (Not a Dotted String)

The `CreateSession` overload takes `component_name` and `file_identifier` as separate parameters rather than a single `"component.file"` string. Dotted strings have escaping pitfalls (what if an identifier contains a dot?), weaker validation, and awkward language bindings. Two parameters also match how the rest of the query APIs already address files.

### `OrtEnv` Only Needed at Options Creation

`CreateModelPackageOptionsFromSessionOptions` needs `env` to resolve a policy against the registered `OrtEpDevice`s (when the session_options uses a policy rather than explicit `Append_V2`). Once options is created, the captured `OrtEpDevice` list is all that variant matching needs — `CreateModelPackageContext` is pure parsing + matching and takes no env. At session-creation time, env is provided again on `CreateSession`, and the `OrtEpDevice` identity preserved by the context's catalog lookup ensures match-time EP == session-time EP. This keeps each entry point focused on the state it actually needs.

### Partial Match Is Not an Error

If a package contains components that don't have a matching variant for the captured EP selection, those components are silently excluded from the context. The consumer sees only what's usable. This matches the mental model of the API: "tell me what's available for my EP."

Components that *are* available can still be loaded. If a consumer expects a specific component and it's missing, that shows up at session creation time with a clear error.

One exception: when the consumer passes **NULL options** and any component has more than one variant, context creation fails. That's ambiguity, not absence — see *NULL Options Means "Single-Variant Package"* above.

### Per-File EP Compatibility List

A file declares its compatible EPs as a list (`ep_compatibility`), always — even when the file only supports one EP, the list has a single entry. Each entry carries its own `ep` name (or `null` for CPU/neutral), `compatibility_info`, `session_options`, and `provider_options`. The variant selector walks the captured EP list against each file's `ep_compatibility` and picks the best-scoring variant where every file either matches or has a null fallback.

Keeping a uniform list shape — rather than a flat "single-EP" shorthand — means consumers, tooling, and the selection algorithm always see one schema. The cost is one extra level of nesting for single-EP files, which is negligible given metadata.json is typically generated by tools rather than hand-written.

Storing per-EP options alongside each compatible EP keeps the options close to the execution target they apply to — when the selector picks CUDA for a file, it uses the CUDA-specific options attached to that entry. The null entry lets neutral/CPU files carry their own options too.

### `ep: null` for Neutral/CPU Files

An `ep_compatibility` entry with `ep: null` runs on CPU and matches any captured EP as a neutral fallback. This serves two use cases:

1. **Base models meant for JIT compilation** — the package ships a single generic ONNX file that any EP can consume at runtime via JIT. The file has a single null entry, so it matches any captured EP and scores neutrally.
2. **Mixed-EP variants** — a QNN decoder pipeline where embeddings and `lm_head` run on CPU while the rest run on QNN. The CPU-resident files have a null entry; they ride along with the variant without affecting its match decision against non-CPU EPs.

Null entries can still carry session options (and those are applied by ORT when the file is loaded on CPU), so a CPU file with specific threading or optimization settings can have them declared alongside any EP-specific entries.

Scoring null entries as `0` (neutral) means a purely EP-matched variant will outscore a mixed variant when both are viable, which is the right preference: if the producer ships a full-CUDA variant alongside a CUDA-decoder-with-CPU-embeddings variant, the full-CUDA one wins.

This subsumes the old `force_cpu` flag — a null entry expresses the same intent without an extra field, and it naturally allows session options on neutral files.

### Variant Score Is the File-Score Average

A multi-file variant is either fully compatible with the candidate EP (every file has a matching EP entry or a null entry) or rejected outright. When compatible, the variant's score is the mean of its file scores. Averaging gives a simple, monotonic score that:

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
- [ ] Public opaque handles `OrtModelPackageOptions` and `OrtModelPackageContext`.
- [ ] Error codes and status messages for malformed packages and for NULL-options ambiguity.
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
- [ ] NULL-options path (single-variant-per-component requirement; ambiguity error otherwise).
- [ ] Multi-EP walk (two passes: strict first, permissive fallback; commit one EP globally). Handles both `Append_V2`-derived and policy-derived lists uniformly.
- [ ] Deterministic tie-break (insertion order).
- [ ] Pure function over WS0 structs plus an ordered `OrtEpDevice` list — no I/O, no ORT runtime deps. EP resolution (policy → ordered list) happens in WS-C before calling WS-B.
- [ ] Unit tests: single-EP variants, multi-EP files, mixed neutral+EP variants, all-rejected case, NULL-options, multi-EP two-pass (including strict-pass anti-hijack case), ties, device-type filter.

**WS-C. Public C API — options, context & queries** *(Dev 3)*
- [ ] `OrtModelPackageApi` sub-API obtained via `OrtApi::GetModelPackageApi` (mirrors `OrtCompileApi`).
- [ ] `CreateModelPackageOptionsFromSessionOptions` — deep-copy the non-EP session settings from `OrtSessionOptions` onto the options handle; capture the `OrtEpDevice` list (from `Append_V2` appends, or by resolving the configured policy/delegate against the registered `OrtEpDevice`s on `env`).
- [ ] `ReleaseModelPackageOptions`.
- [ ] `CreateModelPackageContext` (calls WS-A loader + WS-B selector; returns opaque handle). `options` is optional (NULL = NULL-options path). Does NOT take `OrtEnv` — env is only needed at options creation and at session creation.
- [ ] `ReleaseModelPackageContext`.
- [ ] Context must deep-copy whatever it needs from `options` so the caller can release `options` immediately after the call.
- [ ] `GetComponentNames`, `GetSelectedVariantName`.
- [ ] `GetFileCount`, `GetFileIdentifiers`, `GetFilePath`.
- [ ] `GetFileEp`, `GetFileDeviceType`.
- [ ] `GetFileSessionOptions`, `GetFileProviderOptions`.
- [ ] `GetConsumerMetadata`.
- [ ] Memory ownership, release semantics, thread-safety notes documented in the headers.
- [ ] C API smoke tests (single-EP appended, multi-EP appended, policy-resolved, NULL-options paths).

**WS-D. Session creation integration** *(Dev 4)*
- [ ] `CreateSession` overload taking `OrtModelPackageContext*`, `component_name`, `file_identifier`, optional `OrtSessionOptions*`.
- [ ] Session options precedence per the [Session Options Precedence](#session-options-precedence) table — four cases (NULL/non-NULL context options × NULL/provided caller session_options).
- [ ] Non-EP settings replay: when the context has captured options and caller passed NULL session_options, replay the non-EP settings (threading, optimization level, custom ops, config entries) on the built session_options.
- [ ] Catalog lookup: register the single captured `OrtEpDevice` whose EP name matches the file's selected EP. If the file's selected EP is `null` (CPU), register nothing (default CPU).
- [ ] ORT-side applier for per-file `session_options` — generic passthrough via `AddConfigEntry`.
- [ ] Known-setter dispatch for session-option keys that require dedicated setters (e.g., `intra_op_num_threads`, `graph_optimization_level`) — fixed mapping from key name → setter. See [Appendix B](#appendix-b-session-options-reference) and [Open Question #3](#3-known-session-options-setters).
- [ ] ORT-side applier for per-file `provider_options` — wire the provider options into the registered EP.
- [ ] Integration tests that go from a sample package on disk to a runnable session (all four precedence cases).
- [ ] This workstream depends on A/B/C landing; can develop against stubs in parallel.

#### Post-integration (serial, once A–D are merged)

**WS-E. Language bindings**
- [ ] C++ wrappers (`Ort::ModelPackageOptions`, `Ort::ModelPackageContext`, `Ort::Session` overload).
- [ ] Python bindings (`onnxruntime.ModelPackageContext`, `InferenceSession` from context).
- [ ] Binding-level tests (C++ and Python).

**WS-F. Sample packages & end-to-end tests**
- [ ] Fixture package: single-file CPU.
- [ ] Fixture package: multi-EP single-file (CUDA + WebGPU).
- [ ] Fixture package: multi-file QNN pipeline with CPU-resident files.
- [ ] Fixture package: neutral-only JIT variant (exercises policy Pass 2 fallback).
- [ ] Fixture package: no matching variant (negative case).
- [ ] E2E tests driving the public API across all fixtures, covering single-EP appended, multi-EP appended, policy-resolved, and NULL-options paths.
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
- [ ] For package path: build `OrtSessionOptions` from GenAI's existing EP config, then call `CreateModelPackageOptionsFromSessionOptions` to capture it.
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

The current design captures one `OrtModelPackageOptions` handle (derived from a single `OrtSessionOptions`) for the entire context. What if a consumer needs decoder on QNN but embedding on CPU?

Options:
- Create two contexts, one per EP.
- Extend the API to accept multiple options or per-component overrides.

Most real deployments use one EP for the whole pipeline, so this is an edge case. We can start with a single captured options handle and extend later if needed.

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

### 7. `GetSelectedVariantPath` with NULL Options and EP-Declaring Variants

When the context was created with NULL options and the (single) selected variant's files declare specific EPs, does `GetSelectedVariantPath` still return the variant path? Proposed: yes — trust what's on disk, consistent with the "flat directory" semantics. Consumer who asked for NULL-options opted into this.

### 8. Metadata-Aware Policy Delegates (Deferred)

A custom `EpSelectionDelegate` that factors in package-level signals (e.g., "which EP yields the best component coverage for this specific package?") would let advanced consumers override the default "first EP with at least one match wins" behavior. Deferred — straightforward to layer on once the single-EP-commit default is validated.

### 9. Package Inspection Mode (Deferred)

The v1 query APIs are *selection-scoped*: they surface the single selected variant per component and filter out components that have no match for the captured EP selection. This is the consumer shape (run a package) but not the tooling shape (validate a package, diff two packages, list every variant a package ships, drive a UI that lets the user pick a variant manually).

True inspection would need a second surface:

- A construction entry point that skips variant selection entirely and never errors on multi-variant components — e.g., `CreateModelPackageInspectionContext(package_path, &ctx)`, or a flag on `CreateModelPackageContext`.
- Traversal APIs that enumerate *all* components, *all* variants per component, and *all* files per variant — independent of any EP filtering.
- Getters keyed by the full `(component, variant, file)` tuple rather than `(component, file)` against the selected variant.

Out of scope for v1 because (a) the consumer-run-package flow is the primary driver, (b) the data is all in the file formats (`manifest.json` + per-component `metadata.json`), so tooling can parse directly in the short term, and (c) adding the traversal surface prematurely risks coupling it to selection-context internals that are still evolving. Worth reserving naming so `CreateModelPackageContext` and related APIs stay future-compatible with an inspection entry point layered alongside.

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
