# Model Package API — Proposal

## Table of Contents

- [Problem Statement](#problem-statement)
- [Design Principles](#design-principles)
- [Proposed API](#proposed-api)
  - [1. Package Context (Parse, Traverse, Mutate)](#1-package-context-parse-traverse-mutate)
  - [2. Package Options (EP Capture)](#2-package-options-ep-capture)
  - [3. Component Selection](#3-component-selection)
  - [4. Component Instance Queries](#4-component-instance-queries)
  - [5. Session Creation](#5-session-creation)
  - [API Summary](#api-summary)
- [Package Layout](#package-layout)
- [Metadata Schemas](#metadata-schemas)
  - [Top-Level `manifest.json`](#top-level-manifestjson)
  - [Component `metadata.json`](#component-metadatajson)
  - [Variant `variant.json`](#variant-variantjson)
- [Variant Selection Algorithm](#variant-selection-algorithm)
- [Usage Examples](#usage-examples)
  - [Example 1: Single-File Component](#example-1-single-file-component)
  - [Example 2: Multi-File Component (Pipeline Consumer)](#example-2-multi-file-component-pipeline-consumer)
  - [Example 3: Inspecting a Package](#example-3-inspecting-a-package)
  - [Example 4: Mutating a Package](#example-4-mutating-a-package)
- [Key Design Decisions](#key-design-decisions)
  - [Opaque Handles for ABI Stability](#opaque-handles-for-abi-stability)
  - [Component Metadata Is Selection-Only](#component-metadata-is-selection-only)
  - [Per-Variant `variant.json` Owns Runtime Detail](#per-variant-variantjson-owns-runtime-detail)
  - [EP Compatibility Is Per-Variant, Not Per-File](#ep-compatibility-is-per-variant-not-per-file)
  - [Compatibility String Is Owned by the EP](#compatibility-string-is-owned-by-the-ep)
  - [No File-Set Consistency Across Variants](#no-file-set-consistency-across-variants)
  - [Index-Based Per-File Access (Default 0)](#index-based-per-file-access-default-0)
  - [Single-File-Only `CreateSession`](#single-file-only-createsession)
  - [Shared Weights as a Name-To-Path Mapping](#shared-weights-as-a-name-to-path-mapping)
  - [`consumer_metadata` Is a Single Opaque Blob](#consumer_metadata-is-a-single-opaque-blob)
  - [Manifest Carries Merge Provenance](#manifest-carries-merge-provenance)
- [Implementation Plan](#implementation-plan)
  - [Phase 1: Core API and Read Path](#phase-1-core-api-and-read-path)
  - [Phase 2: Mutation and Authoring](#phase-2-mutation-and-authoring)
  - [Phase 3: Compile and Merge/Unmerge](#phase-3-compile-and-mergeunmerge)
- [Open Questions](#open-questions)
  - [1. EP-Side Preference ABI](#1-ep-side-preference-abi)
  - [2. JIT Compilation Granularity](#2-jit-compilation-granularity)
  - [3. Merge / Unmerge Semantics](#3-merge--unmerge-semantics)
  - [4. Manifest Schema Versioning](#4-manifest-schema-versioning)
  - [5. Cross-Component Consistency](#5-cross-component-consistency)
  - [6. Authoring Tools](#6-authoring-tools)
  - [7. Shared-Weight Verification on Add](#7-shared-weight-verification-on-add)
- [Appendix A: ORT-GenAI Integration](#appendix-a-ort-genai-integration)
  - [Where GenAI Assets Live in the Package](#where-genai-assets-live-in-the-package)
  - [`genai_config.json` in the Package World](#genai_configjson-in-the-package-world)
  - [Per-Variant Overlays (RFC 7386 JSON Merge Patch)](#per-variant-overlays-rfc-7386-json-merge-patch)
  - [EP Defaulting in `og.Model` and `og.Config`](#ep-defaulting-in-ogmodel-and-ogconfig)
  - [Resolution Flow at `Model::Create`](#resolution-flow-at-modelcreate)
  - [What Overlays Should and Should Not Contain](#what-overlays-should-and-should-not-contain)
  - [Single-File Components (CPU, CUDA, WebGPU, Vitis, OpenVINO)](#single-file-components-cpu-cuda-webgpu-vitis-openvino)
  - [Multi-File Components (QNN-Style Pipelines)](#multi-file-components-qnn-style-pipelines)
  - [Per-File CPU Override](#per-file-cpu-override)
  - [Backward Compatibility (Flat-Directory Models)](#backward-compatibility-flat-directory-models)
- [Appendix B: Session Options Reference](#appendix-b-session-options-reference)
  - [Known Session Options (Dispatched via `AddSessionOption`)](#known-session-options-dispatched-via-addsessionoption)
  - [Value Encoding Rules](#value-encoding-rules)
  - [Excluded Setters](#excluded-setters)

---

## Problem Statement

A model that ships in production is rarely a single ONNX file. The same model exists as multiple variants targeting different execution providers (CPU, CUDA, WebGPU, QNN, OpenVINO, …), each variant may consist of multiple ONNX files (split decoders, embedding stages, language-model heads), and consumers (frameworks like ORT-GenAI, applications like Foundry Local) need a single distributable that captures all of this without forcing every consumer to re-implement variant discovery, EP-aware selection, and per-file session setup.

The model package format addresses this by laying out variants on disk under a known structure with a small amount of metadata. The API around it has to:

1. **Open** a package directory and let callers explore its structure.
2. **Mutate** a package — add or remove components and variants — so authoring tools and consumers can compose packages.
3. **Select** a component variant given the consumer's intended EP, returning enough information for any consumer to load it (single ORT session, multi-session pipeline, or anything else).
4. **Open a session** when the variant is a single ONNX file, as a convenience over the lower-level path.
5. Stay **stable across format evolution** so changes to package internals (new fields, new directory shapes) do not break callers.

This proposal defines that API.

---

## Design Principles

1. **Folder is the source of truth.** A model package is a directory. Every API takes a folder path; no archive parsing, no implicit extraction. The format is explorable with normal filesystem tools.
2. **Opaque handles, narrow accessors.** All public types are forward-declared structs accessed through API functions. Internal addressing (file identifiers, sub-component grouping, on-disk layout details) is *not* part of the ABI. This keeps callers stable across format changes.
3. **Generic mechanism, not consumer semantics.** The package API surfaces ORT-relevant primitives — file paths, session options, provider options, external-data resolution — and an opaque `consumer_metadata` blob that ORT does not interpret. Anything that requires consumer-specific knowledge (pipeline shape, file roles, overlay merge rules) is the consumer's responsibility.
4. **Component metadata is for selection only.** A component's top-level `metadata.json` declares variants and their EP compatibility — nothing else. Anything that varies per-file or per-variant beyond selection lives inside the variant's directory. This makes adding or removing a variant a one-entry edit plus a directory move.
5. **EP plugins own EP-specific knowledge.** ORT's selection logic is EP-agnostic: it filters variants by EP-name match. Choosing among compatible variants based on hardware-specific compatibility hints is delegated to the EP itself via a dedicated ABI.
6. **Single-file is the simple path; multi-file is the consumer's path.** ORT's `CreateSession` over a component instance handles single-file variants directly. Multi-file variants are the consumer's responsibility — but the consumer uses the same generic accessors ORT uses internally, so there is no duplicated parsing.
7. **Designed for evolution.** The format must support shared weights, JIT compilation, package merging across producers, and variants that can be downloaded standalone. Where the implementation is deferred, the schema and API leave room.

---

## Proposed API

The API is layered into five concerns: opening a package, capturing EP intent, selecting a component, querying the selected component, and creating a session.

### 1. Package Context (Parse, Traverse, Mutate)

```c
OrtStatus* CreateModelPackageContext(const ORTCHAR_T* path,
                                     OrtModelPackageContext** out);

void ReleaseModelPackageContext(OrtModelPackageContext* ctx);
```

The context is an opaque handle backed by a parsed view of the package directory. Construction validates the directory structure, parses the top-level manifest (if present), and indexes all components and their variants.

**Traversal queries** (read-only, no selection):

```c
OrtStatus* ModelPackage_GetComponentCount(const OrtModelPackageContext* ctx,
                                          size_t* out_count);
OrtStatus* ModelPackage_GetComponentNames(const OrtModelPackageContext* ctx,
                                          const char* const** out_names,
                                          size_t* out_count);

OrtStatus* ModelPackage_GetVariantCount(const OrtModelPackageContext* ctx,
                                        const char* component_name,
                                        size_t* out_count);
OrtStatus* ModelPackage_GetVariantNames(const OrtModelPackageContext* ctx,
                                        const char* component_name,
                                        const char* const** out_variant_names,
                                        size_t* out_count);

// Per-variant EP compatibility — needed before SelectComponent so callers
// can build an OrtSessionOptions / OrtModelPackageOptions with the right EP.
OrtStatus* ModelPackage_GetVariantEpCompatibilityCount(
    const OrtModelPackageContext* ctx,
    const char* component_name,
    const char* variant_name,
    size_t* out_count);

// Returns (ep, device, compatibility_string) for a single ep_compatibility entry.
// Any out-pointer may be NULL if the caller doesn't need that field.
// When the JSON omits a field, the returned pointer is NULL.
OrtStatus* ModelPackage_GetVariantEpCompatibility(
    const OrtModelPackageContext* ctx,
    const char* component_name,
    const char* variant_name,
    size_t ep_idx,
    const char** out_ep,
    const char** out_device,
    const char** out_compatibility_string);
```

Pre-selection traversal exposes exactly the contents of each component's `metadata.json` — variant names and their EP compatibility. Anything beyond that (file count, per-file paths and options, consumer_metadata) requires a `SelectComponent` call and lives on the resulting `OrtModelPackageComponentContext` (see [Component Instance Queries](#4-component-instance-queries)).

**Mutation operations** (authoring path):

```c
OrtStatus* ModelPackageAddComponent(OrtModelPackageContext* ctx,
                                    const char* component_name,
                                    const ORTCHAR_T* source_dir);
OrtStatus* ModelPackageRemoveComponent(OrtModelPackageContext* ctx,
                                       const char* component_name);

OrtStatus* ModelPackageAddVariant(OrtModelPackageContext* ctx,
                                  const char* component_name,
                                  const ORTCHAR_T* source_dir);
OrtStatus* ModelPackageRemoveVariant(OrtModelPackageContext* ctx,
                                     const char* component_name,
                                     const char* variant_name);

OrtStatus* ModelPackageCommit(OrtModelPackageContext* ctx);
```

Mutations are staged on the context and written to disk by `Commit`. The exact transactional guarantees are an implementation detail; the public contract is that `Commit` is the only call that touches the filesystem in a write capacity.

**Add semantics: source mirrors a component directory.** Both `AddComponent` and `AddVariant` take a `source_dir` shaped like a component on disk:

```
<source_dir>/
├── metadata.json                  # selection metadata
├── shared_weights/                # optional; only blobs the included variants reference
│   └── <checksum>/<blob>
└── <variant_name>/                # one or more variant subdirs
    ├── variant.json
    └── *.onnx
```

`AddComponent` accepts any number of variants in the source. `AddVariant` requires *exactly one* variant subdir and a `metadata.json` declaring exactly one variant entry whose key matches that subdir name. (Bulk add is the `AddComponent` operation; if you want to add several variants at once to an existing component, drop the existing component and add a fresh one — or call `AddVariant` per variant.)

ORT then merges:

1. **Validate.** `metadata.json` is well-formed; the variant subdir(s) match the declared variant entries; every checksum referenced by the variants' `variant.json` `shared_files` resolves under `<source>/shared_weights/`.
2. **Merge shared weights.** For each `<source>/shared_weights/<checksum>/`:
    - If the destination component already has the same checksum directory, skip — the checksum identifies content, so collision is identity. (ORT optionally verifies by re-hashing on copy as a producer-error safeguard; trusted authoring tools may opt out.)
    - Otherwise copy the checksum directory verbatim, blob filename and all.
3. **Merge variant files.** Copy each variant subdir into the destination component. Reject if a destination variant of that name already exists. (Replacement is not provided in the v1 surface; remove + add gives the same effect with explicit intent.)
4. **Merge metadata.** Add the variant entries from source `metadata.json` into the destination component's `metadata.json`.

For `AddComponent`, if no component of `component_name` exists in the package yet, the source is materialized as a new component directory in one step; the merge logic above is the same.

**Remove semantics: shared-weight GC at commit.** `RemoveVariant` removes the variant directory and its entry from `metadata.json`. It does *not* immediately delete blobs from `<component>/shared_weights/` — `Commit` does that:

- After all staged mutations are applied, `Commit` walks every remaining variant in the package, collects the union of checksums referenced by each variant's `variant.json` `shared_files`, and deletes any `<component>/shared_weights/<checksum>/` directory whose checksum is not in that set.
- Deferring GC to commit avoids redundant delete-then-re-copy when a sequence like `RemoveVariant("foo") + AddVariant("foo'")` shares blobs.
- `RemoveComponent` deletes the entire component directory, including its `shared_weights/`. There is no cross-component sharing, so no cross-component GC is needed.

This contract gives producers a clean self-describing source format ("variant package") that the same tooling can produce, consume, split, and merge — at the format level, the operations are uniform.

**Compile** (deferred — see [Open Questions](#2-jit-compilation-granularity)) and **merge / unmerge** (deferred — see [Open Questions](#3-merge--unmerge-semantics)) sit on the context as additional methods. Their signatures will be filled in once their granularity is decided; the manifest is designed today to record the provenance information they will need.

### 2. Package Options (EP Capture)

EP intent is captured from a session-options template. The same shape used to build a normal `OrtSession` is the shape used to declare "this is the EP I want a variant for."

```c
OrtStatus* CreateModelPackageOptionsFromSessionOptions(
    OrtEnv* env,
    const OrtSessionOptions* template,
    OrtModelPackageOptions** out);

void ReleaseModelPackageOptions(OrtModelPackageOptions* options);
```

Internally this snapshots the captured EP list (an ordered sequence of `(ep_name, OrtEpDevice*)` pairs) from the template's `Append_V2` calls or its policy-resolved selection. The handle is consumed by component selection and can be reused across components or discarded after use.

The handle is required for `SelectComponent`. Inspection — listing components, listing variants, reading per-variant EP compatibility — does not need EP intent and uses the [traversal accessors](#1-package-context-parse-traverse-mutate) directly on the context.

### 3. Component Selection

```c
OrtStatus* SelectComponent(OrtModelPackageContext* ctx,
                           const char* component_name,
                           const OrtModelPackageOptions* options,  // required, non-NULL
                           OrtModelPackageComponentContext** out);

ORT_CLASS_RELEASE(ModelPackageComponentContext);
```

Selection runs the [variant selection algorithm](#variant-selection-algorithm) and returns an opaque `OrtModelPackageComponentContext*` (`cix`) representing the chosen variant. The instance is independent of the context's lifetime — release it when the consumer is done with the variant.

`options` is required and per-call (not stored on the context). A consumer can select different components with different captured EP intents from the same context.

### 4. Component Instance Queries

The instance exposes a deliberately small accessor surface. Anything beyond what's listed here is read directly from the variant directory by the consumer — `variant.json` is part of this proposal's public schema (see [Variant `variant.json`](#variant-variantjson)).

```c
OrtStatus* ModelPackageComponent_GetSelectedVariantFolderPath(
    const OrtModelPackageComponentContext* ctx,
    const ORTCHAR_T** folder_path);

OrtStatus* ModelPackageComponent_GetSelectedVariantFileCount(
    const OrtModelPackageComponentContext* ctx,
    size_t* num_files);

OrtStatus* ModelPackageComponent_GetSelectedVariantFilePath(
    const OrtModelPackageComponentContext* ctx,
    size_t file_idx,
    const ORTCHAR_T** out_path);

OrtStatus* ModelPackageComponent_GetSelectedVariantFileSessionOptions(
    const OrtModelPackageComponentContext* ctx,
    size_t file_idx,
    const char* const** option_keys,
    const char* const** option_values,
    size_t* num_entries);

OrtStatus* ModelPackageComponent_GetSelectedVariantFileProviderOptions(
    const OrtModelPackageComponentContext* ctx,
    size_t file_idx,
    const char* const** option_keys,
    const char* const** option_values,
    size_t* num_entries);

OrtStatus* ModelPackageComponent_GetSelectedVariantConsumerMetadata(
    const OrtModelPackageComponentContext* ctx,
    const char** out_json_str);

// TODO: Not yet implemented. Resolve a shared-weight checksum (as it appears in variant.json's
// `shared_files`) to its absolute on-disk path. Encapsulates the <component>/shared_weights/<checksum>/<blob>
// layout — the blob filename is producer-chosen, so consumers should not construct the path
// themselves. Each checksum directory is expected to contain exactly one blob file.
// OrtStatus* ModelPackageComponent_GetSharedWeightPath(
//     const OrtModelPackageComponentContext* ctx,
//     const char* checksum,
//     const ORTCHAR_T** out_path);
```

Notes:

- `GetSelectedVariantFolderPath` is the escape hatch. Multi-file consumers walk this directory and parse `variant.json` for filenames, per-file options, and `shared_files` mappings.
- `GetSelectedVariantFileCount` lets consumers dispatch single-file vs. multi-file paths without speculatively calling `CreateSession` and handling its rejection.
- `GetSelectedVariantFilePath` returns the absolute path to the ONNX file at `file_idx`. Avoids the need to join folder path + filename from `variant.json`.
- `GetSelectedVariantFileSessionOptions` returns the per-file session options as flat key/value arrays. Returns NULL/empty when none are specified. Memory is owned by the context.
- `GetSelectedVariantFileProviderOptions` returns the per-file provider options as flat key/value arrays. Same ownership semantics as session options.
- `GetSelectedVariantConsumerMetadata` returns the variant's `consumer_metadata` blob serialized as a JSON string. ORT does not interpret it. Variants without consumer metadata return an empty string.

What's *not* exposed (intentionally):

- The selected variant name, the matched EP, and the matched compatibility string. Selection state is internal. The variant name is derivable as the basename of `GetSelectedVariantFolderPath` for the rare consumer that wants it for logging.

This minimal surface is deliberate. APIs are easy to add when a real need surfaces and hard to remove without an ABI break, so the proposal starts narrow.

### 5. Session Creation

```c
OrtStatus* CreateSession(const OrtEnv* env,
                         OrtModelPackageComponentContext* cix,
                         const OrtSessionOptions* opt_session_options, // may be NULL
                         OrtSession** out);
```

This is the single-file convenience. Behavior is a clean binary on whether the caller supplies session options:

1. If `ModelPackageComponent_GetSelectedVariantFileCount(cix) != 1`, fail with a clear diagnostic. Multi-file variants are the consumer's responsibility (see [Example 2](#example-2-multi-file-component-pipeline-consumer)).
2. **`opt_session_options == NULL` (convenience path).** ORT reads the variant's `variant.json` internally and:
    - applies the file's `session_options` key/value pairs via `AddSessionOption`,
    - registers the EP that selection matched, with the file's `provider_options` key/value pairs,
    - resolves every entry in the file's `shared_files` map to its absolute path and sets up the external-initializers mapping accordingly,
    - opens a session on the file.

   ORT owns the entire setup. This is the default path for typical consumers running single-file variants.
3. **`opt_session_options != NULL` (advanced path).** ORT uses the caller's options *as-is* and opens a session on the variant's ONNX file. ORT does **not** apply `variant.json`'s per-file session_options, does **not** apply per-file provider_options, does **not** resolve `shared_files`, and does **not** validate the EP. The caller is responsible for everything they want from the variant — they parse `variant.json` themselves, resolve shared weights, and configure the session_options before passing them in.

The advanced path exists for callers that need surgical control (custom EP setup, custom external-initializer plumbing, profiling-related options that shouldn't be overridden by the package). It uses the same machinery as the multi-file consumer path — parse `variant.json`, resolve shared weights, build session_options — so consumers who outgrow the convenience path graduate naturally to the advanced path without learning a different mental model.

This is a deliberate "all or nothing" contract. Merging caller options with package options would require ORT to define precedence for every key and every EP, and that policy would itself become a stable surface that's hard to evolve. Saying "you provided options, you own them" sidesteps that entirely.

### API Summary

| API | Purpose | Notes |
|---|---|---|
| `CreateModelPackageContext(path)` | Parse a package directory | Returns opaque context |
| `ReleaseModelPackageContext` | Free the context | |
| `ModelPackage_GetComponent{Count,Names}` | Traversal — list components | |
| `ModelPackage_GetVariant{Count,Names}` | Traversal — list variants per component | |
| `ModelPackage_GetVariantEpCompatibility{Count,Info}` | Per-variant EP compatibility (pre-selection) | Returns (ep, device, compat_string) tuples |
| `ModelPackageAddComponent(ctx, name, source_dir)` | Stage component-add; source mirrors a component dir | Authoring path; merges shared weights by checksum |
| `ModelPackageAddVariant(ctx, component, source_dir)` | Stage variant-add; source is a single-variant component slice | Variant name comes from source `metadata.json` |
| `ModelPackageRemoveComponent` / `…RemoveVariant` | Stage removals | Shared-weight GC happens at `Commit` |
| `ModelPackageCommit` | Persist staged mutations and GC orphan shared weights | |
| `CreateModelPackageOptionsFromSessionOptions` | Capture EP intent from a template | |
| `ReleaseModelPackageOptions` | Free options | |
| `SelectComponent(ctx, component_name, options)` | Pick the best variant for a component | Requires a non-NULL options handle |
| `ReleaseModelPackageComponentContext` | Free the instance | |
| `ModelPackageComponent_GetSelectedVariantFolderPath` | Variant directory on disk | Escape hatch; consumers parse `variant.json` here for per-file detail |
| `ModelPackageComponent_GetSelectedVariantFileCount` | Number of ONNX files in the variant | Lets consumers dispatch single- vs. multi-file paths |
| `ModelPackageComponent_GetSelectedVariantFilePath(ctx, file_idx)` | Absolute path to the ONNX file at index | Avoids manual folder + filename join |
| `ModelPackageComponent_GetSelectedVariantFileSessionOptions(ctx, file_idx)` | Per-file session options as key/value arrays | Returns NULL/empty when none specified |
| `ModelPackageComponent_GetSelectedVariantFileProviderOptions(ctx, file_idx)` | Per-file provider options as key/value arrays | Returns NULL/empty when none specified |
| `ModelPackageComponent_GetSelectedVariantConsumerMetadata` | Opaque consumer blob from `variant.json` | ORT does not interpret |
| ~~`ModelPackageComponent_GetSharedWeightPath(cix, checksum)`~~ | ~~Resolve a shared-weight checksum to its absolute path~~ | Not yet implemented |
| `CreateSession(env, cix, opt_session_options)` | Open an ORT session for a single-file variant | Rejects multi-file |

---

## Package Layout

```
<package>/
├── manifest.json                       # optional content; lists components and merge provenance
├── configs/                            # consumer-shared configs (e.g. genai_config.json base, tokenizer assets)
│   └── ...
└── models/                             # all component models live under here
    └── <component_name>/               # one directory per component; ≥ 1 components per package
        ├── metadata.json               # selection-only metadata
        ├── shared_weights/             # per-component shared weight blobs, addressed by checksum
        │   └── <checksum>/             # the checksum names the directory; the file inside can use any name
        │       └── <blob_filename>
        └── <variant_name>/             # one directory per variant
            ├── variant.json            # file list, per-file SO/PO, shared_files map, consumer_metadata
            ├── model.onnx
            ├── model.onnx.data         # may be a shared_weights checksum reference (see variant.json)
            └── ...                     # additional ONNX files for multi-file variants
```

Notes:

- Components live under the reserved `models/` directory at the package root. This keeps the package root free for additional reserved siblings (`configs/`, `manifest.json`, future extensions) without colliding with component names.
- Component directory names are the component identifiers used by `SelectComponent`. They are arbitrary strings (the `decoder` shown in examples is illustrative, not reserved).
- A package can contain any number of components ≥ 1. A single-component package is a normal case, not a special case.
- A variant directory is self-contained enough that it could be downloaded on its own and dropped into a sibling package directory. This supports "minimal download" workflows where consumers pull only the variant they need.
- `configs/` is reserved for consumer-shared assets that are common across variants. The package format does not interpret its contents; consumers know what to look for there.
- `shared_weights/` is per-component. Cross-component sharing is not supported at this layer.
- The checksum names the per-blob directory; the blob filename inside is unconstrained. Producers may use a generic name (`weight.data`) or one that encodes useful information (`embeddings.fp16.safetensors`). Consumers never construct this path directly — once `ModelPackageComponent_GetSharedWeightPath` is implemented, it will scan the directory and return the absolute path to the single blob inside.

---

## Metadata Schemas

### Top-Level `manifest.json`

```jsonc
{
    "schema_version": 1,                         // package format version
    "components": ["decoder"],                   // optional; if absent, components are discovered from sub-directories
    "merge_provenance": [                        // optional; populated when this package was produced by merging others
        {
            "source_id": "phi4-cpu-only",
            "source_version": "1.0.0",
            "components": ["decoder"],
            "variants": { "decoder": ["cpu"] }
        },
        {
            "source_id": "phi4-qnn",
            "source_version": "1.0.0",
            "components": ["decoder"],
            "variants": { "decoder": ["qnn-npu"] }
        }
    ]
}
```

The manifest's *content* is optional — every field can be omitted and the package will still parse. An absent file is equivalent to an empty manifest. When the `components` list is absent, the context discovers components by scanning sub-directories of the reserved `models/` directory.

`merge_provenance` is the data unmerge needs to reconstruct standalone sub-packages from a merged package. It is populated by `merge` and consumed by `unmerge`. Producers writing a fresh package do not need to populate it.

### Component `metadata.json`

```json
{
    "variants": {
        "cpu": {
            "ep_compatibility": [
                { "ep": "CPUExecutionProvider" }
            ]
        },
        "gpu": {
            "ep_compatibility": [
                { "ep": "CUDAExecutionProvider", "compatibility_string": "sm_80" },
                { "ep": "WebGpuExecutionProvider" }
            ]
        },
        "openvino-npu": {
            "ep_compatibility": [
                { "ep": "OpenVINOExecutionProvider", "device": "NPU" }
            ]
        },
        "qnn-npu": {
            "ep_compatibility": [
                { "ep": "QNNExecutionProvider", "compatibility_string": "soc_60" }
            ]
        }
    }
}
```

**Schema field reference:**

| Field | Required | Description |
|---|---|---|
| `variants` | yes | Map from variant name to variant entry. Variant name is the variant directory name. |
| `variants.<name>.ep_compatibility` | yes | List of EP-compatibility entries declaring the EPs this variant can run on. List shape (instead of EP-keyed map) lets a variant pin a specific device for one EP entry while still listing other EPs separately. |
| `variants.<name>.ep_compatibility[].ep` | yes | EP name (e.g. `CUDAExecutionProvider`). Must be a non-empty string — wildcards / omitted EPs are not supported. Every supported EP must be declared explicitly so variant selection is unambiguous. (Future: if a portable variant is ever needed, the planned shape is to omit the entire `ep_compatibility` array on that variant — likely pinned to CPU at load time. Per-entry wildcards stay disallowed even then.) |
| `variants.<name>.ep_compatibility[].device` | no | Optional device discriminator scoped to this entry's EP (e.g. OpenVINO `"GPU"` vs `"NPU"`). ORT plumbs this to the EP's preference ABI alongside the captured `OrtEpDevice`; ORT does not interpret it. |
| `variants.<name>.ep_compatibility[].compatibility_string` | no | Single opaque string describing hardware/version constraints for this EP entry (e.g. CUDA SM, QNN SoC model). ORT does not interpret it; the EP's preference ABI does. May be omitted (means "no additional constraints, any device of this EP works"). A variant whose artifacts cover multiple compile targets encodes them inside this string using whatever syntax the EP defines (e.g. comma-separated SoC list); ORT never inspects the encoding. |

A variant compatible with multiple EPs (e.g. CUDA *and* WebGPU) has multiple entries under `ep_compatibility`. A variant compatible with one EP across multiple hardware revisions encodes those revisions inside the single `compatibility_string` string for that EP entry, in an EP-defined format. A duplicate `ep` key across two entries is allowed only when the entries differ in `device` (e.g. one entry pinning `OpenVINOExecutionProvider`/`GPU`, another pinning `OpenVINOExecutionProvider`/`NPU` for the same variant).

The schema deliberately uses a single string per EP entry, not a list. A variant is a unit of compilation — its files are co-produced, often share weights, and target one set of compile-time settings. They are not independently selectable, so per-file or per-string ranking at the variant boundary buys no real flexibility but adds ABI surface and runtime aggregation policy (see the "Compatibility String Is Owned by the EP" rationale below).

This schema deliberately holds nothing else. No file lists, no per-file options, no consumer metadata, no human-readable descriptions. Adding or removing a variant is a one-entry edit plus a directory move.

### Variant `variant.json`

```jsonc
{
    "files": [
        {
            "filename": "model.onnx",
            "session_options": {
                "intra_op_num_threads": 4,
                "graph_optimization_level": "ORT_ENABLE_ALL"
            },
            "provider_options": {
                "htp_performance_mode": "burst",
                "htp_graph_finalization_optimization_mode": "3",
                "soc_model": "60"
            },
            "shared_files": {
                "model.onnx.data": "abc123def456..."   // checksum -> resolves to the blob inside <component>/shared_weights/abc123def456.../
            }
        }
    ],
    "consumer_metadata": {
        // Free-form. ORT does not interpret. Returned verbatim by ModelPackageComponent_GetSelectedVariantConsumerMetadata.
        // The convention for ORT-GenAI consumers is a key like "genai_config_overlay".
        "genai_config_overlay": { "...": "..." }
    }
}
```

**Schema field reference:**

| Field | Required | Description |
|---|---|---|
| `files` | yes | List of ONNX files in this variant. Consumers parse this array directly when handling multi-file variants. |
| `files[].filename` | yes | Filename relative to the variant directory. |
| `files[].session_options` | no | Flat key/value session options applied to this file when ORT creates a session. Keys are dispatched through `OrtApi::AddSessionOption` — known keys (see [Appendix B](#appendix-b-session-options-reference)) call the dedicated `OrtSessionOptions` setter (with string→typed parsing); unknown keys fall through to `AddSessionConfigEntry`. Consumers building multi-file pipelines apply these to their own `OrtSessionOptions*` via the same dispatcher. Authoring guidance: do not store consumer-side debug tags (e.g. GenAI `log_id`) here — those are synthesized at runtime by the consumer. |
| `files[].provider_options` | no | Flat key/value provider options for the variant's EP. No EP-name nesting. |
| `files[].shared_files` | no | Map from filename-as-referenced-by-the-onnx-graph to the checksum of a shared-weight blob. Consumers resolve checksums to absolute paths under `<component>/shared_weights/<checksum>/` (a dedicated accessor, `ModelPackageComponent_GetSharedWeightPath`, is planned but not yet implemented). Filenames not listed in `shared_files` are assumed to live next to the ONNX file in the variant directory. |
| `consumer_metadata` | no | Opaque JSON blob for consumer frameworks. Single blob per variant; no nesting structure imposed by ORT. |

Notes:

- File identifiers across variants are not enforced to match. A `cpu` variant may have one file named `model.onnx`; a `qnn-npu` variant may have four files with totally different names. Consumers that care about cross-variant identity rely on filenames inside their own metadata (e.g. ORT-GenAI's pipeline references) rather than ORT-imposed identifiers.
- `variant.json` is part of this proposal's public schema. ORT's internal single-file `CreateSession` reads it; multi-file consumers parse it directly.

---

## Variant Selection Algorithm

Selection runs at `SelectComponent` time. Inputs are the captured EP list on `OrtModelPackageOptions` (always non-NULL) and the component's variants.

**Step 1 — Filter by EP match.**

For each variant, walk the entries of its `ep_compatibility` list. The variant is *eligible* if any entry's `ep` matches any EP in the captured list. Variants with no matching entry are rejected.

**Step 2 — Score by EP order (within ORT).**

Among eligible variants, ORT sorts by the ordinal position of the matched EP in the captured list — earlier EPs preferred. This gives consumers a deterministic preference dial: list EPs in priority order on the session_options template.

**Step 3 — Delegate compatibility-string disambiguation to the EP.**

If multiple variants tie at the same captured-EP rank, ORT calls into the matched EP's preference ABI, passing the compatibility string of each still-eligible variant (one per variant, possibly NULL) and the `OrtEpDevice` from the captured options. The EP returns its preferred index. The exact ABI signature is an [open question](#1-ep-side-preference-abi); the contract is that ORT plumbs one opaque string per variant without interpreting it.

**Step 4 — Tie-break by insertion order.**

If the EP declines to choose (returns no preference), ORT falls back to the order variants appear in `metadata.json`.

**Empty-captured-EP-list edge case.**

If the options handle's captured EP list is empty (consumer built options without `Append_V2` or a policy result), this is treated as captured `[CPUExecutionProvider]` for selection purposes. Variants that match CPU win normally; non-CPU-only variants are rejected.

**Selection determinism.**

For a fixed package and a fixed captured EP list, selection is deterministic. The EP preference ABI is required to be deterministic for the same `OrtEpDevice` and string list.

**Single-variant packages.**

A package with one variant per component is not a special case at the algorithm level. Producers declare the variant's `ep_compatibility` honestly; consumers capture the matching EP into options; the lone variant wins because nothing competes. Consumers that need to discover what EP to register can read variant EP compatibility through the [traversal accessors](#1-package-context-parse-traverse-mutate) before building options.

---

## Usage Examples

### Example 1: Single-File Component

The simple consumer path. A package with one component (`decoder`) and three variants (`cpu`, `gpu`, `qnn-npu`); the consumer wants CUDA.

```c
OrtSessionOptions* template_so;
OrtCreateSessionOptions(&template_so);
OrtSessionOptionsAppendExecutionProvider_V2(template_so, env, "CUDAExecutionProvider", ...);

OrtModelPackageOptions* pkg_options;
CreateModelPackageOptionsFromSessionOptions(env, template_so, &pkg_options);

OrtModelPackageContext* ctx;
CreateModelPackageContext("/path/to/phi4.ortpackage", &ctx);

OrtModelPackageComponentContext* cix;
SelectComponent(ctx, "decoder", pkg_options, &cix);

OrtSession* session;
CreateSession(env, cix, /*opt_session_options=*/ NULL, &session);

// Use session as normal.

ReleaseModelPackageComponentContext(cix);
ReleaseModelPackageContext(ctx);
ReleaseModelPackageOptions(pkg_options);
ReleaseSessionOptions(template_so);
```

The `cix` defaults the file index to 0 internally; the consumer never sees indexes.

### Example 2: Multi-File Component (Pipeline Consumer)

ORT-GenAI handling a QNN-NPU multi-file decoder. ORT's `CreateSession(cix, …)` rejects multi-file, so GenAI walks the variant directory itself.

```c
OrtModelPackageComponentContext* cix;
SelectComponent(ctx, "decoder", pkg_options, &cix);

size_t file_count;
ModelPackageComponent_GetSelectedVariantFileCount(cix, &file_count);

if (file_count > 1) {
    const ORTCHAR_T* folder;
    ModelPackageComponent_GetSelectedVariantFolderPath(cix, &folder);

    for (size_t i = 0; i < file_count; ++i) {
        const ORTCHAR_T* file_path;
        ModelPackageComponent_GetSelectedVariantFilePath(cix, i, &file_path);

        const char* const* so_keys; const char* const* so_values; size_t so_count;
        ModelPackageComponent_GetSelectedVariantFileSessionOptions(cix, i, &so_keys, &so_values, &so_count);

        const char* const* po_keys; const char* const* po_values; size_t po_count;
        ModelPackageComponent_GetSelectedVariantFileProviderOptions(cix, i, &po_keys, &po_values, &po_count);

        // Build session options from the per-file key/value pairs.
        OrtSessionOptions* so;
        OrtCreateSessionOptions(&so);
        apply_known_session_option_setters(so, so_keys, so_values, so_count);  // GenAI helper
        OrtSessionOptionsAppendExecutionProvider_V2(so, env,
            /* selected EP — GenAI registered it on its template options already */ ep_name,
            po_keys, po_values, po_count);

        // TODO: shared_files resolution not yet available. For now, external data files
        // are expected to live next to the ONNX file in the variant directory.

        OrtSession* stage_session;
        OrtCreateSession(env, file_path, so, &stage_session);

        // GenAI keys the session by basename, matched against its own genai_config
        // overlay's pipeline[].<stage>.filename.
        genai_pipeline_register(stage_session, basename(file_path));
    }
}

const char* consumer_json;
ModelPackageComponent_GetSelectedVariantConsumerMetadata(cix, &consumer_json);
// GenAI parses the JSON string, finds genai_config_overlay, and merges into base config.
```

The consumer uses the per-file accessors to get paths and options directly. Shared-weight resolution (via a future `GetSharedWeightPath` accessor) is not yet available; for now external data files live alongside the ONNX file in the variant directory.

### Example 3: Inspecting a Package

Walking a package to print its layout — component names, variants, and per-variant EP compatibility. No selection needed.

```c
OrtModelPackageContext* ctx;
CreateModelPackageContext(path, &ctx);

size_t cn;
ModelPackage_GetComponentCount(ctx, &cn);
const char** component_names;
ModelPackage_GetComponentNames(ctx, &component_names);

for (size_t i = 0; i < cn; ++i) {
    printf("Component: %s\n", component_names[i]);
    size_t vn;
    ModelPackage_GetVariantCount(ctx, component_names[i], &vn);
    const char** variant_names;
    ModelPackage_GetVariantNames(ctx, component_names[i], &variant_names);
    for (size_t j = 0; j < vn; ++j) {
        printf("  Variant: %s\n", variant_names[j]);
        size_t en;
        ModelPackage_GetVariantEpCompatibilityCount(ctx, component_names[i], variant_names[j], &en);
        for (size_t k = 0; k < en; ++k) {
            const char* ep;
            const char* device;
            const char* compat_string;
            ModelPackage_GetVariantEpCompatibility(ctx, component_names[i], variant_names[j], k,
                                                   &ep, &device, &compat_string);
            printf("    EP: %s [device=%s, compat=%s]\n", ep,
                   device ? device : "", compat_string ? compat_string : "");
        }
    }
}
```

To see file-level details (paths, session/provider options, consumer_metadata), the inspector calls `SelectComponent` with an options handle and queries the resulting `cix`. Selection always requires a real options handle; tooling that doesn't intend to run anything can still build a minimal options handle with whatever EP it wants to drive selection toward.

### Example 4: Mutating a Package

Adding a new variant to an existing component.

```c
OrtModelPackageContext* ctx;
CreateModelPackageContext("/path/to/phi4.ortpackage", &ctx);

ModelPackageAddVariant(ctx,
                       "decoder",
                       "/staging/phi4-openvino-npu-slice/");
//
// Source layout:
//   /staging/phi4-openvino-npu-slice/
//     metadata.json                 # one variant entry: "openvino-npu"
//     shared_weights/               # optional; only blobs the variant references
//       <checksum>/<blob>
//     openvino-npu/                 # variant subdir, name matches metadata entry
//       variant.json
//       *.onnx
//
// On commit, ORT copies the variant subdir into <package>/decoder/openvino-npu/,
// merges the metadata.json variant entry into <package>/decoder/metadata.json,
// and for each shared_weights/<checksum>/ that isn't already in <package>/decoder/shared_weights/,
// copies the blob over.

ModelPackageCommit(ctx);
ReleaseModelPackageContext(ctx);
```

The producer is responsible for staging a source directory that conforms to the component-mirror layout described in [Mutation API](#1-package-context-parse-traverse-mutate): `metadata.json` declaring exactly one variant for `AddVariant`, the variant subdir, and any shared-weight blobs the variant references. ORT validates the structure and does not synthesize `metadata.json` or `variant.json`.

---

## Key Design Decisions

### Opaque Handles for ABI Stability

Every public type — `OrtModelPackageContext`, `OrtModelPackageOptions`, `OrtModelPackageComponentContext` — is a forward-declared struct accessed through API functions. Internal addressing (file identifiers, sub-component grouping, on-disk directory shape, manifest version, etc.) is *not* part of the ABI. Adding a new field to `metadata.json`, changing how shared weights are laid out, or introducing a new internal index does not require an API break. Callers stay compatible.

This is a conscious response to APIs that hard-code internal addressing into call signatures (e.g. requiring callers to pass strings naming internal entities). Strings the ORT API takes are limited to *consumer-defined names* (component names, variant names, EP names) — names the consumer chose or that come from a stable EP registry. They are never names ORT made up about its own internals.

### Component Metadata Is Selection-Only

`<component>/metadata.json` declares variants and their EP compatibility. Nothing else. The justification:

- **Adding or removing a variant is a single-entry edit + directory move.** No need to re-flow per-file options, consumer metadata, or anything else.
- **Variant directories are independently movable.** A producer can build a single-variant package, and a downstream tool can drop that variant directory into a multi-variant package without touching the new variant's runtime details.
- **Selection is fast and cheap.** The selection algorithm only ever reads component metadata; it does not page through variant directories.

### Per-Variant `variant.json` Owns Runtime Detail

File lists, per-file session_options, per-file provider_options, shared-weight references, and the consumer_metadata blob all live in `<component>/<variant>/variant.json`. The variant directory is the unit of "everything ORT or a consumer needs to know to actually load this variant," and it is self-contained.

This separation also means the *selection-time* read path is small (just `metadata.json` per component) while the *load-time* read path is local to the chosen variant.

### EP Compatibility Is Per-Variant, Not Per-File

A variant declares one EP-compatibility list. The list says which EPs the variant runs on; it does not say which EP each individual file runs on. Reasons:

- **Multi-file variants ship as a unit.** The QNN-NPU variant of a decoder is one variant — its four ONNX files run together as a pipeline on QNN. There is no meaningful "this file runs on QNN, that file runs on CPU" decision for ORT to make at selection time.
- **Per-file CPU/EP overrides, when needed, are the consumer's domain.** ORT-GenAI can express "run the embedding on CPU even though the rest of the variant is on QNN" through its own genai_config without polluting the package format with mixed-EP variant declarations.
- **Selection stays simple.** The algorithm filters by single-string EP match per variant. No per-file score averaging, no neutral-fallback bookkeeping.

### Compatibility String Is Owned by the EP

ORT does not interpret the compatibility string. It plumbs one string per variant through to the matched EP via the EP-side preference ABI ([Open Question 1](#1-ep-side-preference-abi)). The EP — which already owns hardware enumeration, capability detection, and device-specific tuning — is in the right place to decide whether a string like `"sm_80"` (or an EP-defined multi-value encoding such as `"sm_80,sm_86,sm_90"`) matches the user's actual GPU.

This keeps EP-specific knowledge (SoC model, driver versions, JIT vs AOT compatibility, architecture tier) inside the EP plugin and out of the package format. Adding a new EP, or extending an existing one with a new compatibility dimension, is a producer-and-EP coordination — the package format and ORT core do not change.

**One string per variant, not a list.** A variant ships as a unit of compilation: its files are co-produced with one set of compile settings (often sharing weights, calibration, or HTP context). The package format does not expose per-file or per-string ranking at the variant boundary, because doing so would either invite invalid cross-variant file mosaics or push EPs toward heuristic aggregation policies (min/sum across roles, averaging enum values) that the EP is better positioned to encode internally as it sees fit. If an EP needs to express multiple compile targets per variant, it defines the encoding inside the string and parses it itself; ORT remains a transparent plumber.

### No File-Set Consistency Across Variants

A `cpu` variant may have one ONNX file. A `qnn-npu` variant may have four. They are different packagings of the same component, with different file structures. ORT does not enforce that variants of the same component declare the same files.

The reason is that the variants genuinely have different file structures (single monolithic file vs. multi-stage pipeline) and forcing a uniform identifier scheme on top would either create fictitious "file identity" (one variant's `decoder` is another variant's `prompt-processor` + `token-generator` + …) or restrict what producers can ship. Consumers that need cross-variant identity rely on their own conventions (e.g. ORT-GenAI keys files by filename in its config overlay).

### Minimal Component-Instance Surface

The component instance exposes the smallest accessor set that supports both the single-file `CreateSession` path and the multi-file consumer path:

- `GetVariantFolderPath` — the escape hatch and the root for parsing `variant.json`.
- `GetFileCount` — for dispatch.
- `GetConsumerMetadata` — blessed access to the overlay/consumer blob.
- `GetSelectedVariantFilePath(file_idx)`, `...FileSessionOptions(file_idx)`, `...FileProviderOptions(file_idx)` — per-file accessors for path and options without needing to parse `variant.json` manually.
- `CreateSession` — single-file convenience.

What's deliberately *not* exposed: the matched EP, the matched compatibility string, the selected variant name. Selection state is internal. The variant name is derivable as the basename of `GetSelectedVariantFolderPath` for logging.

The bias is toward narrowness. Adding an accessor later is cheap; removing one without an ABI break is not.

### Single-File-Only `CreateSession`

`CreateSession(env, cix, opt_so)` rejects variants with more than one file. The rationale:

- A single-file variant is a pure ORT concept: open a session, run it. The package API can handle this end-to-end.
- A multi-file variant has *consumer-specific orchestration* (which file runs first, when each is invoked, how state is shared between sessions). That orchestration is not the package API's job.
- Multi-file consumers parse `variant.json` (a documented schema in this proposal) and build sessions themselves. ORT's internal single-file convenience path uses the same `variant.json` data internally, so the format is the single source of truth — there is no duplicate code path.

Refusing the multi-file case explicitly (rather than silently picking the first file) prevents subtle bugs where a consumer accidentally runs only one stage of a pipeline.

### `CreateSession` Is All-Or-Nothing on Caller Options

When the caller passes `opt_session_options == NULL`, ORT owns the entire session setup from `variant.json`. When the caller passes their own `OrtSessionOptions*`, ORT uses it as-is — no merging, no precedence resolution, no per-file pair application from the package. The caller takes full responsibility for parsing `variant.json` themselves and feeding whatever they want into their options.

The alternative — merging caller options on top of package options — would require ORT to define precedence rules for every session-option key, every provider-option key, every EP, and every external-initializer source. That policy would itself become a stable, observable surface that consumers depend on, which is exactly the kind of implicit contract that is painful to evolve.

The all-or-nothing rule also keeps the advanced caller's mental model consistent with the multi-file consumer's: in both cases, you parse `variant.json`, resolve shared weights, and build session_options yourself. There is one "manual" path, and the convenience function exists *next to* it, not layered on top of it.

### Shared Weights as a Resolved Path

Shared weights live under `<component>/shared_weights/<checksum>/<blob_filename>`. The checksum names the per-blob directory; the blob filename inside is producer's choice (a plain `weight.data`, or a name that encodes format/version like `embeddings.fp16.safetensors`). Allowing arbitrary filenames keeps options open for future tooling that wants to read intent from the name without a separate manifest. When a variant's ONNX file references an external-data filename, the variant's `variant.json` maps that filename to a checksum, and the layout under `shared_weights/` is the resolution.

The intended API for this is `ModelPackageComponent_GetSharedWeightPath(cix, checksum, &path)` (not yet implemented). Critically:

- ORT does not symlink, hardlink, or copy shared weights into the variant directory.
- ORT does not intercept the ONNX loader's path resolution.
- The caller (ORT's internal session creation, or an external consumer building its own session_options) plumbs the resolved path into ORT through the existing external-initializers session-options API.
- The accessor encapsulates the `<component>/shared_weights/<checksum>/<blob>` layout — including the fact that the blob filename is unconstrained — so the format can evolve (different blob filenames, additional levels of nesting) without consumer changes.

This avoids filesystem-mutation surprises, sidesteps platform-specific symlink quirks, and keeps the package's internal layout an implementation detail.

### `consumer_metadata` Is a Single Opaque Blob

The variant's `variant.json` carries a single `consumer_metadata` field. It is returned verbatim by `GetConsumerMetadata(cix)` — no key, no namespace, no parsing. ORT stores the blob, ORT returns the blob, and ORT does not interpret the blob.

Consumers that share a package layout (e.g. all ORT-GenAI consumers on a Phi-4 package) coordinate on the blob's internal shape — typically a JSON object with framework-named keys like `genai_config_overlay`. Coordination happens *outside* ORT.

This keeps the package format fully neutral toward any one consumer framework. Adding a new consumer means defining a key inside the blob; no schema change to the package format.

### Manifest Carries Merge Provenance

The top-level `manifest.json`'s content is optional, but when a package is the result of merging two or more upstream packages, the `merge_provenance` field records which components and variants came from which source. Unmerge — extracting a standalone sub-package — uses this provenance to reconstruct each upstream package faithfully.

Both merge and unmerge are deferred to a later workstream, but the manifest schema is designed today to carry the data they need. Producers writing fresh packages can ignore `merge_provenance` entirely.

---

## Implementation Plan

### Phase 1: Core API and Read Path

- `CreateModelPackageContext` parses the package directory (manifest, components, variants) into an in-memory representation. Read-only.
- Traversal accessors (`GetComponentCount`, `GetComponentName`, `GetVariantCount`, `GetVariantName`).
- `CreateModelPackageOptionsFromSessionOptions` snapshots EP intent.
- `SelectComponent` runs the selection algorithm. EP-side preference ABI is initially stubbed with insertion-order tie-break only (Open Question 1 lands later).
- `OrtModelPackageComponentContext` accessors: `ModelPackageComponent_GetSelectedVariantFolderPath`, `...FileCount`, `...FilePath`, `...FileSessionOptions`, `...FileProviderOptions`, `...ConsumerMetadata`.
- `CreateSession(env, cix, opt_so)` for single-file variants — rejects multi-file with a diagnostic. Reads `variant.json` internally for the file's session_options, provider_options, and shared_files.
- ORT-internal external-data plumbing: `CreateSession` resolves shared-weight checksums internally and feeds the resulting paths to the existing external-initializers infrastructure.

### Phase 2: Mutation and Authoring

- `ModelPackageAddComponent`, `RemoveComponent`, `AddVariant`, `RemoveVariant`, `Commit`. Staged-then-committed model.
- Validation on `Add`: source directory conforms to the component-mirror layout, `metadata.json` is well-formed (and declares exactly one variant for `AddVariant`), each variant's `variant.json` is well-formed, `ep_compatibility` entries name known EPs (warning-only — unknown EPs may be intentional for future-proofing), every checksum referenced by `variant.json` `shared_files` resolves under the source's `shared_weights/`.
- Shared-weight merge logic: content-addressable dedup (skip on checksum collision; optionally re-hash on copy as a producer-error safeguard).
- Shared-weight GC at `Commit`: walk all remaining variants, collect referenced checksums, delete unreferenced `<component>/shared_weights/<checksum>/` directories.
- A small authoring CLI (or Python tooling) on top of the C API to make package construction usable from a workflow without writing C.

### Phase 3: Compile and Merge/Unmerge

- `ModelPackageCompile(ctx, …)` — granularity is [Open Question 2](#2-jit-compilation-granularity). Once decided, the API materializes a compilable variant in place, updates its compatibility string, and writes the result back through `Commit`.
- `ModelPackageMerge(target, source, …)` — combine packages, populate `merge_provenance`.
- `ModelPackageUnmerge(ctx, source_id, out_path)` — extract a standalone sub-package using `merge_provenance`.

---

## Open Questions

### 1. EP-Side Preference ABI

ORT delegates compatibility-string disambiguation to the matched EP. The ABI shape needs to land before the selection algorithm is fully wired:

```c
// Strawman: one opaque compatibility string per candidate variant (may be NULL
// when the variant's ep_compatibility entry omits `compatibility_string`).
OrtStatus* OrtEpPreferVariant(const OrtEp* ep,
                              const OrtEpDevice* device,
                              size_t num_variants,
                              const char* const* variant_names,
                              const char* const* compat_strings,  // one per variant; entries may be NULL
                              size_t* out_preferred_idx,
                              bool* out_has_preference);
```

Open: should the EP return a single preferred index, or a ranking? Should it have access to the entire captured device list, or only the matched device? How does it signal "no preference" vs. "this one is mandatory"?

The single-string-per-variant shape mirrors the metadata.json schema and keeps the ABI flat. If an EP author needs to encode multiple compile-target attributes (e.g. several supported SoCs), they define the encoding inside the string and parse it on the EP side. ORT does not standardize a multi-value list shape because doing so would push toward a runtime aggregation policy (which file's compat wins, how to score partial matches) that the EP is in a strictly better position to encode internally for its own hardware semantics.

### 2. JIT Compilation Granularity

Whiteboarded options:

- **Variant-level** — the variant has a "compilable" flag and a target output filename. The whole variant is JIT-compiled in place.
- **File-level** — `variant.json` marks specific files as compilable, each with its own target output filename.
- **Component-level** — the component has a "base" variant that is compiled into a per-EP variant on first use.

For multi-file variants, the design also has to answer "is each file JIT'd independently, or as a unit?" — likely file-level if the host EP supports it.

### 3. Merge / Unmerge Semantics

Merging two single-variant packages into one two-variant package is straightforward: copy directories, append entries to component `metadata.json`, append `merge_provenance` entries to the manifest. Open questions:

- What happens when sources conflict on the same component + variant name? Reject? Rename?
- Does merge consolidate `configs/` (e.g. shared `genai_config.json` base)? If so, by what rule?
- Does merge across producers require a common manifest schema version?
- Does unmerge fail if a downstream consumer mutated the merged package's component metadata after the merge?

### 4. Manifest Schema Versioning

The manifest carries `schema_version` today as a placeholder. The upgrade story (forward and backward compatibility, what counts as a breaking change) is not yet specified. Keeping the manifest's content optional until v2 lands gives producers a way to ship without committing to a version they don't know yet.

### 5. Cross-Component Consistency

A package can have multiple components. Today, selection runs per-component, and there is no enforcement that the EP chosen for `decoder` matches the EP chosen for `vision_encoder`. For most cases this is desirable (e.g. CPU embedding + GPU decoder). For some cases — multi-component models that must share a session memory pool, or that have hard runtime dependencies on co-located EPs — the consumer wants a guarantee. Whether ORT should provide that guarantee, or whether it remains the consumer's responsibility, is an open question.

### 6. Authoring Tools

The C API supports mutation, but day-to-day producers (model-export scripts, CI pipelines) will not call C directly. A Python authoring layer on top of the C API — `package = ModelPackage("/path"); package.add_variant("decoder", staging_dir); package.commit()` — is in scope but unspecified at this point.

### 7. Shared-Weight Verification on Add

`AddComponent` / `AddVariant` skip copying a shared-weight blob when the destination already has the same checksum directory, on the assumption that "checksum identifies content." If a producer accidentally mislabels a blob (wrong checksum filename), the destination's pre-existing blob silently wins. ORT can guard against this by re-hashing on copy:

- **Always re-hash:** safest, but adds I/O proportional to total shared-weight size on every add.
- **Re-hash by default; opt-out flag for trusted authoring tools:** good middle ground.
- **Never re-hash; rely on producer correctness:** fastest; matches how content-addressable stores typically operate.

Default policy is yet to land. Recommend "re-hash by default, opt-out flag" for v1.

---

## Appendix A: ORT-GenAI Integration

ORT-GenAI is the first consumer of the model package API and the primary multi-file consumer. This appendix specifies how GenAI sits on top of the package API: where its assets live in the package, how its per-variant configuration is layered, and how its session-creation flow consumes the package's generic accessors.

The package format itself is generic. Nothing below requires schema changes to the package — it is entirely a convention layered on top of `consumer_metadata` and the `configs/` directory.

### Where GenAI Assets Live in the Package

```
<package>/                              # ← model_root for GenAI
├── manifest.json
├── configs/                            # ← consumer-shared (model-level) assets
│   ├── genai_config.json               # GenAI BASE — merged into for every variant
│   ├── tokenizer.json                  # tokenizer assets
│   ├── tokenizer_config.json
│   ├── special_tokens_map.json
│   ├── added_tokens.json
│   ├── vocab.json
│   ├── merges.txt
│   ├── chat_template.jinja             # if present
│   ├── processor_config.json           # vision processor (multimodal models)
│   └── audio_processor_config.json     # speech processor (multimodal/whisper)
└── models/
    └── <component>/
        ├── metadata.json
        └── <variant>/                      # ← variant_folder for GenAI
            ├── variant.json                # consumer_metadata.genai_config_overlay carried here
            ├── *.onnx, *.onnx.data         # ORT-loadable model files
            ├── <custom_ops>.so/.dll        # if `custom_ops_library` referenced
            └── <lora>.onnx                 # per-variant LoRA adapters (e.g. phi multimodal)
```

**Two filesystem roots for GenAI:**

| What GenAI is loading | Where it looks | Consumer of the path |
|---|---|---|
| `genai_config.json` (base), tokenizer assets, processor configs, chat template | `<package>/configs/` | `Generators::Config`, `Tokenizer`, image/audio processors, constrained logits |
| ONNX model files, external data, custom ops library, LoRA adapters | `ModelPackageComponent_GetSelectedVariantFolderPath(cix, &folder)` | `OrtSession::Create`, `RegisterCustomOpsLibrary`, `Adapters::LoadAdapter` |

The package root is the model root for any consumer that thinks of itself as loading "the model" (config, tokenizer, processors). The variant folder is the per-(component, variant) directory for anything tied to a specific ONNX file.

- **`configs/genai_config.json`** is the shared base configuration. One per package. It captures the GenAI-architecture-level fields that don't vary by variant.
- **Tokenizer assets** are shared across variants and live in `configs/`. A package that mixes incompatible tokenizers across variants is a producer error; the package format does not police it but GenAI documents it as a constraint.
- **Processor configs** (`processor_config.json`, `audio_processor_config.json`) are model-level preprocessing descriptors and live in `configs/`. They don't vary by EP or quantization. Per-component preprocessing differences (rare) can be expressed via overlay overrides on `model.<component>.config_filename` if needed.
- **Per-variant overlays** are stored in the variant's `variant.json` under `consumer_metadata.genai_config_overlay`. ORT returns the entire `consumer_metadata` as a JSON string through `ModelPackageComponent_GetSelectedVariantConsumerMetadata`; GenAI parses it and pulls `genai_config_overlay` out.
- **LoRA adapters and custom ops libraries** live in the variant folder. Both are tied to specific ONNX files (the LoRA matches a base model graph; the custom ops library matches the EP build that the variant targets), so they cannot be shared across variants.

### Component Discovery Is Driven by `genai_config.json`, Not the User

Important: GenAI users never tell `og.Model(path)` which components to load. The merged `genai_config.json` declares the model architecture, and from `model.type` GenAI knows which component roles exist — `decoder`, or `{vision, speech, embedding, decoder}` for multimodal, or `{encoder, decoder}` for encoder-decoder, etc. GenAI's role-to-package-component mapping comes from the `"component": "<name>"` field carried inside each `model.<role>` block in `genai_config.json`.

The flow is:

1. Open the package, read manifest + scan components.
2. Load `<package>/configs/genai_config.json` as the base.
3. For each component listed in the package, run `SelectComponent` (using the resolved EP from defaulting or user choice) to obtain a `cix`. Pull its consumer_metadata, extract `genai_config_overlay`, and JSON-merge into the base.
4. With the merged config in hand, GenAI consults `model.type` to instantiate the right `Model` subclass (decoder-only, multimodal, whisper, etc.) and walks its expected component roles. For each role, look up `model.<role>.component` to find the right `cix`, then call `cix->GetVariantFolderPath()` and load files from there.

This means the user-facing API stays exactly as it is today: `og.Model(path[, ep])`. The model package integration adds component discovery and variant folder resolution under the covers; nothing surfaces in the C API.

### `genai_config.json` in the Package World

The package-world `genai_config.json` is trimmed compared to the flat-directory shape: anything ORT now owns lives in the package's per-file `variant.json`, and anything that varies by variant lives in the overlay. What remains is the GenAI-architecture description.

| Legacy flat-dir field | Package-world equivalent |
|---|---|
| `model.type` (nested) | `model.type` (nested, unchanged) |
| `model.<component>.filename` | dropped — GenAI matches `pipeline[].<stage>.filename` against the basenames listed in the variant's `variant.json` `files[]` |
| `model.<component>.session_options` (any sub-key) | dropped from the package-shipped `genai_config.json`; static SO/PO are owned by per-file `variant.json` and parsed by GenAI when it walks the variant directory. The same path *does* exist in the *merged-at-runtime* config object as the layer-2 runtime-overrides channel populated by `OgaConfigOverlay` / `RuntimeSettings` — see [Resolution Flow at `Model::Create`](#resolution-flow-at-modelcreate). |
| `model.<component>.session_options.provider_options` | same as above |
| `model.<component>` block | adds a `"component": "<name>"` field carrying the package component name |

What stays in `genai_config.json`: GenAI architecture identity (`model.type`, `head_size`, `hidden_size`, `num_attention_heads`, `num_hidden_layers`, `num_key_value_heads`, `vocab_size`), tokenizer-id constants (`bos_token_id`, `eos_token_id`, `pad_token_id`), capacity (`context_length`), per-component I/O name maps, GenAI-specific structures (`pipeline[]`, `sliding_window`), and `search` defaults.

Pipeline stages reference component files **by filename** (relative to the variant directory). GenAI walks the variant's `variant.json`, builds a `{filename → file-entry}` map, and matches each `pipeline[].<stage>.filename` against it. ORT does not impose a separate logical file identifier.

Concrete base example (Phi-4-mini-reasoning shape):

```json
{
    "model": {
        "type": "phi3",
        "context_length": 131072,
        "bos_token_id": 199999,
        "eos_token_id": [200020, 199999],
        "pad_token_id": 199999,
        "vocab_size": 200064,
        "decoder": {
            "component": "decoder",
            "head_size": 128,
            "hidden_size": 3072,
            "num_attention_heads": 24,
            "num_hidden_layers": 32,
            "num_key_value_heads": 8,
            "inputs":  { "input_ids": "input_ids",
                         "attention_mask": "attention_mask",
                         "past_key_names":  "past_key_values.%d.key",
                         "past_value_names": "past_key_values.%d.value" },
            "outputs": { "logits": "logits",
                         "present_key_names":  "present.%d.key",
                         "present_value_names": "present.%d.value" }
        }
    },
    "search": { "max_length": 131072, "past_present_share_buffer": true,
                "temperature": 1.0, "top_k": 1, "top_p": 1.0 }
}
```

No `session_options`. No `provider_options`. No `filename`. The decoder block declares architectural shape and which package component it maps to. This is the document overlays merge into.

### Per-Variant Overlays (RFC 7386 JSON Merge Patch)

Each variant's overlay is an RFC 7386 JSON Merge Patch stored as `consumer_metadata.genai_config_overlay` in the variant's `variant.json`. At `Model::Create` time, GenAI merges the selected variant's overlay into the base `genai_config.json`.

The RFC is two pages; the merge rules are:

1. Object keys recurse — `{"a":{"b":2}}` patched onto `{"a":{"c":1}}` yields `{"a":{"b":2,"c":1}}`.
2. Scalars and arrays replace wholesale — no element-level array merging.
3. `null` deletes — `{"a": null}` removes key `a`.
4. Anything not mentioned is unchanged.

JSON Merge Patch is widely deployed (Kubernetes `--type=merge`, Azure ARM PATCH, GitHub PATCH) and implementations are short (~20 lines in any language). Arrays replacing wholesale is fine for our case: the largest array in practice (`pipeline[]`) is whole-variant-specific by nature — there is no meaningful per-stage merge.

Why not alternatives:

| Alternative | Why not |
|---|---|
| Full per-variant `genai_config.json` (no merging) | 7× duplication of identical content; drift inevitable. |
| RFC 6902 JSON Patch (operation list) | Verbose; only useful when precise array-element ops are needed, which they aren't. |
| Strategic merge (Kubernetes-style) | Requires schema annotations on every field. Overkill. |
| Custom DSL (Jsonnet, CUE) | Adds a dependency for marginal gain; opaque to reviewers. |

### EP Defaulting in `og.Model` and `og.Config`

GenAI's `Model` constructor, and the lower-level `Config` constructor that wraps the same package-loading flow without immediately building sessions, both accept an EP argument. With the pre-selection traversal accessors, that argument can be made optional with sensible defaulting:

```python
# explicit
model  = og.Model("/path/to/phi4.ortpackage", ep="CUDAExecutionProvider")
config = og.Config("/path/to/phi4.ortpackage", ep="CUDAExecutionProvider")

# defaulted — let GenAI pick if the package leaves no ambiguity
model  = og.Model("/path/to/phi4.ortpackage")
config = og.Config("/path/to/phi4.ortpackage")
```

The defaulting algorithm is the same for both entry points (it runs in GenAI before any package option is captured) — `Model` just chains into session creation after the EP is resolved, while `Config` stops at the merged-config view.

Algorithm:

1. Open the package as `OrtModelPackageContext`.
2. For each component, walk its variants and collect the union of EP names declared in each entry of `ep_compatibility` (via `ModelPackage_GetVariantEpCompatibility`, consuming only the `ep_name` out-parameter). Defaulting only cares about which EPs the package can run on, not the device or compatibility string.
3. Intersect those per-component sets. The result is the set of EPs that *every* component can run on — a session built on any of those EPs can load every component.
4. If the intersection has exactly one EP, use it. This is the common "the package only ships one EP across components" case (e.g. a CPU-only package, or a CUDA-only package).
5. If the intersection has more than one EP, GenAI may pick by a fixed preference order (e.g. CUDA > QNN > WebGPU > CPU) or fail with a diagnostic listing the candidates and asking the user to choose. Recommend fail-with-diagnostic for the first cut — implicit policy is the kind of thing that's hard to change later.
6. If the intersection is empty, fail with a diagnostic. The package has no EP that supports every component; the user must pick per-component, which isn't expressible through the simple top-level `ep` argument and likely indicates a malformed package.

Failure path 5 and 6 use the *intersection*, not the *union*, because GenAI builds one set of `OrtModelPackageOptions` for the whole `Model` (one EP-list captured from one session-options template). If the chosen EP isn't compatible with one of the components, that component's `SelectComponent` will return no eligible variants.

This works because the API exposes per-variant EP compatibility on the *context* (pre-selection), not just on the component context (post-selection). Without those traversal accessors, GenAI would have to either require the EP argument always or speculatively call `SelectComponent` and unwind on failure.

Edge cases:

- **Single-component package with multi-EP variants** (e.g. one component with cpu and cuda variants): defaulting works — intersection is `{cpu, cuda}`, fall through to step 5.
- **Single-EP package** (e.g. a Vitis-only edge build): intersection is `{vitis}`, defaulting picks Vitis automatically. The constructor "just works" without the user knowing or caring.
- **Mixed-EP multi-component package** where every component happens to ship a CPU variant alongside its EP-specific variants: intersection includes CPU, and CPU wins iff GenAI's tie-break prefers it. Recommend GenAI list the candidate EPs and prefer the user's explicit choice over implicit defaulting in this case.

### Resolution Flow at `Model::Create`

```
                        ┌─────────────────────────────────────────────┐
                        │ ORT model package context (already built)   │
                        │   - selected component instance per         │
                        │     component                               │
                        │   - per-file paths, EPs, session/provider   │
                        │     options, external-data mapping          │
                        └─────────────────────────────────────────────┘
                                            │
                                            ▼
   1. Load base                ┌───────────────────────────┐
      <pkg>/configs/           │ genai_config (base copy)  │
      genai_config.json        └───────────────────────────┘
                                            │
   2. For each component,                   │  ModelPackageComponent_GetSelectedVariantConsumerMetadata(cix, &json)
      pull consumer blob,      ┌───────────────────────────┐
      extract                  │ overlay (JSON merge patch)│
      genai_config_overlay     └───────────────────────────┘
                                            │
   3. Apply RFC 7386 merge                  ▼
      patch                    ┌───────────────────────────┐
                               │ merged genai_config       │  ──►  drives model.type
                               │  (final, per-variant view)│       dispatch, pipeline
                               └───────────────────────────┘       construction, search
                                                                   defaults, etc.
```

GenAI then walks the merged config to construct sessions. **GenAI uses the advanced path of `CreateSession` (or, equivalently, calls `OrtCreateSession` directly) — never the convenience path with `NULL` options.** GenAI's session setup is not a pass-through of the package's flat KV pairs into `AppendExecutionProvider_V2`; it runs framework-level logic on top.

The session-options resolution for any one file is layered:

| Layer | Source | When applied |
|---|---|---|
| 1. Baseline | variant.json `files[].session_options` and `files[].provider_options` | Package-baked; static for the variant |
| 2. Runtime overrides | Merged genai_config's `model.<component>.session_options` subtree | Comes from `OgaConfigOverlay` (user JSON) or `RuntimeSettings::GenerateConfigOverlay()` (framework-generated, e.g. `dawnProcTable`) |
| 3. Framework knobs | GenAI's `SetProviderSessionOptions()` machinery | Cross-session state, typed structs, V2/V1 fallback, key translations, graph-capture detection |

Layer 2 wins over layer 1 per-key. Layer 3 runs last because it injects state (CUDA stream pointers, HTP allocators) that only exists at runtime in the GenAI process. The framework-level concerns are unchanged from before:

- **Cross-session state.** `user_compute_stream` is shared across encoder/decoder/embedding sessions on CUDA. The QNN HTP shared-memory allocator is registered against the global `OrtEnv` and reused across stages. The convenience path knows nothing about cross-session sharing because it only sees one `cix` at a time.
- **Typed provider-options structs.** Some EPs (CUDA `OrtCUDAProviderOptionsV2`, NvTensorRtRtx multi-profile shapes) have V2 ABI surface that isn't fully expressible as flat KV. GenAI builds the typed struct from the merged KV map.
- **V2-plugin → V1-fallback dispatch.** GenAI tries `AppendExecutionProvider_V2` first, falls back to legacy named-EP registration for EPs that aren't yet plugin-shaped.
- **Special key translations.** OpenVINO `cache_dir` → `CACHE_DIR` config entry; custom-ops library path resolution searches model dir → EP lib dir → cwd.
- **Graph-capture detection.** GenAI scans the merged `provider_options` for `enable_cuda_graph` / DML implicit / `multi_profile` and threads the flag through to runtime input placement.

#### Why the Merged Config Still Carries a `session_options` Subtree

The package-shipped `genai_config.json` does not carry `session_options` — that's variant.json's job. But the *merged* genai_config — the in-memory object GenAI works with at runtime — does have a `session_options` slot, used exclusively as the layer-2 runtime override channel.

This is required to preserve two existing GenAI mechanisms:

- **`OgaConfigOverlay(config, json)`**: a public API that merges a user-supplied JSON into the config tree. Existing callers write to `model.<component>.session_options.*` paths (e.g. to set thread counts, EP-specific knobs, or pass platform handles). Reinterpreting these writes as runtime overrides keeps the surface working without a breaking change.
- **`RuntimeSettings::GenerateConfigOverlay()`**: GenAI itself emits an overlay to inject runtime-only handles like WebGPU's `dawnProcTable` — an opaque process-local pointer that *cannot* be baked into a static package. The overlay path it generates is `model.decoder.session_options.provider_options[*].WebGPU.dawnProcTable`, and it must keep working.

So variant.json owns the static config (what the package author shipped); the merged genai_config's `session_options` subtree owns the runtime overrides (what the calling app or the framework injects). Both flow into GenAI's `SetProviderSessionOptions()` as a single merged KV map.

#### What Goes Where

| Concern | Where it lives |
|---|---|
| Static, EP-mechanical knobs (intra-op threads, graph optimization level, `htp_performance_mode`) | variant.json `files[].session_options` / `files[].provider_options` |
| Process-local handles that can't be baked (e.g. `dawnProcTable`) | Layer 2 — emitted by `RuntimeSettings::GenerateConfigOverlay()` into the overlay |
| User-supplied per-deployment overrides (e.g. raise threading on a beefier host) | Layer 2 — `OgaConfigOverlay(config, json)` writes to `model.<component>.session_options.*` |
| Cross-session state (CUDA stream sharing, custom-op library paths) | Layer 3 — GenAI internal, computed at session-build time |

For both single-file and multi-file components, GenAI's flow is uniform:

1. Get the file count and iterate using per-file accessors (`ModelPackageComponent_GetSelectedVariantFilePath`, `...FileSessionOptions`, `...FileProviderOptions`).
2. For each file in the variant (one entry for cpu/cuda/webgpu/vitis/openvino; multiple for QNN-style pipelines):
    - Start with the file's session_options and provider_options from the per-file accessors as the layer-1 baseline.
    - Merge layer-2 overrides on top: walk the merged genai_config's `model.<component>.session_options` subtree (populated by `OgaConfigOverlay` and/or `RuntimeSettings::GenerateConfigOverlay()`) and apply per-key, layer-2 winning. For multi-file variants where overrides need to target a specific stage, the convention is `model.<component>.session_options.pipeline.<filename>.{session_options,provider_options}` (overrides without that nesting apply to every file in the variant).
    - Run the resulting merged map through GenAI's `SetProviderSessionOptions()` machinery (layer 3) to build `OrtSessionOptions*`: cross-session state injection, typed-struct construction, V2/V1 fallback, key translations, graph-capture detection.
    - Call `OrtCreateSession(env, file_path, so, &session)`.
3. Match files to roles: for `model.type: "phi3"`-style configs, the single file *is* the decoder; for `pipeline[]` configs, GenAI matches `pipeline[].<stage>.filename` against the variant's file basenames.

The single-file case is the multi-file case with `files.size() == 1`. There is no separate code path inside GenAI.

ORT's convenience path (`CreateSession(cix, NULL)`) exists for simpler consumers — sample apps, Foundry's pure-inference flow, tools that don't need cross-session orchestration. GenAI is a framework consumer, not a simple consumer, and it stays in the manual lane.

#### How `p_device_` Is Determined

GenAI's `Model` carries a single `DeviceInterface* p_device_` — the "main device" that anchors KV-cache allocation, generation state, search-state buffers, and tensor placement decisions. Exactly one device per `Model`, regardless of how many sessions or pipeline stages exist underneath.

In flat-directory GenAI today, this is determined by the `is_primary_session_options` flag passed to `SetProviderSessionOptions()`:

- The decoder's main `session_options` block is built with `is_primary_session_options = true`. The first registered EP returns a `DeviceInterface*` and that becomes `p_device_`.
- Pipeline stages, vision, speech, and any other secondary sessions are built with `is_primary_session_options = false`. Their EPs do not contribute to `p_device_` and they also pick up an implicit "all listed provider_options are providers" expansion (`session_options.cpp:164-173`).

In the package world, both halves of that flag stop making sense:

- There is no "higher-level" `session_options` block to mark as primary. variant.json carries per-file SO/PO; every file is built with the same dispatch loop.
- Provider-list inference is moot — variant.json `files[].provider_options[].name` already names the EP for each file explicitly. Nothing to infer.

The rule is: **`p_device_` derives from the EP GenAI captured for the *primary component role* of the model.** GenAI already knows this EP — it captured it into the `OrtModelPackageOptions` it passed to `SelectComponent`. No new ORT API is needed.

The "primary component role" is determined by `model.type` in the merged genai_config:

| `model.type` | Primary role | Source of `p_device_` |
|---|---|---|
| LLM (`phi3`, `llama`, `gpt2`, …) | `decoder` | EP captured for the decoder component |
| Encoder-decoder (`marian`, `whisper` ALM-style) | `decoder` | Same |
| Multimodal (`fara`, `qwen2_5_vl`, `phi3_v`, …) | `decoder` | Same — generation runs on the decoder |
| RNN-T speech (`nemotron_speech`) | `decoder` (or whichever component owns generation state) | Same |
| Pipeline (`decoder-pipeline`) | `decoder` (the variant containing the pipeline) | EP captured for the decoder variant — applies to all files in the pipeline that are not `run_on_cpu`-pinned |

Consequences:

- **Mixed-EP multi-component packages** (e.g. vision on CPU, decoder on QNN) still produce a single `p_device_` from the decoder. Vision sessions allocate their own buffers on CPU and copy outputs to/from `p_device_` exactly as today. No change in tensor flow.
- **`run_on_cpu` stages** inside a pipeline use the CPU EP for their session, but they do not influence `p_device_`. They allocate inputs/outputs on CPU and copy across the device boundary as needed (existing behavior).
- **The `is_primary_session_options` parameter is dropped from `SetProviderSessionOptions()`.** Its two effects (provider-list expansion and device assignment) are both replaced: variant.json makes provider lists explicit, and `p_device_` is set once at `Model::Create` time from the captured decoder EP rather than as a side-effect of an SO-building call.
- **No new component context accessor.** Deriving the device from the component context's per-file EPs is unnecessary — GenAI already holds the EP it asked for. This keeps the `OrtModelPackageComponentContext` surface trimmed.

### What Overlays Should and Should Not Contain

| Field | In overlay? | Rationale |
|---|---|---|
| `model.type` | yes | GenAI model-class dispatch — variant-specific (e.g. `phi3` vs `decoder-pipeline`). |
| `model.context_length`, `search.max_length` | yes | EP-capability-driven (e.g. 4096 / 4224 / 131072). |
| `model.<component>.inputs / outputs` name maps | yes | I/O contract differs per variant (QNN renames; OpenVINO adds `position_ids`). |
| `model.<component>.pipeline[]` | yes | Whole-array replace; stages reference files by `filename`. Multi-file variants only. |
| `model.<component>.sliding_window` | yes | EP-specific runtime structure (currently QNN-only). |
| `search.past_present_share_buffer` | yes | KV-layout differs on some EPs. |
| `model.pad_token_id` | avoid unless exporter-required | Should be a tokenizer-level property in the base. |
| `search.{temperature, top_k, top_p, do_sample}` | avoid unless EP-required | User-tunable defaults; producer should pick one canonical value in the base. |
| `model.<component>.filename` | n/a | Not in `genai_config.json` anymore — files are addressed by basename, looked up in the variant's `variant.json`. |
| `session_options`, `provider_options`, `graph_optimization_level`, `custom_ops_library`, threading | n/a in package overlay; **yes** in runtime overlay | Static EP knobs belong in `variant.json`'s per-file fields, not in the package-shipped overlay. The *runtime* overlay (`OgaConfigOverlay` / `RuntimeSettings::GenerateConfigOverlay`) does write `model.<component>.session_options.*` — see the layer-2 mechanism above. |

The table covers what producers ship inside the package (the consumer_metadata-borne overlay applied at config load). Runtime overrides — the layer-2 channel — are a separate stack populated at session-build time by app code and the framework itself.

Rule of thumb: **if the field describes how the model is shaped or what it produces, it is overlay material. If the field describes how ORT runs it, the package's per-file `variant.json` owns it.**

### Single-File Components (CPU, CUDA, WebGPU, Vitis, OpenVINO)

For variants where the only differences from the base are EP-mechanical (different binary, different provider options, different threading), the GenAI overlay is empty (`{}`) or near-empty. Everything that varies lives in the variant's per-file `session_options` and `provider_options` inside `variant.json`.

Example (cuda overlay): `{}`. CUDA-specific knobs (e.g. `enable_cuda_graph`, `enable_skip_layer_norm_strict_mode`) live in per-file `provider_options` in `variant.json`. From GenAI's perspective the cuda variant runs the same `phi3` architecture as cpu — only the binary on disk and EP setup differ.

Example (webgpu overlay) — minimal interesting case (one extra graph input):

```json
{ "model": { "decoder": { "inputs": { "position_ids": "position_ids" } } } }
```

GenAI uses the per-file accessors (`ModelPackageComponent_GetSelectedVariantFilePath`, `...FileSessionOptions`, `...FileProviderOptions`) to get the single file's path and options, runs `provider_options` through `SetProviderSessionOptions()` to handle CUDA stream sharing / typed structs / V2-V1 fallback / key translations, applies `session_options`, and calls `OrtCreateSession`. Same loop structure as the multi-file case below; the loop just runs once.

### Multi-File Components (QNN-Style Pipelines)

For pipeline-style variants (e.g. QNN with embedding / prompt-processor / token-generator / transformer-head), the overlay carries the full `pipeline[]` array. Each stage references a file by `filename` relative to the variant directory.

```jsonc
{
  "model": {
    "type": "decoder-pipeline",
    "context_length": 4096,
    "decoder": {
      "inputs":  { /* renamed past_keys_%d, past_seq_len, total_seq_len */ },
      "outputs": { /* renamed present_keys_%d, etc.                   */ },
      "sliding_window": { "window_size": 64, "alignment": "left",
                          "pad_value": 0, "slide_key_value_cache": false },
      "pipeline": [
        { "embedding":        { "filename": "phi_4_mini_embeddings.all.quant.onnx",
                                "inputs":  ["input_ids"],
                                "outputs": ["input_hidden_states"] } },
        { "prompt-processor": { "filename": "phi_4_mini_ctx.onnx_ctx.onnx",
                                "inputs":  [/* past_keys/values_0..N, ... */],
                                "outputs": [/* present_keys/values_0..N, ... */],
                                "run_on_token_gen": false } },
        { "token-generator":  { "filename": "phi_4_mini_iter.onnx_ctx.onnx",
                                /* same I/O shape as prompt-processor */
                                "run_on_prompt": false } },
        { "transformer-head": { "filename": "phi_4_mini_lm_head.onnx",
                                "inputs":  ["output_hidden_states"],
                                "outputs": ["logits"] } }
      ]
    }
  },
  "search": { "max_length": 4096 }
}
```

The four pipeline stages map 1:1 to the four file entries in the QNN variant's `variant.json` `files[]` array. GenAI's pipeline runner uses the per-file accessors to iterate and creates a session per stage:

```c
size_t file_count;
ModelPackageComponent_GetSelectedVariantFileCount(cix, &file_count);
const char* selected_ep = /* GenAI knows this — it built pkg_options with its EP list */;

for (size_t s = 0; s < merged_genai_config.pipeline_count; ++s) {
    PipelineStage* stage = &merged_genai_config.pipeline[s];

    // Find the file index that matches this pipeline stage's filename.
    size_t file_idx = find_file_by_basename(cix, file_count, stage->filename);
    if (file_idx == SIZE_MAX) { /* producer error: stage references a file not in the variant */ continue; }

    const ORTCHAR_T* file_path;
    ModelPackageComponent_GetSelectedVariantFilePath(cix, file_idx, &file_path);

    const char* const* so_keys; const char* const* so_values; size_t so_count;
    ModelPackageComponent_GetSelectedVariantFileSessionOptions(cix, file_idx, &so_keys, &so_values, &so_count);

    const char* const* po_keys; const char* const* po_values; size_t po_count;
    ModelPackageComponent_GetSelectedVariantFileProviderOptions(cix, file_idx, &po_keys, &po_values, &po_count);

    OrtSessionOptions* so;
    OrtCreateSessionOptions(&so);

    // Layer 1 + Layer 2: merge per-file baseline with runtime overrides
    // pulled from the merged genai_config (model.<component>.session_options).
    KvMap merged_so = layer_merge(so_keys, so_values, so_count,
                                   runtime_overrides_for_file(merged_genai_config, "decoder", stage->filename));
    KvMap merged_po = layer_merge(po_keys, po_values, po_count,
                                   runtime_overrides_for_file_po(merged_genai_config, "decoder", stage->filename));

    apply_known_session_option_setters(so, merged_so);   // GenAI helper

    // Layer 3: SetProviderSessionOptions injects cross-session state, builds typed structs,
    // does V2/V1 fallback, key translations, and graph-capture detection on top of merged_po.
    SetProviderSessionOptions(so, env, selected_ep, merged_po, framework_state);

    OrtSession* stage_session;
    OrtCreateSession(env, file_path, so, &stage_session);
    pipeline_attach(stage, stage_session);
}
```

No static `session_options` or `provider_options` need to appear in the package-shipped overlay — the package owns the static baseline per-file in `variant.json`. The runtime overlay channel (`OgaConfigOverlay`, `RuntimeSettings`) is still available for layer-2 overrides at session-build time.

### Per-File CPU Override

A common pattern in QNN-style pipelines is "the decoder runs on QNN, but the embedding and lm_head should run on CPU because they don't fit the NPU well." This is **not** a package-format fact — `variant.json` `files[]` does not carry a per-file EP pin. It is a consumer-side orchestration concern, expressed by the consumer's overlay. For ORT-GenAI, that means a per-stage `run_on_cpu: true` flag inside the pipeline definition under `consumer_metadata.genai_config_overlay` — the same key already used by `qwen_vl_model.cpp` and the existing `decoder.pipeline[]` config block:

```jsonc
{
  "model": {
    "decoder": {
      "pipeline": [
        { "embedding": { "filename": "phi_4_mini_embeddings.all.quant.onnx",
                         "run_on_cpu": true,
                         "inputs": ["input_ids"], "outputs": ["input_hidden_states"] } },
        { "prompt-processor": { "filename": "phi_4_mini_ctx.onnx_ctx.onnx", ... } },
        ...
      ]
    }
  }
}
```

When GenAI sees `run_on_cpu: true` on a stage, the only thing it changes is **which EP it appends** when building that stage's session — `CPUExecutionProvider` instead of the variant's selected EP. The file's `session_options` and `provider_options` from `variant.json` are still applied as-is: producers authoring a multi-file QNN variant with one CPU-bound stage already ship CPU-shaped (or empty) SO/PO for that file in `variant.json`.

This keeps the per-file CPU override as a GenAI-level concern, not a package-level one, and reuses an existing schema key — packages produced today already round-trip through this path without invention. A non-GenAI consumer that wants to pin specific files to CPU expresses it the same way in its own overlay key, or in whatever pipeline schema it owns.

### Backward Compatibility (Flat-Directory Models)

Models in the legacy flat-directory layout (a single directory with `genai_config.json`, ONNX files, tokenizer assets) are not packages. GenAI continues to support them through the existing `Model::Create(path)` path that does not go through the package API.

A flat-directory model can be wrapped into a single-component, single-variant package without code changes by:

1. Creating `<pkg>/configs/` and moving `genai_config.json` + tokenizer assets in.
2. Trimming `genai_config.json` per the [package-world rules](#genai_configjson-in-the-package-world).
3. Creating `<pkg>/<component>/` with a `metadata.json` that has one variant entry, and the variant directory with a `variant.json` listing the ONNX file(s) and any external data.

When opened, the consumer reads the variant's declared EP from the traversal accessors (or just registers the producer-documented EP), captures it into options, calls `SelectComponent`, and builds sessions exactly as it would for a multi-variant package. The flat-directory and package-world paths converge once a model is wrapped.

---

## Appendix B: Session Options Reference

The variant's per-file `session_options` block in `variant.json` is a flat string-to-string JSON object. ORT applies it through a single helper, `OrtApi::AddSessionOption(opts, key, value)`, that dispatches each key in one of two ways:

- **Known key** — the helper parses the string value into the right type and calls the dedicated `OrtSessionOptions` setter (e.g. `SetIntraOpNumThreads`). Parse failures return a non-OK status; the session is not created with a misconfigured value.
- **Unknown key** — the helper falls through to `AddSessionConfigEntry`, preserving the historical "opaque key-value bag" semantics.

This lets package authors set things like thread counts and graph-optimization levels from a static JSON manifest without the consumer having to special-case each one — and it gives every ORT-side consumer (model-package internals, future tools, eventually ORT-GenAI) a single chokepoint instead of each rebuilding its own typed mapping.

The canonical key names below match the [Python binding's `SessionOptions` property names](https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/python/onnxruntime_pybind_state.cc), which are the most established user-facing precedent in the codebase. Where ORT-GenAI's `Config::SessionOptions` struct uses a slightly different spelling (only `log_id` vs `logid`), we accept it as a documented alias.

### Known Session Options (Dispatched via `AddSessionOption`)

| Key | Value type | C API setter | Internal field |
|---|---|---|---|
| `intra_op_num_threads` | int | `SetIntraOpNumThreads` | `intra_op_param.thread_pool_size` |
| `inter_op_num_threads` | int | `SetInterOpNumThreads` | `inter_op_param.thread_pool_size` |
| `enable_cpu_mem_arena` | bool | `EnableCpuMemArena` / `DisableCpuMemArena` | `enable_cpu_mem_arena` |
| `enable_mem_pattern` | bool | `EnableMemPattern` / `DisableMemPattern` | `enable_mem_pattern` |
| `logid` (alias: `log_id`) | string | `SetSessionLogId` | `session_logid` |
| `log_severity_level` | int (0–4) | `SetSessionLogSeverityLevel` | `session_log_severity_level` |
| `log_verbosity_level` | int | `SetSessionLogVerbosityLevel` | `session_log_verbosity_level` |
| `enable_profiling` | string (path prefix) | `EnableProfiling(prefix)` / `DisableProfiling` | `enable_profiling` + profile-file prefix |
| `graph_optimization_level` | enum string | `SetSessionGraphOptimizationLevel` | `graph_optimization_level` |
| `optimized_model_filepath` | string (path) | `SetOptimizedModelFilePath` | `optimized_model_filepath` |
| `execution_mode` | enum string | `SetSessionExecutionMode` | `execution_mode` |
| `use_per_session_threads` | bool | `DisablePerSessionThreads` (one-way) | `use_per_session_threads` |
| `use_deterministic_compute` | bool | `SetDeterministicCompute` | `use_deterministic_compute` |

All other keys are passed through unchanged to `AddSessionConfigEntry`, so the existing reserved keys in `onnxruntime_session_options_config_keys.h` (e.g. `session.use_env_allocators`, `session.disable_prepacking`, etc.) keep working without being enumerated here.

### Value Encoding Rules

- **bool** — accepts `"0"` / `"1"` and `"true"` / `"false"` (case-insensitive). Any other value is a parse error. Encoders should emit `"0"` / `"1"` to match the existing reserved-key conventions in `onnxruntime_session_options_config_keys.h`.
- **int** — base-10 string. Negative values are accepted where the setter accepts them (e.g. `log_severity_level = -1` means "use default").
- **string** — passed through. Path-typed values are platform-native paths; on Windows ORT internally converts to wide.
- **`graph_optimization_level`** — one of `DISABLE_ALL`, `ENABLE_BASIC`, `ENABLE_EXTENDED`, `ENABLE_ALL` (case-insensitive). These map to `ORT_DISABLE_ALL` / `ORT_ENABLE_BASIC` / `ORT_ENABLE_EXTENDED` / `ORT_ENABLE_ALL`.
- **`execution_mode`** — one of `SEQUENTIAL`, `PARALLEL` (case-insensitive). Maps to `ORT_SEQUENTIAL` / `ORT_PARALLEL`.
- **`enable_profiling`** — non-empty string enables profiling with that path prefix (`EnableProfiling`). Empty string or absent key leaves profiling at its default (off). To explicitly disable after a prior enable, use `""`; the dispatcher calls `DisableProfiling` in that case.
- **`use_per_session_threads`** — `true` is a no-op (this is already the default). `false` calls `DisablePerSessionThreads`. There is no public C API to re-enable, so flipping back to `true` after disable returns an error.

### Excluded Setters

The following `OrtSessionOptions` setters are intentionally *not* part of the dispatcher because their argument shape doesn't fit a single string-keyed scalar:

- `AddFreeDimensionOverride` / `AddFreeDimensionOverrideByName` — two-arg (denotation/name + int64 value), list-valued. If variant manifests ever need to declare free-dimension overrides, add a sibling `free_dimensions` block to `variant.json` with its own structured shape rather than overloading `session_options`.
- `AddExternalInitializers` / `AddExternalInitializersFromFilesInMemory` — need `OrtValue` handles and memory buffers. External-initializer plumbing is owned by the package's `shared_files` mechanism and handled separately by `CreateSession` / multi-file consumers.
- `SetUserLoggingFunction` — function pointer.
- `SessionOptionsAppendExecutionProvider` — EP registration is handled out-of-band via the captured EPs in `ModelPackageOptions` and the variant's `provider_options` map (see [Session Creation](#5-session-creation)).
- `RegisterCustomOpsLibrary` — DLL/shared-library path with side effects; loading custom-op libraries from package metadata would be a security footgun.
- `enable_mem_reuse`, `execution_order`, `thread_pool_allow_spinning` — exist as struct fields but have no per-session public C API setter. They remain reachable via `AddSessionConfigEntry` if the existing reserved-key namespace covers them.

---
