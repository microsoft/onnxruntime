# Model Package Library

A standalone C library for **reading, authoring, validating, and committing**
ONNX Runtime model packages. The library has no dependency on ONNX Runtime
itself, so any consumer (ORT, publisher tools, ...) can compile it in
without dragging in a session runtime. It is distributed and consumed as
**source** (see [Versioning and compatibility](#versioning-and-compatibility)).

The library owns three things:

1. The **on-disk layout** of a model package (directory + manifest + shared
   assets).
2. The **schema** of `manifest.json` and `component.json`, including the
   `executor_info` extension point.
3. The **resolution rules** for paths and content-addressed shared assets,
   including portable vs installed confinement.

It deliberately does **not** know about ONNX, execution providers, sessions,
or the JSON payload that lives under any `executor_info["<consumer>"]` slot.
Each consumer owns its own slot and parses it itself.

---

## On-disk layout

A package is a directory containing a top-level `manifest.json`. Components
live under the package root, either declared inline in the manifest or as
external `component.json` files. Variants are directories under their
component. Shared assets are content-addressed directories under
`shared_assets/`.

```
package_root/
├── manifest.json                       # required
├── decoder/                            # external component (directory)
│   ├── component.json                  # required when external
│   └── cpu/                            # variant_directory
│       ├── model.onnx
│       └── ort_info.json               # executor_info["ort"], external form
├── encoder/                            # inline component (no component.json)
│   └── cuda/
│       └── model.onnx
└── shared_assets/
    └── sha256-<64hex>/                 # content-addressed asset directory
        ├── tokenizer.json
        └── chat_template.jinja
```

- The package root must be a directory. A single file is **not** a package.
- A package has at least one component. A component has at least one variant.
- A variant always corresponds to a directory on disk (`variant_directory`).
  Files inside that directory are referenced by `executor_info` payloads, not
  by the manifest.
- `shared_assets/` is optional and only needs to exist if at least one
  shared asset is published.

### Portable vs installed layout

`manifest.layout` declares how the package may use paths:

- `"portable"` (default): every path is a `package_root`-relative POSIX path
  with no `..` segments and no absolute paths. The package is self-contained
  and movable. This is the format you ship.
- `"installed"`: absolute paths and `..` segments are allowed. This is for
  packages that have been "installed" onto a system that links shared assets
  to a system-wide cache, or that reference pre-existing files outside the
  package root.

The library enforces these rules at parse time. `ModelPackageOpenOptions.
allow_external_paths` can additionally relax portable confinement for read
operations, but the parser still rejects absolute paths inside the manifest
unless `layout == "installed"`.

---

## `manifest.json`

```jsonc
{
  "schema_version": "1.0",               // required, "<major>.<minor>" (major gates compat)
  "package_name":   "phi-4-mini",        // optional, free-form
  "package_version":"4.0.0",             // optional, free-form
  "description":    "Phi-4 mini reasoning model.",  // optional
  "layout":         "portable",          // optional: "portable" (default) | "installed"

  "components": {                        // required, at least one entry
    "decoder":  "decoder",               // external — path relative to package_root
    "encoder":  { /* inline component body */ }
  },

  "shared_assets": {                     // optional
    "sha256:<64hex>": "shared_assets/sha256-<64hex>"  // optional path override
  },

  "additional_metadata": { /* free-form */ }   // optional
}
```

Field reference:

| Field                | Type            | Required | Notes |
| -------------------- | --------------- | -------- | ----- |
| `schema_version`     | string          | yes      | `"<major>.<minor>"` (e.g. `"1.0"`). The library accepts any package whose **major** is in its supported range and any **minor**; a major outside the range is an `ERR_VERSION`. A bare integer is accepted as `"<major>.0"`. Major gates compatibility; minor tells consumers which optional fields may be present. |
| `package_name`       | string          | no       | Human label. Not used for resolution. |
| `package_version`    | string          | no       | Human label. Not used for resolution. |
| `description`        | string          | no       | Free-form. |
| `layout`             | string          | no       | `"portable"` (default) or `"installed"`. |
| `components`         | object          | yes      | Map of component name → component value. See below. |
| `shared_assets`      | object          | no       | Map of `sha256:<hex>` URI → path override (string). |
| `additional_metadata`| any JSON value  | no       | Opaque to this library. Round-tripped verbatim. |

By default the parser rejects unknown top-level keys (`strict_unknown_fields`,
on by default). Disable it via `ModelPackageOpenOptions` to round-trip
manifests authored against a newer schema.

### Components

The value under `components[name]` is either:

- **A string** — the path to an external component, resolved against
  `package_root`. The path may be:
  - **A directory.** The loader appends `component.json` and reads that
    file. The filename is fixed in this form (must be exactly
    `component.json`).
  - **A file.** Loaded directly. The filename is not enforced and may be
    anything (e.g. `decoder.json`). Useful when one directory holds
    multiple component definitions.
- **A JSON object** — an inline component body matching the
  [component schema](#componentjson) below.

The component's "directory" is:

- For an inline component, the package root itself.
- For an external component pointed at by a directory path, that directory.
- For an external component pointed at by a file path, the file's parent.

Variant paths in the component body are resolved against this directory.

### Shared assets

`shared_assets[uri]` is an **override**: it says "the asset with this URI
lives at this path", overriding the default convention of
`<package_root>/shared_assets/sha256-<hex>/`. Overrides are eagerly rejected
in portable layout when they would escape `package_root` (e.g. absolute paths,
`..` segments).

Variants reference shared assets only by embedding `sha256:<hex>[/sub/path]`
strings inside their `executor_info` payloads. Consumers resolve those
references through [`ModelPackage_ResolveStringRef`](#path-resolution-rules).
The library never parses `executor_info` payloads, so it has no manifest-level
list of which variant uses which asset.

---

## `component.json`

When a component is external, `component.json` is the file referenced from
the manifest. When inline, the same body is embedded directly in
`manifest.components[name]`.

```jsonc
{
  "component_name": "decoder",           // optional, descriptive only
  "variants": {                          // required, may be empty
    "cpu":  { /* variant body */ },
    "cuda": { /* variant body */ }
  },
  "additional_metadata": { /* free-form */ }   // optional
}
```

Field reference:

| Field                | Type   | Required | Notes |
| -------------------- | ------ | -------- | ----- |
| `component_name`     | string | no       | Sanity-checked as a string; not used for lookup. The map key in `components` wins. |
| `variants`           | object | yes      | Map of variant name → variant body. May be empty (placeholder component). |
| `additional_metadata`| any    | no       | Free-form. |

---

## Variant body

A variant binds a single (EP, device, compatibility) triple to a single
on-disk directory plus zero or more per-consumer `executor_info` payloads.

```jsonc
{
  "variant_directory":    "cuda",                          // optional — defaults to variant name
  "ep":                   "CUDAExecutionProvider",          // optional
  "device":               "gpu",                            // optional ("cpu" | "gpu" | "npu")
  "compatibility_string": "<EP-defined opaque token>",      // optional, opaque to library
  "executor_info": {                                        // optional
    "ort":   "ort_info.json",                               // string → external file
    "other": { "filename": "model.onnx" }                    // object → inline JSON
  },
  "additional_metadata": { /* free-form */ }                 // optional
}
```

Field reference:

| Field                  | Type             | Required | Notes |
| ---------------------- | ---------------- | -------- | ----- |
| `variant_directory`    | string           | no       | Path relative to the component directory. Defaults to the variant name. If declared but missing on disk, parse fails. |
| `ep`                   | string           | no       | Single ONNX Runtime EP name (e.g. `CPUExecutionProvider`). |
| `device`               | string           | no       | Lower-case `cpu` / `gpu` / `npu`. ORT uses this for variant selection. |
| `compatibility_string` | string           | no       | Opaque to the library. ORT hands it to the EP's `ValidateCompiledModelCompatibilityInfo` callback. |
| `executor_info`        | object           | no       | Map of consumer namespace → string (external file) or object (inline JSON). |
| `additional_metadata`  | any              | no       | Free-form. |

#### `variant_directory`

- Always interpreted as a directory.
- Resolved against the **component directory** (not the package root).
- The library does not validate the directory's contents; consumers resolve
  their own file references relative to it.

#### `executor_info`

This is the extension point that lets ORT and any future consumer share a
package without colliding. Keys are consumer namespaces; values are either:

- **A string** — a path to a JSON file. Resolved against the variant
  directory. The file must exist (in strict mode) and parse as JSON.
- **An inline JSON object** — embedded directly in the manifest.

The library round-trips the payload but never interprets it. See
[`onnxruntime/core/session/model_package/README.md`](../onnxruntime/core/session/model_package/README.md)
for the `"ort"` namespace schema.

Consumers can embed `sha256:<hex>[/sub/path]` references inside their
`executor_info` payload and resolve them through
`ModelPackage_ResolveStringRef`. The library does not maintain a per-variant
list of consumed assets; see [Shared assets](#shared-assets) for how URIs
enter the resolvable set.

---

## Shared assets

Shared assets are **directories** identified by a content hash. Two packages
that ship the same tokenizer will reuse the same asset directory on disk in
an installed layout, dedup-ing storage and downloads.

### Canonical asset URI

`ModelPackage_ComputeDirectoryHash(source_dir)` computes the canonical URI:

1. Walk `source_dir` recursively, collecting regular files. Empty
   subdirectories are ignored.
2. Reject symlinks (portability hazard).
3. For each file, compute `sha256(file_bytes)` → per-file hex digest.
4. Build a manifest text of lines `<sha256_hex>  <relative_posix_path>\n`
   sorted lexicographically by path. Paths use forward slashes, no leading
   `./`. Non-ASCII paths must be NFC-normalized by the caller.
5. `asset_uri = "sha256:" + sha256(manifest_text)`, lowercase hex.

The scheme hashes **both** file contents and file names, so renaming a file
inside an asset changes the URI. The on-disk directory name follows the
convention `sha256-<hex>` (dash, not colon) to keep the path filesystem-safe.

### Default location

`<package_root>/shared_assets/sha256-<hex>/`. Override per-asset by adding an
entry to `manifest.shared_assets`.

### How URIs enter the resolvable set

At Open time the library populates the resolvable shared-asset table from
three sources, in order. Within each tier an already-seen URI is skipped:

1. **Manifest overrides.** Every entry under `manifest.shared_assets` lands
   first. These can also point at non-default paths (subject to the
   layout's portability rules).
2. **On-disk discovery.** The library lists `<package_root>/shared_assets/`
   and admits each `sha256-<hex>` subdirectory it finds (sorted
   lexicographically). The resolved path is the default
   `<package_root>/shared_assets/sha256-<hex>/`. A missing `shared_assets/`
   directory is fine.
3. **Pending authoring stages.** Any `copy_in=true` source registered via
   `ModelPackage_AddSharedAsset` is surfaced at its staged source path so
   `ResolveStringRef` works before `Commit`.

This means the manifest does not need to enumerate the assets that ship in
the conventional `shared_assets/` directory. The override list is only
needed when an asset lives outside the default convention.

### Adding a shared asset programmatically

```c
const char* uri = NULL;
ModelPackageStatus* st = ModelPackage_AddSharedAsset(
    pkg,
    "/path/to/tokenizer",     // source_dir
    NULL,                     // expected_uri_or_null (reproducible-build check)
    /*copy_in=*/true,         // stage for copy at Commit time
    &uri);
```

`copy_in == false` stores an override path in the manifest and is rejected
eagerly in portable layout (the path is unlikely to be portable). `copy_in
== true` stages the source for copy when `ModelPackage_Commit()` runs.

---

## Path resolution rules

`ModelPackage_ResolveStringRef(pkg, base_dir, input, must_exist, &out)` is
the canonical path resolver. It accepts:

| Input form                  | Resolution |
| --------------------------- | ---------- |
| `sha256:<hex>`              | Returns the on-disk directory for that shared asset. Error if the asset isn't registered. |
| `sha256:<hex>/sub/path`     | Returns `<asset_dir>/sub/path`. The subpath is confined to the asset folder (no absolute, no `..`). |
| Relative path               | Resolved against `base_dir` (or `package_root` when `base_dir` is NULL). |
| Absolute path / `..` segments | Allowed only in `installed` layout or when the package was opened with `allow_external_paths = true`. |

In portable layout the resolver enforces that the resolved path stays
underneath `package_root`. Symlinks are followed by default
(`follow_symlinks`).

`out_path` is a NUL-terminated thread-local pointer; copy it if it must
outlive the next `ResolveStringRef` call on the same thread.

---

## C API quick tour

All public entry points are declared in `include/model_package.h`. Reading a
package and walking the info tree:

```c
#include "model_package.h"

ModelPackage* pkg = NULL;
if (ModelPackageStatus* st = ModelPackage_Open("/path/to/pkg", NULL, &pkg)) {
    fprintf(stderr, "open failed: %s\n", ModelPackageStatus_Message(st));
    ModelPackageStatus_Release(st);
    return 1;
}

const ModelPackageInfo* info = ModelPackage_Info(pkg);
printf("schema=%lld.%lld layout=%s\n",
       (long long)info->schema_version_major, (long long)info->schema_version_minor, info->layout);
for (size_t i = 0; i < info->num_components; ++i) {
    const ModelComponentInfo* c = &info->components[i];
    printf("component %s (%zu variants)\n", c->name, c->num_variants);
    for (size_t v = 0; v < c->num_variants; ++v) {
        const ModelVariantInfo* var = &c->variants[v];
        printf("  variant %s  dir=%s  ep=%s\n",
               var->name,
               var->variant_directory ? var->variant_directory : "(unset)",
               var->ep ? var->ep : "(unset)");
        for (size_t e = 0; e < var->num_executor_infos; ++e) {
            const ModelExecutorInfoEntry* ei = &var->executor_infos[e];
            printf("    executor_info[%s] = %s\n", ei->namespace_key, ei->json);
        }
    }
}

ModelPackage_Close(pkg);
```

Authoring a new package from scratch:

```c
ModelPackage* pkg = NULL;
ModelPackage_New(&pkg);
ModelPackage_SetMetadata(pkg, "phi-4-mini", "4.0.0", "Phi-4 mini.");

ModelPackage_SetComponentInline(pkg, "decoder", "{\"variants\": {}}");
ModelPackage_SetVariant(pkg, "decoder", "cpu",
    "{\"variant_directory\":\"decoder/cpu\","
    " \"ep\":\"CPUExecutionProvider\","
    " \"device\":\"cpu\"}");
ModelPackage_SetVariantExecutorInfoInline(
    pkg, "decoder", "cpu", "ort", "{\"model_file\":\"model.onnx\"}");

const char* asset_uri = NULL;
ModelPackage_AddSharedAsset(pkg, "/src/tokenizer", NULL, /*copy_in=*/true, &asset_uri);
// asset_uri is owned by pkg; copy it if you need it past the next mutation.

ModelPackage_Commit(pkg, "/path/to/new_pkg", MODEL_PACKAGE_WRITE_PRESERVE);
ModelPackage_Close(pkg);
```

### Lifetime contract

Every `const char*` and every `const ModelPackageInfo*` (plus sub-arrays)
returned by the read API is owned by the `ModelPackage` handle and remains
valid **until the next mutation of that scope** or until
`ModelPackage_Close()`. Any `Set*` / `Remove*` / `Add*` / `Commit` call
invalidates cached pointers in the mutated scope; re-read `Info()` after
mutating.

`ModelPackage_AddSharedAsset`'s `out_uri` follows the same "valid until next
mutation" rule.

`ModelPackage_ResolveStringRef` and `ModelPackage_ComputeDirectoryHash`
return pointers into a per-thread scratch slot; copy before the next call on
the same thread.

### Commit modes

`ModelPackage_Commit(pkg, dest, mode)`:

- `dest == NULL` → in-place commit at `package_root`.
- `dest != NULL` → write a self-contained "save as". `dest` must be empty or
  nonexistent. On success the package's root is updated to `dest`, so
  subsequent in-place commits go there.

`mode`:

- `MODEL_PACKAGE_WRITE_PRESERVE` (default) — each component and
  `executor_info` entry keeps its current inline-or-external shape.
- `MODEL_PACKAGE_WRITE_DENSE` — flatten every external component back inline
  into `manifest.json`. Useful for single-file authoring inspection.

### Prune

`ModelPackage_Prune(pkg)` reclaims storage that the library itself manages:

- Tracked orphan variant and component directories left behind by
  `RemoveVariant`, `RemoveComponent`, `SetVariant`, or
  `SetComponentExternal`.
- Stale `.tmp.<suffix>` staging directories from interrupted commits, after
  a short grace window.

`Prune` deliberately never removes `shared_assets/sha256-<hex>/` directories.
Consumers freely embed `sha256:` references inside their own `executor_info`
payloads, and the library cannot prove an asset is unused without parsing
every consumer's namespace. Use `ModelPackage_RemoveSharedAsset(uri)` to
delete a shared asset explicitly when the caller knows it is unreferenced.

Only paths registered through this API and strictly inside `package_root`
are touched.

### Validate

`ModelPackage_Validate(pkg, flags, &report_json)` runs a configurable set of
structural checks and returns a JSON report
`{"errors": [...], "warnings": [...]}`:

| Flag                                    | Checks |
| --------------------------------------- | ------ |
| `MODEL_PACKAGE_VALIDATE_SCHEMA`         | Required keys, types, value ranges. |
| `MODEL_PACKAGE_VALIDATE_PATHS`          | Every recorded path resolves under the configured layout. |
| `MODEL_PACKAGE_VALIDATE_ASSET_REHASH`   | Recompute every asset directory hash and compare to its URI (slow). |
| `MODEL_PACKAGE_VALIDATE_UNKNOWN_FIELDS` | Surface unknown JSON fields as warnings. |
| `MODEL_PACKAGE_VALIDATE_ALL`            | All of the above. |

Errors cause a non-NULL status return; warnings alone return success.

---

## Versioning and compatibility

### Distributed as source

The library is meant to be **vendored and compiled into each consumer's own
binary** (ORT, publisher tooling, third-party loaders). No prebuilt shared
library (`.so`/`.dll`) is published as the supported interface.

A direct consequence is that the public POD structs in `model_package.h` have
**no binary boundary** to defend: within any single build there is exactly one
definition of every struct, so there is nothing for two separately-compiled
artifacts to disagree about. The library therefore carries **none** of the usual
ABI machinery — no per-struct `struct_size`/`cbSize`, no `abi_version`, no
library SOVERSION, and no offset `static_assert`s. Collections are exposed as
plain array members (`components`/`num_components`, `variants`/`num_variants`,
…) rather than count+index accessors, since accessors only earn their keep when
the library owns the struct stride across a binary boundary.

The **only** compatibility contract is the on-disk data format, expressed by
`schema_version`. Everything a consumer needs to know about which fields and
objects a package may contain follows from that one value.

### `schema_version`

`schema_version` is a `"<major>.<minor>"` string in `manifest.json` (a bare
integer `N` is accepted and treated as `N.0`). It is parsed into
`ModelPackageInfo.schema_version_major` and `schema_version_minor`.

- **major** — the data contract. Incremented only for a **breaking** change
  (a field removed, renamed, retyped, or given new semantics). A consumer that
  understands major *N* can read any `N.x` package.
- **minor** — additive evolution within a major. Incremented when a new
  **optional** field or object is added. It never removes or reinterprets
  anything, so it is fully backward- and forward-compatible within the major.

Consumers should branch **solely on `schema_version_major` / `schema_version_minor`**
to decide which optional fields a package may carry — not on the presence or
absence of individual fields, and never on any library version.

### What the parser enforces

Each build declares the majors it understands as a closed range
(`kMinSupportedSchemaMajor … kMaxSupportedSchemaMajor` in `manifest_parser.cc`)
plus the highest minor it authored (`kMaxKnownSchemaMinor`):

- **Unsupported major** → `ModelPackage_Open` fails with
  `MODEL_PACKAGE_ERR_VERSION`. A consumer never silently misreads a package
  whose contract it does not understand.
- **Any minor is accepted.** When the minor is **newer** than this build knows
  (`minor > kMaxKnownSchemaMinor`), unknown-field strictness is relaxed for that
  package so the additive fields a newer authoring tool wrote are **tolerated**
  (read through, preserved on round-trip via the JSON getters) instead of
  rejected. An older library can therefore load a newer-minor package and ignore
  the fields it does not recognize.

### Supporting a major version bump

When a breaking change requires a new major, deployed packages do **not** have to
be rewritten and consumers do **not** have to upgrade in lockstep. The library is
designed to support a **range** of majors simultaneously:

1. Bump `kMaxSupportedSchemaMajor` and add the new major's parse/serialize path,
   keeping the existing major's path in place. The supported range now spans both.
2. Existing `N.x` packages keep loading unchanged through the old path; new
   `(N+1).x` packages load through the new path.
3. Consumers branch on `schema_version_major` to pick the field set they read.
   Code that only supports major *N* simply declines `(N+1).x` packages (the open
   call returns `MODEL_PACKAGE_ERR_VERSION` for it) rather than misreading them.
4. A major is dropped from the supported range only when its packages are no
   longer in circulation — an explicit, opt-in deprecation, never an implicit
   break.

This keeps already-published packages valid for as long as the library advertises
their major, which is the backward-compatibility guarantee external publishers
depend on.

### How a major bump maps onto the structs

A natural question is how a single C struct can represent two majors with
different fields. It can't — and it never has to, because **there is only one
struct definition in any given build**. The "old major" exists only as JSON on
disk; it is never a second C type in the consumer's binary. Since the library is
compiled from source, every consumer compiles exactly one definition of
`ModelPackageInfo`/`ModelVariantInfo`/etc. — the current one. Reconciling an
old-major package with that one definition is a **parse-time** job, not a
struct-layout one.

The single struct is the **superset / newest** shape, and divergence between
majors is absorbed in three places:

1. **Additive differences (common).** A field a new major added is present in the
   struct and is simply `NULL`/`0`/empty when an older-major package lacks it —
   the same mechanism as a minor bump. The consumer treats absence as "not
   provided".

2. **Parse-time normalization (preferred).** When a new major is added, its
   parser path is added alongside the existing one, and **both populate the same
   struct**. An older-major package is mapped up to the current in-memory model
   (defaults filled, renamed fields mapped to their current names) before the
   consumer sees it, so reads are uniform. `schema_version_major` then records the
   *source* contract — useful for write-back and provenance — rather than
   selecting a layout.

3. **Non-migratable changes (rare).** A field whose *type* changes, or one
   removed with no equivalent, cannot reuse the same name (C gives one field one
   type). Add a new field for the new representation, populate the old field only
   for old-major packages and the new field only for new-major packages, and let
   the consumer branch on `schema_version_major`:

   ```c
   // e.g. major 1 stored a single compatibility string; major 2 stores a list
   const char* compatibility_string;    // set when schema_version_major == 1
   const char* const* compatibilities;  // set when schema_version_major == 2
   size_t num_compatibilities;
   ```

**Escape hatch.** If a major bump is sweeping enough that the superset becomes
unwieldy, the standard move is **per-major typed structs** (e.g. a
`ModelPackageInfoV2` returned by a versioned accessor) — a deliberate API
expansion reserved for a wholesale redesign, not the default. In practice: prefer
normalizing old majors up to the newest struct at parse time; fall back to extra
nullable fields plus `schema_version_major` branching only when a change cannot be
auto-migrated.

---

## What the library deliberately does NOT do

- **Variant selection.** Picking which variant best matches the EPs the
  caller has available requires EP factory introspection and is owned by the
  executor. ORT's selector lives in
  `onnxruntime/core/session/model_package/` and uses each EP's
  `ValidateCompiledModelCompatibilityInfo` callback.
- **Session creation.** Building an `OrtSession` is ORT's job.
- **Interpreting `executor_info` payloads.** Each consumer namespace owns
  its own slot. The library only validates that values are either strings
  (paths) or objects.
- **Interpreting `compatibility_string`.** The format is owned by the EP
  declared in `ep`. The library never parses it.

---

## Building

```bash
cmake -B build -S . [-DMODEL_PACKAGE_BUILD_TESTS=ON]
cmake --build build -j
ctest --test-dir build --output-on-failure   # requires BUILD_TESTS=ON
```

CMake options:

- `MODEL_PACKAGE_BUILD_SHARED` (default `ON`) — shared vs static.
- `MODEL_PACKAGE_BUILD_TESTS` (default `OFF`) — build the unit-test
  executables (`test_asset_hashing`, `test_inspection`, `test_authoring`,
  `test_commit`).

The only build-time dependency is a vendored copy of nlohmann/json (header
only).

---

## See also

- `onnxruntime/core/session/model_package/README.md` — how ORT consumes this
  library and the `executor_info["ort"]` schema.
- `model_package_redesign.md` in the `archive` repo — original design
  rationale (extension fields, content addressing, portable vs installed,
  shared-asset overrides).
