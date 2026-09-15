// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

/// \file model_package.h
/// \brief Public C API for the ONNX Runtime Model Package library.
///
/// A model package is a directory with a top-level `manifest.json` that
/// declares a set of components; each component declares a set of variants;
/// each variant points at a directory containing the model files and may
/// carry executor-specific configuration under per-namespace
/// `executor_info` entries.
///
/// Error handling: functions that can fail return `ModelPackageStatus*`.
/// A `nullptr` return indicates success. Use `ModelPackageStatus_Message`,
/// `ModelPackageStatus_Code`, and `ModelPackageStatus_Release` to inspect and
/// release statuses.
///
/// Object lifetime: every `const char*` and every `const ModelPackageInfo*`
/// (and its sub-arrays) returned by this API is owned by the `ModelPackage`
/// handle and remains valid until the next mutation of that scope or until
/// the package is closed. Mutations invalidate cached pointers in the mutated
/// scope and its descendants; callers must re-read `ModelPackage_Info()`
/// after any mutation.

#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "model_package_api.h"

#ifdef __cplusplus
extern "C" {
#endif

// The library is consumed as source (compiled into each consumer's own binary),
// so the structs below have no binary boundary to maintain: there is no
// struct_size/ABI versioning. Compatibility with on-disk packages is governed
// solely by `schema_version` (see ModelPackageInfo).

// ─────────────────────────────────────────────────────────────────────────────
// Opaque handle
// ─────────────────────────────────────────────────────────────────────────────

typedef struct ModelPackage ModelPackage;

// ─────────────────────────────────────────────────────────────────────────────
// Status helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Get the error message from a status object. Returns NULL if `status` is NULL.
/// The returned string is owned by the status object.
MODEL_PACKAGE_API const char* ModelPackageStatus_Message(const ModelPackageStatus*);
/// Get the categorical error code. Returns `MODEL_PACKAGE_OK` when `status` is NULL.
MODEL_PACKAGE_API ModelPackageErrorCode ModelPackageStatus_Code(const ModelPackageStatus*);
/// Release a status object. Safe to call with NULL.
MODEL_PACKAGE_API void ModelPackageStatus_Release(ModelPackageStatus*);

// ─────────────────────────────────────────────────────────────────────────────
// Lifecycle
// ─────────────────────────────────────────────────────────────────────────────

typedef struct ModelPackageOpenOptions {
  bool allow_external_paths;   ///< default false; unlocks absolute paths and `..` segments
  bool follow_symlinks;        ///< default true
  bool strict_unknown_fields;  ///< default true; relax to round-trip newer schemas
} ModelPackageOpenOptions;

/// Open an existing model package directory. `opts` may be NULL for defaults.
MODEL_PACKAGE_API ModelPackageStatus* ModelPackage_Open(const char* package_root,
                                                        const ModelPackageOpenOptions* opts,
                                                        ModelPackage** out);

/// Create a new empty in-memory package for from-scratch authoring.
MODEL_PACKAGE_API ModelPackageStatus* ModelPackage_New(ModelPackage** out);

/// Release a ModelPackage handle and all its caches. Safe on NULL.
MODEL_PACKAGE_API void ModelPackage_Close(ModelPackage* pkg);

// ─────────────────────────────────────────────────────────────────────────────
// Data model — POD structs read from ModelPackage_Info()
// ─────────────────────────────────────────────────────────────────────────────

typedef struct ModelExecutorInfoEntry {
  const char* namespace_key;  ///< executor namespace name (e.g. "ort")
  const char* json;           ///< canonical JSON value as string (object, array, etc.)
} ModelExecutorInfoEntry;

typedef struct ModelVariantInfo {
  const char* name;
  /// Resolved absolute path to the variant's on-disk directory, or NULL when
  /// no directory has been declared and the default location does not exist.
  const char* variant_directory;
  const char* ep;                        ///< NULL when unset
  const char* device;                    ///< NULL when unset
  const char* compatibility_string;      ///< NULL when unset
  const char* additional_metadata_json;  ///< NULL when unset
  size_t num_executor_infos;
  const ModelExecutorInfoEntry* executor_infos;
} ModelVariantInfo;

typedef struct ModelComponentInfo {
  const char* name;
  const char* additional_metadata_json;  ///< NULL when unset
  size_t num_variants;
  const ModelVariantInfo* variants;
} ModelComponentInfo;

typedef struct ModelSharedAssetInfo {
  const char* uri;            ///< "sha256:<hex>"
  const char* resolved_path;  ///< absolute on-disk directory path
} ModelSharedAssetInfo;

typedef struct ModelPackageInfo {
  int64_t schema_version_major;          ///< parsed from on-disk "<major>.<minor>"; gates compatibility
  int64_t schema_version_minor;          ///< informational; indicates which optional fields may be present
  const char* package_name;              ///< NULL when unset
  const char* package_version;           ///< NULL when unset
  const char* description;               ///< NULL when unset
  const char* layout;                    ///< "portable" or "installed"
  const char* additional_metadata_json;  ///< NULL when unset
  size_t num_components;
  const ModelComponentInfo* components;
  size_t num_shared_assets;
  const ModelSharedAssetInfo* shared_assets;
} ModelPackageInfo;

/// Return the package-level info tree. Pointer is owned by the package and is
/// invalidated by any mutation.
MODEL_PACKAGE_API const ModelPackageInfo* ModelPackage_Info(const ModelPackage* pkg);

// ─────────────────────────────────────────────────────────────────────────────
// Convenience lookups
// ─────────────────────────────────────────────────────────────────────────────

/// Find a component by name. Returns NULL when not found.
MODEL_PACKAGE_API const ModelComponentInfo* ModelPackage_FindComponent(const ModelPackageInfo*,
                                                                       const char* name);
/// Find a variant within a component by name. Returns NULL when not found.
MODEL_PACKAGE_API const ModelVariantInfo* ModelComponentInfo_FindVariant(const ModelComponentInfo*,
                                                                         const char* name);
/// Find an executor_info entry by namespace. Returns NULL when not declared.
MODEL_PACKAGE_API const ModelExecutorInfoEntry* ModelVariantInfo_FindExecutorInfo(
    const ModelVariantInfo*, const char* namespace_key);

// ─────────────────────────────────────────────────────────────────────────────
// Round-trip JSON getters
// ─────────────────────────────────────────────────────────────────────────────

/// Get the canonical schema-shaped JSON for the named component. Preserves
/// fields unknown to this build. The returned pointer is owned by the package.
MODEL_PACKAGE_API ModelPackageStatus* ModelPackage_GetComponentJson(const ModelPackage*,
                                                                    const char* component_name,
                                                                    const char** out_json);

/// Get the canonical schema-shaped JSON for the named variant.
MODEL_PACKAGE_API ModelPackageStatus* ModelPackage_GetVariantJson(const ModelPackage*,
                                                                  const char* component_name,
                                                                  const char* variant_name,
                                                                  const char** out_json);

// ─────────────────────────────────────────────────────────────────────────────
// Asset resolution + hashing
// ─────────────────────────────────────────────────────────────────────────────

/// Resolve a `sha256:<hex>` URI to an on-disk directory. Errors with
/// `MODEL_PACKAGE_ERR_ASSET_MISSING` when not resolvable.
MODEL_PACKAGE_API ModelPackageStatus* ModelPackage_ResolveAssetUri(const ModelPackage*,
                                                                   const char* uri,
                                                                   const char** out_path);

/// Resolve a string reference using the model package's path resolution rules.
/// `input` may be:
///   - `sha256:<hex>`               -> shared-asset folder
///   - `sha256:<hex>/sub/path`      -> file or subdir inside a shared-asset folder
///                                     (sub/path is resolved with portable-mode
///                                     confinement under the asset folder: no
///                                     absolute, no `..`)
///   - relative path                -> resolved against `base_dir` (or
///                                     `package_root` when `base_dir == NULL`),
///                                     confined to `package_root` in portable layout
///   - absolute path / `..` segments -> only allowed in installed layout, or in
///                                     any layout when the package was opened with
///                                     `ModelPackageOpenOptions.allow_external_paths`
///
/// `must_exist` controls whether a missing target is `MODEL_PACKAGE_ERR_NOT_FOUND`
/// or the lexically-normalized path is returned anyway.
/// On success `*out_path` points to a NUL-terminated thread-local string; copy
/// it if you need it to outlive the next `ModelPackage_ResolveStringRef` call on
/// the same thread.
MODEL_PACKAGE_API ModelPackageStatus* ModelPackage_ResolveStringRef(const ModelPackage*,
                                                                    const char* base_dir,
                                                                    const char* input,
                                                                    bool must_exist,
                                                                    const char** out_path);

/// Compute the canonical `sha256:<hex>` URI for a directory. On success,
/// `*out_uri` is set to a NUL-terminated string owned by an internal
/// thread-local slot; the caller must copy if it must outlive the next call
/// on the same thread.
MODEL_PACKAGE_API ModelPackageStatus* ModelPackage_ComputeDirectoryHash(const char* source_dir,
                                                                        const char** out_uri);

// ─────────────────────────────────────────────────────────────────────────────
// Authoring — mutation API
// ─────────────────────────────────────────────────────────────────────────────
//
// Each mutation invalidates info pointers in the mutated scope and its
// descendants. Strict unknown-field rejection follows the open-time option
// `strict_unknown_fields` (default true).

/// Set or replace an inline component. `component_json` must be a JSON object
/// matching the component schema. An existing component with the same name is
/// replaced.
MODEL_PACKAGE_API ModelPackageStatus* ModelPackage_SetComponentInline(ModelPackage*,
                                                                      const char* name,
                                                                      const char* component_json);

/// Set or replace an external component. `path` is recorded in the manifest
/// (relative to package_root, or absolute in installed layout). If the file
/// exists, it is loaded; otherwise the component is initialized empty
/// (`{"variants": {}}`). `path` may be a directory (resolves to
/// `<dir>/component.json`).
MODEL_PACKAGE_API ModelPackageStatus* ModelPackage_SetComponentExternal(ModelPackage*,
                                                                        const char* name,
                                                                        const char* path);

/// Remove a component by name. No-op when the name is not present.
MODEL_PACKAGE_API ModelPackageStatus* ModelPackage_RemoveComponent(ModelPackage*, const char* name);

/// Upsert a variant inside a component. `variant_json` must be a JSON object
/// matching the variant schema. The library does not validate that
/// `variant_directory` exists on disk; executors are responsible for resolving
/// their own file references at load time.
MODEL_PACKAGE_API ModelPackageStatus* ModelPackage_SetVariant(ModelPackage*,
                                                              const char* component_name,
                                                              const char* variant_name,
                                                              const char* variant_json);

MODEL_PACKAGE_API ModelPackageStatus* ModelPackage_RemoveVariant(ModelPackage*,
                                                                 const char* component_name,
                                                                 const char* variant_name);

MODEL_PACKAGE_API ModelPackageStatus* ModelPackage_SetVariantExecutorInfoInline(ModelPackage*,
                                                                                const char* component,
                                                                                const char* variant,
                                                                                const char* ns,
                                                                                const char* info_json);

MODEL_PACKAGE_API ModelPackageStatus* ModelPackage_SetVariantExecutorInfoExternal(ModelPackage*,
                                                                                  const char* component,
                                                                                  const char* variant,
                                                                                  const char* ns,
                                                                                  const char* path);

MODEL_PACKAGE_API ModelPackageStatus* ModelPackage_RemoveVariantExecutorInfo(ModelPackage*,
                                                                             const char* component,
                                                                             const char* variant,
                                                                             const char* ns);

/// Add a content-addressed shared asset. When `expected_uri_or_null` is
/// non-NULL, the computed URI must match (reproducible-build check). With
/// `copy_in == false`, an override path is stored in the manifest; this is
/// rejected eagerly in portable layout. With `copy_in == true`, the source
/// directory is staged for copy at `_Commit` time.
///
/// `out_uri` is set to a NUL-terminated string owned by the package. The
/// pointer is only guaranteed to remain valid until the next mutation
/// (any ModelPackage_Set*, ModelPackage_Remove*, ModelPackage_AddSharedAsset,
/// or ModelPackage_Commit call), since those calls may rebuild the
/// shared-asset table or rehash the pending-copies map. Callers that need to
/// retain the URI must copy it into their own storage.
MODEL_PACKAGE_API ModelPackageStatus* ModelPackage_AddSharedAsset(ModelPackage*,
                                                                  const char* source_dir,
                                                                  const char* expected_uri_or_null,
                                                                  bool copy_in,
                                                                  const char** out_uri);

MODEL_PACKAGE_API ModelPackageStatus* ModelPackage_RemoveSharedAsset(ModelPackage*, const char* uri);

/// Set or clear package-level metadata. Any argument may be NULL to leave the
/// existing value untouched. Passing an empty string clears the field.
MODEL_PACKAGE_API ModelPackageStatus* ModelPackage_SetMetadata(ModelPackage*,
                                                               const char* name_or_null,
                                                               const char* version_or_null,
                                                               const char* description_or_null);

/// Set the layout. Valid values: "portable" or "installed".
MODEL_PACKAGE_API ModelPackageStatus* ModelPackage_SetLayout(ModelPackage*, const char* layout);

/// Set or clear `additional_metadata` at a given scope.
///   scope = "manifest"  — component_or_null and variant_or_null must be NULL
///   scope = "component" — component_or_null is required, variant_or_null is NULL
///   scope = "variant"   — component_or_null and variant_or_null are both required
/// `json_or_null == NULL` clears the field at that scope.
MODEL_PACKAGE_API ModelPackageStatus* ModelPackage_SetAdditionalMetadataJson(ModelPackage*,
                                                                             const char* scope,
                                                                             const char* component_or_null,
                                                                             const char* variant_or_null,
                                                                             const char* json_or_null);

// ─────────────────────────────────────────────────────────────────────────────
// Commit / Prune / Validate
// ─────────────────────────────────────────────────────────────────────────────

typedef enum {
  MODEL_PACKAGE_WRITE_PRESERVE = 0,  ///< each component/executor-info keeps its current shape
  MODEL_PACKAGE_WRITE_DENSE = 1,     ///< flatten all external components inline
} ModelPackageWriteMode;

/// Persist the in-memory model to disk. `dest_root_or_null == NULL` commits
/// in-place at `package_root`. Otherwise `dest_root` must be empty or
/// nonexistent and the entire package is materialized there (self-contained
/// "save as"). On a successful dest_root commit, the package's root is
/// updated to `dest_root` so subsequent in-place commits go there.
MODEL_PACKAGE_API ModelPackageStatus* ModelPackage_Commit(ModelPackage*,
                                                          const char* dest_root_or_null,
                                                          ModelPackageWriteMode mode);

/// Reclaim stale `.tmp.<suffix>` staging directories under
/// `<package_root>/shared_assets/` (left by interrupted commits, after a grace
/// window) and tracked orphan variant/component directories left behind by
/// RemoveVariant, RemoveComponent, SetVariant or SetComponentExternal. Only
/// paths registered through this API and inside `package_root` are touched.
/// Content-addressed shared-asset (`sha256-<hex>`) directories are never removed
/// — use ModelPackage_RemoveSharedAsset to reclaim those.
MODEL_PACKAGE_API ModelPackageStatus* ModelPackage_Prune(ModelPackage*);

typedef enum {
  MODEL_PACKAGE_VALIDATE_SCHEMA = 1 << 0,
  MODEL_PACKAGE_VALIDATE_PATHS = 1 << 1,
  MODEL_PACKAGE_VALIDATE_ASSET_REHASH = 1 << 2,
  MODEL_PACKAGE_VALIDATE_UNKNOWN_FIELDS = 1 << 3,
  MODEL_PACKAGE_VALIDATE_ALL = ~0,
} ModelPackageValidateFlags;

/// Run structural and reachability checks. `*out_report_json` is set to a
/// JSON string owned by the package describing findings:
///   `{"errors": [{"code": "...", "message": "..."}, ...],
///     "warnings": [...]}`
/// Returns non-NULL status when any error-level finding fired; warnings alone
/// still return success.
MODEL_PACKAGE_API ModelPackageStatus* ModelPackage_Validate(ModelPackage*,
                                                            int flags,
                                                            const char** out_report_json);

#ifdef __cplusplus
}  // extern "C"
#endif
