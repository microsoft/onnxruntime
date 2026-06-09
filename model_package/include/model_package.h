// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

/// \file model_package.h
/// \brief Public C API for the ONNX Runtime Model Package library.
///
/// This is the new API per model_package_redesign.md. The legacy
/// model_package_api.h coexists during the in-progress redesign.
///
/// Error handling: functions that can fail return `ModelPackageStatus*`.
/// `nullptr` means success. Use `ModelPackageStatus_Message`,
/// `ModelPackageStatus_Code`, and `ModelPackageStatus_Release` from the legacy
/// header to inspect and release statuses; the type is shared.

#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "model_package_api.h"  // for MODEL_PACKAGE_API, ModelPackageStatus, ModelPackageErrorCode

#ifdef __cplusplus
extern "C" {
#endif

// ─────────────────────────────────────────────────────────────────────────────
// Opaque handles
// ─────────────────────────────────────────────────────────────────────────────

typedef struct ModelPackage   ModelPackage;
typedef struct ModelComponent ModelComponent;
typedef struct ModelVariant   ModelVariant;

// ─────────────────────────────────────────────────────────────────────────────
// Status helpers (alias names matching §7.1)
// ─────────────────────────────────────────────────────────────────────────────

/// Same as ModelPackage_GetErrorMessage. Provided under the §7.1 name.
MODEL_PACKAGE_API const char*           ModelPackageStatus_Message(const ModelPackageStatus*);
/// Same as ModelPackage_GetErrorCode. Provided under the §7.1 name.
MODEL_PACKAGE_API ModelPackageErrorCode ModelPackageStatus_Code(const ModelPackageStatus*);
/// Same as ModelPackage_ReleaseStatus. Provided under the §7.1 name.
MODEL_PACKAGE_API void                  ModelPackageStatus_Release(ModelPackageStatus*);

// ─────────────────────────────────────────────────────────────────────────────
// Lifecycle
// ─────────────────────────────────────────────────────────────────────────────

typedef struct ModelPackageOpenOptions {
  size_t struct_size;          ///< sizeof(ModelPackageOpenOptions)
  int    abi_version;          ///< 1
  bool   allow_external_paths; ///< default false; unlocks absolute paths + `..` segments
  bool   follow_symlinks;      ///< default true
  bool   strict_unknown_fields;///< default true; relax to round-trip newer schemas
} ModelPackageOpenOptions;

/// Open an existing model package directory.
/// `opts` may be NULL to use defaults.
MODEL_PACKAGE_API ModelPackageStatus* ModelPackage_Open(const char* package_root,
                                                        const ModelPackageOpenOptions* opts,
                                                        ModelPackage** out);

/// Create a new empty in-memory package (for from-scratch authoring).
/// Not yet implemented in Phase 1; reserved.
MODEL_PACKAGE_API ModelPackageStatus* ModelPackage_New(ModelPackage** out);

/// Release a ModelPackage handle and all its caches. Safe on NULL.
MODEL_PACKAGE_API void ModelPackage_Close(ModelPackage* pkg);

// ─────────────────────────────────────────────────────────────────────────────
// Package-level inspection
// ─────────────────────────────────────────────────────────────────────────────

typedef struct ModelPackageInfo {
  size_t      struct_size;
  int         abi_version;
  int64_t     schema_version;
  const char* package_name;             ///< may be NULL
  const char* package_version;          ///< may be NULL
  const char* description;              ///< may be NULL
  const char* layout;                   ///< "portable" | "installed"
  const char* additional_metadata_json; ///< may be NULL
  size_t      num_components;
  size_t      num_shared_assets;
} ModelPackageInfo;

/// Return a pointer to the package-level info. Owned by the package; valid
/// until the package is closed (Phase 1) or its manifest scope is mutated.
MODEL_PACKAGE_API const ModelPackageInfo* ModelPackage_Info(const ModelPackage* pkg);

// ─────────────────────────────────────────────────────────────────────────────
// Components
// ─────────────────────────────────────────────────────────────────────────────

/// Get a component by 0-based declaration order. NULL on out-of-range.
MODEL_PACKAGE_API const ModelComponent* ModelPackage_GetComponent(const ModelPackage*, size_t idx);
/// Find a component by name. NULL on not-found.
MODEL_PACKAGE_API const ModelComponent* ModelPackage_FindComponent(const ModelPackage*, const char* name);

MODEL_PACKAGE_API const char* ModelComponent_Name(const ModelComponent*);
MODEL_PACKAGE_API size_t      ModelComponent_VariantCount(const ModelComponent*);
MODEL_PACKAGE_API const ModelVariant* ModelComponent_GetVariant(const ModelComponent*, size_t idx);
MODEL_PACKAGE_API const ModelVariant* ModelComponent_FindVariant(const ModelComponent*, const char* name);

// ─────────────────────────────────────────────────────────────────────────────
// Variants
// ─────────────────────────────────────────────────────────────────────────────

MODEL_PACKAGE_API const char* ModelVariant_Name(const ModelVariant*);
/// NULL if the variant did not declare an `ep` field.
MODEL_PACKAGE_API const char* ModelVariant_EpName(const ModelVariant*);
/// NULL if the variant did not declare a `device` field.
MODEL_PACKAGE_API const char* ModelVariant_Device(const ModelVariant*);
/// NULL if the variant did not declare `compatibility_string`.
MODEL_PACKAGE_API const char* ModelVariant_CompatibilityString(const ModelVariant*);

/// Resolve `variant_directory` to an absolute on-disk path. Errors with
/// MODEL_PACKAGE_ERR_NOT_FOUND if the directory does not exist on disk.
MODEL_PACKAGE_API ModelPackageStatus* ModelVariant_ResolveDirectoryPath(const ModelVariant*,
                                                                       const char** out_path);

/// Get a specific executor-info namespace's JSON for this variant. Sets
/// *out_json to NULL (and returns nullptr) when the namespace is not declared
/// on this variant — that is not treated as an error.
MODEL_PACKAGE_API ModelPackageStatus* ModelVariant_GetExecutorInfoJson(const ModelVariant*,
                                                                      const char* namespace_,
                                                                      const char** out_json);

/// Number of entries in the variant's declared `uses_assets` list.
MODEL_PACKAGE_API size_t      ModelVariant_UsedAssetCount(const ModelVariant*);
/// Get the i-th entry of `uses_assets`. NULL on out-of-range.
MODEL_PACKAGE_API const char* ModelVariant_UsedAssetUri(const ModelVariant*, size_t idx);

// ─────────────────────────────────────────────────────────────────────────────
// Shared assets
// ─────────────────────────────────────────────────────────────────────────────

typedef struct ModelSharedAsset {
  size_t      struct_size;
  int         abi_version;
  const char* uri;            ///< "sha256:<hex>"
  const char* resolved_path;  ///< absolute on-disk directory path
} ModelSharedAsset;

MODEL_PACKAGE_API const ModelSharedAsset* ModelPackage_GetSharedAsset(const ModelPackage*, size_t idx);

/// Resolve a `sha256:<hex>` URI to an on-disk directory. Errors with
/// MODEL_PACKAGE_ERR_ASSET_MISSING if not resolvable.
MODEL_PACKAGE_API ModelPackageStatus* ModelPackage_ResolveAssetUri(const ModelPackage*,
                                                                  const char* uri,
                                                                  const char** out_path);

// ─────────────────────────────────────────────────────────────────────────────
// Round-trip JSON getters and additional_metadata accessors
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

/// Manifest-scope additional_metadata. NULL when absent.
MODEL_PACKAGE_API const char* ModelPackage_AdditionalMetadataJson(const ModelPackage*);
/// Component-scope additional_metadata. NULL when absent.
MODEL_PACKAGE_API const char* ModelComponent_AdditionalMetadataJson(const ModelComponent*);
/// Variant-scope additional_metadata. NULL when absent.
MODEL_PACKAGE_API const char* ModelVariant_AdditionalMetadataJson(const ModelVariant*);

// ─────────────────────────────────────────────────────────────────────────────
// Shared asset hashing utility
// ─────────────────────────────────────────────────────────────────────────────

/// Compute the canonical sha256:<hex> URI for a directory per §4.3.1.
/// On success, *out_uri is set to a NUL-terminated string owned by an internal
/// per-call slot; the caller must copy if it needs to outlive the next call.
/// (Phase 2: the slot is thread-local so a single thread's repeated calls each
///  invalidate the previous return.)
MODEL_PACKAGE_API ModelPackageStatus* ModelPackage_ComputeDirectoryHash(const char* source_dir,
                                                                       const char** out_uri);

// ─────────────────────────────────────────────────────────────────────────────
// Authoring — mutation API (Phase 3)
// ─────────────────────────────────────────────────────────────────────────────
//
// All mutations follow the §7.2 pointer-invalidation contract: a mutation on
// entity X invalidates pointers into X and its descendants. Callers must
// re-fetch handles within X's subtree after mutating it.
//
// Strict unknown-field rejection follows the open option `strict_unknown_fields`
// (default true). Newly created packages from ModelPackage_New default to strict.

/// Set or replace an inline component. `component_json` must be a JSON object
/// matching the §5.2 schema. Existing component with the same name is replaced.
MODEL_PACKAGE_API ModelPackageStatus* ModelPackage_SetComponentInline(ModelPackage*,
                                                                     const char* name,
                                                                     const char* component_json);

/// Set or replace an external component. `path` is recorded in the manifest
/// (relative to package_root, or absolute in installed layout). If the file
/// exists, it is loaded; otherwise the component is initialized empty
/// ({"variants": {}}). The path is library-owned until removed.
/// `path` may be a directory (resolves to `<dir>/component.json`).
MODEL_PACKAGE_API ModelPackageStatus* ModelPackage_SetComponentExternal(ModelPackage*,
                                                                       const char* name,
                                                                       const char* path);

/// Remove a component by name. No-op on missing name.
MODEL_PACKAGE_API ModelPackageStatus* ModelPackage_RemoveComponent(ModelPackage*, const char* name);

/// Upsert a variant inside a component. `variant_json` must be a JSON object
/// matching the §5.2 variant schema. Errors with ERR_STATE when the new variant
/// declares any inline executor_info but has no resolvable variant_directory.
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
                                                                               const char* namespace_,
                                                                               const char* info_json);

MODEL_PACKAGE_API ModelPackageStatus* ModelPackage_SetVariantExecutorInfoExternal(ModelPackage*,
                                                                                 const char* component,
                                                                                 const char* variant,
                                                                                 const char* namespace_,
                                                                                 const char* path);

MODEL_PACKAGE_API ModelPackageStatus* ModelPackage_RemoveVariantExecutorInfo(ModelPackage*,
                                                                            const char* component,
                                                                            const char* variant,
                                                                            const char* namespace_);

/// Add a content-addressed shared asset. If `expected_uri_or_null` is non-NULL,
/// the computed URI must match it (reproducible-build check). With
/// `copy_in=false`, an override path is stored in the manifest; this is
/// rejected eagerly in portable layout. With `copy_in=true`, the source
/// directory is staged for copy at _Commit time.
/// `out_uri` is set to a NUL-terminated string owned by the package; remains
/// valid until the asset is removed or the package is closed.
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

/// Set layout. Valid values: "portable" or "installed".
MODEL_PACKAGE_API ModelPackageStatus* ModelPackage_SetLayout(ModelPackage*, const char* layout);

/// Set or clear `additional_metadata` at a given scope.
/// scope: "manifest" (component and variant must be NULL),
///        "component" (component required, variant NULL),
///        "variant" (component and variant required).
/// `json_or_null = NULL` clears the field at that scope.
MODEL_PACKAGE_API ModelPackageStatus* ModelPackage_SetAdditionalMetadataJson(ModelPackage*,
                                                                            const char* scope,
                                                                            const char* component_or_null,
                                                                            const char* variant_or_null,
                                                                            const char* json_or_null);

// ─────────────────────────────────────────────────────────────────────────────
// Commit / Vacuum / Validate (Phase 4)
// ─────────────────────────────────────────────────────────────────────────────

typedef enum {
  MODEL_PACKAGE_WRITE_PRESERVE = 0, ///< each component/executor-info keeps current shape
  MODEL_PACKAGE_WRITE_DENSE    = 1, ///< flatten all external components inline
} ModelPackageWriteMode;

/// Persist the in-memory model to disk. `dest_root_or_null = NULL` commits
/// in-place at `package_root`. Otherwise `dest_root` must be empty or
/// nonexistent; the entire package is materialized there (self-contained "save
/// as"). On a successful dest_root commit, `package_root` is updated to
/// `dest_root` so subsequent in-place commits go to the new location.
MODEL_PACKAGE_API ModelPackageStatus* ModelPackage_Commit(ModelPackage*,
                                                          const char* dest_root_or_null,
                                                          ModelPackageWriteMode mode);

/// Reclaim files under `<package_root>/shared_assets/` that are no longer
/// reachable from the current manifest. Files outside `<package_root>` are
/// never touched per §4.2.
MODEL_PACKAGE_API ModelPackageStatus* ModelPackage_Vacuum(ModelPackage*);

typedef enum {
  MODEL_PACKAGE_VALIDATE_SCHEMA           = 1 << 0,
  MODEL_PACKAGE_VALIDATE_PATHS            = 1 << 1,
  MODEL_PACKAGE_VALIDATE_ASSET_REACH      = 1 << 2,
  MODEL_PACKAGE_VALIDATE_ASSET_REHASH     = 1 << 3,
  MODEL_PACKAGE_VALIDATE_UNKNOWN_FIELDS   = 1 << 4,
  MODEL_PACKAGE_VALIDATE_ALL              = ~0,
} ModelPackageValidateFlags;

/// Run structural and reachability checks. `*out_report_json` is set to a
/// JSON string owned by the package describing findings:
///   `{"errors": [{"code": "...", "message": "..."}, ...],
///     "warnings": [...]}` — empty arrays when nothing was found at that level.
/// Returns non-NULL status when any error-level finding fired; warnings alone
/// still return success.
MODEL_PACKAGE_API ModelPackageStatus* ModelPackage_Validate(ModelPackage*,
                                                            int flags,
                                                            const char** out_report_json);

#ifdef __cplusplus
}  // extern "C"
#endif
