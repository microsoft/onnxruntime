// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

/// \file model_package_api.h
/// \brief Standalone C API for parsing and inspecting ONNX Runtime Model Packages.
///
/// This library has no dependency on ONNX Runtime. It provides read-only access to
/// model package structure: components, variants, EP compatibility declarations,
/// model files, session/provider options, and consumer metadata.
///
/// Error handling: Functions that can fail return `ModelPackageStatus*`.
/// A nullptr return indicates success. On failure, use `ModelPackage_GetErrorMessage()`
/// to retrieve the error string, and `ModelPackage_ReleaseStatus()` to free it.
///
/// Lifetime: All `const char*` pointers returned by this API are owned by the
/// `ModelPackageContext` and remain valid until it is released.

#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// ─────────────────────────────────────────────────────────────────────────────
// Export macros
// ─────────────────────────────────────────────────────────────────────────────

#ifdef _WIN32
#ifdef MODEL_PACKAGE_DLL_EXPORT
#define MODEL_PACKAGE_API __declspec(dllexport)
#elif defined(MODEL_PACKAGE_DLL_IMPORT)
#define MODEL_PACKAGE_API __declspec(dllimport)
#else
#define MODEL_PACKAGE_API
#endif
#else
#ifdef MODEL_PACKAGE_DLL_EXPORT
#define MODEL_PACKAGE_API __attribute__((visibility("default")))
#else
#define MODEL_PACKAGE_API
#endif
#endif

// ─────────────────────────────────────────────────────────────────────────────
// Opaque types
// ─────────────────────────────────────────────────────────────────────────────

/// Opaque status type. nullptr indicates success.
typedef struct ModelPackageStatus ModelPackageStatus;

/// Opaque context holding a parsed model package.
typedef struct ModelPackageContext ModelPackageContext;

// ─────────────────────────────────────────────────────────────────────────────
// Status API
// ─────────────────────────────────────────────────────────────────────────────

/// Release a status object. Safe to call with nullptr.
MODEL_PACKAGE_API void ModelPackage_ReleaseStatus(ModelPackageStatus* status);

/// Get the error message from a status object. Returns nullptr if status is nullptr.
/// The returned string is owned by the status object.
MODEL_PACKAGE_API const char* ModelPackage_GetErrorMessage(const ModelPackageStatus* status);

// ─────────────────────────────────────────────────────────────────────────────
// Context lifecycle
// ─────────────────────────────────────────────────────────────────────────────

/// Parse a model package from a directory path and create a context.
///
/// \param[in] package_root_path  Null-terminated UTF-8 path to the package root directory.
/// \param[out] out_context       On success, receives the created context. Caller must release
///                               via ModelPackage_ReleaseContext().
/// \return nullptr on success, or a status object describing the error.
MODEL_PACKAGE_API ModelPackageStatus* ModelPackage_CreateContext(
    const char* package_root_path,
    ModelPackageContext** out_context);

/// Release a model package context and all associated resources.
/// Safe to call with nullptr.
MODEL_PACKAGE_API void ModelPackage_ReleaseContext(ModelPackageContext* context);

// ─────────────────────────────────────────────────────────────────────────────
// Package-level queries
// ─────────────────────────────────────────────────────────────────────────────

/// Get the schema version declared in manifest.json.
MODEL_PACKAGE_API ModelPackageStatus* ModelPackage_GetSchemaVersion(
    const ModelPackageContext* context,
    int64_t* out_version);

// ─────────────────────────────────────────────────────────────────────────────
// Component queries
// ─────────────────────────────────────────────────────────────────────────────

/// Get the number of components in the package.
MODEL_PACKAGE_API ModelPackageStatus* ModelPackage_GetComponentCount(
    const ModelPackageContext* context,
    size_t* out_count);

/// Get the name of a component by index.
///
/// \param[in] context        The package context.
/// \param[in] component_idx  Zero-based index (must be < component count).
/// \param[out] out_name      Receives a pointer to the component name string.
///                           Lifetime is tied to the context.
MODEL_PACKAGE_API ModelPackageStatus* ModelPackage_GetComponentName(
    const ModelPackageContext* context,
    size_t component_idx,
    const char** out_name);

// ─────────────────────────────────────────────────────────────────────────────
// Variant queries
// ─────────────────────────────────────────────────────────────────────────────

/// Get the number of variants for a component.
MODEL_PACKAGE_API ModelPackageStatus* ModelPackage_GetVariantCount(
    const ModelPackageContext* context,
    const char* component_name,
    size_t* out_count);

/// Get the name of a variant by index.
MODEL_PACKAGE_API ModelPackageStatus* ModelPackage_GetVariantName(
    const ModelPackageContext* context,
    const char* component_name,
    size_t variant_idx,
    const char** out_name);

/// Get the folder path for a variant (resolved absolute path).
MODEL_PACKAGE_API ModelPackageStatus* ModelPackage_GetVariantFolderPath(
    const ModelPackageContext* context,
    const char* component_name,
    const char* variant_name,
    const char** out_path);

// ─────────────────────────────────────────────────────────────────────────────
// EP compatibility queries
// ─────────────────────────────────────────────────────────────────────────────

/// Get the EP name declared for a variant.
///
/// Each variant targets a single EP. When the variant does not declare an EP,
/// the returned pointer is set to nullptr.
MODEL_PACKAGE_API ModelPackageStatus* ModelPackage_GetVariantEpName(
    const ModelPackageContext* context,
    const char* component_name,
    const char* variant_name,
    const char** out_ep);

#ifdef __cplusplus
}  // extern "C"
#endif
