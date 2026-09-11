// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

/// \file model_package_api.h
/// \brief Core types shared by the model_package public API surface.
///
/// This header defines the export macro, the opaque `ModelPackageStatus` type,
/// and the `ModelPackageErrorCode` enum used by every entry point in the
/// library. The actual API entry points live in `model_package.h`.
///
/// Error handling: functions that can fail return `ModelPackageStatus*`. A
/// `nullptr` return indicates success. Use the `ModelPackageStatus_*` helpers
/// in `model_package.h` to inspect and release statuses.

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
// Opaque status type
// ─────────────────────────────────────────────────────────────────────────────

/// Opaque status type. nullptr indicates success.
typedef struct ModelPackageStatus ModelPackageStatus;

// ─────────────────────────────────────────────────────────────────────────────
// Error codes
// ─────────────────────────────────────────────────────────────────────────────

/// Categorical error codes attached to every non-OK ModelPackageStatus.
/// Stable additive enum: new codes will be appended at the end; existing
/// values will not be renumbered.
typedef enum ModelPackageErrorCode {
  MODEL_PACKAGE_OK = 0,
  MODEL_PACKAGE_ERR_IO = 1,                   ///< Filesystem read/write/sync failure.
  MODEL_PACKAGE_ERR_SCHEMA = 2,               ///< JSON value has wrong shape or wrong type.
  MODEL_PACKAGE_ERR_VERSION = 3,              ///< Unsupported schema_version.
  MODEL_PACKAGE_ERR_PATH_CONFINEMENT = 4,     ///< Path resolution escaped the allowed base.
  MODEL_PACKAGE_ERR_ASSET_MISSING = 5,        ///< Declared shared asset not resolvable.
  MODEL_PACKAGE_ERR_ASSET_HASH_MISMATCH = 6,  ///< Existing asset directory failed rehash.
  MODEL_PACKAGE_ERR_NOT_FOUND = 7,            ///< Named entity not present.
  MODEL_PACKAGE_ERR_INVALID_ARG = 8,          ///< Null pointer or otherwise invalid argument.
  MODEL_PACKAGE_ERR_STATE = 9                 ///< Operation not legal in current state.
} ModelPackageErrorCode;

#ifdef __cplusplus
}  // extern "C"
#endif
