// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

/// \file path_resolver.h
/// \brief Path-resolution and confinement helpers.

#pragma once

#include <filesystem>
#include <string>

#include "model_package_api.h"  // for ModelPackageStatus

namespace model_package {

struct PathResolverOptions {
  bool allow_external_paths{false};
  bool follow_symlinks{true};
};

/// Resolve a relative-or-absolute path string under a given base directory.
/// In portable mode (`allow_external_paths == false`):
///   - Reject absolute inputs (ERR_PATH_CONFINEMENT).
///   - Reject any path that, after canonicalization, escapes `package_root`.
///   - Reject `..` segments syntactically before resolution.
/// In installed mode:
///   - Absolute and `..` allowed.
///   - No confinement check.
///
/// `must_exist` controls whether a missing target is an error (ERR_NOT_FOUND)
/// or whether the resolved (non-canonical) path is returned anyway.
/// Symlinks are followed when `follow_symlinks` is true.
ModelPackageStatus* ResolvePath(const std::filesystem::path& base_dir,
                                const std::filesystem::path& package_root,
                                const std::string& input,
                                const PathResolverOptions& opts,
                                bool must_exist,
                                std::filesystem::path* out);

/// True if `uri` matches `^sha256:[0-9a-f]{64}$`.
bool IsSha256AssetUri(const std::string& uri);

/// If `input` begins with a `sha256:<hex>` token followed by end-of-string or
/// '/', split into `uri` (the bare URI) and `tail` (substring after '/', or
/// empty). Returns true on a match, false otherwise.
bool TrySplitAssetUriPrefix(const std::string& input, std::string& uri, std::string& tail);

/// Default on-disk directory name for a shared asset URI, i.e. the basename
/// under `<package_root>/shared_assets/`. For `sha256:<hex>` this is
/// `sha256-<hex>`. Returns empty string if `uri` is not a valid sha256 URI.
std::string DefaultSharedAssetDirName(const std::string& uri);

/// Inverse of `DefaultSharedAssetDirName`. If `dir_name` matches `sha256-<hex>`
/// returns the corresponding `sha256:<hex>` URI; otherwise returns empty string.
std::string SharedAssetUriFromDirName(const std::string& dir_name);

/// Prefix shared by every default-convention shared-asset directory name.
constexpr const char* kSharedAssetOnDiskPrefix = "sha256-";

}  // namespace model_package
