// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

/// \file path_resolver.h
/// \brief Path-resolution and confinement helpers per §4.2 of the redesign.

#pragma once

#include <filesystem>
#include <string>

#include "model_package_api.h"  // for ModelPackageStatus

namespace model_package_v2 {

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

}  // namespace model_package_v2
