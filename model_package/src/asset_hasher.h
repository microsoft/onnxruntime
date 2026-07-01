// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

/// \file asset_hasher.h
/// \brief Directory Merkle hash for content-addressed shared assets.

#pragma once

#include <filesystem>
#include <string>

#include "model_package_api.h"

namespace model_package {

/// Compute the canonical asset URI for a directory:
///   1. Walk recursively, collect regular files (ignore empty dirs).
///   2. Reject symlinks (ERR_SCHEMA: portability hazard).
///   3. For each file, compute sha256(file_bytes) → per-file hex.
///   4. Build manifest text: `<sha256_hex>  <relative_posix_path>\n` lines,
///      sorted lexicographically by path. Paths are POSIX (`/`), no leading
///      `./`. NFC normalization is the caller's responsibility for non-ASCII
///      paths; ASCII is identity.
///   5. asset_uri = "sha256:" + sha256(manifest_text), lowercase hex.
///
/// On success, *out_uri is set to the URI string.
ModelPackageStatus* ComputeDirectoryAssetUri(const std::filesystem::path& source_dir,
                                             std::string* out_uri);

}  // namespace model_package
