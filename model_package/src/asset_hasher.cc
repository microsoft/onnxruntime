// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "asset_hasher.h"

#include <algorithm>
#include <string>
#include <system_error>
#include <vector>

#include "sha256.h"
#include "status_impl.h"

namespace fs = std::filesystem;

namespace model_package {

using model_package::MakeStatus;

namespace {

std::string ToPosix(const fs::path& rel) {
  std::string s = rel.generic_string();  // generic_string uses '/'
  // Strip leading "./" if any (lexical normalization edge case).
  if (s.size() >= 2 && s[0] == '.' && s[1] == '/') s.erase(0, 2);
  return s;
}

}  // namespace

ModelPackageStatus* ComputeDirectoryAssetUri(const fs::path& source_dir,
                                             std::string* out_uri) {
  if (!out_uri) {
    return MakeStatus(MODEL_PACKAGE_ERR_INVALID_ARG, "ComputeDirectoryAssetUri: out_uri is null.");
  }
  std::error_code ec;
  if (!fs::exists(source_dir, ec) || !fs::is_directory(source_dir, ec)) {
    return MakeStatus(MODEL_PACKAGE_ERR_NOT_FOUND,
                      "ComputeDirectoryAssetUri: '" + source_dir.string() + "' is not a directory.");
  }

  // Collect (relative_posix_path, absolute_path) pairs.
  std::vector<std::pair<std::string, fs::path>> entries;

  auto walker = fs::recursive_directory_iterator(
      source_dir, fs::directory_options::none, ec);
  if (ec) {
    return MakeStatus(MODEL_PACKAGE_ERR_IO,
                      "ComputeDirectoryAssetUri: cannot iterate '" + source_dir.string() +
                          "': " + ec.message());
  }
  for (; walker != fs::recursive_directory_iterator(); walker.increment(ec)) {
    if (ec) {
      return MakeStatus(MODEL_PACKAGE_ERR_IO,
                        "ComputeDirectoryAssetUri: iteration error: " + ec.message());
    }
    const fs::directory_entry& de = *walker;
    if (de.is_symlink(ec)) {
      return MakeStatus(MODEL_PACKAGE_ERR_SCHEMA,
                        "ComputeDirectoryAssetUri: symlink not allowed: '" + de.path().string() + "'.");
    }
    if (de.is_regular_file(ec)) {
      fs::path rel = fs::relative(de.path(), source_dir, ec);
      if (ec) {
        return MakeStatus(MODEL_PACKAGE_ERR_IO,
                          "ComputeDirectoryAssetUri: relative path failed: " + ec.message());
      }
      entries.emplace_back(ToPosix(rel), de.path());
    } else if (!de.is_directory(ec)) {
      return MakeStatus(MODEL_PACKAGE_ERR_SCHEMA,
                        "ComputeDirectoryAssetUri: unsupported file kind: '" +
                            de.path().string() + "' (only regular files and directories allowed).");
    }
  }

  std::sort(entries.begin(), entries.end(),
            [](const auto& a, const auto& b) { return a.first < b.first; });

  std::string manifest_text;
  manifest_text.reserve(entries.size() * 96);
  for (const auto& entry : entries) {
    std::string file_hex = Sha256::HashFileHex(entry.second.string());
    if (file_hex.empty()) {
      return MakeStatus(MODEL_PACKAGE_ERR_IO,
                        "ComputeDirectoryAssetUri: failed to hash file '" + entry.second.string() + "'.");
    }
    manifest_text.append(file_hex);
    manifest_text.append("  ");
    manifest_text.append(entry.first);
    manifest_text.append("\n");
  }

  *out_uri = "sha256:" + Sha256::HashStringHex(manifest_text);
  return nullptr;
}

}  // namespace model_package
