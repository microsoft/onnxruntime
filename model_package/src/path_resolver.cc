// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "path_resolver.h"

#include <cctype>
#include <string>
#include <system_error>

#include "status_impl.h"

namespace fs = std::filesystem;

namespace model_package {

namespace {

bool IsHexLower(char c) { return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f'); }

bool ContainsParentRefSegment(const fs::path& p) {
  for (const auto& seg : p) {
    if (seg == "..") return true;
  }
  return false;
}

}  // namespace

bool IsSha256AssetUri(const std::string& uri) {
  static constexpr const char* kPrefix = "sha256:";
  static constexpr size_t kPrefixLen = 7;
  static constexpr size_t kHexLen = 64;
  if (uri.size() != kPrefixLen + kHexLen) return false;
  if (uri.compare(0, kPrefixLen, kPrefix) != 0) return false;
  for (size_t i = kPrefixLen; i < uri.size(); ++i) {
    if (!IsHexLower(uri[i])) return false;
  }
  return true;
}

ModelPackageStatus* ResolvePath(const fs::path& base_dir,
                                const fs::path& package_root,
                                const std::string& input,
                                const PathResolverOptions& opts,
                                bool must_exist,
                                fs::path* out) {
  if (!out) {
    return model_package::MakeStatus(MODEL_PACKAGE_ERR_INVALID_ARG,
                                     "ResolvePath: out must not be null.");
  }
  if (input.empty()) {
    return model_package::MakeStatus(MODEL_PACKAGE_ERR_INVALID_ARG,
                                     "ResolvePath: input must not be empty.");
  }

  fs::path raw(input);

  if (!opts.allow_external_paths) {
    if (raw.is_absolute()) {
      return model_package::MakeStatus(
          MODEL_PACKAGE_ERR_PATH_CONFINEMENT,
          std::string("ResolvePath: absolute path '") + input +
              "' is not allowed in portable layout.");
    }
    if (ContainsParentRefSegment(raw)) {
      return model_package::MakeStatus(
          MODEL_PACKAGE_ERR_PATH_CONFINEMENT,
          std::string("ResolvePath: '..' segments are not allowed in portable layout: '") +
              input + "'.");
    }
  }

  fs::path joined = raw.is_absolute() ? raw : (base_dir / raw);

  std::error_code ec;
  fs::path canonical;
  bool exists_on_disk = fs::exists(joined, ec);
  if (!exists_on_disk) {
    if (must_exist) {
      return model_package::MakeStatus(
          MODEL_PACKAGE_ERR_NOT_FOUND,
          std::string("ResolvePath: '") + joined.string() + "' does not exist.");
    }
    // Best-effort: lexically-normalize so we at least drop redundant separators.
    canonical = joined.lexically_normal();
  } else if (opts.follow_symlinks) {
    canonical = fs::canonical(joined, ec);
    if (ec) {
      return model_package::MakeStatus(
          MODEL_PACKAGE_ERR_IO,
          std::string("ResolvePath: canonical('") + joined.string() + "') failed: " + ec.message());
    }
  } else {
    canonical = fs::weakly_canonical(joined, ec);
    if (ec) {
      canonical = joined.lexically_normal();
    }
  }

  if (!opts.allow_external_paths && exists_on_disk) {
    // Confinement check: canonical must live under package_root's canonical form.
    fs::path canonical_root = fs::weakly_canonical(package_root, ec);
    if (ec) canonical_root = package_root.lexically_normal();

    auto root_str = canonical_root.lexically_normal().string();
    auto can_str = canonical.lexically_normal().string();
    if (can_str.size() < root_str.size() ||
        can_str.compare(0, root_str.size(), root_str) != 0 ||
        (can_str.size() > root_str.size() &&
         can_str[root_str.size()] != fs::path::preferred_separator &&
         can_str[root_str.size()] != '/')) {
      return model_package::MakeStatus(
          MODEL_PACKAGE_ERR_PATH_CONFINEMENT,
          std::string("ResolvePath: '") + can_str +
              "' escapes package_root '" + root_str + "'.");
    }
  }

  *out = canonical;
  return nullptr;
}

}  // namespace model_package
