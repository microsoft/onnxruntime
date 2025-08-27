// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cstdint>
#include <optional>
#include <string_view>

#include "core/common/status.h"

namespace onnxruntime {

// Semantic Versioning version utilities.
// See https://github.com/semver/semver/blob/v2.0.0/semver.md.

// Semantic Versioning version components.
struct SemVerVersion {
  uint32_t major{};
  uint32_t minor{};
  uint32_t patch{};
  std::optional<std::string_view> prerelease{};
  std::optional<std::string_view> build_metadata{};
};

// Parse a Semantic Versioning version from `version_string`.
// If provided, the parsed version components will be written to `semver_version`.
Status ParseSemVerVersion(std::string_view version_string, SemVerVersion* semver_version);

// Parse a Semantic Versioning version from `version_string`.
SemVerVersion ParseSemVerVersion(std::string_view version_string);

}  // namespace onnxruntime
