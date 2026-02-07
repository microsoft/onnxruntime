// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/session/onnxruntime_cxx_api.h"

#include <cstdint>
#include <charconv>
#include <optional>
#include <string>
#include <vector>

#include "absl/strings/str_split.h"
#include "gtest/gtest.h"

TEST(CApiTest, VersionConsistencyWithApiVersion) {
  const auto version_string = Ort::GetVersionString();
  const std::vector<std::string> version_string_components = absl::StrSplit(version_string, '.');
  ASSERT_EQ(version_string_components.size(), size_t{3});

  auto to_uint32_t = [](const std::string& s) -> std::optional<uint32_t> {
    uint32_t result{};
    if (std::from_chars(s.data(), s.data() + s.size(), result).ec == std::errc{}) {
      return result;
    }
    return std::nullopt;
  };

  ASSERT_NE(to_uint32_t(version_string_components[0]), std::nullopt);
  ASSERT_EQ(to_uint32_t(version_string_components[1]), uint32_t{ORT_API_VERSION});
  ASSERT_NE(to_uint32_t(version_string_components[0]), std::nullopt);
}
