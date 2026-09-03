// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <algorithm>
#include <charconv>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>

namespace onnxruntime::telemetry_internal {

inline constexpr size_t kMaxDeviceCensusEntries = 24;
inline constexpr size_t kMaxDeviceCensusStateSize = 4096;
inline constexpr int64_t kDeviceCensusSchemaVersion = 1;

struct DeviceCensusState {
  int64_t schema_version;
  int64_t utc_day;
  bool emitted;
  std::vector<std::string> versions;
};

inline bool IsValidDeviceCensusVersion(std::string_view library_version) {
  if (library_version.empty() || library_version.size() > 128) {
    return false;
  }

  return std::all_of(
      library_version.begin(), library_version.end(), [](unsigned char c) {
        return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
               (c >= '0' && c <= '9') || c == '.' || c == '-' ||
               c == '_' || c == '+';
      });
}

inline bool AddDeviceCensusVersion(DeviceCensusState& state,
                                   std::string version) {
  const auto position = std::lower_bound(
      state.versions.begin(), state.versions.end(), version);
  if (position != state.versions.end() && *position == version) {
    return false;
  }
  if (state.versions.size() >= kMaxDeviceCensusEntries) {
    return false;
  }

  state.versions.insert(position, std::move(version));
  return true;
}

inline std::optional<DeviceCensusState> ParseDeviceCensusState(
    std::string_view serialized) {
  if (serialized.empty() || serialized.size() > kMaxDeviceCensusStateSize) {
    return std::nullopt;
  }

  const size_t first_newline = serialized.find('\n');
  if (first_newline == std::string_view::npos) {
    return std::nullopt;
  }

  int64_t schema_version = 0;
  const std::string_view schema = serialized.substr(0, first_newline);
  const auto [schema_end, schema_error] =
      std::from_chars(schema.data(), schema.data() + schema.size(),
                      schema_version);
  if (schema_error != std::errc{} ||
      schema_end != schema.data() + schema.size() || schema_version < 0) {
    return std::nullopt;
  }

  const size_t second_newline = serialized.find('\n', first_newline + 1);
  if (second_newline == std::string_view::npos) {
    return std::nullopt;
  }

  int64_t utc_day = 0;
  const std::string_view day =
      serialized.substr(first_newline + 1, second_newline - first_newline - 1);
  const auto [day_end, error] =
      std::from_chars(day.data(), day.data() + day.size(), utc_day);
  if (error != std::errc{} || day_end != day.data() + day.size() ||
      utc_day < 0) {
    return std::nullopt;
  }

  const size_t third_newline = serialized.find('\n', second_newline + 1);
  if (third_newline == std::string_view::npos) {
    return std::nullopt;
  }

  const std::string_view emitted = serialized.substr(
      second_newline + 1, third_newline - second_newline - 1);
  if (emitted != "0" && emitted != "1") {
    return std::nullopt;
  }

  DeviceCensusState state{schema_version, utc_day, emitted == "1", {}};
  size_t offset = third_newline + 1;
  while (offset < serialized.size()) {
    size_t newline = serialized.find('\n', offset);
    if (newline == std::string_view::npos) {
      newline = serialized.size();
    }

    const std::string_view entry = serialized.substr(offset, newline - offset);
    if (!IsValidDeviceCensusVersion(entry) ||
        !AddDeviceCensusVersion(state, std::string(entry))) {
      return std::nullopt;
    }

    offset = newline + 1;
  }

  return state.versions.empty()
             ? std::nullopt
             : std::optional<DeviceCensusState>{std::move(state)};
}

inline std::string SerializeDeviceCensusState(const DeviceCensusState& state) {
  std::string serialized = std::to_string(state.schema_version) + "\n" +
                           std::to_string(state.utc_day) + "\n" +
                           (state.emitted ? "1\n" : "0\n");
  for (const std::string& entry : state.versions) {
    serialized += entry;
    serialized += '\n';
  }
  return serialized;
}

}  // namespace onnxruntime::telemetry_internal
