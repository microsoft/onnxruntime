// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <array>
#include <cstddef>
#include <string_view>

namespace onnxruntime::propagate_cast_ops_internal {

struct RelevantOpArgs {
  std::string_view op_type;
  std::array<int, 3> input_indices;
  size_t input_count;
  std::array<int, 1> output_indices;
  size_t output_count;

  constexpr bool IsRelevantInput(int index) const noexcept {
    for (size_t i = 0; i < input_count; ++i) {
      if (input_indices[i] == index) {
        return true;
      }
    }

    return false;
  }

  constexpr bool IsRelevantOutput(int index) const noexcept {
    for (size_t i = 0; i < output_count; ++i) {
      if (output_indices[i] == index) {
        return true;
      }
    }

    return false;
  }
};

inline constexpr std::array<RelevantOpArgs, 7> kRelevantOpArgs{{
    {"Dropout", {0}, 1, {0}, 1},
    {"Expand", {0}, 1, {0}, 1},
    {"Gather", {0}, 1, {0}, 1},
    {"LayerNormalization", {0, 1, 2}, 3, {0}, 1},
    {"Reshape", {0}, 1, {0}, 1},
    {"Squeeze", {0}, 1, {0}, 1},
    {"Unsqueeze", {0}, 1, {0}, 1},
}};

template <size_t N>
consteval bool HasValidIndices(const std::array<int, N>& indices, size_t count) {
  if (count == 0 || count > indices.size()) {
    return false;
  }

  for (size_t i = 0; i < count; ++i) {
    if (indices[i] < 0) {
      return false;
    }

    for (size_t j = i + 1; j < count; ++j) {
      if (indices[i] == indices[j]) {
        return false;
      }
    }
  }

  return true;
}

consteval bool IsRelevantOpArgsTableValid() {
  for (size_t index = 0; index < kRelevantOpArgs.size(); ++index) {
    const auto& metadata = kRelevantOpArgs[index];
    if (metadata.op_type.empty() ||
        (index > 0 && kRelevantOpArgs[index - 1].op_type >= metadata.op_type) ||
        !HasValidIndices(metadata.input_indices, metadata.input_count) ||
        !HasValidIndices(metadata.output_indices, metadata.output_count)) {
      return false;
    }
  }

  return true;
}

static_assert(IsRelevantOpArgsTableValid(),
              "PropagateCastOps metadata must have sorted unique operators and valid argument indices.");

constexpr const RelevantOpArgs* FindRelevantOpArgs(std::string_view op_type) noexcept {
  for (const auto& metadata : kRelevantOpArgs) {
    if (metadata.op_type == op_type) {
      return &metadata;
    }
  }

  return nullptr;
}

}  // namespace onnxruntime::propagate_cast_ops_internal