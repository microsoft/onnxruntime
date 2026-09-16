// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <optional>
#include <string_view>

namespace onnxruntime {
namespace webgpu {

enum class MatMulAlgorithm {
  SubgroupMatrix,
  Naive,
  IntelSubgroup,
  Packed,
  PackedSplitK,
};

inline std::optional<MatMulAlgorithm> ParseMatMulAlgorithm(std::string_view name) {
  if (name == "subgroup_matrix") {
    return MatMulAlgorithm::SubgroupMatrix;
  }
  if (name == "naive") {
    return MatMulAlgorithm::Naive;
  }
  if (name == "intel_subgroup") {
    return MatMulAlgorithm::IntelSubgroup;
  }
  if (name == "packed") {
    return MatMulAlgorithm::Packed;
  }
  if (name == "packed_split_k") {
    return MatMulAlgorithm::PackedSplitK;
  }
  return std::nullopt;
}

inline std::string_view MatMulAlgorithmName(MatMulAlgorithm algorithm) {
  switch (algorithm) {
    case MatMulAlgorithm::SubgroupMatrix:
      return "subgroup_matrix";
    case MatMulAlgorithm::Naive:
      return "naive";
    case MatMulAlgorithm::IntelSubgroup:
      return "intel_subgroup";
    case MatMulAlgorithm::Packed:
      return "packed";
    case MatMulAlgorithm::PackedSplitK:
      return "packed_split_k";
  }
  return "unknown";
}

}  // namespace webgpu
}  // namespace onnxruntime
