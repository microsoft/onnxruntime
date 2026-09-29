// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/webgpu/math/matmul_algorithm.h"

namespace onnxruntime {
namespace webgpu {

std::optional<MatMulAlgorithm> ParseMatMulAlgorithm(std::string_view name) {
  if (name == "subgroup_matrix") {
    return MatMulAlgorithm::SubgroupMatrix;
  }
  if (name == "naive") {
    return MatMulAlgorithm::Naive;
  }
  if (name == "subgroup") {
    return MatMulAlgorithm::Subgroup;
  }
  if (name == "packed") {
    return MatMulAlgorithm::Packed;
  }
  if (name == "packed_split_k") {
    return MatMulAlgorithm::PackedSplitK;
  }
  return std::nullopt;
}

std::string_view MatMulAlgorithmName(MatMulAlgorithm algorithm) {
  switch (algorithm) {
    case MatMulAlgorithm::SubgroupMatrix:
      return "subgroup_matrix";
    case MatMulAlgorithm::Naive:
      return "naive";
    case MatMulAlgorithm::Subgroup:
      return "subgroup";
    case MatMulAlgorithm::Packed:
      return "packed";
    case MatMulAlgorithm::PackedSplitK:
      return "packed_split_k";
  }
  return "unknown";
}

}  // namespace webgpu
}  // namespace onnxruntime