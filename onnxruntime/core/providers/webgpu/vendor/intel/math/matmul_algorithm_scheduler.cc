// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/webgpu/vendor/intel/math/matmul_algorithm_scheduler.h"

#include <utility>

namespace onnxruntime {
namespace webgpu {
namespace intel {

IntelMatMulAlgorithmScheduler::IntelMatMulAlgorithmScheduler(SplitKConfig split_k_config)
    : MatMulAlgorithmScheduler{std::move(split_k_config)} {}

std::optional<MatMulAlgorithm> IntelMatMulAlgorithmScheduler::SelectVendorAlgorithm(
    const MatMulAlgorithmSelectionParams& params) const {
  if (params.can_use_subgroup_matrix) {
    return MatMulAlgorithm::SubgroupMatrix;
  }
  if (params.has_subgroup_capability &&
      params.m >= 64 && params.n >= 512 && params.k >= 32) {
    return MatMulAlgorithm::Subgroup;
  }
  return std::nullopt;
}

}  // namespace intel
}  // namespace webgpu
}  // namespace onnxruntime