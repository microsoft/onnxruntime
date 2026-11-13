// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/webgpu/vendor/intel/math/matmul_execution_planner.h"

#include <utility>

#include "core/providers/webgpu/vendor/intel/math/gemm_subgroup_utils.h"

namespace onnxruntime {
namespace webgpu {
namespace intel {

IntelMatMulExecutionPlanner::IntelMatMulExecutionPlanner(SplitKConfig split_k_config)
    : MatMulExecutionPlanner{std::move(split_k_config)} {}

std::optional<MatMulAlgorithm> IntelMatMulExecutionPlanner::SelectVendorAlgorithm(
    const MatMulAlgorithmSelectionParams& params) const {
  if (params.can_use_subgroup_matrix) {
    return MatMulAlgorithm::SubgroupMatrix;
  }
  if (params.has_subgroup_capability &&
      params.m >= 64 && params.n >= 512 && params.k >= 32) {
    // Split-K beats the subgroup-16 kernel for small outputs with a long K.
    if (UseMatMulSubgroup16Kernel(params.adapter_architecture, params.subgroup_size) && ShouldUseSplitK(params)) {
      return std::nullopt;
    }
    return MatMulAlgorithm::Subgroup;
  }
  return std::nullopt;
}

}  // namespace intel
}  // namespace webgpu
}  // namespace onnxruntime