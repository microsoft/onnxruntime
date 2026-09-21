// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <utility>

#include "core/providers/webgpu/math/matmul_algorithm_scheduler.h"

namespace onnxruntime {
namespace webgpu {
namespace intel {

class IntelMatMulAlgorithmScheduler final : public MatMulAlgorithmScheduler {
 public:
  IntelMatMulAlgorithmScheduler() = default;
  explicit IntelMatMulAlgorithmScheduler(SplitKConfig split_k_config)
      : MatMulAlgorithmScheduler{std::move(split_k_config)} {}

 protected:
  std::optional<MatMulAlgorithm> SelectVendorAlgorithm(
      const MatMulAlgorithmSelectionParams& params) const override {
    if (params.can_use_subgroup_matrix) {
      return MatMulAlgorithm::SubgroupMatrix;
    }
    if (params.has_intel_subgroup_capability &&
        params.m >= 64 && params.n >= 512 && params.k >= 32) {
      return MatMulAlgorithm::IntelSubgroup;
    }
    return std::nullopt;
  }
};

}  // namespace intel
}  // namespace webgpu
}  // namespace onnxruntime
