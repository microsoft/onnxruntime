// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cstdint>
#include <optional>

#include "core/providers/webgpu/math/matmul_algorithm.h"

namespace onnxruntime {
namespace webgpu {

struct MatMulAlgorithmSelectionParams {
  int64_t m = 0;
  int64_t n = 0;
  int64_t k = 0;
  bool can_use_subgroup_matrix = false;
  bool has_intel_subgroup_capability = false;
  bool use_split_k = false;
};

class MatMulAlgorithmScheduler {
 public:
  virtual ~MatMulAlgorithmScheduler() = default;

  MatMulAlgorithm Select(
      const MatMulAlgorithmSelectionParams& params,
      std::optional<MatMulAlgorithm> forced_algorithm = std::nullopt) const {
    if (forced_algorithm.has_value()) {
      return *forced_algorithm;
    }
    if (params.can_use_subgroup_matrix) {
      return MatMulAlgorithm::SubgroupMatrix;
    }
    if (params.n < 8 && params.k < 8) {
      return MatMulAlgorithm::Naive;
    }
    if (const auto vendor_algorithm = SelectVendorAlgorithm(params); vendor_algorithm.has_value()) {
      return *vendor_algorithm;
    }
    if (params.use_split_k) {
      return MatMulAlgorithm::PackedSplitK;
    }
    return MatMulAlgorithm::Packed;
  }

 protected:
  virtual std::optional<MatMulAlgorithm> SelectVendorAlgorithm(
      const MatMulAlgorithmSelectionParams& /*params*/) const {
    return std::nullopt;
  }
};

}  // namespace webgpu
}  // namespace onnxruntime
