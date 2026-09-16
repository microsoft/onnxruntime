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

struct MatMulAlgorithmPrerequisites {
  bool can_use_subgroup_matrix = false;
  bool has_intel_subgroup_capability = false;
  bool split_k_configured = false;
  bool deterministic_compute = false;
  bool is_vec4 = false;
  bool has_fused_activation = false;
  bool split_k_bias_layout_supported = true;
};

inline bool MeetsMatMulAlgorithmPrerequisites(
    MatMulAlgorithm algorithm,
    const MatMulAlgorithmPrerequisites& prerequisites) {
  switch (algorithm) {
    case MatMulAlgorithm::SubgroupMatrix:
      return prerequisites.can_use_subgroup_matrix;
    case MatMulAlgorithm::IntelSubgroup:
      return prerequisites.has_intel_subgroup_capability;
    case MatMulAlgorithm::PackedSplitK:
      return prerequisites.split_k_configured &&
             !prerequisites.deterministic_compute &&
             prerequisites.is_vec4 &&
             !prerequisites.has_fused_activation &&
             prerequisites.split_k_bias_layout_supported;
    case MatMulAlgorithm::Naive:
    case MatMulAlgorithm::Packed:
      return true;
  }
  return false;
}

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
