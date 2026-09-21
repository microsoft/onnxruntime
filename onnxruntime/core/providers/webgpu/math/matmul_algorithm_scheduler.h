// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <array>
#include <cstdint>
#include <limits>
#include <optional>
#include <string_view>
#include <utility>
#include <variant>

#include "core/providers/webgpu/math/matmul_algorithm.h"

namespace onnxruntime {
namespace webgpu {

struct MatMulAlgorithmSelectionParams {
  int64_t m = 0;
  int64_t n = 0;
  int64_t k = 0;
  int64_t packed_m = 0;
  uint64_t batch_size = 1;
  uint64_t packed_batch_size = 1;
  std::string_view adapter_architecture;
  int32_t a_data_type = 0;
  int32_t b_data_type = 0;
  bool can_use_subgroup_matrix = false;
  bool has_intel_subgroup_capability = false;
  bool is_vec4 = false;
  bool deterministic_compute = false;
  bool has_fused_activation = false;
  bool has_bias = false;
  bool is_channels_last = true;
  bool common_use_split_k = false;
  uint32_t split_dim_inner = 0;
};

struct MatMulSubgroupMatrixConfiguration {};
struct MatMulNaiveConfiguration {};
struct MatMulIntelSubgroupConfiguration {};

struct MatMulPackedConfiguration {
  std::array<uint32_t, 3> workgroup_size{8, 8, 1};
  std::array<uint32_t, 3> elements_per_thread{4, 4, 1};
  uint32_t tile_inner = 32;
  uint32_t split_dim_inner = 1;
};

inline bool IsMatMulPackedConfigurationValid(
    const MatMulPackedConfiguration& configuration,
    bool use_split_k) {
  if (configuration.workgroup_size[0] == 0 ||
      configuration.workgroup_size[1] == 0 ||
      configuration.workgroup_size[2] != 1 ||
      configuration.elements_per_thread[0] == 0 ||
      configuration.elements_per_thread[1] == 0 ||
      configuration.elements_per_thread[2] != 1 ||
      configuration.tile_inner == 0) {
    return false;
  }

  return !use_split_k ||
         (configuration.split_dim_inner > 1 &&
          configuration.split_dim_inner % configuration.tile_inner == 0);
}

inline std::optional<uint32_t> TryGetMatMulPackedDispatchGroupCount(
    uint64_t dimension,
    uint32_t workgroup_size,
    uint32_t elements_per_thread) {
  if (workgroup_size == 0 || elements_per_thread == 0) {
    return std::nullopt;
  }

  const uint64_t elements_per_group =
      static_cast<uint64_t>(workgroup_size) * elements_per_thread;
  const uint64_t group_count =
      dimension == 0 ? 0 : 1 + (dimension - 1) / elements_per_group;
  if (group_count > std::numeric_limits<uint32_t>::max()) {
    return std::nullopt;
  }

  return static_cast<uint32_t>(group_count);
}

using MatMulAlgorithmConfiguration =
    std::variant<MatMulSubgroupMatrixConfiguration,
                 MatMulNaiveConfiguration,
                 MatMulIntelSubgroupConfiguration,
                 MatMulPackedConfiguration>;

struct MatMulExecutionPlan {
  MatMulAlgorithm algorithm;
  MatMulAlgorithmConfiguration configuration;
};

inline bool IsMatMulAlgorithmConfigurationCompatible(
    const MatMulExecutionPlan& plan) {
  switch (plan.algorithm) {
    case MatMulAlgorithm::SubgroupMatrix:
      return std::holds_alternative<MatMulSubgroupMatrixConfiguration>(plan.configuration);
    case MatMulAlgorithm::Naive:
      return std::holds_alternative<MatMulNaiveConfiguration>(plan.configuration);
    case MatMulAlgorithm::IntelSubgroup:
      return std::holds_alternative<MatMulIntelSubgroupConfiguration>(plan.configuration);
    case MatMulAlgorithm::Packed:
    case MatMulAlgorithm::PackedSplitK:
      return std::holds_alternative<MatMulPackedConfiguration>(plan.configuration);
  }
  return false;
}

struct MatMulAlgorithmPrerequisites {
  bool can_use_subgroup_matrix = false;
  bool has_intel_subgroup_capability = false;
  bool has_nonzero_k = false;
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
      return prerequisites.has_intel_subgroup_capability &&
             prerequisites.has_nonzero_k;
    case MatMulAlgorithm::PackedSplitK:
      return prerequisites.has_nonzero_k &&
             prerequisites.split_k_configured &&
             !prerequisites.deterministic_compute &&
             prerequisites.is_vec4 &&
             !prerequisites.has_fused_activation &&
             prerequisites.split_k_bias_layout_supported;
    case MatMulAlgorithm::Packed:
      return prerequisites.has_nonzero_k;
    case MatMulAlgorithm::Naive:
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
    if (params.k == 0) {
      return MatMulAlgorithm::Naive;
    }
    if (const auto vendor_algorithm = SelectVendorAlgorithm(params); vendor_algorithm.has_value()) {
      return *vendor_algorithm;
    }
    return SelectCommonAlgorithm(params);
  }

  MatMulExecutionPlan CreateExecutionPlan(
      const MatMulAlgorithmSelectionParams& params,
      std::optional<MatMulAlgorithm> forced_algorithm = std::nullopt) const {
    const MatMulAlgorithm algorithm = Select(params, forced_algorithm);
    std::optional<MatMulAlgorithmConfiguration> configuration =
        SelectVendorConfiguration(algorithm, params);
    if (!configuration.has_value()) {
      configuration = SelectCommonConfiguration(algorithm, params);
    }
    return MatMulExecutionPlan{algorithm, std::move(*configuration)};
  }

 protected:
  virtual std::optional<MatMulAlgorithm> SelectVendorAlgorithm(
      const MatMulAlgorithmSelectionParams& /*params*/) const {
    return std::nullopt;
  }

  virtual std::optional<MatMulAlgorithmConfiguration> SelectVendorConfiguration(
      MatMulAlgorithm /*algorithm*/,
      const MatMulAlgorithmSelectionParams& /*params*/) const {
    return std::nullopt;
  }

 private:
  MatMulAlgorithm SelectCommonAlgorithm(const MatMulAlgorithmSelectionParams& params) const {
    if (params.can_use_subgroup_matrix) {
      return MatMulAlgorithm::SubgroupMatrix;
    }
    if (params.n < 8 && params.k < 8) {
      return MatMulAlgorithm::Naive;
    }
    if (params.common_use_split_k) {
      return MatMulAlgorithm::PackedSplitK;
    }
    return MatMulAlgorithm::Packed;
  }

  MatMulAlgorithmConfiguration SelectCommonConfiguration(
      MatMulAlgorithm algorithm,
      const MatMulAlgorithmSelectionParams& params) const {
    switch (algorithm) {
      case MatMulAlgorithm::SubgroupMatrix:
        return MatMulSubgroupMatrixConfiguration{};
      case MatMulAlgorithm::Naive:
        return MatMulNaiveConfiguration{};
      case MatMulAlgorithm::IntelSubgroup:
        return MatMulIntelSubgroupConfiguration{};
      case MatMulAlgorithm::Packed:
      case MatMulAlgorithm::PackedSplitK: {
        MatMulPackedConfiguration configuration{};
        configuration.elements_per_thread =
            params.packed_m <= 8 ? std::array<uint32_t, 3>{4, 1, 1}
                                 : std::array<uint32_t, 3>{4, 4, 1};
        configuration.split_dim_inner =
            algorithm == MatMulAlgorithm::PackedSplitK ? params.split_dim_inner : 1;
        return configuration;
      }
    }
    return MatMulNaiveConfiguration{};
  }
};

}  // namespace webgpu
}  // namespace onnxruntime
