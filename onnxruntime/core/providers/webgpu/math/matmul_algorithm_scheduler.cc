// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/webgpu/math/matmul_algorithm_scheduler.h"

#include <limits>
#include <utility>

namespace onnxruntime {
namespace webgpu {

bool IsMatMulPackedConfigurationValid(
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

std::optional<uint32_t> TryGetMatMulPackedDispatchGroupCount(
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

bool IsMatMulAlgorithmConfigurationCompatible(const MatMulExecutionPlan& plan) {
  switch (plan.algorithm) {
    case MatMulAlgorithm::SubgroupMatrix:
      return std::holds_alternative<MatMulSubgroupMatrixConfiguration>(plan.configuration);
    case MatMulAlgorithm::Naive:
      return std::holds_alternative<MatMulNaiveConfiguration>(plan.configuration);
    case MatMulAlgorithm::Subgroup:
      return std::holds_alternative<MatMulSubgroupConfiguration>(plan.configuration);
    case MatMulAlgorithm::Packed:
    case MatMulAlgorithm::PackedSplitK:
      return std::holds_alternative<MatMulPackedConfiguration>(plan.configuration);
  }
  return false;
}

bool MeetsMatMulAlgorithmPrerequisites(
    MatMulAlgorithm algorithm,
    const MatMulAlgorithmPrerequisites& prerequisites) {
  switch (algorithm) {
    case MatMulAlgorithm::SubgroupMatrix:
      return prerequisites.can_use_subgroup_matrix;
    case MatMulAlgorithm::Subgroup:
      return prerequisites.has_subgroup_capability &&
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

MatMulAlgorithmScheduler::MatMulAlgorithmScheduler(SplitKConfig split_k_config)
    : split_k_config_{std::move(split_k_config)} {}

MatMulAlgorithmScheduler::~MatMulAlgorithmScheduler() = default;

MatMulAlgorithm MatMulAlgorithmScheduler::Select(
    const MatMulAlgorithmSelectionParams& params,
    std::optional<MatMulAlgorithm> forced_algorithm) const {
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

MatMulExecutionPlan MatMulAlgorithmScheduler::CreateExecutionPlan(
    const MatMulAlgorithmSelectionParams& params,
    std::optional<MatMulAlgorithm> forced_algorithm) const {
  const MatMulAlgorithm algorithm = Select(params, forced_algorithm);
  std::optional<MatMulAlgorithmConfiguration> configuration =
      SelectVendorConfiguration(algorithm, params);
  if (!configuration.has_value()) {
    configuration = SelectCommonConfiguration(algorithm, params);
  }
  return MatMulExecutionPlan{algorithm, std::move(*configuration)};
}

std::optional<MatMulAlgorithm> MatMulAlgorithmScheduler::SelectVendorAlgorithm(
    const MatMulAlgorithmSelectionParams& /*params*/) const {
  return std::nullopt;
}

std::optional<MatMulAlgorithmConfiguration> MatMulAlgorithmScheduler::SelectVendorConfiguration(
    MatMulAlgorithm /*algorithm*/,
    const MatMulAlgorithmSelectionParams& /*params*/) const {
  return std::nullopt;
}

MatMulAlgorithm MatMulAlgorithmScheduler::SelectCommonAlgorithm(
    const MatMulAlgorithmSelectionParams& params) const {
  if (params.can_use_subgroup_matrix) {
    return MatMulAlgorithm::SubgroupMatrix;
  }
  if (params.n < 8 && params.k < 8) {
    return MatMulAlgorithm::Naive;
  }
  if (ShouldUseSplitK(params)) {
    return MatMulAlgorithm::PackedSplitK;
  }
  return MatMulAlgorithm::Packed;
}

MatMulAlgorithmConfiguration MatMulAlgorithmScheduler::SelectCommonConfiguration(
    MatMulAlgorithm algorithm,
    const MatMulAlgorithmSelectionParams& params) const {
  switch (algorithm) {
    case MatMulAlgorithm::SubgroupMatrix:
      return MatMulSubgroupMatrixConfiguration{};
    case MatMulAlgorithm::Naive:
      return MatMulNaiveConfiguration{};
    case MatMulAlgorithm::Subgroup:
      return MatMulSubgroupConfiguration{params.subgroup_size};
    case MatMulAlgorithm::Packed:
    case MatMulAlgorithm::PackedSplitK: {
      MatMulPackedConfiguration configuration{};
      configuration.elements_per_thread =
          params.packed_m <= 8 ? std::array<uint32_t, 3>{4, 1, 1}
                               : std::array<uint32_t, 3>{4, 4, 1};
      configuration.split_dim_inner =
          algorithm == MatMulAlgorithm::PackedSplitK
              ? split_k_config_.GetSplitDimInner()
              : 1;
      return configuration;
    }
  }
  return MatMulNaiveConfiguration{};
}

bool MatMulAlgorithmScheduler::ShouldUseSplitK(
    const MatMulAlgorithmSelectionParams& params) const {
  if (params.deterministic_compute || params.has_fused_activation ||
      params.packed_m < 0 || params.n < 0 || params.k < 0) {
    return false;
  }

  return split_k_config_.UseSplitK(
      params.is_vec4,
      ActivationKind::None,
      params.packed_batch_size,
      static_cast<uint64_t>(params.packed_m),
      static_cast<uint64_t>(params.n),
      static_cast<uint64_t>(params.k),
      params.is_channels_last);
}

}  // namespace webgpu
}  // namespace onnxruntime