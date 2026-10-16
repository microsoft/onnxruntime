// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/webgpu/math/matmul_execution_planner.h"

#include <limits>
#include <utility>

#include "core/common/safeint.h"
#include "core/providers/webgpu/vendor/intel/math/gemm_subgroup.h"

namespace onnxruntime {
namespace webgpu {

namespace {

constexpr uint64_t kMaxUint32 = std::numeric_limits<uint32_t>::max();

bool IsUint32Dimension(int64_t dimension) {
  return dimension >= 0 && static_cast<uint64_t>(dimension) <= kMaxUint32;
}

bool IsMatMulNaiveOutputSizeValid(const MatMulAlgorithmSelectionParams& params) {
  if (!IsUint32Dimension(params.m) ||
      !IsUint32Dimension(params.n) ||
      !IsUint32Dimension(params.k)) {
    return false;
  }

  const uint64_t output_rows = static_cast<uint64_t>(params.m) / GetMaxComponents(params.m);
  const uint64_t output_columns = static_cast<uint64_t>(params.n) / GetMaxComponents(params.n);
  uint64_t output_size = params.batch_size;
  for (const uint64_t factor : {output_rows, output_columns}) {
    uint64_t product{};
    if (!SafeMultiply(output_size, factor, product) || product > kMaxUint32) {
      return false;
    }
    output_size = product;
  }
  return true;
}

bool IsDispatchGroupCountProductValid(uint32_t x, uint32_t y, uint32_t z) {
  // ProgramManager normalizes device-specific per-axis limits. Plans only need to
  // guarantee that ShaderHelper's flattened logical workgroup ID fits in uint32.
  uint32_t xy{};
  uint32_t xyz{};
  return SafeMultiply(x, y, xy) && SafeMultiply(xy, z, xyz);
}

bool IsMatMulAlgorithmConfigurationCompatible(const MatMulExecutionPlan& plan) {
  switch (plan.algorithm) {
    case MatMulAlgorithm::SubgroupMatrix:
      return std::holds_alternative<MatMulSubgroupMatrixConfiguration>(plan.configuration);
    case MatMulAlgorithm::Gemv:
      return std::holds_alternative<MatMulGemvConfiguration>(plan.configuration);
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

}  // namespace

bool IsMatMulPackedConfigurationValid(
    const MatMulPackedConfiguration& configuration,
    bool is_vec4,
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

  if (configuration.tile_inner % configuration.workgroup_size[0] != 0 ||
      configuration.tile_inner % configuration.workgroup_size[1] != 0) {
    return false;
  }

  if (is_vec4) {
    const uint32_t a_vector_width = configuration.tile_inner / configuration.workgroup_size[0];
    if (configuration.elements_per_thread[0] != 4 ||
        a_vector_width != 4) {
      return false;
    }
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

MatMulExecutionPlanner::MatMulExecutionPlanner(SplitKConfig split_k_config)
    : split_k_config_{std::move(split_k_config)} {}

MatMulExecutionPlanner::~MatMulExecutionPlanner() = default;

MatMulAlgorithm MatMulExecutionPlanner::SelectAlgorithm(
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

Status MatMulExecutionPlanner::CreateExecutionPlan(
    const MatMulAlgorithmSelectionParams& params,
    std::optional<MatMulAlgorithm> forced_algorithm,
    MatMulExecutionPlan& plan) const {
  const MatMulAlgorithm algorithm = SelectAlgorithm(params, forced_algorithm);
  std::optional<MatMulAlgorithmConfiguration> configuration =
      SelectVendorConfiguration(algorithm, params);
  if (!configuration.has_value()) {
    configuration = SelectCommonConfiguration(algorithm, params);
  }
  MatMulExecutionPlan candidate{algorithm, std::move(*configuration)};
  ORT_RETURN_IF_ERROR(ValidateExecutionPlan(params, candidate));
  plan = std::move(candidate);
  return Status::OK();
}

std::optional<MatMulAlgorithm> MatMulExecutionPlanner::SelectVendorAlgorithm(
    const MatMulAlgorithmSelectionParams& /*params*/) const {
  return std::nullopt;
}

std::optional<MatMulAlgorithmConfiguration> MatMulExecutionPlanner::SelectVendorConfiguration(
    MatMulAlgorithm /*algorithm*/,
    const MatMulAlgorithmSelectionParams& /*params*/) const {
  return std::nullopt;
}

MatMulAlgorithm MatMulExecutionPlanner::SelectCommonAlgorithm(
    const MatMulAlgorithmSelectionParams& params) const {
  if (params.can_use_subgroup_matrix) {
    return MatMulAlgorithm::SubgroupMatrix;
  }
  if (ShouldUseGemv(params)) {
    return MatMulAlgorithm::Gemv;
  }
  if (params.n < 8 && params.k < 8) {
    return MatMulAlgorithm::Naive;
  }
  if (ShouldUseSplitK(params)) {
    return MatMulAlgorithm::PackedSplitK;
  }
  return MatMulAlgorithm::Packed;
}

MatMulAlgorithmConfiguration MatMulExecutionPlanner::SelectCommonConfiguration(
    MatMulAlgorithm algorithm,
    const MatMulAlgorithmSelectionParams& params) const {
  switch (algorithm) {
    case MatMulAlgorithm::SubgroupMatrix:
      return MatMulSubgroupMatrixConfiguration{};
    case MatMulAlgorithm::Gemv:
      return MatMulGemvConfiguration{};
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

Status MatMulExecutionPlanner::ValidateExecutionPlan(
    const MatMulAlgorithmSelectionParams& params,
    const MatMulExecutionPlan& plan) const {
  ORT_RETURN_IF_NOT(IsMatMulAlgorithmConfigurationCompatible(plan),
                    "MatMul algorithm ", MatMulAlgorithmName(plan.algorithm),
                    " received an incompatible vendor configuration.");

  switch (plan.algorithm) {
    case MatMulAlgorithm::SubgroupMatrix:
      ORT_RETURN_IF_NOT(params.can_use_subgroup_matrix,
                        "MatMul algorithm subgroup_matrix does not support these inputs or this device.");
      break;
    case MatMulAlgorithm::Gemv:
      ORT_RETURN_IF_NOT(params.can_run_gemv &&
                            IsUint32Dimension(params.n) &&
                            IsUint32Dimension(params.k),
                        "MatMul algorithm gemv does not support these inputs or this device.");
      break;
    case MatMulAlgorithm::Naive:
      ORT_RETURN_IF_NOT(IsMatMulNaiveOutputSizeValid(params),
                        "MatMul algorithm naive dimensions or vectorized output size exceed uint32 limits.");
      break;
    case MatMulAlgorithm::Subgroup: {
      const auto& configuration = std::get<MatMulSubgroupConfiguration>(plan.configuration);
      const bool supported_size = configuration.subgroup_size == 8 ||
                                  configuration.subgroup_size == 16 ||
                                  configuration.subgroup_size == 32;
      ORT_RETURN_IF_NOT(params.has_subgroup_capability && params.k > 0 && supported_size &&
                            configuration.subgroup_size == params.subgroup_size,
                        "MatMul algorithm subgroup does not support these inputs, configuration, or device.");

      const bool can_fold_batch_into_m = params.packed_batch_size != params.batch_size;
      const bool fold_batch_into_m = intel::ShouldFoldMatMulSubgroupBatch(
          params.adapter_architecture, params.m, can_fold_batch_into_m);
      const int64_t effective_m = fold_batch_into_m ? params.packed_m : params.m;
      const uint64_t effective_batch_size = fold_batch_into_m ? params.packed_batch_size : params.batch_size;
      ORT_RETURN_IF_NOT(effective_m > 0 && IsUint32Dimension(effective_m) &&
                            params.n > 0 && IsUint32Dimension(params.n) &&
                            IsUint32Dimension(params.k) &&
                            effective_batch_size > 0 && effective_batch_size <= kMaxUint32,
                        "MatMul subgroup dimensions must fit in uint32.");

      const uint32_t elements_per_thread_y = static_cast<uint32_t>(
          intel::MatMulSubgroupElementsPerThreadY(params.adapter_architecture,
                                                  static_cast<uint32_t>(effective_m)));
      const auto dispatch_x = TryGetMatMulPackedDispatchGroupCount(
          static_cast<uint64_t>(params.n), intel::kSubgroupLogicalWorkGroupSizeX, 4);
      const auto dispatch_y = TryGetMatMulPackedDispatchGroupCount(
          static_cast<uint64_t>(effective_m), intel::kSubgroupLogicalWorkGroupSizeY,
          elements_per_thread_y);
      const auto dispatch_z = TryGetMatMulPackedDispatchGroupCount(
          effective_batch_size, intel::kSubgroupLogicalWorkGroupSizeZ, 1);
      ORT_RETURN_IF_NOT(dispatch_x.has_value() && dispatch_y.has_value() && dispatch_z.has_value() &&
                            IsDispatchGroupCountProductValid(*dispatch_x, *dispatch_y, *dispatch_z),
                        "MatMul subgroup dispatch group count exceeds uint32 limits.");
      break;
    }
    case MatMulAlgorithm::Packed:
    case MatMulAlgorithm::PackedSplitK: {
      const bool use_split_k = plan.algorithm == MatMulAlgorithm::PackedSplitK;
      const auto& configuration = std::get<MatMulPackedConfiguration>(plan.configuration);
      ORT_RETURN_IF_NOT(params.k > 0 &&
                            IsMatMulPackedConfigurationValid(configuration, params.is_vec4, use_split_k),
                        "MatMul algorithm ", MatMulAlgorithmName(plan.algorithm),
                        " has invalid inputs or packed configuration.");
      if (use_split_k) {
        ORT_RETURN_IF_NOT(!params.deterministic_compute && params.is_vec4 &&
                              !params.has_fused_activation &&
                              (!params.has_bias || params.is_channels_last),
                          "MatMul algorithm packed_split_k does not support these inputs.");
      }

      ORT_RETURN_IF_NOT(IsUint32Dimension(params.packed_m) &&
                            IsUint32Dimension(params.n) &&
                            IsUint32Dimension(params.k),
                        "MatMul packed dimensions must fit in uint32.");
      const auto dispatch_x = TryGetMatMulPackedDispatchGroupCount(
          static_cast<uint64_t>(params.n), configuration.workgroup_size[0],
          configuration.elements_per_thread[0]);
      const auto dispatch_y = TryGetMatMulPackedDispatchGroupCount(
          static_cast<uint64_t>(params.packed_m), configuration.workgroup_size[1],
          configuration.elements_per_thread[1]);
      const auto dispatch_z = TryGetMatMulPackedDispatchGroupCount(
          params.packed_batch_size, configuration.workgroup_size[2],
          configuration.elements_per_thread[2]);
      ORT_RETURN_IF_NOT(dispatch_x.has_value() && dispatch_y.has_value() && dispatch_z.has_value(),
                        "MatMul packed dispatch dimensions exceed uint32 limits.");

      uint32_t final_dispatch_z = *dispatch_z;
      if (use_split_k) {
        const auto splits_per_batch = TryGetMatMulPackedDispatchGroupCount(
            static_cast<uint64_t>(params.k), configuration.split_dim_inner, 1);
        uint32_t split_dispatch_z{};
        ORT_RETURN_IF_NOT(params.packed_batch_size > 0 && splits_per_batch.has_value() &&
                              SafeMultiply(final_dispatch_z, *splits_per_batch, split_dispatch_z),
                          "MatMul packed Split-K dispatch dimension exceeds uint32 limits.");
        final_dispatch_z = split_dispatch_z;
      }
      ORT_RETURN_IF_NOT(IsDispatchGroupCountProductValid(*dispatch_x, *dispatch_y, final_dispatch_z),
                        "MatMul packed combined dispatch group count exceeds uint32 limits.");
      break;
    }
  }

  return Status::OK();
}

bool MatMulExecutionPlanner::ShouldUseSplitK(
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

bool MatMulExecutionPlanner::ShouldUseGemv(
    const MatMulAlgorithmSelectionParams& params) const {
  return params.can_run_gemv &&
         params.n >= 16 && params.n <= 64 &&
         params.k >= 2048 && params.k <= 8192;
}

}  // namespace webgpu
}  // namespace onnxruntime