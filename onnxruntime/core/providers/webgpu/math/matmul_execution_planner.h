// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <array>
#include <cstdint>
#include <optional>
#include <string_view>
#include <variant>

#include "core/common/common.h"
#include "core/providers/webgpu/math/matmul_algorithm.h"
#include "core/providers/webgpu/webgpu_utils.h"

namespace onnxruntime {
namespace webgpu {

// Immutable problem facts used by automatic selection and execution tuning.
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
  bool can_run_gemv = false;
  bool has_subgroup_capability = false;
  uint32_t subgroup_size = 0;
  bool is_vec4 = false;
  bool deterministic_compute = false;
  bool has_fused_activation = false;
  bool has_bias = false;
  bool is_channels_last = true;
};

// Algorithm-specific tuning selected together with the implementation.
struct MatMulSubgroupMatrixConfiguration {};
struct MatMulGemvConfiguration {};
struct MatMulNaiveConfiguration {};
struct MatMulSubgroupConfiguration {
  uint32_t subgroup_size = 0;
};

struct MatMulPackedConfiguration {
  std::array<uint32_t, 3> workgroup_size{8, 8, 1};
  std::array<uint32_t, 3> elements_per_thread{4, 4, 1};
  uint32_t tile_inner = 32;
  uint32_t split_dim_inner = 1;
};

bool IsMatMulPackedConfigurationValid(
    const MatMulPackedConfiguration& configuration,
    bool is_vec4,
    bool use_split_k);

std::optional<uint32_t> TryGetMatMulPackedDispatchGroupCount(
    uint64_t dimension,
    uint32_t workgroup_size,
    uint32_t elements_per_thread);

using MatMulAlgorithmConfiguration =
    std::variant<MatMulSubgroupMatrixConfiguration,
                 MatMulGemvConfiguration,
                 MatMulNaiveConfiguration,
                 MatMulSubgroupConfiguration,
                 MatMulPackedConfiguration>;

// Complete per-invocation decision consumed by the compute dispatcher.
struct MatMulExecutionPlan {
  MatMulAlgorithm algorithm;
  MatMulAlgorithmConfiguration configuration;
};

bool IsMatMulAlgorithmConfigurationCompatible(const MatMulExecutionPlan& plan);

// Runtime correctness constraints validated immediately before dispatch.
struct MatMulAlgorithmPrerequisites {
  bool can_use_subgroup_matrix = false;
  bool can_run_gemv = false;
  bool has_subgroup_capability = false;
  uint32_t subgroup_size = 0;
  bool has_nonzero_k = false;
  bool split_k_configured = false;
  bool deterministic_compute = false;
  bool is_vec4 = false;
  bool has_fused_activation = false;
  bool split_k_bias_layout_supported = true;
};

bool MeetsMatMulAlgorithmPrerequisites(
    MatMulAlgorithm algorithm,
    const MatMulAlgorithmPrerequisites& prerequisites);

// Pure policy object that selects an algorithm and its tuning configuration to create an execution plan.
// Runtime prerequisite validation and execution remain in MatMulComputeDispatcher.
class MatMulExecutionPlanner {
 public:
  explicit MatMulExecutionPlanner(SplitKConfig split_k_config = {});
  virtual ~MatMulExecutionPlanner();
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(MatMulExecutionPlanner);

  // Selects one algorithm from immutable invocation facts. A forced algorithm bypasses
  // automatic policy but remains subject to dispatcher prerequisite validation.
  MatMulAlgorithm SelectAlgorithm(
      const MatMulAlgorithmSelectionParams& params,
      std::optional<MatMulAlgorithm> forced_algorithm = std::nullopt) const;

  // Selects an algorithm and its typed tuning configuration as one immutable plan.
  // A forced plan may intentionally violate runtime prerequisites; the dispatcher reports
  // an error naming the requested algorithm instead of silently changing the selection.
  MatMulExecutionPlan CreateExecutionPlan(
      const MatMulAlgorithmSelectionParams& params,
      std::optional<MatMulAlgorithm> forced_algorithm = std::nullopt) const;

 protected:
  virtual std::optional<MatMulAlgorithm> SelectVendorAlgorithm(
      const MatMulAlgorithmSelectionParams& params) const;

  virtual std::optional<MatMulAlgorithmConfiguration> SelectVendorConfiguration(
      MatMulAlgorithm algorithm,
      const MatMulAlgorithmSelectionParams& params) const;

 private:
  MatMulAlgorithm SelectCommonAlgorithm(const MatMulAlgorithmSelectionParams& params) const;

  MatMulAlgorithmConfiguration SelectCommonConfiguration(
      MatMulAlgorithm algorithm,
      const MatMulAlgorithmSelectionParams& params) const;

  bool ShouldUseSplitK(const MatMulAlgorithmSelectionParams& params) const;
  bool ShouldUseGemv(const MatMulAlgorithmSelectionParams& params) const;

  SplitKConfig split_k_config_;
};

}  // namespace webgpu
}  // namespace onnxruntime
