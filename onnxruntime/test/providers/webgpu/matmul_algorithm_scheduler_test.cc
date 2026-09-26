// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <limits>
#include <type_traits>

#include "gtest/gtest.h"

#include "core/providers/webgpu/math/matmul_algorithm.h"
#include "core/providers/webgpu/math/matmul_algorithm_scheduler.h"
#include "core/providers/webgpu/math/matmul_compute_dispatcher.h"
#include "core/providers/webgpu/vendor/intel/math/gemm_subgroup_utils.h"
#include "core/providers/webgpu/vendor/intel/math/matmul_algorithm_scheduler.h"
#include "core/providers/webgpu/vendor/intel/math/split_k_config.h"

namespace onnxruntime {
namespace webgpu {
namespace test {

namespace {

static_assert(!std::is_copy_constructible_v<MatMulAlgorithmScheduler>);
static_assert(!std::is_copy_assignable_v<MatMulAlgorithmScheduler>);
static_assert(!std::is_move_constructible_v<MatMulAlgorithmScheduler>);
static_assert(!std::is_move_assignable_v<MatMulAlgorithmScheduler>);
static_assert(!std::is_copy_constructible_v<SubgroupMatrixMatMulImpl>);
static_assert(!std::is_copy_assignable_v<SubgroupMatrixMatMulImpl>);
static_assert(!std::is_move_constructible_v<SubgroupMatrixMatMulImpl>);
static_assert(!std::is_move_assignable_v<SubgroupMatrixMatMulImpl>);
static_assert(!std::is_copy_constructible_v<MatMulComputeDispatcher>);
static_assert(!std::is_copy_assignable_v<MatMulComputeDispatcher>);
static_assert(!std::is_move_constructible_v<MatMulComputeDispatcher>);
static_assert(!std::is_move_assignable_v<MatMulComputeDispatcher>);

class AlwaysPackedVendorScheduler final : public MatMulAlgorithmScheduler {
 protected:
  std::optional<MatMulAlgorithm> SelectVendorAlgorithm(
      const MatMulAlgorithmSelectionParams& /*params*/) const override {
    return MatMulAlgorithm::Packed;
  }
};

class TunedPackedVendorScheduler final : public MatMulAlgorithmScheduler {
 protected:
  std::optional<MatMulAlgorithm> SelectVendorAlgorithm(
      const MatMulAlgorithmSelectionParams& /*params*/) const override {
    return MatMulAlgorithm::Naive;
  }

  std::optional<MatMulAlgorithmConfiguration> SelectVendorConfiguration(
      MatMulAlgorithm algorithm,
      const MatMulAlgorithmSelectionParams& params) const override {
    const bool is_packed_algorithm = algorithm == MatMulAlgorithm::Packed ||
                                     algorithm == MatMulAlgorithm::PackedSplitK;
    if (!is_packed_algorithm ||
        params.adapter_architecture != "test-architecture" ||
        params.a_data_type != 10 || params.b_data_type != 10) {
      return std::nullopt;
    }

    MatMulPackedConfiguration configuration{};
    configuration.workgroup_size = {16, 4, 1};
    configuration.elements_per_thread = {4, 2, 1};
    configuration.tile_inner = 16;
    configuration.split_dim_inner = 128;
    return configuration;
  }
};

class VendorSplitKThresholdScheduler final : public MatMulAlgorithmScheduler {
 public:
  VendorSplitKThresholdScheduler()
      : MatMulAlgorithmScheduler{SplitKConfig{
            /*max_batch_size=*/8,
            /*split_dim_inner=*/128,
            /*min_dim_inner_with_split_k=*/256,
            {{4096, 16.0}}}} {}

 protected:
  std::optional<MatMulAlgorithm> SelectVendorAlgorithm(
      const MatMulAlgorithmSelectionParams& params) const override {
    if (params.batch_size <= 16 && params.is_vec4 &&
        !params.deterministic_compute && !params.has_fused_activation &&
        (!params.has_bias || params.is_channels_last)) {
      return MatMulAlgorithm::PackedSplitK;
    }
    return std::nullopt;
  }
};

}  // namespace

TEST(MatMulAlgorithmParsingTest, RoundTripsEveryAlgorithmName) {
  struct TestCase {
    std::string_view name;
    MatMulAlgorithm algorithm;
  };

  constexpr TestCase test_cases[] = {
      {"subgroup_matrix", MatMulAlgorithm::SubgroupMatrix},
      {"naive", MatMulAlgorithm::Naive},
      {"intel_subgroup", MatMulAlgorithm::IntelSubgroup},
      {"packed", MatMulAlgorithm::Packed},
      {"packed_split_k", MatMulAlgorithm::PackedSplitK},
  };

  for (const auto& test_case : test_cases) {
    SCOPED_TRACE(test_case.name);
    EXPECT_EQ(ParseMatMulAlgorithm(test_case.name), test_case.algorithm);
    EXPECT_EQ(MatMulAlgorithmName(test_case.algorithm), test_case.name);
  }
}

TEST(MatMulAlgorithmParsingTest, RejectsUnknownAlgorithmName) {
  EXPECT_EQ(ParseMatMulAlgorithm("unknown"), std::nullopt);
}

TEST(SplitKConfigTest, IntelArchitectureProfilesPreserveCurrentBoundaries) {
  const SplitKConfig discrete_config = intel::CreateSplitKConfig("xe-2lpg");
  EXPECT_EQ(discrete_config.GetSplitDimInner(), 256u);
  EXPECT_TRUE(discrete_config.UseSplitK(
      /*is_vec4=*/true, ActivationKind::None, /*batch_size=*/1,
      /*dim_a_outer=*/192, /*dim_b_outer=*/160, /*dim_inner=*/1024));

  const SplitKConfig xe3_config = intel::CreateSplitKConfig("xe-3lpg");
  EXPECT_FALSE(xe3_config.UseSplitK(
      /*is_vec4=*/true, ActivationKind::None, /*batch_size=*/1,
      /*dim_a_outer=*/192, /*dim_b_outer=*/160, /*dim_inner=*/1024));
  EXPECT_TRUE(xe3_config.UseSplitK(
      /*is_vec4=*/true, ActivationKind::None, /*batch_size=*/1,
      /*dim_a_outer=*/128, /*dim_b_outer=*/128, /*dim_inner=*/1024));

  const SplitKConfig default_config = intel::CreateSplitKConfig("gen-12lp");
  EXPECT_FALSE(default_config.UseSplitK(
      /*is_vec4=*/true, ActivationKind::None, /*batch_size=*/1,
      /*dim_a_outer=*/128, /*dim_b_outer=*/128, /*dim_inner=*/1024));

  const SplitKConfig legacy_config = intel::CreateSplitKConfig("gen-9");
  EXPECT_EQ(legacy_config.GetSplitDimInner(), 0u);
  EXPECT_FALSE(legacy_config.UseSplitK(
      /*is_vec4=*/true, ActivationKind::None, /*batch_size=*/1,
      /*dim_a_outer=*/1, /*dim_b_outer=*/1, /*dim_inner=*/1024));
}

TEST(SplitKConfigTest, FactoryRoutesOnlySupportedVendorProfiles) {
  EXPECT_EQ(CreateSplitKConfig("intel", "xe-2lpg").GetSplitDimInner(), 256u);
  EXPECT_EQ(CreateSplitKConfig("nvidia", "pascal").GetSplitDimInner(), 0u);
}

TEST(MatMulAlgorithmSchedulerTest, ForcedAlgorithmTakesPrecedence) {
  AlwaysPackedVendorScheduler scheduler;
  MatMulAlgorithmSelectionParams params{};
  params.can_use_subgroup_matrix = true;

  EXPECT_EQ(scheduler.Select(params, MatMulAlgorithm::Naive), MatMulAlgorithm::Naive);
}

TEST(MatMulAlgorithmSchedulerTest, ReevaluatesSelectionForEachRuntimeShape) {
  MatMulAlgorithmScheduler scheduler;
  MatMulAlgorithmSelectionParams params{};
  params.m = 4;
  params.packed_m = 4;
  params.n = 7;
  params.k = 7;

  EXPECT_EQ(scheduler.CreateExecutionPlan(params).algorithm, MatMulAlgorithm::Naive);

  params.m = 64;
  params.packed_m = 64;
  params.n = 64;
  params.k = 64;
  EXPECT_EQ(scheduler.CreateExecutionPlan(params).algorithm, MatMulAlgorithm::Packed);
}

TEST(MatMulAlgorithmSchedulerTest, VendorCanTuneForcedAlgorithmConfiguration) {
  TunedPackedVendorScheduler scheduler;
  MatMulAlgorithmSelectionParams params{};
  params.adapter_architecture = "test-architecture";
  params.a_data_type = 10;
  params.b_data_type = 10;

  const MatMulExecutionPlan plan =
      scheduler.CreateExecutionPlan(params, MatMulAlgorithm::Packed);

  EXPECT_EQ(plan.algorithm, MatMulAlgorithm::Packed);
  const auto& configuration =
      std::get<MatMulPackedConfiguration>(plan.configuration);
  EXPECT_EQ(configuration.workgroup_size,
            (std::array<uint32_t, 3>{16, 4, 1}));
  EXPECT_EQ(configuration.elements_per_thread,
            (std::array<uint32_t, 3>{4, 2, 1}));
  EXPECT_EQ(configuration.tile_inner, 16u);
  EXPECT_EQ(configuration.split_dim_inner, 128u);
}

TEST(MatMulAlgorithmSchedulerTest, VendorCanSetIndependentSplitKThresholds) {
  VendorSplitKThresholdScheduler scheduler;
  MatMulAlgorithmSelectionParams params{};
  params.m = 32;
  params.n = 64;
  params.k = 1024;
  params.batch_size = 16;
  params.packed_batch_size = 16;
  params.is_vec4 = true;
  params.is_channels_last = true;

  const MatMulExecutionPlan plan = scheduler.CreateExecutionPlan(params);
  EXPECT_EQ(plan.algorithm, MatMulAlgorithm::PackedSplitK);
  EXPECT_EQ(std::get<MatMulPackedConfiguration>(plan.configuration).split_dim_inner, 128u);

  params.batch_size = 17;
  params.packed_batch_size = 17;
  EXPECT_EQ(scheduler.Select(params), MatMulAlgorithm::Packed);
}

TEST(MatMulAlgorithmSchedulerTest, InjectedSplitKPolicyCreatesPackedSplitKPlan) {
  intel::IntelMatMulAlgorithmScheduler scheduler{intel::CreateSplitKConfig("xe-2lpg")};
  MatMulAlgorithmSelectionParams params{};
  params.m = 192;
  params.packed_m = 192;
  params.n = 160;
  params.k = 1024;
  params.is_vec4 = true;

  const MatMulExecutionPlan plan = scheduler.CreateExecutionPlan(params);

  EXPECT_EQ(plan.algorithm, MatMulAlgorithm::PackedSplitK);
  const auto& configuration =
      std::get<MatMulPackedConfiguration>(plan.configuration);
  EXPECT_EQ(configuration.split_dim_inner, 256u);
}

TEST(MatMulAlgorithmSchedulerTest, CommonPackedConfigurationPreservesCurrentTuning) {
  MatMulAlgorithmScheduler scheduler;
  MatMulAlgorithmSelectionParams params{};
  params.m = 8;
  params.packed_m = 8;
  params.n = 64;
  params.k = 64;

  const MatMulExecutionPlan small_m_plan = scheduler.CreateExecutionPlan(params);
  const auto& small_m_configuration =
      std::get<MatMulPackedConfiguration>(small_m_plan.configuration);
  EXPECT_EQ(small_m_configuration.workgroup_size,
            (std::array<uint32_t, 3>{8, 8, 1}));
  EXPECT_EQ(small_m_configuration.elements_per_thread,
            (std::array<uint32_t, 3>{4, 1, 1}));
  EXPECT_EQ(small_m_configuration.tile_inner, 32u);

  params.m = 9;
  params.packed_m = 9;
  const MatMulExecutionPlan large_m_plan = scheduler.CreateExecutionPlan(params);
  const auto& large_m_configuration =
      std::get<MatMulPackedConfiguration>(large_m_plan.configuration);
  EXPECT_EQ(large_m_configuration.elements_per_thread,
            (std::array<uint32_t, 3>{4, 4, 1}));
}

TEST(MatMulAlgorithmConfigurationTest, PackedConfigurationKeepsBatchAxesUntiled) {
  MatMulPackedConfiguration configuration{};
  EXPECT_TRUE(IsMatMulPackedConfigurationValid(configuration, /*use_split_k=*/false));

  configuration.workgroup_size[2] = 2;
  EXPECT_FALSE(IsMatMulPackedConfigurationValid(configuration, /*use_split_k=*/false));

  configuration.workgroup_size[2] = 1;
  configuration.elements_per_thread[2] = 2;
  EXPECT_FALSE(IsMatMulPackedConfigurationValid(configuration, /*use_split_k=*/false));
}

TEST(MatMulAlgorithmConfigurationTest, SplitKConfigurationRequiresTileAlignedSplits) {
  MatMulPackedConfiguration configuration{};
  configuration.tile_inner = 32;
  configuration.split_dim_inner = 64;
  EXPECT_TRUE(IsMatMulPackedConfigurationValid(configuration, /*use_split_k=*/true));

  configuration.split_dim_inner = 48;
  EXPECT_FALSE(IsMatMulPackedConfigurationValid(configuration, /*use_split_k=*/true));
}

TEST(MatMulAlgorithmConfigurationTest, PackedDispatchArithmeticIsOverflowSafe) {
  const auto maximum_tuning_dispatch = TryGetMatMulPackedDispatchGroupCount(
      std::numeric_limits<uint32_t>::max(),
      std::numeric_limits<uint32_t>::max(),
      std::numeric_limits<uint32_t>::max());
  ASSERT_TRUE(maximum_tuning_dispatch.has_value());
  EXPECT_EQ(*maximum_tuning_dispatch, 1u);

  EXPECT_EQ(TryGetMatMulPackedDispatchGroupCount(
                std::numeric_limits<uint64_t>::max(), 1, 1),
            std::nullopt);
}

TEST(MatMulAlgorithmSchedulerTest, CommonFallbackPrefersSubgroupMatrix) {
  MatMulAlgorithmScheduler scheduler{intel::CreateSplitKConfig("xe-2lpg")};
  MatMulAlgorithmSelectionParams params{};
  params.m = 64;
  params.packed_m = 64;
  params.n = 64;
  params.k = 1024;
  params.can_use_subgroup_matrix = true;
  params.is_vec4 = true;

  EXPECT_EQ(scheduler.Select(params), MatMulAlgorithm::SubgroupMatrix);
}

TEST(MatMulAlgorithmSchedulerTest, VendorPolicyPrecedesCommonHeuristics) {
  AlwaysPackedVendorScheduler scheduler;
  MatMulAlgorithmSelectionParams params{};
  params.n = 4;
  params.k = 4;
  params.can_use_subgroup_matrix = true;

  EXPECT_EQ(scheduler.Select(params), MatMulAlgorithm::Packed);
}

TEST(MatMulAlgorithmSchedulerTest, ZeroContractionDimensionPrecedesVendorPolicy) {
  AlwaysPackedVendorScheduler scheduler;
  MatMulAlgorithmSelectionParams params{};
  params.n = 8;
  params.k = 0;

  EXPECT_EQ(scheduler.Select(params), MatMulAlgorithm::Naive);
}

TEST(MatMulAlgorithmSchedulerTest, NaiveUsesStrictSmallDimensionBoundaries) {
  MatMulAlgorithmScheduler scheduler;

  MatMulAlgorithmSelectionParams small_params{};
  small_params.n = 7;
  small_params.k = 7;
  EXPECT_EQ(scheduler.Select(small_params), MatMulAlgorithm::Naive);

  MatMulAlgorithmSelectionParams n_boundary = small_params;
  n_boundary.n = 8;
  EXPECT_EQ(scheduler.Select(n_boundary), MatMulAlgorithm::Packed);

  MatMulAlgorithmSelectionParams k_boundary = small_params;
  k_boundary.k = 8;
  EXPECT_EQ(scheduler.Select(k_boundary), MatMulAlgorithm::Packed);
}

TEST(MatMulAlgorithmSchedulerTest, ZeroContractionDimensionUsesNaive) {
  MatMulAlgorithmScheduler scheduler;
  MatMulAlgorithmSelectionParams params{};
  params.n = 8;
  params.k = 0;

  EXPECT_EQ(scheduler.Select(params), MatMulAlgorithm::Naive);
}

TEST(MatMulAlgorithmSchedulerTest, IntelSchedulerAppliesCurrentVendorRule) {
  intel::IntelMatMulAlgorithmScheduler scheduler;
  MatMulAlgorithmSelectionParams params{};
  params.m = 64;
  params.n = 512;
  params.k = 32;
  params.has_intel_subgroup_capability = true;

  EXPECT_EQ(scheduler.Select(params), MatMulAlgorithm::IntelSubgroup);

  params.n = 511;
  EXPECT_EQ(scheduler.Select(params), MatMulAlgorithm::Packed);
}

TEST(MatMulAlgorithmSchedulerTest, IntelSchedulerPreservesSubgroupMatrixPrecedence) {
  intel::IntelMatMulAlgorithmScheduler scheduler;
  MatMulAlgorithmSelectionParams params{};
  params.m = 64;
  params.n = 512;
  params.k = 32;
  params.can_use_subgroup_matrix = true;
  params.has_intel_subgroup_capability = true;

  EXPECT_EQ(scheduler.Select(params), MatMulAlgorithm::SubgroupMatrix);
}

TEST(MatMulAlgorithmSchedulerTest, SplitKPrecedesPackedFallback) {
  MatMulAlgorithmScheduler scheduler{intel::CreateSplitKConfig("xe-2lpg")};
  MatMulAlgorithmSelectionParams params{};
  params.m = 64;
  params.packed_m = 64;
  params.n = 64;
  params.k = 1024;
  params.is_vec4 = true;
  EXPECT_EQ(scheduler.Select(params), MatMulAlgorithm::PackedSplitK);

  MatMulAlgorithmScheduler disabled_scheduler;
  EXPECT_EQ(disabled_scheduler.Select(params), MatMulAlgorithm::Packed);
}

TEST(MatMulAlgorithmPrerequisiteTest, SplitKRejectsEachHardConstraint) {
  MatMulAlgorithmPrerequisites prerequisites{};
  prerequisites.has_nonzero_k = true;
  prerequisites.split_k_configured = true;
  prerequisites.is_vec4 = true;
  prerequisites.split_k_bias_layout_supported = true;
  EXPECT_TRUE(MeetsMatMulAlgorithmPrerequisites(MatMulAlgorithm::PackedSplitK, prerequisites));

  prerequisites.deterministic_compute = true;
  EXPECT_FALSE(MeetsMatMulAlgorithmPrerequisites(MatMulAlgorithm::PackedSplitK, prerequisites));
  prerequisites.deterministic_compute = false;

  prerequisites.is_vec4 = false;
  EXPECT_FALSE(MeetsMatMulAlgorithmPrerequisites(MatMulAlgorithm::PackedSplitK, prerequisites));
  prerequisites.is_vec4 = true;

  prerequisites.has_fused_activation = true;
  EXPECT_FALSE(MeetsMatMulAlgorithmPrerequisites(MatMulAlgorithm::PackedSplitK, prerequisites));
  prerequisites.has_fused_activation = false;

  prerequisites.split_k_bias_layout_supported = false;
  EXPECT_FALSE(MeetsMatMulAlgorithmPrerequisites(MatMulAlgorithm::PackedSplitK, prerequisites));
}

TEST(MatMulAlgorithmPrerequisiteTest, PackedAlgorithmsRejectZeroContractionDimension) {
  MatMulAlgorithmPrerequisites prerequisites{};
  prerequisites.split_k_configured = true;
  prerequisites.is_vec4 = true;
  prerequisites.split_k_bias_layout_supported = true;

  EXPECT_FALSE(MeetsMatMulAlgorithmPrerequisites(MatMulAlgorithm::Packed, prerequisites));
  EXPECT_FALSE(MeetsMatMulAlgorithmPrerequisites(MatMulAlgorithm::PackedSplitK, prerequisites));

  prerequisites.has_nonzero_k = true;
  EXPECT_TRUE(MeetsMatMulAlgorithmPrerequisites(MatMulAlgorithm::Packed, prerequisites));
  EXPECT_TRUE(MeetsMatMulAlgorithmPrerequisites(MatMulAlgorithm::PackedSplitK, prerequisites));
}

TEST(MatMulAlgorithmPrerequisiteTest, IntelAVec4RequiresCompatibleRowsPerThread) {
  EXPECT_FALSE(intel::CanUseAVec4CooperativeLoad(intel::gpu_arch::kXe3Lpg, 32, 1));
  EXPECT_FALSE(intel::CanUseAVec4CooperativeLoad(intel::gpu_arch::kXe3Lpg, 32, 2));
  EXPECT_TRUE(intel::CanUseAVec4CooperativeLoad(intel::gpu_arch::kXe3Lpg, 32, 4));
  EXPECT_FALSE(intel::CanUseAVec4CooperativeLoad(intel::gpu_arch::kXe3Lpg, 31, 4));
  EXPECT_FALSE(intel::CanUseAVec4CooperativeLoad(intel::gpu_arch::kXeLpg, 32, 4));
}

TEST(MatMulAlgorithmPrerequisiteTest, IntelCapabilityDoesNotIncludeAutomaticThresholds) {
  MatMulAlgorithmPrerequisites prerequisites{};
  prerequisites.has_intel_subgroup_capability = true;
  prerequisites.has_nonzero_k = true;
  EXPECT_TRUE(MeetsMatMulAlgorithmPrerequisites(MatMulAlgorithm::IntelSubgroup, prerequisites));

  MatMulAlgorithmSelectionParams below_heuristic_threshold{};
  below_heuristic_threshold.m = 1;
  below_heuristic_threshold.n = 1;
  below_heuristic_threshold.k = 1;
  below_heuristic_threshold.has_intel_subgroup_capability = true;
  intel::IntelMatMulAlgorithmScheduler scheduler;
  EXPECT_NE(scheduler.Select(below_heuristic_threshold), MatMulAlgorithm::IntelSubgroup);
}

TEST(MatMulAlgorithmPrerequisiteTest, IntelSubgroupRejectsZeroContractionDimension) {
  MatMulAlgorithmPrerequisites prerequisites{};
  prerequisites.has_intel_subgroup_capability = true;

  EXPECT_FALSE(MeetsMatMulAlgorithmPrerequisites(MatMulAlgorithm::IntelSubgroup, prerequisites));

  prerequisites.has_nonzero_k = true;
  EXPECT_TRUE(MeetsMatMulAlgorithmPrerequisites(MatMulAlgorithm::IntelSubgroup, prerequisites));
}

}  // namespace test
}  // namespace webgpu
}  // namespace onnxruntime
