// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <limits>
#include <type_traits>

#include "gtest/gtest.h"

#include "core/providers/webgpu/math/matmul_algorithm.h"
#include "core/providers/webgpu/math/matmul_execution_planner.h"
#include "core/providers/webgpu/math/matmul_compute_dispatcher.h"
#include "core/providers/webgpu/vendor/intel/math/gemm_subgroup_utils.h"
#include "core/providers/webgpu/vendor/intel/math/matmul_execution_planner.h"
#include "core/providers/webgpu/vendor/intel/math/split_k_config.h"

namespace onnxruntime {
namespace webgpu {
namespace test {

namespace {

static_assert(!std::is_copy_constructible_v<MatMulExecutionPlanner>);
static_assert(!std::is_copy_assignable_v<MatMulExecutionPlanner>);
static_assert(!std::is_move_constructible_v<MatMulExecutionPlanner>);
static_assert(!std::is_move_assignable_v<MatMulExecutionPlanner>);
static_assert(!std::is_copy_constructible_v<SubgroupMatrixMatMulImpl>);
static_assert(!std::is_copy_assignable_v<SubgroupMatrixMatMulImpl>);
static_assert(!std::is_move_constructible_v<SubgroupMatrixMatMulImpl>);
static_assert(!std::is_move_assignable_v<SubgroupMatrixMatMulImpl>);
static_assert(!std::is_copy_constructible_v<MatMulComputeDispatcher>);
static_assert(!std::is_copy_assignable_v<MatMulComputeDispatcher>);
static_assert(!std::is_move_constructible_v<MatMulComputeDispatcher>);
static_assert(!std::is_move_assignable_v<MatMulComputeDispatcher>);

class AlwaysPackedVendorPlanner final : public MatMulExecutionPlanner {
 protected:
  std::optional<MatMulAlgorithm> SelectVendorAlgorithm(
      const MatMulAlgorithmSelectionParams& /*params*/) const override {
    return MatMulAlgorithm::Packed;
  }
};

class TunedPackedVendorPlanner final : public MatMulExecutionPlanner {
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
    configuration.tile_inner = 64;
    configuration.split_dim_inner = 128;
    return configuration;
  }
};

class VendorSplitKThresholdPlanner final : public MatMulExecutionPlanner {
 public:
  VendorSplitKThresholdPlanner()
      : MatMulExecutionPlanner{SplitKConfig{
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

class IncompatibleConfigurationPlanner final : public MatMulExecutionPlanner {
 protected:
  std::optional<MatMulAlgorithmConfiguration> SelectVendorConfiguration(
      MatMulAlgorithm algorithm,
      const MatMulAlgorithmSelectionParams& /*params*/) const override {
    return algorithm == MatMulAlgorithm::Packed
               ? std::optional<MatMulAlgorithmConfiguration>{MatMulGemvConfiguration{}}
               : std::nullopt;
  }
};

class InvalidPackedConfigurationPlanner final : public MatMulExecutionPlanner {
 protected:
  std::optional<MatMulAlgorithmConfiguration> SelectVendorConfiguration(
      MatMulAlgorithm algorithm,
      const MatMulAlgorithmSelectionParams& /*params*/) const override {
    if (algorithm != MatMulAlgorithm::Packed) {
      return std::nullopt;
    }
    MatMulPackedConfiguration configuration{};
    configuration.workgroup_size[0] = 0;
    return configuration;
  }
};

class MismatchedSubgroupConfigurationPlanner final : public MatMulExecutionPlanner {
 protected:
  std::optional<MatMulAlgorithmConfiguration> SelectVendorConfiguration(
      MatMulAlgorithm algorithm,
      const MatMulAlgorithmSelectionParams& /*params*/) const override {
    return algorithm == MatMulAlgorithm::Subgroup
               ? std::optional<MatMulAlgorithmConfiguration>{MatMulSubgroupConfiguration{8}}
               : std::nullopt;
  }
};

MatMulExecutionPlan CreatePlan(
    const MatMulExecutionPlanner& planner,
    const MatMulAlgorithmSelectionParams& params,
    std::optional<MatMulAlgorithm> forced_algorithm = std::nullopt) {
  MatMulExecutionPlan plan{MatMulAlgorithm::Naive, MatMulNaiveConfiguration{}};
  const Status status = planner.CreateExecutionPlan(params, forced_algorithm, plan);
  EXPECT_TRUE(status.IsOK()) << status.ErrorMessage();
  return plan;
}

}  // namespace

TEST(MatMulAlgorithmNameTest, ReturnsEveryAlgorithmName) {
  struct TestCase {
    std::string_view name;
    MatMulAlgorithm algorithm;
  };

  constexpr TestCase test_cases[] = {
      {"subgroup_matrix", MatMulAlgorithm::SubgroupMatrix},
      {"gemv", MatMulAlgorithm::Gemv},
      {"naive", MatMulAlgorithm::Naive},
      {"subgroup", MatMulAlgorithm::Subgroup},
      {"packed", MatMulAlgorithm::Packed},
      {"packed_split_k", MatMulAlgorithm::PackedSplitK},
  };

  for (const auto& test_case : test_cases) {
    SCOPED_TRACE(test_case.name);
    EXPECT_EQ(MatMulAlgorithmName(test_case.algorithm), test_case.name);
  }
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

TEST(MatMulExecutionPlannerTest, ForcedAlgorithmTakesPrecedence) {
  AlwaysPackedVendorPlanner planner;
  MatMulAlgorithmSelectionParams params{};
  params.can_use_subgroup_matrix = true;

  EXPECT_EQ(planner.SelectAlgorithm(params, MatMulAlgorithm::Naive), MatMulAlgorithm::Naive);
}

TEST(MatMulExecutionPlannerTest, ReevaluatesSelectionForEachRuntimeShape) {
  MatMulExecutionPlanner planner;
  MatMulAlgorithmSelectionParams params{};
  params.m = 4;
  params.packed_m = 4;
  params.n = 7;
  params.k = 7;

  EXPECT_EQ(CreatePlan(planner, params).algorithm, MatMulAlgorithm::Naive);

  params.m = 64;
  params.packed_m = 64;
  params.n = 64;
  params.k = 64;
  EXPECT_EQ(CreatePlan(planner, params).algorithm, MatMulAlgorithm::Packed);
}

TEST(MatMulExecutionPlannerTest, VendorCanTuneForcedAlgorithmConfiguration) {
  TunedPackedVendorPlanner planner;
  MatMulAlgorithmSelectionParams params{};
  params.adapter_architecture = "test-architecture";
  params.a_data_type = 10;
  params.b_data_type = 10;
  params.k = 64;

  const MatMulExecutionPlan plan =
      CreatePlan(planner, params, MatMulAlgorithm::Packed);

  EXPECT_EQ(plan.algorithm, MatMulAlgorithm::Packed);
  const auto& configuration =
      std::get<MatMulPackedConfiguration>(plan.configuration);
  EXPECT_EQ(configuration.workgroup_size,
            (std::array<uint32_t, 3>{16, 4, 1}));
  EXPECT_EQ(configuration.elements_per_thread,
            (std::array<uint32_t, 3>{4, 2, 1}));
  EXPECT_EQ(configuration.tile_inner, 64u);
  EXPECT_EQ(configuration.split_dim_inner, 128u);
}

TEST(MatMulExecutionPlannerTest, VendorCanSetIndependentSplitKThresholds) {
  VendorSplitKThresholdPlanner planner;
  MatMulAlgorithmSelectionParams params{};
  params.m = 32;
  params.n = 64;
  params.k = 1024;
  params.batch_size = 16;
  params.packed_batch_size = 16;
  params.is_vec4 = true;
  params.is_channels_last = true;

  const MatMulExecutionPlan plan = CreatePlan(planner, params);
  EXPECT_EQ(plan.algorithm, MatMulAlgorithm::PackedSplitK);
  EXPECT_EQ(std::get<MatMulPackedConfiguration>(plan.configuration).split_dim_inner, 128u);

  params.batch_size = 17;
  params.packed_batch_size = 17;
  EXPECT_EQ(planner.SelectAlgorithm(params), MatMulAlgorithm::Packed);
}

TEST(MatMulExecutionPlannerTest, InjectedSplitKPolicyCreatesPackedSplitKPlan) {
  intel::IntelMatMulExecutionPlanner planner{intel::CreateSplitKConfig("xe-2lpg")};
  MatMulAlgorithmSelectionParams params{};
  params.m = 192;
  params.packed_m = 192;
  params.n = 160;
  params.k = 1024;
  params.is_vec4 = true;

  const MatMulExecutionPlan plan = CreatePlan(planner, params);

  EXPECT_EQ(plan.algorithm, MatMulAlgorithm::PackedSplitK);
  const auto& configuration =
      std::get<MatMulPackedConfiguration>(plan.configuration);
  EXPECT_EQ(configuration.split_dim_inner, 256u);
}

TEST(MatMulExecutionPlannerTest, CommonPackedConfigurationPreservesCurrentTuning) {
  MatMulExecutionPlanner planner;
  MatMulAlgorithmSelectionParams params{};
  params.m = 8;
  params.packed_m = 8;
  params.n = 64;
  params.k = 64;

  const MatMulExecutionPlan small_m_plan = CreatePlan(planner, params);
  const auto& small_m_configuration =
      std::get<MatMulPackedConfiguration>(small_m_plan.configuration);
  EXPECT_EQ(small_m_configuration.workgroup_size,
            (std::array<uint32_t, 3>{8, 8, 1}));
  EXPECT_EQ(small_m_configuration.elements_per_thread,
            (std::array<uint32_t, 3>{4, 1, 1}));
  EXPECT_EQ(small_m_configuration.tile_inner, 32u);

  params.m = 9;
  params.packed_m = 9;
  const MatMulExecutionPlan large_m_plan = CreatePlan(planner, params);
  const auto& large_m_configuration =
      std::get<MatMulPackedConfiguration>(large_m_plan.configuration);
  EXPECT_EQ(large_m_configuration.elements_per_thread,
            (std::array<uint32_t, 3>{4, 4, 1}));
}

TEST(MatMulAlgorithmConfigurationTest, PackedConfigurationKeepsBatchAxesUntiled) {
  MatMulPackedConfiguration configuration{};
  EXPECT_TRUE(IsMatMulPackedConfigurationValid(configuration, /*is_vec4=*/false,
                                               /*use_split_k=*/false));
  EXPECT_TRUE(IsMatMulPackedConfigurationValid(configuration, /*is_vec4=*/true,
                                               /*use_split_k=*/false));

  configuration.workgroup_size[2] = 2;
  EXPECT_FALSE(IsMatMulPackedConfigurationValid(configuration, /*is_vec4=*/false,
                                                /*use_split_k=*/false));

  configuration.workgroup_size[2] = 1;
  configuration.elements_per_thread[2] = 2;
  EXPECT_FALSE(IsMatMulPackedConfigurationValid(configuration, /*is_vec4=*/false,
                                                /*use_split_k=*/false));
}

TEST(MatMulAlgorithmConfigurationTest, PackedVec4RequiresSupportedCooperativeLoads) {
  MatMulPackedConfiguration configuration{};

  configuration.elements_per_thread[0] = 8;
  EXPECT_FALSE(IsMatMulPackedConfigurationValid(configuration, /*is_vec4=*/true,
                                                /*use_split_k=*/false));

  configuration.elements_per_thread[0] = 4;
  configuration.workgroup_size = {16, 4, 1};
  configuration.tile_inner = 16;
  EXPECT_FALSE(IsMatMulPackedConfigurationValid(configuration, /*is_vec4=*/true,
                                                /*use_split_k=*/false));

  configuration.tile_inner = 48;
  EXPECT_FALSE(IsMatMulPackedConfigurationValid(configuration, /*is_vec4=*/true,
                                                /*use_split_k=*/false));

  configuration.tile_inner = 60;
  EXPECT_FALSE(IsMatMulPackedConfigurationValid(configuration, /*is_vec4=*/true,
                                                /*use_split_k=*/false));
}

TEST(MatMulAlgorithmConfigurationTest, SubgroupConfigurationCarriesSelectedSize) {
  MatMulExecutionPlanner planner;
  MatMulAlgorithmSelectionParams params{};
  params.m = 1;
  params.n = 1;
  params.k = 1;
  params.has_subgroup_capability = true;
  params.subgroup_size = 16;

  const auto plan = CreatePlan(planner, params, MatMulAlgorithm::Subgroup);
  const auto* configuration = std::get_if<MatMulSubgroupConfiguration>(&plan.configuration);
  ASSERT_NE(configuration, nullptr);
  EXPECT_EQ(configuration->subgroup_size, 16u);
}

TEST(MatMulAlgorithmConfigurationTest, SubgroupSizeSelectionRequiresSupportedSize) {
  EXPECT_EQ(intel::SelectMatMulSubgroupSize(8, 8, false), 8u);
  EXPECT_EQ(intel::SelectMatMulSubgroupSize(16, 16, false), 16u);
  EXPECT_EQ(intel::SelectMatMulSubgroupSize(32, 32, false), 32u);
  EXPECT_FALSE(intel::SelectMatMulSubgroupSize(64, 64, false).has_value());
  EXPECT_FALSE(intel::SelectMatMulSubgroupSize(8, 32, false).has_value());
  EXPECT_EQ(intel::SelectMatMulSubgroupSize(8, 32, true), 32u);
  EXPECT_EQ(intel::SelectMatMulSubgroupSize(8, 16, true), 16u);
  EXPECT_FALSE(intel::SelectMatMulSubgroupSize(4, 4, true).has_value());
}

TEST(MatMulAlgorithmConfigurationTest, SplitKConfigurationRequiresTileAlignedSplits) {
  MatMulPackedConfiguration configuration{};
  configuration.tile_inner = 32;
  configuration.split_dim_inner = 64;
  EXPECT_TRUE(IsMatMulPackedConfigurationValid(configuration, /*is_vec4=*/true,
                                               /*use_split_k=*/true));

  configuration.split_dim_inner = 48;
  EXPECT_FALSE(IsMatMulPackedConfigurationValid(configuration, /*is_vec4=*/true,
                                                /*use_split_k=*/true));
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

TEST(MatMulExecutionPlannerTest, CommonFallbackPrefersSubgroupMatrix) {
  MatMulExecutionPlanner planner{intel::CreateSplitKConfig("xe-2lpg")};
  MatMulAlgorithmSelectionParams params{};
  params.m = 64;
  params.packed_m = 64;
  params.n = 64;
  params.k = 1024;
  params.can_use_subgroup_matrix = true;
  params.is_vec4 = true;

  EXPECT_EQ(planner.SelectAlgorithm(params), MatMulAlgorithm::SubgroupMatrix);
}

TEST(MatMulExecutionPlannerTest, GemvPrecedesGenericFallbacks) {
  MatMulExecutionPlanner planner{intel::CreateSplitKConfig("xe-2lpg")};
  MatMulAlgorithmSelectionParams params{};
  params.m = 1;
  params.packed_m = 1;
  params.n = 48;
  params.k = 5120;
  params.can_run_gemv = true;
  params.is_vec4 = true;

  EXPECT_EQ(planner.SelectAlgorithm(params), MatMulAlgorithm::Gemv);

  params.can_use_subgroup_matrix = true;
  EXPECT_EQ(planner.SelectAlgorithm(params), MatMulAlgorithm::SubgroupMatrix);
}

TEST(MatMulExecutionPlannerTest, GemvAutomaticSelectionUsesPerformanceThresholds) {
  MatMulExecutionPlanner planner;
  MatMulAlgorithmSelectionParams params{};
  params.m = 1;
  params.n = 16;
  params.k = 2048;
  params.can_run_gemv = true;

  EXPECT_EQ(planner.SelectAlgorithm(params), MatMulAlgorithm::Gemv);

  params.n = 12;
  EXPECT_EQ(planner.SelectAlgorithm(params), MatMulAlgorithm::Packed);
  params.n = 68;
  EXPECT_EQ(planner.SelectAlgorithm(params), MatMulAlgorithm::Packed);
  params.n = 16;

  params.k = 2047;
  EXPECT_EQ(planner.SelectAlgorithm(params), MatMulAlgorithm::Packed);
  params.k = 8193;
  EXPECT_EQ(planner.SelectAlgorithm(params), MatMulAlgorithm::Packed);
}

TEST(MatMulAlgorithmConfigurationTest, GemvUsesTypedEmptyConfiguration) {
  MatMulExecutionPlanner planner;
  MatMulAlgorithmSelectionParams params{};
  params.m = 1;
  params.n = 48;
  params.k = 5120;
  params.can_run_gemv = true;

  const MatMulExecutionPlan plan = CreatePlan(planner, params);
  EXPECT_EQ(plan.algorithm, MatMulAlgorithm::Gemv);
  EXPECT_TRUE(std::holds_alternative<MatMulGemvConfiguration>(plan.configuration));
}

TEST(MatMulExecutionPlannerTest, VendorPolicyPrecedesCommonHeuristics) {
  AlwaysPackedVendorPlanner planner;
  MatMulAlgorithmSelectionParams params{};
  params.n = 4;
  params.k = 4;
  params.can_use_subgroup_matrix = true;

  EXPECT_EQ(planner.SelectAlgorithm(params), MatMulAlgorithm::Packed);
}

TEST(MatMulExecutionPlannerTest, ZeroContractionDimensionPrecedesVendorPolicy) {
  AlwaysPackedVendorPlanner planner;
  MatMulAlgorithmSelectionParams params{};
  params.n = 8;
  params.k = 0;

  EXPECT_EQ(planner.SelectAlgorithm(params), MatMulAlgorithm::Naive);
}

TEST(MatMulExecutionPlannerTest, NaiveUsesStrictSmallDimensionBoundaries) {
  MatMulExecutionPlanner planner;

  MatMulAlgorithmSelectionParams small_params{};
  small_params.n = 7;
  small_params.k = 7;
  EXPECT_EQ(planner.SelectAlgorithm(small_params), MatMulAlgorithm::Naive);

  MatMulAlgorithmSelectionParams n_boundary = small_params;
  n_boundary.n = 8;
  EXPECT_EQ(planner.SelectAlgorithm(n_boundary), MatMulAlgorithm::Packed);

  MatMulAlgorithmSelectionParams k_boundary = small_params;
  k_boundary.k = 8;
  EXPECT_EQ(planner.SelectAlgorithm(k_boundary), MatMulAlgorithm::Packed);
}

TEST(MatMulExecutionPlannerTest, ZeroContractionDimensionUsesNaive) {
  MatMulExecutionPlanner planner;
  MatMulAlgorithmSelectionParams params{};
  params.n = 8;
  params.k = 0;

  EXPECT_EQ(planner.SelectAlgorithm(params), MatMulAlgorithm::Naive);
}

TEST(MatMulExecutionPlannerTest, IntelPlannerAppliesCurrentVendorRule) {
  intel::IntelMatMulExecutionPlanner planner;
  MatMulAlgorithmSelectionParams params{};
  params.m = 64;
  params.n = 512;
  params.k = 32;
  params.has_subgroup_capability = true;

  EXPECT_EQ(planner.SelectAlgorithm(params), MatMulAlgorithm::Subgroup);

  params.n = 511;
  EXPECT_EQ(planner.SelectAlgorithm(params), MatMulAlgorithm::Packed);
}

TEST(MatMulExecutionPlannerTest, IntelPlannerPreservesSubgroupMatrixPrecedence) {
  intel::IntelMatMulExecutionPlanner planner;
  MatMulAlgorithmSelectionParams params{};
  params.m = 64;
  params.n = 512;
  params.k = 32;
  params.can_use_subgroup_matrix = true;
  params.has_subgroup_capability = true;

  EXPECT_EQ(planner.SelectAlgorithm(params), MatMulAlgorithm::SubgroupMatrix);
}

TEST(MatMulExecutionPlannerTest, SplitKPrecedesPackedFallback) {
  MatMulExecutionPlanner planner{intel::CreateSplitKConfig("xe-2lpg")};
  MatMulAlgorithmSelectionParams params{};
  params.m = 64;
  params.packed_m = 64;
  params.n = 64;
  params.k = 1024;
  params.is_vec4 = true;
  EXPECT_EQ(planner.SelectAlgorithm(params), MatMulAlgorithm::PackedSplitK);

  MatMulExecutionPlanner disabled_planner;
  EXPECT_EQ(disabled_planner.SelectAlgorithm(params), MatMulAlgorithm::Packed);
}

TEST(MatMulExecutionPlanValidationTest, SplitKRejectsEachHardConstraint) {
  MatMulExecutionPlanner planner{SplitKConfig{
      /*max_batch_size=*/8,
      /*split_dim_inner=*/128,
      /*min_dim_inner_with_split_k=*/256,
      {{4096, 16.0}}}};
  MatMulAlgorithmSelectionParams params{};
  params.m = 32;
  params.packed_m = 32;
  params.n = 64;
  params.k = 256;
  params.is_vec4 = true;
  MatMulExecutionPlan plan{MatMulAlgorithm::Naive, MatMulNaiveConfiguration{}};

  EXPECT_TRUE(planner.CreateExecutionPlan(params, MatMulAlgorithm::PackedSplitK, plan).IsOK());

  params.deterministic_compute = true;
  EXPECT_FALSE(planner.CreateExecutionPlan(params, MatMulAlgorithm::PackedSplitK, plan).IsOK());
  params.deterministic_compute = false;

  params.is_vec4 = false;
  EXPECT_FALSE(planner.CreateExecutionPlan(params, MatMulAlgorithm::PackedSplitK, plan).IsOK());
  params.is_vec4 = true;

  params.has_fused_activation = true;
  EXPECT_FALSE(planner.CreateExecutionPlan(params, MatMulAlgorithm::PackedSplitK, plan).IsOK());
  params.has_fused_activation = false;

  params.has_bias = true;
  params.is_channels_last = false;
  EXPECT_FALSE(planner.CreateExecutionPlan(params, MatMulAlgorithm::PackedSplitK, plan).IsOK());
}

TEST(MatMulExecutionPlanValidationTest, ForcedFailureDoesNotFallBackOrModifyOutput) {
  MatMulExecutionPlanner planner;
  MatMulAlgorithmSelectionParams params{};
  params.m = 1;
  params.n = 48;
  params.k = 5120;
  MatMulExecutionPlan plan{MatMulAlgorithm::Naive, MatMulNaiveConfiguration{}};

  const Status status = planner.CreateExecutionPlan(params, MatMulAlgorithm::Gemv, plan);

  EXPECT_FALSE(status.IsOK());
  EXPECT_EQ(plan.algorithm, MatMulAlgorithm::Naive);
  EXPECT_TRUE(std::holds_alternative<MatMulNaiveConfiguration>(plan.configuration));
}

TEST(MatMulExecutionPlanValidationTest, GemvRequiresCompatibleInputs) {
  MatMulExecutionPlanner planner;
  MatMulAlgorithmSelectionParams params{};
  params.m = 1;
  params.n = 48;
  params.k = 5120;
  MatMulExecutionPlan plan{MatMulAlgorithm::Naive, MatMulNaiveConfiguration{}};

  EXPECT_FALSE(planner.CreateExecutionPlan(params, MatMulAlgorithm::Gemv, plan).IsOK());
  params.can_run_gemv = true;
  EXPECT_TRUE(planner.CreateExecutionPlan(params, MatMulAlgorithm::Gemv, plan).IsOK());
}

TEST(MatMulExecutionPlanValidationTest, GemvRejectsDimensionsOutsideUint32Range) {
  MatMulExecutionPlanner planner;
  MatMulAlgorithmSelectionParams params{};
  params.m = 1;
  params.n = std::numeric_limits<uint32_t>::max() - 3;
  params.k = std::numeric_limits<uint32_t>::max();
  params.can_run_gemv = true;
  MatMulExecutionPlan plan{MatMulAlgorithm::Naive, MatMulNaiveConfiguration{}};

  EXPECT_TRUE(planner.CreateExecutionPlan(params, MatMulAlgorithm::Gemv, plan).IsOK());

  plan = {MatMulAlgorithm::Naive, MatMulNaiveConfiguration{}};
  params.n = static_cast<int64_t>(std::numeric_limits<uint32_t>::max()) + 1;
  EXPECT_FALSE(planner.CreateExecutionPlan(params, MatMulAlgorithm::Gemv, plan).IsOK());
  EXPECT_EQ(plan.algorithm, MatMulAlgorithm::Naive);
  EXPECT_TRUE(std::holds_alternative<MatMulNaiveConfiguration>(plan.configuration));

  params.n = 4;
  params.k = static_cast<int64_t>(std::numeric_limits<uint32_t>::max()) + 1;
  EXPECT_FALSE(planner.CreateExecutionPlan(params, MatMulAlgorithm::Gemv, plan).IsOK());
  EXPECT_EQ(plan.algorithm, MatMulAlgorithm::Naive);
  EXPECT_TRUE(std::holds_alternative<MatMulNaiveConfiguration>(plan.configuration));
}

TEST(MatMulExecutionPlanValidationTest, NaiveRejectsDimensionsAndVectorizedOutputOutsideUint32Range) {
  MatMulExecutionPlanner planner;
  MatMulAlgorithmSelectionParams params{};
  params.m = std::numeric_limits<uint32_t>::max();
  params.n = 1;
  MatMulExecutionPlan plan{MatMulAlgorithm::Packed, MatMulPackedConfiguration{}};

  EXPECT_TRUE(planner.CreateExecutionPlan(params, MatMulAlgorithm::Naive, plan).IsOK());

  plan = {MatMulAlgorithm::Packed, MatMulPackedConfiguration{}};
  params.m = static_cast<int64_t>(std::numeric_limits<uint32_t>::max()) + 1;
  EXPECT_FALSE(planner.CreateExecutionPlan(params, MatMulAlgorithm::Naive, plan).IsOK());
  EXPECT_EQ(plan.algorithm, MatMulAlgorithm::Packed);
  EXPECT_TRUE(std::holds_alternative<MatMulPackedConfiguration>(plan.configuration));

  params.m = 1;
  params.n = static_cast<int64_t>(std::numeric_limits<uint32_t>::max()) + 1;
  EXPECT_FALSE(planner.CreateExecutionPlan(params, MatMulAlgorithm::Naive, plan).IsOK());
  EXPECT_EQ(plan.algorithm, MatMulAlgorithm::Packed);
  EXPECT_TRUE(std::holds_alternative<MatMulPackedConfiguration>(plan.configuration));

  params.n = 1;
  params.k = static_cast<int64_t>(std::numeric_limits<uint32_t>::max()) + 1;
  EXPECT_FALSE(planner.CreateExecutionPlan(params, MatMulAlgorithm::Naive, plan).IsOK());
  EXPECT_EQ(plan.algorithm, MatMulAlgorithm::Packed);
  EXPECT_TRUE(std::holds_alternative<MatMulPackedConfiguration>(plan.configuration));

  params.k = 0;
  params.m = std::numeric_limits<uint32_t>::max();
  params.batch_size = 2;
  EXPECT_FALSE(planner.CreateExecutionPlan(params, MatMulAlgorithm::Naive, plan).IsOK());
  EXPECT_EQ(plan.algorithm, MatMulAlgorithm::Packed);
  EXPECT_TRUE(std::holds_alternative<MatMulPackedConfiguration>(plan.configuration));
}

TEST(MatMulExecutionPlanValidationTest, PackedAlgorithmsRejectZeroContractionDimension) {
  MatMulExecutionPlanner planner{SplitKConfig{
      /*max_batch_size=*/8,
      /*split_dim_inner=*/128,
      /*min_dim_inner_with_split_k=*/256,
      {{4096, 16.0}}}};
  MatMulAlgorithmSelectionParams params{};
  params.m = 32;
  params.packed_m = 32;
  params.n = 64;
  params.is_vec4 = true;
  MatMulExecutionPlan plan{MatMulAlgorithm::Naive, MatMulNaiveConfiguration{}};

  EXPECT_FALSE(planner.CreateExecutionPlan(params, MatMulAlgorithm::Packed, plan).IsOK());
  EXPECT_FALSE(planner.CreateExecutionPlan(params, MatMulAlgorithm::PackedSplitK, plan).IsOK());

  params.k = 256;
  EXPECT_TRUE(planner.CreateExecutionPlan(params, MatMulAlgorithm::Packed, plan).IsOK());
  EXPECT_TRUE(planner.CreateExecutionPlan(params, MatMulAlgorithm::PackedSplitK, plan).IsOK());
}

TEST(MatMulExecutionPlanValidationTest, RejectsInvalidVendorConfigurations) {
  MatMulAlgorithmSelectionParams params{};
  params.k = 64;
  MatMulExecutionPlan plan{MatMulAlgorithm::Naive, MatMulNaiveConfiguration{}};

  IncompatibleConfigurationPlanner incompatible_planner;
  EXPECT_FALSE(incompatible_planner.CreateExecutionPlan(
                                       params, MatMulAlgorithm::Packed, plan)
                   .IsOK());

  InvalidPackedConfigurationPlanner invalid_packed_planner;
  EXPECT_FALSE(invalid_packed_planner.CreateExecutionPlan(
                                         params, MatMulAlgorithm::Packed, plan)
                   .IsOK());

  MismatchedSubgroupConfigurationPlanner mismatched_subgroup_planner;
  params.has_subgroup_capability = true;
  params.subgroup_size = 16;
  EXPECT_FALSE(mismatched_subgroup_planner.CreateExecutionPlan(
                                              params, MatMulAlgorithm::Subgroup, plan)
                   .IsOK());
}

TEST(MatMulExecutionPlanValidationTest, RejectsPackedDimensionsOutsideUint32Range) {
  MatMulExecutionPlanner planner;
  MatMulAlgorithmSelectionParams params{};
  params.packed_m = 32;
  params.n = 64;
  params.k = 64;
  MatMulExecutionPlan plan{MatMulAlgorithm::Naive, MatMulNaiveConfiguration{}};
  constexpr int64_t max_shader_dimension = std::numeric_limits<uint32_t>::max();

  params.packed_m = max_shader_dimension;
  EXPECT_TRUE(planner.CreateExecutionPlan(params, MatMulAlgorithm::Packed, plan).IsOK());
  params.packed_m = max_shader_dimension + 1;
  EXPECT_FALSE(planner.CreateExecutionPlan(params, MatMulAlgorithm::Packed, plan).IsOK());

  params.packed_m = 32;
  params.n = max_shader_dimension;
  EXPECT_TRUE(planner.CreateExecutionPlan(params, MatMulAlgorithm::Packed, plan).IsOK());
  params.n = max_shader_dimension + 1;
  EXPECT_FALSE(planner.CreateExecutionPlan(params, MatMulAlgorithm::Packed, plan).IsOK());

  params.n = 64;
  params.k = max_shader_dimension;
  EXPECT_TRUE(planner.CreateExecutionPlan(params, MatMulAlgorithm::Packed, plan).IsOK());
  params.k = max_shader_dimension + 1;
  EXPECT_FALSE(planner.CreateExecutionPlan(params, MatMulAlgorithm::Packed, plan).IsOK());
}

TEST(MatMulExecutionPlanValidationTest, RejectsPackedDispatchOverflow) {
  MatMulExecutionPlanner planner;
  MatMulAlgorithmSelectionParams params{};
  params.k = 64;
  params.packed_batch_size = static_cast<uint64_t>(std::numeric_limits<uint32_t>::max()) + 1;
  MatMulExecutionPlan plan{MatMulAlgorithm::Naive, MatMulNaiveConfiguration{}};

  EXPECT_FALSE(planner.CreateExecutionPlan(params, MatMulAlgorithm::Packed, plan).IsOK());
}

TEST(MatMulExecutionPlanValidationTest, RejectsPackedCombinedDispatchOverflow) {
  MatMulExecutionPlanner planner;
  MatMulAlgorithmSelectionParams params{};
  params.packed_m = 2097152;
  params.n = 2097152;
  params.k = 64;
  MatMulExecutionPlan plan{MatMulAlgorithm::Naive, MatMulNaiveConfiguration{}};

  EXPECT_FALSE(planner.CreateExecutionPlan(params, MatMulAlgorithm::Packed, plan).IsOK());
  EXPECT_EQ(plan.algorithm, MatMulAlgorithm::Naive);
  EXPECT_TRUE(std::holds_alternative<MatMulNaiveConfiguration>(plan.configuration));
}

TEST(MatMulExecutionPlanValidationTest, RejectsSplitKDispatchOverflow) {
  MatMulExecutionPlanner planner{SplitKConfig{
      /*max_batch_size=*/std::numeric_limits<uint32_t>::max(),
      /*split_dim_inner=*/128,
      /*min_dim_inner_with_split_k=*/256,
      {{4096, 16.0}}}};
  MatMulAlgorithmSelectionParams params{};
  params.m = 32;
  params.packed_m = 32;
  params.n = 64;
  params.k = 256;
  params.packed_batch_size = std::numeric_limits<uint32_t>::max();
  params.is_vec4 = true;
  MatMulExecutionPlan plan{MatMulAlgorithm::Naive, MatMulNaiveConfiguration{}};

  EXPECT_FALSE(planner.CreateExecutionPlan(
                          params, MatMulAlgorithm::PackedSplitK, plan)
                   .IsOK());
}

TEST(MatMulExecutionPlanValidationTest, RejectsSplitKCombinedDispatchOverflow) {
  MatMulExecutionPlanner planner{SplitKConfig{
      /*max_batch_size=*/8,
      /*split_dim_inner=*/128,
      /*min_dim_inner_with_split_k=*/256,
      {{4096, 16.0}}}};
  MatMulAlgorithmSelectionParams params{};
  params.packed_m = 1048576;
  params.n = 2097152;
  params.k = 256;
  params.is_vec4 = true;
  MatMulExecutionPlan plan{MatMulAlgorithm::Naive, MatMulNaiveConfiguration{}};

  EXPECT_FALSE(planner.CreateExecutionPlan(
                          params, MatMulAlgorithm::PackedSplitK, plan)
                   .IsOK());
  EXPECT_EQ(plan.algorithm, MatMulAlgorithm::Naive);
  EXPECT_TRUE(std::holds_alternative<MatMulNaiveConfiguration>(plan.configuration));
}

TEST(MatMulAlgorithmPrerequisiteTest, IntelAVec4RequiresCompatibleRowsPerThread) {
  EXPECT_FALSE(intel::CanUseAVec4CooperativeLoad(intel::gpu_arch::kXe3Lpg, 32, 1));
  EXPECT_FALSE(intel::CanUseAVec4CooperativeLoad(intel::gpu_arch::kXe3Lpg, 32, 2));
  EXPECT_TRUE(intel::CanUseAVec4CooperativeLoad(intel::gpu_arch::kXe3Lpg, 32, 4));
  EXPECT_FALSE(intel::CanUseAVec4CooperativeLoad(intel::gpu_arch::kXe3Lpg, 31, 4));
  EXPECT_FALSE(intel::CanUseAVec4CooperativeLoad(intel::gpu_arch::kXeLpg, 32, 4));
}

TEST(MatMulExecutionPlanValidationTest, IntelCapabilityDoesNotIncludeAutomaticThresholds) {
  MatMulAlgorithmSelectionParams below_heuristic_threshold{};
  below_heuristic_threshold.m = 1;
  below_heuristic_threshold.n = 1;
  below_heuristic_threshold.k = 1;
  below_heuristic_threshold.has_subgroup_capability = true;
  below_heuristic_threshold.subgroup_size = 16;
  intel::IntelMatMulExecutionPlanner planner;
  MatMulExecutionPlan plan{MatMulAlgorithm::Naive, MatMulNaiveConfiguration{}};

  EXPECT_TRUE(planner.CreateExecutionPlan(
                         below_heuristic_threshold, MatMulAlgorithm::Subgroup, plan)
                  .IsOK());
  EXPECT_NE(planner.SelectAlgorithm(below_heuristic_threshold), MatMulAlgorithm::Subgroup);
}

TEST(MatMulExecutionPlanValidationTest, SubgroupRejectsZeroContractionDimension) {
  MatMulExecutionPlanner planner;
  MatMulAlgorithmSelectionParams params{};
  params.m = 1;
  params.n = 1;
  params.has_subgroup_capability = true;
  params.subgroup_size = 16;
  MatMulExecutionPlan plan{MatMulAlgorithm::Naive, MatMulNaiveConfiguration{}};

  EXPECT_FALSE(planner.CreateExecutionPlan(params, MatMulAlgorithm::Subgroup, plan).IsOK());
  params.k = 1;
  EXPECT_TRUE(planner.CreateExecutionPlan(params, MatMulAlgorithm::Subgroup, plan).IsOK());
}

TEST(MatMulExecutionPlanValidationTest, SubgroupRejectsUnsupportedConfiguredSize) {
  MatMulExecutionPlanner planner;
  MatMulAlgorithmSelectionParams params{};
  params.m = 1;
  params.n = 1;
  params.has_subgroup_capability = true;
  params.k = 1;
  MatMulExecutionPlan plan{MatMulAlgorithm::Naive, MatMulNaiveConfiguration{}};

  for (uint32_t subgroup_size : {8u, 16u, 32u}) {
    params.subgroup_size = subgroup_size;
    EXPECT_TRUE(planner.CreateExecutionPlan(params, MatMulAlgorithm::Subgroup, plan).IsOK());
  }

  params.subgroup_size = 64;
  EXPECT_FALSE(planner.CreateExecutionPlan(params, MatMulAlgorithm::Subgroup, plan).IsOK());
}

TEST(MatMulExecutionPlanValidationTest, SubgroupRejectsDimensionsOutsideUint32Range) {
  MatMulExecutionPlanner planner;
  MatMulAlgorithmSelectionParams params{};
  params.m = static_cast<int64_t>(std::numeric_limits<uint32_t>::max()) + 1;
  params.packed_m = params.m;
  params.n = 1;
  params.k = 1;
  params.has_subgroup_capability = true;
  params.subgroup_size = 16;
  MatMulExecutionPlan plan{MatMulAlgorithm::Naive, MatMulNaiveConfiguration{}};

  EXPECT_FALSE(planner.CreateExecutionPlan(params, MatMulAlgorithm::Subgroup, plan).IsOK());
  EXPECT_EQ(plan.algorithm, MatMulAlgorithm::Naive);
  EXPECT_TRUE(std::holds_alternative<MatMulNaiveConfiguration>(plan.configuration));
}

TEST(MatMulExecutionPlanValidationTest, SubgroupRejectsCombinedDispatchOverflow) {
  MatMulExecutionPlanner planner;
  MatMulAlgorithmSelectionParams params{};
  params.m = 4194304;
  params.packed_m = params.m;
  params.n = 8388608;
  params.k = 1;
  params.has_subgroup_capability = true;
  params.subgroup_size = 16;
  MatMulExecutionPlan plan{MatMulAlgorithm::Naive, MatMulNaiveConfiguration{}};

  EXPECT_FALSE(planner.CreateExecutionPlan(params, MatMulAlgorithm::Subgroup, plan).IsOK());
  EXPECT_EQ(plan.algorithm, MatMulAlgorithm::Naive);
  EXPECT_TRUE(std::holds_alternative<MatMulNaiveConfiguration>(plan.configuration));
}

TEST(MatMulExecutionPlanValidationTest, SubgroupRejectsFoldedBatchDispatchOverflow) {
  MatMulExecutionPlanner planner;
  MatMulAlgorithmSelectionParams params{};
  params.m = 2097152;
  params.packed_m = 4194304;
  params.n = 8388608;
  params.k = 1;
  params.batch_size = 2;
  params.packed_batch_size = 1;
  params.has_subgroup_capability = true;
  params.subgroup_size = 16;
  MatMulExecutionPlan plan{MatMulAlgorithm::Naive, MatMulNaiveConfiguration{}};

  EXPECT_FALSE(planner.CreateExecutionPlan(params, MatMulAlgorithm::Subgroup, plan).IsOK());
  EXPECT_EQ(plan.algorithm, MatMulAlgorithm::Naive);
  EXPECT_TRUE(std::holds_alternative<MatMulNaiveConfiguration>(plan.configuration));
}

}  // namespace test
}  // namespace webgpu
}  // namespace onnxruntime
