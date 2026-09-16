// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"

#include "core/providers/webgpu/math/matmul_algorithm.h"
#include "core/providers/webgpu/math/matmul_algorithm_scheduler.h"
#include "core/providers/webgpu/vendor/intel/math/matmul_algorithm_scheduler.h"

namespace onnxruntime {
namespace webgpu {
namespace test {

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

TEST(MatMulAlgorithmSchedulerTest, ForcedAlgorithmTakesPrecedence) {
  MatMulAlgorithmScheduler scheduler;
  MatMulAlgorithmSelectionParams params{};
  params.can_use_subgroup_matrix = true;

  EXPECT_EQ(scheduler.Select(params, MatMulAlgorithm::Packed), MatMulAlgorithm::Packed);
}

TEST(MatMulAlgorithmSchedulerTest, SubgroupMatrixTakesAutomaticPrecedence) {
  MatMulAlgorithmScheduler scheduler;
  MatMulAlgorithmSelectionParams params{};
  params.n = 4;
  params.k = 4;
  params.can_use_subgroup_matrix = true;
  params.use_split_k = true;

  EXPECT_EQ(scheduler.Select(params), MatMulAlgorithm::SubgroupMatrix);
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

TEST(MatMulAlgorithmSchedulerTest, SplitKPrecedesPackedFallback) {
  MatMulAlgorithmScheduler scheduler;
  MatMulAlgorithmSelectionParams params{};
  params.n = 64;
  params.k = 1024;
  params.use_split_k = true;
  EXPECT_EQ(scheduler.Select(params), MatMulAlgorithm::PackedSplitK);

  params.use_split_k = false;
  EXPECT_EQ(scheduler.Select(params), MatMulAlgorithm::Packed);
}

}  // namespace test
}  // namespace webgpu
}  // namespace onnxruntime
