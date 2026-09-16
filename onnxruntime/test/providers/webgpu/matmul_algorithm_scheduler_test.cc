// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"

#include "core/providers/webgpu/math/matmul_algorithm.h"

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

}  // namespace test
}  // namespace webgpu
}  // namespace onnxruntime
