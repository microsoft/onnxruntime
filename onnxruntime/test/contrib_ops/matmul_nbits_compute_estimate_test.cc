// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <array>
#include <limits>
#include <optional>
#include <string>
#include <variant>
#include <vector>

#include "gtest/gtest.h"

#include "contrib_ops/cpu/quantization/matmul_nbits_compute_estimate.h"

#if !defined(ORT_MINIMAL_BUILD)
#include "core/graph/constants.h"
#include "core/graph/model.h"
#include "test/test_environment.h"
#include "test/unittest_util/graph_transform_test_builder.h"
#include "test/util/include/asserts.h"
#endif

namespace onnxruntime {
namespace test {
namespace {

using contrib::ComputeEstimateUnavailableReason;
using contrib::EstimateMatMulNBitsCompute;
using contrib::MatMulNBitsComputeEstimate;
using contrib::MatMulNBitsComputeEstimateResult;

void ExpectCount(const MatMulNBitsComputeEstimateResult& result, uint64_t expected) {
  const auto* estimate = std::get_if<MatMulNBitsComputeEstimate>(&result);
  ASSERT_NE(estimate, nullptr);
  EXPECT_EQ(estimate->dense_equivalent_ops, expected);
  EXPECT_EQ(estimate->kConventionVersion, 1u);
}

void ExpectUnavailable(const MatMulNBitsComputeEstimateResult& result,
                       ComputeEstimateUnavailableReason expected) {
  const auto* reason = std::get_if<ComputeEstimateUnavailableReason>(&result);
  ASSERT_NE(reason, nullptr);
  EXPECT_EQ(*reason, expected);
}

TEST(MatMulNBitsComputeEstimateTest, MatrixAndBatchedCounts) {
  ExpectCount(EstimateMatMulNBitsCompute(std::array<int64_t, 2>{3, 64}, 64, 32), 12288);
  ExpectCount(EstimateMatMulNBitsCompute(std::array<int64_t, 3>{2, 3, 64}, 64, 32), 24576);
  ExpectCount(EstimateMatMulNBitsCompute(std::array<int64_t, 4>{2, 3, 4, 64}, 64, 32), 98304);
}

TEST(MatMulNBitsComputeEstimateTest, RankOneInput) {
  ExpectCount(EstimateMatMulNBitsCompute(std::array<int64_t, 1>{64}, 64, 32), 4096);
}

TEST(MatMulNBitsComputeEstimateTest, EmptyWorkloads) {
  ExpectCount(EstimateMatMulNBitsCompute(std::array<int64_t, 2>{0, 64}, 64, 32), 0);
  ExpectCount(EstimateMatMulNBitsCompute(std::array<int64_t, 2>{3, 64}, 64, 0), 0);
  ExpectCount(EstimateMatMulNBitsCompute(std::array<int64_t, 2>{3, 0}, 0, 32), 0);
}

TEST(MatMulNBitsComputeEstimateTest, ZeroPrecedesProductOverflow) {
  constexpr auto max = std::numeric_limits<int64_t>::max();
  ExpectCount(EstimateMatMulNBitsCompute(std::array<int64_t, 4>{max, max, 0, 64}, 64, max), 0);
}

TEST(MatMulNBitsComputeEstimateTest, UnknownDimensionsAreNotZero) {
  ExpectUnavailable(EstimateMatMulNBitsCompute(std::array<int64_t, 2>{-1, 64}, 64, 32),
                    ComputeEstimateUnavailableReason::UnknownShape);
  ExpectUnavailable(EstimateMatMulNBitsCompute(std::array<int64_t, 2>{3, -1}, 64, 32),
                    ComputeEstimateUnavailableReason::UnknownShape);
  ExpectUnavailable(EstimateMatMulNBitsCompute(std::array<int64_t, 3>{0, -1, 64}, 64, 32),
                    ComputeEstimateUnavailableReason::UnknownShape);
}

TEST(MatMulNBitsComputeEstimateTest, InvalidMetadata) {
  ExpectUnavailable(EstimateMatMulNBitsCompute({}, 64, 32), ComputeEstimateUnavailableReason::InvalidMetadata);
  ExpectUnavailable(EstimateMatMulNBitsCompute(std::array<int64_t, 2>{3, 63}, 64, 32),
                    ComputeEstimateUnavailableReason::InvalidMetadata);
  ExpectUnavailable(EstimateMatMulNBitsCompute(std::array<int64_t, 2>{-2, 64}, 64, 32),
                    ComputeEstimateUnavailableReason::InvalidMetadata);
  ExpectUnavailable(EstimateMatMulNBitsCompute(std::array<int64_t, 2>{3, 64}, -1, 32),
                    ComputeEstimateUnavailableReason::InvalidMetadata);
  ExpectUnavailable(EstimateMatMulNBitsCompute(std::array<int64_t, 2>{3, 64}, 64, -1),
                    ComputeEstimateUnavailableReason::InvalidMetadata);
  ExpectUnavailable(EstimateMatMulNBitsCompute(std::array<int64_t, 3>{0, -2, 64}, 64, 32),
                    ComputeEstimateUnavailableReason::InvalidMetadata);
}

TEST(MatMulNBitsComputeEstimateTest, CountsUseUnsigned64BitRange) {
  constexpr auto max = std::numeric_limits<int64_t>::max();
  ExpectCount(EstimateMatMulNBitsCompute(std::array<int64_t, 1>{max}, max, 1),
              std::numeric_limits<uint64_t>::max() - 1);
}

TEST(MatMulNBitsComputeEstimateTest, Overflow) {
  constexpr auto max = std::numeric_limits<int64_t>::max();
  ExpectUnavailable(EstimateMatMulNBitsCompute(std::array<int64_t, 2>{2, max}, max, 1),
                    ComputeEstimateUnavailableReason::Overflow);
  ExpectUnavailable(EstimateMatMulNBitsCompute(std::array<int64_t, 1>{2}, 2, max),
                    ComputeEstimateUnavailableReason::Overflow);
  ExpectUnavailable(EstimateMatMulNBitsCompute(std::array<int64_t, 3>{max, max, 1}, 1, 1),
                    ComputeEstimateUnavailableReason::Overflow);
}

#if !defined(ORT_MINIMAL_BUILD)
class MatMulNBitsComputeEstimateGraphTest : public ::testing::Test {
 protected:
  MatMulNBitsComputeEstimateGraphTest()
      : model_("compute_estimate", false, ModelMetaData(), PathString(),
               IOnnxRuntimeOpSchemaRegistryList(), {{kOnnxDomain, 17}, {kMSDomain, 1}}, {},
               DefaultLoggingManager().DefaultLogger()),
        builder_(model_.MainGraph()) {}

  template <typename T = float>
  Node& AddMatMul(std::optional<std::vector<int64_t>> shape = std::vector<int64_t>{2, 3, 64},
                  int64_t bits = 4, bool bias = false) {
    NodeArg* a = builder_.MakeInput<T>(shape);
    NodeArg* b = builder_.MakeInput<uint8_t>(std::vector<int64_t>{32, 2, 32 * bits / 8});
    NodeArg* scales = builder_.MakeInput<T>(std::vector<int64_t>{32, 2});
    std::vector<NodeArg*> inputs{a, b, scales};
    if (bias) {
      auto& missing = builder_.graph_.GetOrCreateNodeArg("", nullptr);
      inputs.push_back(&missing);
      inputs.push_back(&missing);
      inputs.push_back(builder_.MakeInput<T>(std::vector<int64_t>{32}));
    }
    auto* output = builder_.MakeOutput<T>(std::nullopt);
    auto& node = builder_.AddNode("MatMulNBits", inputs, {output}, kMSDomain);
    node.AddAttribute("K", int64_t{64});
    node.AddAttribute("N", int64_t{32});
    node.AddAttribute("block_size", int64_t{32});
    node.AddAttribute("bits", bits);
    builder_.SetGraphOutputs();
    return node;
  }

  Model model_;
  ModelTestBuilder builder_;
};

TEST_F(MatMulNBitsComputeEstimateGraphTest, ResolvedGraph) {
  auto& node = AddMatMul();
  ASSERT_STATUS_OK(model_.MainGraph().Resolve());
  ExpectCount(EstimateMatMulNBitsCompute(node), 24576);
}

TEST_F(MatMulNBitsComputeEstimateGraphTest, RankOneGraph) {
  auto& node = AddMatMul(std::vector<int64_t>{64});
  ASSERT_STATUS_OK(model_.MainGraph().Resolve());
  ExpectCount(EstimateMatMulNBitsCompute(node), 4096);
}

TEST_F(MatMulNBitsComputeEstimateGraphTest, TwoBitWithBias) {
  auto& node = AddMatMul(std::vector<int64_t>{2, 3, 64}, 2, true);
  ASSERT_STATUS_OK(model_.MainGraph().Resolve());
  ExpectCount(EstimateMatMulNBitsCompute(node), 24576);
}

TEST_F(MatMulNBitsComputeEstimateGraphTest, EightBitHalfPrecision) {
  auto& node = AddMatMul<MLFloat16>(std::vector<int64_t>{2, 3, 64}, 8);
  ASSERT_STATUS_OK(model_.MainGraph().Resolve());
  ExpectCount(EstimateMatMulNBitsCompute(node), 24576);
}

TEST_F(MatMulNBitsComputeEstimateGraphTest, BFloat16) {
  auto& node = AddMatMul<BFloat16>();
  ASSERT_STATUS_OK(model_.MainGraph().Resolve());
  ExpectCount(EstimateMatMulNBitsCompute(node), 24576);
}

TEST_F(MatMulNBitsComputeEstimateGraphTest, UnknownRank) {
  auto& node = AddMatMul(std::nullopt);
  ASSERT_STATUS_OK(model_.MainGraph().Resolve());
  ExpectUnavailable(EstimateMatMulNBitsCompute(node), ComputeEstimateUnavailableReason::UnknownShape);
}

TEST_F(MatMulNBitsComputeEstimateGraphTest, PartiallyKnownShape) {
  auto& node = AddMatMul(std::vector<int64_t>{-1, 64});
  ASSERT_STATUS_OK(model_.MainGraph().Resolve());
  ExpectUnavailable(EstimateMatMulNBitsCompute(node), ComputeEstimateUnavailableReason::UnknownShape);
}

TEST_F(MatMulNBitsComputeEstimateGraphTest, MissingAttribute) {
  auto& node = AddMatMul();
  ASSERT_STATUS_OK(model_.MainGraph().Resolve());
  node.ClearAttribute("K");
  ExpectUnavailable(EstimateMatMulNBitsCompute(node), ComputeEstimateUnavailableReason::InvalidMetadata);
}

TEST_F(MatMulNBitsComputeEstimateGraphTest, WrongAttributeType) {
  auto& node = AddMatMul();
  ASSERT_STATUS_OK(model_.MainGraph().Resolve());
  node.AddAttribute("N", std::string("32"));
  ExpectUnavailable(EstimateMatMulNBitsCompute(node), ComputeEstimateUnavailableReason::InvalidMetadata);
}

TEST_F(MatMulNBitsComputeEstimateGraphTest, UnresolvedVersion) {
  auto& node = AddMatMul();
  ExpectUnavailable(EstimateMatMulNBitsCompute(node), ComputeEstimateUnavailableReason::UnsupportedVersion);
}

TEST_F(MatMulNBitsComputeEstimateGraphTest, UnsupportedVersion) {
  auto& node = AddMatMul();
  ASSERT_STATUS_OK(model_.MainGraph().Resolve());
  node.SetSinceVersion(2);
  ExpectUnavailable(EstimateMatMulNBitsCompute(node), ComputeEstimateUnavailableReason::UnsupportedVersion);
}

TEST_F(MatMulNBitsComputeEstimateGraphTest, EmptyGraphWorkload) {
  auto& node = AddMatMul(std::vector<int64_t>{0, 64});
  ASSERT_STATUS_OK(model_.MainGraph().Resolve());
  ExpectCount(EstimateMatMulNBitsCompute(node), 0);
}

TEST_F(MatMulNBitsComputeEstimateGraphTest, UnsupportedDomain) {
  auto* input = builder_.MakeInput<float>(std::vector<int64_t>{2, 64});
  auto* output = builder_.MakeOutput<float>(std::nullopt);
  auto& node = builder_.AddNode("MatMulNBits", {input}, {output}, "other.domain");
  ExpectUnavailable(EstimateMatMulNBitsCompute(node), ComputeEstimateUnavailableReason::UnsupportedOperator);
}

TEST_F(MatMulNBitsComputeEstimateGraphTest, UnsupportedOperator) {
  auto* input = builder_.MakeInput<float>(std::vector<int64_t>{2, 3});
  auto* output = builder_.MakeOutput<float>(std::nullopt);
  auto& node = builder_.AddNode("Identity", {input}, {output});
  builder_.SetGraphOutputs();
  ASSERT_STATUS_OK(model_.MainGraph().Resolve());
  ExpectUnavailable(EstimateMatMulNBitsCompute(node), ComputeEstimateUnavailableReason::UnsupportedOperator);
}
#endif

}  // namespace
}  // namespace test
}  // namespace onnxruntime
