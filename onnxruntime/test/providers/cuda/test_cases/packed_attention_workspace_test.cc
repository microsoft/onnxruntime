// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"

#include <initializer_list>
#include <limits>
#include <string>
#include <type_traits>

#include "contrib_ops/cuda/bert/packed_attention_workspace.h"

namespace onnxruntime {
namespace test {

using contrib::cuda::BuildPackedAttentionProblem;
using contrib::cuda::BuildPackedMultiHeadAttentionProblem;
using contrib::cuda::CheckedPackedAttentionAdd;
using contrib::cuda::CheckedPackedAttentionAlign;
using contrib::cuda::CheckedPackedAttentionMultiply;
using contrib::cuda::GetPackedAttentionWorkspaceRecipe;
using contrib::cuda::GetPackedMultiHeadAttentionWorkspaceRecipe;
using contrib::cuda::PackedAttentionBackend;
using contrib::cuda::PackedAttentionInputShapes;
using contrib::cuda::PackedAttentionProblem;
using contrib::cuda::PackedAttentionQkvMaterializationIndexWidth;
using contrib::cuda::PackedAttentionQkvWorkspaceLayout;
using contrib::cuda::PackedAttentionShape;
using contrib::cuda::PackedAttentionWorkspaceError;
using contrib::cuda::PackedMultiHeadAttentionInputShapes;
using contrib::cuda::PackedMultiHeadAttentionProblem;
using contrib::cuda::PackedMultiHeadAttentionQkvFormat;
using contrib::cuda::ValidatePackedAttentionWorkspaceRecipe;

static_assert(std::is_trivially_copyable_v<PackedAttentionProblem>);
static_assert(std::is_trivially_copyable_v<PackedMultiHeadAttentionProblem>);

namespace {

PackedAttentionShape Shape(std::initializer_list<int64_t> dimensions) {
  PackedAttentionShape shape;
  shape.rank = dimensions.size();
  size_t index = 0;
  for (int64_t dimension : dimensions) {
    if (index < shape.dimensions.size()) {
      shape.dimensions[index] = dimension;
    }
    ++index;
  }

  return shape;
}

PackedAttentionInputShapes ValidPackedAttentionInputs(int64_t token_count = 6) {
  PackedAttentionInputShapes inputs;
  inputs.input = Shape({token_count, 6});
  inputs.weights = Shape({6, 20});
  inputs.bias = Shape({20});
  inputs.token_offset = Shape({2, 4});
  inputs.cumulative_sequence_length = Shape({3});
  inputs.element_size = 2;
  inputs.num_heads = 2;
  inputs.qkv_hidden_sizes_count = 3;
  inputs.qkv_hidden_sizes = {8, 8, 4};
  return inputs;
}

PackedAttentionInputShapes ValidPackedAttentionEqualHeadsInputs(int64_t token_count = 6) {
  auto inputs = ValidPackedAttentionInputs(token_count);
  inputs.weights = Shape({6, 24});
  inputs.bias = Shape({24});
  inputs.qkv_hidden_sizes = {8, 8, 8};
  return inputs;
}

PackedMultiHeadAttentionInputShapes ValidSeparateQkvInputs(int64_t token_count = 6) {
  PackedMultiHeadAttentionInputShapes inputs;
  inputs.query = Shape({token_count, 8});
  inputs.key = Shape({token_count, 8});
  inputs.value = Shape({token_count, 4});
  inputs.token_offset = Shape({2, 4});
  inputs.cumulative_sequence_length = Shape({3});
  inputs.element_size = 2;
  inputs.num_heads = 2;
  inputs.has_key = true;
  inputs.has_value = true;
  return inputs;
}

PackedMultiHeadAttentionInputShapes ValidSeparateEqualQkvInputs(int64_t token_count = 6) {
  auto inputs = ValidSeparateQkvInputs(token_count);
  inputs.value = Shape({token_count, 8});
  return inputs;
}

PackedMultiHeadAttentionInputShapes ValidPackedQkvInputs(int64_t token_count = 6) {
  PackedMultiHeadAttentionInputShapes inputs;
  inputs.query = Shape({token_count, 2, 3, 4});
  inputs.token_offset = Shape({2, 4});
  inputs.cumulative_sequence_length = Shape({3});
  inputs.element_size = 2;
  inputs.num_heads = 2;
  return inputs;
}

}  // namespace

TEST(PackedAttentionWorkspaceTest, CheckedArithmeticHandlesZeroBoundaryAndOverflow) {
  size_t result = 123;
  EXPECT_TRUE(CheckedPackedAttentionAdd(0, 0, result).IsOK());
  EXPECT_EQ(result, 0U);
  EXPECT_TRUE(CheckedPackedAttentionAdd(7, 9, result).IsOK());
  EXPECT_EQ(result, 16U);
  EXPECT_EQ(CheckedPackedAttentionAdd(std::numeric_limits<size_t>::max(), 1, result).error,
            PackedAttentionWorkspaceError::Overflow);

  EXPECT_TRUE(CheckedPackedAttentionMultiply(0, std::numeric_limits<size_t>::max(), result).IsOK());
  EXPECT_EQ(result, 0U);
  EXPECT_TRUE(CheckedPackedAttentionMultiply(7, 9, result).IsOK());
  EXPECT_EQ(result, 63U);
  EXPECT_EQ(CheckedPackedAttentionMultiply(std::numeric_limits<size_t>::max(), 2, result).error,
            PackedAttentionWorkspaceError::Overflow);

  EXPECT_TRUE(CheckedPackedAttentionAlign(0, 256, result).IsOK());
  EXPECT_EQ(result, 0U);
  EXPECT_TRUE(CheckedPackedAttentionAlign(257, 256, result).IsOK());
  EXPECT_EQ(result, 512U);
  EXPECT_TRUE(CheckedPackedAttentionAlign(std::numeric_limits<size_t>::max() - 255, 256, result).IsOK());
  EXPECT_EQ(result, std::numeric_limits<size_t>::max() - 255);
  EXPECT_EQ(CheckedPackedAttentionAlign(std::numeric_limits<size_t>::max(), 256, result).error,
            PackedAttentionWorkspaceError::Overflow);
  EXPECT_EQ(CheckedPackedAttentionAlign(1, 0, result).error,
            PackedAttentionWorkspaceError::InvalidArgument);
}

enum class LegacyParityOperator {
  PackedAttention,
  PackedMultiHeadAttention,
};

struct LegacyParityCase {
  const char* name;
  LegacyParityOperator op;
  PackedAttentionBackend backend;
  int64_t token_count;
  bool expected_no_qkv_workspace;
  size_t expected_projection_bytes;
  size_t expected_qkv_capacity_bytes;
  size_t expected_planar_q_bytes;
  size_t expected_interleaved_qkv_bytes;
  size_t expected_backend_offset_bytes;
  size_t expected_backend_bytes;
  size_t expected_attention_workspace_bytes;
  PackedAttentionQkvWorkspaceLayout expected_layout;
};

class PackedAttentionLegacyParityTest : public testing::TestWithParam<LegacyParityCase> {};

TEST_P(PackedAttentionLegacyParityTest, MatchesHandCalculatedLegacyComponents) {
  const LegacyParityCase& test_case = GetParam();
  contrib::cuda::PackedAttentionWorkspaceResult workspace_result;

  if (test_case.op == LegacyParityOperator::PackedAttention) {
    auto problem_result = BuildPackedAttentionProblem(
        ValidPackedAttentionEqualHeadsInputs(test_case.token_count));
    ASSERT_TRUE(problem_result.status.IsOK()) << problem_result.status.message;
    problem_result.problem.backend = test_case.backend;
    problem_result.problem.trt_runner_available = test_case.backend == PackedAttentionBackend::Trt;
    workspace_result = GetPackedAttentionWorkspaceRecipe(problem_result.problem);
  } else {
    PackedMultiHeadAttentionInputShapes inputs;
    if (test_case.backend == PackedAttentionBackend::Trt) {
      inputs = test_case.expected_no_qkv_workspace
                   ? ValidPackedQkvInputs(test_case.token_count)
                   : ValidSeparateEqualQkvInputs(test_case.token_count);
    } else if (test_case.backend == PackedAttentionBackend::Flash ||
               test_case.backend == PackedAttentionBackend::MemoryEfficient) {
      inputs = test_case.expected_no_qkv_workspace
                   ? ValidSeparateEqualQkvInputs(test_case.token_count)
                   : ValidPackedQkvInputs(test_case.token_count);
    } else {
      inputs = ValidPackedQkvInputs(test_case.token_count);
    }

    auto problem_result = BuildPackedMultiHeadAttentionProblem(inputs);
    ASSERT_TRUE(problem_result.status.IsOK()) << problem_result.status.message;
    problem_result.problem.backend = test_case.backend;
    problem_result.problem.trt_runner_available = test_case.backend == PackedAttentionBackend::Trt;
    workspace_result = GetPackedMultiHeadAttentionWorkspaceRecipe(problem_result.problem);
  }

  ASSERT_TRUE(workspace_result.status.IsOK()) << workspace_result.status.message;
  const auto& recipe = workspace_result.recipe;
  EXPECT_EQ(recipe.no_qkv_workspace, test_case.expected_no_qkv_workspace);
  EXPECT_EQ(recipe.projection_bytes, test_case.expected_projection_bytes);
  EXPECT_EQ(recipe.qkv_capacity_bytes, test_case.expected_qkv_capacity_bytes);
  EXPECT_EQ(recipe.q_bytes, test_case.expected_planar_q_bytes);
  EXPECT_EQ(recipe.interleaved_qkv_bytes, test_case.expected_interleaved_qkv_bytes);
  EXPECT_EQ(recipe.backend_workspace_offset_bytes, test_case.expected_backend_offset_bytes);
  EXPECT_EQ(recipe.backend_workspace_bytes, test_case.expected_backend_bytes);
  EXPECT_EQ(recipe.attention_workspace_bytes, test_case.expected_attention_workspace_bytes);
  EXPECT_EQ(recipe.qkv_layout, test_case.expected_layout);
  EXPECT_TRUE(ValidatePackedAttentionWorkspaceRecipe(recipe).IsOK());
}

INSTANTIATE_TEST_SUITE_P(
    HandCalculatedMatrix,
    PackedAttentionLegacyParityTest,
    testing::Values(
        // B=2, S=4, N=2, H=Hv=4, sizeof(T)=2:
        // QKV capacity=384, Flash LSE=64, unfused scratch=align256(128)=256.
        LegacyParityCase{"PmhaFlashDirectT6", LegacyParityOperator::PackedMultiHeadAttention,
                         PackedAttentionBackend::Flash, 6, true, 0, 0, 0, 0, 0, 64, 64,
                         PackedAttentionQkvWorkspaceLayout::None},
        LegacyParityCase{"PmhaFlashPackedT6", LegacyParityOperator::PackedMultiHeadAttention,
                         PackedAttentionBackend::Flash, 6, false, 0, 384, 96, 0, 288, 64, 448,
                         PackedAttentionQkvWorkspaceLayout::Planar},
        LegacyParityCase{"PmhaMeaDirectT6", LegacyParityOperator::PackedMultiHeadAttention,
                         PackedAttentionBackend::MemoryEfficient, 6, true, 0, 0, 0, 0, 0, 0, 0,
                         PackedAttentionQkvWorkspaceLayout::None},
        LegacyParityCase{"PmhaMeaPackedT6", LegacyParityOperator::PackedMultiHeadAttention,
                         PackedAttentionBackend::MemoryEfficient, 6, false, 0, 384, 96, 0, 288, 0, 384,
                         PackedAttentionQkvWorkspaceLayout::Planar},
        LegacyParityCase{"PmhaTrtDirectT6", LegacyParityOperator::PackedMultiHeadAttention,
                         PackedAttentionBackend::Trt, 6, true, 0, 0, 0, 0, 0, 0, 0,
                         PackedAttentionQkvWorkspaceLayout::None},
        LegacyParityCase{"PmhaTrtMaterializedT6", LegacyParityOperator::PackedMultiHeadAttention,
                         PackedAttentionBackend::Trt, 6, false, 0, 384, 0, 288, 0, 0, 384,
                         PackedAttentionQkvWorkspaceLayout::InterleavedTn3h},
        LegacyParityCase{"PmhaUnfusedT6", LegacyParityOperator::PackedMultiHeadAttention,
                         PackedAttentionBackend::Unfused, 6, false, 0, 384, 128, 0, 384, 256, 896,
                         PackedAttentionQkvWorkspaceLayout::Planar},
        LegacyParityCase{"PmhaFlashDirectT8", LegacyParityOperator::PackedMultiHeadAttention,
                         PackedAttentionBackend::Flash, 8, true, 0, 0, 0, 0, 0, 64, 64,
                         PackedAttentionQkvWorkspaceLayout::None},
        LegacyParityCase{"PmhaFlashPackedT8", LegacyParityOperator::PackedMultiHeadAttention,
                         PackedAttentionBackend::Flash, 8, false, 0, 384, 128, 0, 384, 64, 448,
                         PackedAttentionQkvWorkspaceLayout::Planar},
        LegacyParityCase{"PmhaMeaDirectT8", LegacyParityOperator::PackedMultiHeadAttention,
                         PackedAttentionBackend::MemoryEfficient, 8, true, 0, 0, 0, 0, 0, 0, 0,
                         PackedAttentionQkvWorkspaceLayout::None},
        LegacyParityCase{"PmhaMeaPackedT8", LegacyParityOperator::PackedMultiHeadAttention,
                         PackedAttentionBackend::MemoryEfficient, 8, false, 0, 384, 128, 0, 384, 0, 384,
                         PackedAttentionQkvWorkspaceLayout::Planar},
        LegacyParityCase{"PmhaTrtDirectT8", LegacyParityOperator::PackedMultiHeadAttention,
                         PackedAttentionBackend::Trt, 8, true, 0, 0, 0, 0, 0, 0, 0,
                         PackedAttentionQkvWorkspaceLayout::None},
        LegacyParityCase{"PmhaTrtMaterializedT8", LegacyParityOperator::PackedMultiHeadAttention,
                         PackedAttentionBackend::Trt, 8, false, 0, 384, 0, 384, 0, 0, 384,
                         PackedAttentionQkvWorkspaceLayout::InterleavedTn3h},
        LegacyParityCase{"PmhaUnfusedT8", LegacyParityOperator::PackedMultiHeadAttention,
                         PackedAttentionBackend::Unfused, 8, false, 0, 384, 128, 0, 384, 256, 896,
                         PackedAttentionQkvWorkspaceLayout::Planar},
        LegacyParityCase{"PaMeaT6", LegacyParityOperator::PackedAttention,
                         PackedAttentionBackend::MemoryEfficient, 6, false, 288, 384, 96, 0, 288, 0, 384,
                         PackedAttentionQkvWorkspaceLayout::Planar},
        LegacyParityCase{"PaTrtT6", LegacyParityOperator::PackedAttention,
                         PackedAttentionBackend::Trt, 6, false, 288, 384, 0, 288, 0, 0, 384,
                         PackedAttentionQkvWorkspaceLayout::InterleavedTn3h},
        LegacyParityCase{"PaUnfusedT6", LegacyParityOperator::PackedAttention,
                         PackedAttentionBackend::Unfused, 6, false, 288, 384, 128, 0, 384, 256, 896,
                         PackedAttentionQkvWorkspaceLayout::Planar},
        LegacyParityCase{"PaMeaT8", LegacyParityOperator::PackedAttention,
                         PackedAttentionBackend::MemoryEfficient, 8, false, 384, 384, 128, 0, 384, 0, 384,
                         PackedAttentionQkvWorkspaceLayout::Planar},
        LegacyParityCase{"PaTrtT8", LegacyParityOperator::PackedAttention,
                         PackedAttentionBackend::Trt, 8, false, 384, 384, 0, 384, 0, 0, 384,
                         PackedAttentionQkvWorkspaceLayout::InterleavedTn3h},
        LegacyParityCase{"PaUnfusedT8", LegacyParityOperator::PackedAttention,
                         PackedAttentionBackend::Unfused, 8, false, 384, 384, 128, 0, 384, 256, 896,
                         PackedAttentionQkvWorkspaceLayout::Planar}),
    [](const testing::TestParamInfo<LegacyParityCase>& info) {
      return std::string(info.param.name);
    });

TEST(PackedAttentionWorkspaceTest, UnfusedGoldenPreservesLegacyCapacityWhenPacked) {
  auto problem_result = BuildPackedAttentionProblem(ValidPackedAttentionInputs());
  ASSERT_TRUE(problem_result.status.IsOK()) << problem_result.status.message;
  problem_result.problem.backend = PackedAttentionBackend::Unfused;

  auto workspace_result = GetPackedAttentionWorkspaceRecipe(problem_result.problem);
  ASSERT_TRUE(workspace_result.status.IsOK()) << workspace_result.status.message;
  const auto& recipe = workspace_result.recipe;

  // Projection: T * (Q + K + V) * 2 = 6 * 20 * 2 = 240.
  EXPECT_EQ(recipe.projection_bytes, 240U);
  EXPECT_EQ(recipe.projection_m, 6);
  EXPECT_EQ(recipe.projection_n, 20);
  EXPECT_EQ(recipe.projection_k, 6);

  // QKV capacity: B*S*N*(H+H+Hv)*2 = 2*4*2*10*2 = 320.
  // Each attention scratch: align256(2*2*2*4*4) = 256. Total = 320 + 2*256 = 832.
  EXPECT_EQ(recipe.qkv_capacity_bytes, 320U);
  EXPECT_EQ(recipe.qkv_layout, PackedAttentionQkvWorkspaceLayout::Planar);
  EXPECT_EQ(recipe.q_offset_bytes, 0U);
  EXPECT_EQ(recipe.q_bytes, 128U);
  EXPECT_EQ(recipe.k_offset_bytes, 128U);
  EXPECT_EQ(recipe.v_offset_bytes, 256U);
  EXPECT_EQ(recipe.v_bytes, 64U);
  EXPECT_EQ(recipe.backend_workspace_offset_bytes, 320U);
  EXPECT_EQ(recipe.backend_workspace_bytes, 256U);
  EXPECT_TRUE(recipe.has_second_scratch);
  EXPECT_EQ(recipe.second_scratch_offset_bytes, 576U);
  EXPECT_EQ(recipe.attention_workspace_bytes, 832U);
}

TEST(PackedAttentionWorkspaceTest, LegacyAttentionComponentMatchesForTEqualPaddedCapacity) {
  auto pa_problem = BuildPackedAttentionProblem(ValidPackedAttentionInputs(8));
  ASSERT_TRUE(pa_problem.status.IsOK()) << pa_problem.status.message;
  pa_problem.problem.backend = PackedAttentionBackend::Unfused;
  auto pa_workspace = GetPackedAttentionWorkspaceRecipe(pa_problem.problem);
  ASSERT_TRUE(pa_workspace.status.IsOK()) << pa_workspace.status.message;

  auto pmha_problem = BuildPackedMultiHeadAttentionProblem(ValidSeparateQkvInputs(8));
  ASSERT_TRUE(pmha_problem.status.IsOK()) << pmha_problem.status.message;
  pmha_problem.problem.backend = PackedAttentionBackend::Unfused;
  auto pmha_workspace = GetPackedMultiHeadAttentionWorkspaceRecipe(pmha_problem.problem);
  ASSERT_TRUE(pmha_workspace.status.IsOK()) << pmha_workspace.status.message;

  // Both operators retain the shared legacy B*S attention allocation.
  EXPECT_EQ(pa_workspace.recipe.attention_workspace_bytes, 832U);
  EXPECT_EQ(pmha_workspace.recipe.attention_workspace_bytes, 832U);
  EXPECT_EQ(pa_workspace.recipe.projection_bytes, 320U);
  EXPECT_EQ(pmha_workspace.recipe.projection_bytes, 0U);
}

TEST(PackedAttentionWorkspaceTest, TokenCountGreaterThanPaddedCapacityIsRejected) {
  auto pa_result = BuildPackedAttentionProblem(ValidPackedAttentionInputs(9));
  EXPECT_EQ(pa_result.status.error, PackedAttentionWorkspaceError::InvalidArgument);

  auto pmha_result = BuildPackedMultiHeadAttentionProblem(ValidSeparateQkvInputs(9));
  EXPECT_EQ(pmha_result.status.error, PackedAttentionWorkspaceError::InvalidArgument);
}

TEST(PackedAttentionWorkspaceTest, PackedMetadataShapesMustMatchBatchAndSequence) {
  auto pa_inputs = ValidPackedAttentionInputs();
  pa_inputs.token_offset = Shape({8});
  EXPECT_EQ(BuildPackedAttentionProblem(pa_inputs).status.error,
            PackedAttentionWorkspaceError::InvalidArgument);

  pa_inputs = ValidPackedAttentionInputs();
  pa_inputs.cumulative_sequence_length = Shape({4});
  EXPECT_EQ(BuildPackedAttentionProblem(pa_inputs).status.error,
            PackedAttentionWorkspaceError::InvalidArgument);

  auto pmha_inputs = ValidSeparateQkvInputs();
  pmha_inputs.cumulative_sequence_length = Shape({2});
  EXPECT_EQ(BuildPackedMultiHeadAttentionProblem(pmha_inputs).status.error,
            PackedAttentionWorkspaceError::InvalidArgument);
}

TEST(PackedAttentionWorkspaceTest, LargeAttentionBiasDimensionIsNotReportedAsRankError) {
  auto inputs = ValidPackedAttentionInputs();
  inputs.has_attention_bias = true;
  inputs.attention_bias =
      Shape({1, 2, static_cast<int64_t>(std::numeric_limits<int32_t>::max()) + 1, 4});

  const auto result = BuildPackedAttentionProblem(inputs);
  EXPECT_EQ(result.status.error, PackedAttentionWorkspaceError::InvalidArgument);
  EXPECT_NE(std::string(result.status.message).find("int32 CUDA ABI"), std::string::npos)
      << result.status.message;
  EXPECT_EQ(std::string(result.status.message).find("rank"), std::string::npos)
      << result.status.message;
}

TEST(PackedAttentionWorkspaceTest, FusedViewsUseTButTotalRetainsBatchSequenceCapacity) {
  auto problem_result = BuildPackedMultiHeadAttentionProblem(ValidPackedQkvInputs());
  ASSERT_TRUE(problem_result.status.IsOK()) << problem_result.status.message;
  problem_result.problem.backend = PackedAttentionBackend::Flash;

  auto workspace_result = GetPackedMultiHeadAttentionWorkspaceRecipe(problem_result.problem);
  ASSERT_TRUE(workspace_result.status.IsOK()) << workspace_result.status.message;
  const auto& recipe = workspace_result.recipe;

  // Legacy QKV capacity uses B*S: 2*4*2*(4+4+4)*2 = 384.
  EXPECT_EQ(recipe.qkv_capacity_bytes, 384U);
  // Runtime views use T: each Q/K/V view is 6*2*4*2 = 96.
  EXPECT_EQ(recipe.qkv_layout, PackedAttentionQkvWorkspaceLayout::Planar);
  EXPECT_EQ(recipe.q_bytes, 96U);
  EXPECT_EQ(recipe.k_offset_bytes, 96U);
  EXPECT_EQ(recipe.v_offset_bytes, 192U);
  EXPECT_EQ(recipe.backend_workspace_offset_bytes, 288U);
  // Flash LSE is 4*B*S*N = 64, while total remains capacity + LSE.
  EXPECT_EQ(recipe.backend_workspace_bytes, 64U);
  EXPECT_EQ(recipe.attention_workspace_bytes, 448U);
}

TEST(PackedAttentionWorkspaceTest, ProjectionUsesActualGemmDimensionsAndPmhaHasNone) {
  auto pa_problem = BuildPackedAttentionProblem(ValidPackedAttentionInputs());
  ASSERT_TRUE(pa_problem.status.IsOK()) << pa_problem.status.message;
  auto pa_workspace = GetPackedAttentionWorkspaceRecipe(pa_problem.problem);
  ASSERT_TRUE(pa_workspace.status.IsOK()) << pa_workspace.status.message;

  // input_hidden=6, Q hidden=8, V hidden=4.
  EXPECT_EQ(pa_workspace.recipe.projection_k, 6);
  EXPECT_EQ(pa_workspace.recipe.projection_n, 20);
  EXPECT_EQ(pa_workspace.recipe.projection_bytes, 240U);

  auto pmha_problem = BuildPackedMultiHeadAttentionProblem(ValidSeparateQkvInputs());
  ASSERT_TRUE(pmha_problem.status.IsOK()) << pmha_problem.status.message;
  auto pmha_workspace = GetPackedMultiHeadAttentionWorkspaceRecipe(pmha_problem.problem);
  ASSERT_TRUE(pmha_workspace.status.IsOK()) << pmha_workspace.status.message;
  EXPECT_EQ(pmha_problem.problem.hidden_size, 8);
  EXPECT_EQ(pmha_problem.problem.v_hidden_size, 4);
  EXPECT_EQ(pmha_workspace.recipe.projection_bytes, 0U);
}

TEST(PackedAttentionWorkspaceTest, PmhaDirectQkvRoutesHaveZeroProjectionAndNoQkvWorkspace) {
  auto packed_problem = BuildPackedMultiHeadAttentionProblem(ValidPackedQkvInputs());
  ASSERT_TRUE(packed_problem.status.IsOK()) << packed_problem.status.message;
  packed_problem.problem.backend = PackedAttentionBackend::Trt;
  packed_problem.problem.trt_runner_available = true;
  auto trt_workspace = GetPackedMultiHeadAttentionWorkspaceRecipe(packed_problem.problem);
  ASSERT_TRUE(trt_workspace.status.IsOK()) << trt_workspace.status.message;
  EXPECT_TRUE(trt_workspace.recipe.no_qkv_workspace);
  EXPECT_EQ(trt_workspace.recipe.qkv_layout, PackedAttentionQkvWorkspaceLayout::None);
  EXPECT_EQ(trt_workspace.recipe.projection_bytes, 0U);
  EXPECT_EQ(trt_workspace.recipe.attention_workspace_bytes, 0U);

  auto separate_problem = BuildPackedMultiHeadAttentionProblem(ValidSeparateQkvInputs());
  ASSERT_TRUE(separate_problem.status.IsOK()) << separate_problem.status.message;
  separate_problem.problem.backend = PackedAttentionBackend::MemoryEfficient;
  auto mea_workspace = GetPackedMultiHeadAttentionWorkspaceRecipe(separate_problem.problem);
  ASSERT_TRUE(mea_workspace.status.IsOK()) << mea_workspace.status.message;
  EXPECT_TRUE(mea_workspace.recipe.no_qkv_workspace);
  EXPECT_EQ(mea_workspace.recipe.qkv_layout, PackedAttentionQkvWorkspaceLayout::None);
  EXPECT_EQ(mea_workspace.recipe.attention_workspace_bytes, 0U);
}

TEST(PackedAttentionWorkspaceTest, DirectQkvFusedRoutesDoNotRequireInt32SequenceSquare) {
  constexpr int64_t kSequenceLength = 46341;  // S*S is greater than INT32_MAX.

  auto separate_inputs = ValidSeparateEqualQkvInputs(1);
  separate_inputs.token_offset = Shape({1, kSequenceLength});
  separate_inputs.cumulative_sequence_length = Shape({2});
  auto separate_problem = BuildPackedMultiHeadAttentionProblem(separate_inputs);
  ASSERT_TRUE(separate_problem.status.IsOK()) << separate_problem.status.message;

  for (PackedAttentionBackend backend :
       {PackedAttentionBackend::Flash, PackedAttentionBackend::MemoryEfficient}) {
    separate_problem.problem.backend = backend;
    auto workspace = GetPackedMultiHeadAttentionWorkspaceRecipe(separate_problem.problem);
    ASSERT_TRUE(workspace.status.IsOK()) << workspace.status.message;
    EXPECT_TRUE(workspace.recipe.no_qkv_workspace);
  }

  auto packed_inputs = ValidPackedQkvInputs(1);
  packed_inputs.token_offset = Shape({1, kSequenceLength});
  packed_inputs.cumulative_sequence_length = Shape({2});
  auto packed_problem = BuildPackedMultiHeadAttentionProblem(packed_inputs);
  ASSERT_TRUE(packed_problem.status.IsOK()) << packed_problem.status.message;
  packed_problem.problem.backend = PackedAttentionBackend::Trt;
  packed_problem.problem.trt_runner_available = true;
  auto trt_workspace = GetPackedMultiHeadAttentionWorkspaceRecipe(packed_problem.problem);
  ASSERT_TRUE(trt_workspace.status.IsOK()) << trt_workspace.status.message;
  EXPECT_TRUE(trt_workspace.recipe.no_qkv_workspace);

  separate_problem.problem.backend = PackedAttentionBackend::Unfused;
  EXPECT_EQ(GetPackedMultiHeadAttentionWorkspaceRecipe(separate_problem.problem).status.error,
            PackedAttentionWorkspaceError::InvalidArgument);
}

TEST(PackedAttentionWorkspaceTest, MeaAttentionBiasStridesUseInt64) {
  constexpr int64_t kSequenceLength = 32768;
  constexpr int64_t kNumHeads = 2;
  static_assert(kNumHeads * kSequenceLength * kSequenceLength >
                std::numeric_limits<int32_t>::max());

  auto inputs = ValidPackedQkvInputs(1);
  inputs.token_offset = Shape({1, kSequenceLength});
  inputs.cumulative_sequence_length = Shape({2});
  inputs.attention_bias = Shape({1, kNumHeads, kSequenceLength, kSequenceLength});
  inputs.has_attention_bias = true;

  auto problem = BuildPackedMultiHeadAttentionProblem(inputs);
  ASSERT_TRUE(problem.status.IsOK()) << problem.status.message;
  problem.problem.backend = PackedAttentionBackend::MemoryEfficient;
  auto workspace = GetPackedMultiHeadAttentionWorkspaceRecipe(problem.problem);
  ASSERT_TRUE(workspace.status.IsOK()) << workspace.status.message;
  EXPECT_EQ(workspace.recipe.qkv_layout, PackedAttentionQkvWorkspaceLayout::Planar);
}

TEST(PackedAttentionWorkspaceTest, TrtMaterializationExposesOnlyInterleavedTn3hRegion) {
  auto problem = BuildPackedMultiHeadAttentionProblem(ValidSeparateEqualQkvInputs());
  ASSERT_TRUE(problem.status.IsOK()) << problem.status.message;
  problem.problem.backend = PackedAttentionBackend::Trt;
  problem.problem.trt_runner_available = true;

  auto workspace = GetPackedMultiHeadAttentionWorkspaceRecipe(problem.problem);
  ASSERT_TRUE(workspace.status.IsOK()) << workspace.status.message;
  const auto& recipe = workspace.recipe;

  EXPECT_EQ(recipe.qkv_layout, PackedAttentionQkvWorkspaceLayout::InterleavedTn3h);
  EXPECT_EQ(recipe.interleaved_qkv_offset_bytes, 0U);
  // Producer address: (((t * N + n) * 3 + component) * H + h) * sizeof(fp16).
  // With T=6, N=2, H=4, the end of V for the last head is byte 288.
  constexpr size_t kProducerRegionEnd = ((((6U - 1) * 2 + (2U - 1)) * 3 + 2) * 4 + 4) * 2;
  EXPECT_EQ(recipe.interleaved_qkv_bytes, kProducerRegionEnd);
  EXPECT_EQ(recipe.q_offset_bytes, 0U);
  EXPECT_EQ(recipe.q_bytes, 0U);
  EXPECT_EQ(recipe.k_offset_bytes, 0U);
  EXPECT_EQ(recipe.k_bytes, 0U);
  EXPECT_EQ(recipe.v_offset_bytes, 0U);
  EXPECT_EQ(recipe.v_bytes, 0U);
}

TEST(PackedAttentionWorkspaceTest, MemoryEfficientAccumulatorUsesLegacyCapacity) {
  auto inputs = ValidSeparateQkvInputs();
  inputs.value = Shape({6, 320});
  inputs.has_bias = true;
  inputs.bias = Shape({336});
  auto problem_result = BuildPackedMultiHeadAttentionProblem(inputs);
  ASSERT_TRUE(problem_result.status.IsOK()) << problem_result.status.message;
  problem_result.problem.backend = PackedAttentionBackend::MemoryEfficient;

  auto workspace_result = GetPackedMultiHeadAttentionWorkspaceRecipe(problem_result.problem);
  ASSERT_TRUE(workspace_result.status.IsOK()) << workspace_result.status.message;
  const auto& recipe = workspace_result.recipe;

  // QKV capacity: 2*4*2*(4+4+160)*2 = 5376. T-view ends at 6*2*168*2 = 4032.
  // FP32 accumulator: 4*2*4*2*160 = 10240. Legacy total = 5376 + 10240 = 15616.
  EXPECT_EQ(recipe.qkv_capacity_bytes, 5376U);
  EXPECT_EQ(recipe.backend_workspace_offset_bytes, 4032U);
  EXPECT_EQ(recipe.backend_workspace_bytes, 10240U);
  EXPECT_EQ(recipe.attention_workspace_bytes, 15616U);
}

TEST(PackedAttentionWorkspaceTest, PackedQkvAxesAreStrictlyValidated) {
  auto inputs = ValidPackedQkvInputs();
  inputs.query = Shape({6, 1, 3, 4});
  EXPECT_EQ(BuildPackedMultiHeadAttentionProblem(inputs).status.error,
            PackedAttentionWorkspaceError::InvalidArgument);

  inputs = ValidPackedQkvInputs();
  inputs.query = Shape({6, 2, 2, 4});
  EXPECT_EQ(BuildPackedMultiHeadAttentionProblem(inputs).status.error,
            PackedAttentionWorkspaceError::InvalidArgument);

  inputs = ValidPackedQkvInputs();
  inputs.query = Shape({6, 2, 3});
  EXPECT_EQ(BuildPackedMultiHeadAttentionProblem(inputs).status.error,
            PackedAttentionWorkspaceError::InvalidArgument);
}

TEST(PackedAttentionWorkspaceTest, HeadDivisibilityAndSeparateQkvConsistencyAreValidated) {
  auto inputs = ValidSeparateQkvInputs();
  inputs.query = Shape({6, 7});
  inputs.key = Shape({6, 7});
  EXPECT_EQ(BuildPackedMultiHeadAttentionProblem(inputs).status.error,
            PackedAttentionWorkspaceError::InvalidArgument);

  inputs = ValidSeparateQkvInputs();
  inputs.key = Shape({6, 4});
  EXPECT_EQ(BuildPackedMultiHeadAttentionProblem(inputs).status.error,
            PackedAttentionWorkspaceError::InvalidArgument);

  inputs = ValidSeparateQkvInputs();
  inputs.value = Shape({5, 4});
  EXPECT_EQ(BuildPackedMultiHeadAttentionProblem(inputs).status.error,
            PackedAttentionWorkspaceError::InvalidArgument);

  inputs = ValidSeparateQkvInputs();
  inputs.has_value = false;
  EXPECT_EQ(BuildPackedMultiHeadAttentionProblem(inputs).status.error,
            PackedAttentionWorkspaceError::InvalidArgument);

  auto pa_inputs = ValidPackedAttentionInputs();
  pa_inputs.qkv_hidden_sizes = {7, 7, 6};
  EXPECT_EQ(BuildPackedAttentionProblem(pa_inputs).status.error,
            PackedAttentionWorkspaceError::InvalidArgument);
}

TEST(PackedAttentionWorkspaceTest, Int32NarrowingAndDerivedProductsAreValidated) {
  auto inputs = ValidPackedQkvInputs();
  inputs.query = Shape({static_cast<int64_t>(std::numeric_limits<int32_t>::max()) + 1, 2, 3, 4});
  EXPECT_EQ(BuildPackedMultiHeadAttentionProblem(inputs).status.error,
            PackedAttentionWorkspaceError::InvalidArgument);

  inputs = ValidPackedQkvInputs(0);
  inputs.query = Shape({0, 50000, 3, 1});
  inputs.num_heads = 50000;
  inputs.token_offset = Shape({1, 50000});
  inputs.cumulative_sequence_length = Shape({2});
  auto large_ns_problem = BuildPackedMultiHeadAttentionProblem(inputs);
  ASSERT_TRUE(large_ns_problem.status.IsOK()) << large_ns_problem.status.message;
  large_ns_problem.problem.backend = PackedAttentionBackend::Unfused;
  EXPECT_EQ(GetPackedMultiHeadAttentionWorkspaceRecipe(large_ns_problem.problem).status.error,
            PackedAttentionWorkspaceError::InvalidArgument);

  inputs.query = Shape({0, 50000, 3, 50000});
  EXPECT_EQ(BuildPackedMultiHeadAttentionProblem(inputs).status.error,
            PackedAttentionWorkspaceError::InvalidArgument);

  auto large_bs_inputs = ValidSeparateEqualQkvInputs(0);
  large_bs_inputs.query = Shape({0, 8});
  large_bs_inputs.key = Shape({0, 8});
  large_bs_inputs.value = Shape({0, 8});
  large_bs_inputs.token_offset = Shape({50000, 50000});
  large_bs_inputs.cumulative_sequence_length = Shape({50001});
  auto large_bs_problem = BuildPackedMultiHeadAttentionProblem(large_bs_inputs);
  ASSERT_TRUE(large_bs_problem.status.IsOK()) << large_bs_problem.status.message;
  large_bs_problem.problem.backend = PackedAttentionBackend::MemoryEfficient;
  EXPECT_TRUE(GetPackedMultiHeadAttentionWorkspaceRecipe(large_bs_problem.problem).status.IsOK());
  large_bs_problem.problem.backend = PackedAttentionBackend::Unfused;
  EXPECT_EQ(GetPackedMultiHeadAttentionWorkspaceRecipe(large_bs_problem.problem).status.error,
            PackedAttentionWorkspaceError::InvalidArgument);

  constexpr int64_t kLargeTokenCount = 100000;
  constexpr int64_t kLargeNumHeads = 4000;
  auto materialized_inputs = ValidPackedQkvInputs(kLargeTokenCount);
  materialized_inputs.query = Shape({kLargeTokenCount, kLargeNumHeads, 3, 8});
  materialized_inputs.num_heads = kLargeNumHeads;
  materialized_inputs.token_offset = Shape({1, kLargeTokenCount});
  materialized_inputs.cumulative_sequence_length = Shape({2});
  auto materialized_problem = BuildPackedMultiHeadAttentionProblem(materialized_inputs);
  ASSERT_TRUE(materialized_problem.status.IsOK()) << materialized_problem.status.message;
  materialized_problem.problem.backend = PackedAttentionBackend::MemoryEfficient;
  EXPECT_EQ(GetPackedMultiHeadAttentionWorkspaceRecipe(materialized_problem.problem).status.error,
            PackedAttentionWorkspaceError::InvalidArgument);

  PackedMultiHeadAttentionInputShapes direct_inputs;
  direct_inputs.query = Shape({kLargeTokenCount, kLargeNumHeads * 8});
  direct_inputs.key = direct_inputs.query;
  direct_inputs.value = direct_inputs.query;
  direct_inputs.token_offset = Shape({1, kLargeTokenCount});
  direct_inputs.cumulative_sequence_length = Shape({2});
  direct_inputs.element_size = 2;
  direct_inputs.num_heads = kLargeNumHeads;
  direct_inputs.has_key = true;
  direct_inputs.has_value = true;
  auto direct_problem = BuildPackedMultiHeadAttentionProblem(direct_inputs);
  ASSERT_TRUE(direct_problem.status.IsOK()) << direct_problem.status.message;
  direct_problem.problem.backend = PackedAttentionBackend::MemoryEfficient;
  auto direct_workspace = GetPackedMultiHeadAttentionWorkspaceRecipe(direct_problem.problem);
  ASSERT_TRUE(direct_workspace.status.IsOK()) << direct_workspace.status.message;
  EXPECT_TRUE(direct_workspace.recipe.no_qkv_workspace);
}

TEST(PackedAttentionWorkspaceTest, AllQkvMaterializationProducersUseSelectedVectorIndexWidth) {
  auto vector4_inputs = ValidSeparateEqualQkvInputs();
  auto vector4_problem = BuildPackedMultiHeadAttentionProblem(vector4_inputs);
  ASSERT_TRUE(vector4_problem.status.IsOK()) << vector4_problem.status.message;
  EXPECT_EQ(vector4_problem.problem.qkv_materialization_index_width,
            PackedAttentionQkvMaterializationIndexWidth::Vector4);

  auto vector2_inputs = ValidSeparateEqualQkvInputs();
  vector2_inputs.query = Shape({6, 12});
  vector2_inputs.key = Shape({6, 12});
  vector2_inputs.value = Shape({6, 12});
  auto vector2_problem = BuildPackedMultiHeadAttentionProblem(vector2_inputs);
  ASSERT_TRUE(vector2_problem.status.IsOK()) << vector2_problem.status.message;
  EXPECT_EQ(vector2_problem.problem.qkv_materialization_index_width,
            PackedAttentionQkvMaterializationIndexWidth::Vector2);

  auto scalar_inputs = ValidSeparateEqualQkvInputs();
  scalar_inputs.query = Shape({6, 6});
  scalar_inputs.key = Shape({6, 6});
  scalar_inputs.value = Shape({6, 6});
  auto scalar_problem = BuildPackedMultiHeadAttentionProblem(scalar_inputs);
  ASSERT_TRUE(scalar_problem.status.IsOK()) << scalar_problem.status.message;
  EXPECT_EQ(scalar_problem.problem.qkv_materialization_index_width,
            PackedAttentionQkvMaterializationIndexWidth::Scalar);
  scalar_problem.problem.backend = PackedAttentionBackend::Trt;
  scalar_problem.problem.trt_runner_available = true;
  scalar_problem.problem.qkv_materialization_index_width =
      PackedAttentionQkvMaterializationIndexWidth::Vector4;
  EXPECT_EQ(GetPackedMultiHeadAttentionWorkspaceRecipe(scalar_problem.problem).status.error,
            PackedAttentionWorkspaceError::InvalidArgument);

  auto packed_problem = BuildPackedMultiHeadAttentionProblem(ValidPackedQkvInputs());
  ASSERT_TRUE(packed_problem.status.IsOK()) << packed_problem.status.message;
  EXPECT_EQ(packed_problem.problem.qkv_materialization_index_width,
            PackedAttentionQkvMaterializationIndexWidth::Vector4);

  auto packed_vector2_inputs = ValidPackedQkvInputs();
  packed_vector2_inputs.query = Shape({6, 2, 3, 2});
  auto packed_vector2_problem = BuildPackedMultiHeadAttentionProblem(packed_vector2_inputs);
  ASSERT_TRUE(packed_vector2_problem.status.IsOK()) << packed_vector2_problem.status.message;
  EXPECT_EQ(packed_vector2_problem.problem.qkv_materialization_index_width,
            PackedAttentionQkvMaterializationIndexWidth::Vector2);

  auto packed_scalar_inputs = ValidPackedQkvInputs();
  packed_scalar_inputs.query = Shape({6, 2, 3, 1});
  auto packed_scalar_problem = BuildPackedMultiHeadAttentionProblem(packed_scalar_inputs);
  ASSERT_TRUE(packed_scalar_problem.status.IsOK()) << packed_scalar_problem.status.message;
  EXPECT_EQ(packed_scalar_problem.problem.qkv_materialization_index_width,
            PackedAttentionQkvMaterializationIndexWidth::Scalar);

  auto pa_problem = BuildPackedAttentionProblem(ValidPackedAttentionEqualHeadsInputs());
  ASSERT_TRUE(pa_problem.status.IsOK()) << pa_problem.status.message;
  EXPECT_EQ(pa_problem.problem.qkv_materialization_index_width,
            PackedAttentionQkvMaterializationIndexWidth::Vector4);

  auto pa_vector2_inputs = ValidPackedAttentionEqualHeadsInputs();
  pa_vector2_inputs.weights = Shape({6, 12});
  pa_vector2_inputs.bias = Shape({12});
  pa_vector2_inputs.qkv_hidden_sizes = {4, 4, 4};
  auto pa_vector2_problem = BuildPackedAttentionProblem(pa_vector2_inputs);
  ASSERT_TRUE(pa_vector2_problem.status.IsOK()) << pa_vector2_problem.status.message;
  EXPECT_EQ(pa_vector2_problem.problem.qkv_materialization_index_width,
            PackedAttentionQkvMaterializationIndexWidth::Vector2);

  auto pa_scalar_inputs = ValidPackedAttentionEqualHeadsInputs();
  pa_scalar_inputs.weights = Shape({6, 6});
  pa_scalar_inputs.bias = Shape({6});
  pa_scalar_inputs.qkv_hidden_sizes = {2, 2, 2};
  auto pa_scalar_problem = BuildPackedAttentionProblem(pa_scalar_inputs);
  ASSERT_TRUE(pa_scalar_problem.status.IsOK()) << pa_scalar_problem.status.message;
  EXPECT_EQ(pa_scalar_problem.problem.qkv_materialization_index_width,
            PackedAttentionQkvMaterializationIndexWidth::Scalar);
}

TEST(PackedAttentionWorkspaceTest, PackedFormatMaterializationBoundariesUseProducerIndexWidth) {
  constexpr int64_t kNumHeads = 32768;
  constexpr int64_t kLargestVectorSafeSequenceLength = 21845;
  static_assert(kLargestVectorSafeSequenceLength * kNumHeads * 3 <=
                std::numeric_limits<int32_t>::max());
  static_assert((kLargestVectorSafeSequenceLength + 1) * kNumHeads * 3 >
                std::numeric_limits<int32_t>::max());
  static_assert((kLargestVectorSafeSequenceLength + 1) *
                    (kLargestVectorSafeSequenceLength + 1) <=
                std::numeric_limits<int32_t>::max());

  const auto make_inputs = [](int64_t sequence_length, int64_t head_size) {
    PackedMultiHeadAttentionInputShapes inputs;
    inputs.query = Shape({0, kNumHeads, 3, head_size});
    inputs.token_offset = Shape({1, sequence_length});
    inputs.cumulative_sequence_length = Shape({2});
    inputs.element_size = 2;
    inputs.num_heads = kNumHeads;
    return inputs;
  };

  struct BoundaryCase {
    int64_t head_size;
    PackedAttentionQkvMaterializationIndexWidth expected_width;
  };

  for (const auto& test_case :
       {BoundaryCase{4, PackedAttentionQkvMaterializationIndexWidth::Vector4},
        BoundaryCase{2, PackedAttentionQkvMaterializationIndexWidth::Vector2},
        BoundaryCase{1, PackedAttentionQkvMaterializationIndexWidth::Scalar}}) {
    SCOPED_TRACE(test_case.head_size);
    auto boundary_problem = BuildPackedMultiHeadAttentionProblem(
        make_inputs(kLargestVectorSafeSequenceLength, test_case.head_size));
    ASSERT_TRUE(boundary_problem.status.IsOK()) << boundary_problem.status.message;
    ASSERT_EQ(boundary_problem.problem.qkv_materialization_index_width,
              test_case.expected_width);
    boundary_problem.problem.backend = PackedAttentionBackend::Unfused;
    const auto boundary_workspace =
        GetPackedMultiHeadAttentionWorkspaceRecipe(boundary_problem.problem);
    ASSERT_TRUE(boundary_workspace.status.IsOK()) << boundary_workspace.status.message;

    auto beyond_problem = BuildPackedMultiHeadAttentionProblem(
        make_inputs(kLargestVectorSafeSequenceLength + 1, test_case.head_size));
    ASSERT_TRUE(beyond_problem.status.IsOK()) << beyond_problem.status.message;
    ASSERT_EQ(beyond_problem.problem.qkv_materialization_index_width,
              test_case.expected_width);
    beyond_problem.problem.backend = PackedAttentionBackend::Unfused;
    EXPECT_EQ(GetPackedMultiHeadAttentionWorkspaceRecipe(beyond_problem.problem).status.error,
              PackedAttentionWorkspaceError::InvalidArgument);
  }
}

TEST(PackedAttentionWorkspaceTest, PackedAttentionProjectionUsesVectorUnitMaterializationBoundary) {
  constexpr int64_t kNumHeads = 32768;
  constexpr int64_t kHeadSize = 4;
  constexpr int64_t kLargestVectorSafeTokenCount = 21845;
  constexpr int64_t kHiddenSize = kNumHeads * kHeadSize;
  constexpr int64_t kProjectionSize = 3 * kHiddenSize;
  static_assert(kLargestVectorSafeTokenCount * kNumHeads * 3 <=
                std::numeric_limits<int32_t>::max());
  static_assert((kLargestVectorSafeTokenCount + 1) * kNumHeads * 3 >
                std::numeric_limits<int32_t>::max());
  static_assert(kLargestVectorSafeTokenCount * kProjectionSize >
                std::numeric_limits<int32_t>::max());

  const auto make_inputs = [](int64_t token_count) {
    PackedAttentionInputShapes inputs;
    inputs.input = Shape({token_count, 1});
    inputs.weights = Shape({1, kProjectionSize});
    inputs.bias = Shape({kProjectionSize});
    inputs.token_offset = Shape({1, token_count});
    inputs.cumulative_sequence_length = Shape({2});
    inputs.element_size = 2;
    inputs.num_heads = kNumHeads;
    inputs.qkv_hidden_sizes_count = 3;
    inputs.qkv_hidden_sizes = {kHiddenSize, kHiddenSize, kHiddenSize};
    return inputs;
  };

  auto boundary_problem =
      BuildPackedAttentionProblem(make_inputs(kLargestVectorSafeTokenCount));
  ASSERT_TRUE(boundary_problem.status.IsOK()) << boundary_problem.status.message;
  ASSERT_EQ(boundary_problem.problem.qkv_materialization_index_width,
            PackedAttentionQkvMaterializationIndexWidth::Vector4);
  boundary_problem.problem.backend = PackedAttentionBackend::Unfused;
  const auto boundary_workspace = GetPackedAttentionWorkspaceRecipe(boundary_problem.problem);
  ASSERT_TRUE(boundary_workspace.status.IsOK()) << boundary_workspace.status.message;
  EXPECT_EQ(boundary_workspace.recipe.projection_m, kLargestVectorSafeTokenCount);
  EXPECT_EQ(boundary_workspace.recipe.projection_n, kProjectionSize);

  auto beyond_problem =
      BuildPackedAttentionProblem(make_inputs(kLargestVectorSafeTokenCount + 1));
  ASSERT_TRUE(beyond_problem.status.IsOK()) << beyond_problem.status.message;
  beyond_problem.problem.backend = PackedAttentionBackend::Unfused;
  EXPECT_EQ(GetPackedAttentionWorkspaceRecipe(beyond_problem.problem).status.error,
            PackedAttentionWorkspaceError::InvalidArgument);
}

TEST(PackedAttentionWorkspaceTest, MemoryEfficientQueryBlockRoundingIsValidatedAtInt32Boundary) {
  constexpr int64_t kLargestRound64SafeSequenceLength =
      static_cast<int64_t>(std::numeric_limits<int32_t>::max()) - 63;

  const auto make_problem = [](int64_t sequence_length) {
    auto inputs = ValidSeparateEqualQkvInputs(0);
    inputs.query = Shape({0, 8});
    inputs.key = inputs.query;
    inputs.value = inputs.query;
    inputs.token_offset = Shape({1, sequence_length});
    inputs.cumulative_sequence_length = Shape({2});
    return BuildPackedMultiHeadAttentionProblem(inputs);
  };

  auto boundary_problem = make_problem(kLargestRound64SafeSequenceLength);
  ASSERT_TRUE(boundary_problem.status.IsOK()) << boundary_problem.status.message;
  boundary_problem.problem.backend = PackedAttentionBackend::MemoryEfficient;
  EXPECT_TRUE(GetPackedMultiHeadAttentionWorkspaceRecipe(boundary_problem.problem).status.IsOK());

  auto beyond_problem = make_problem(kLargestRound64SafeSequenceLength + 1);
  ASSERT_TRUE(beyond_problem.status.IsOK()) << beyond_problem.status.message;
  beyond_problem.problem.backend = PackedAttentionBackend::MemoryEfficient;
  EXPECT_EQ(GetPackedMultiHeadAttentionWorkspaceRecipe(beyond_problem.problem).status.error,
            PackedAttentionWorkspaceError::InvalidArgument);
}

TEST(PackedAttentionWorkspaceTest, WorkspaceViewsAreContainedInAttentionAllocation) {
  auto problem = BuildPackedMultiHeadAttentionProblem(ValidPackedQkvInputs());
  ASSERT_TRUE(problem.status.IsOK()) << problem.status.message;
  problem.problem.backend = PackedAttentionBackend::Unfused;
  auto workspace = GetPackedMultiHeadAttentionWorkspaceRecipe(problem.problem);
  ASSERT_TRUE(workspace.status.IsOK()) << workspace.status.message;
  EXPECT_TRUE(ValidatePackedAttentionWorkspaceRecipe(workspace.recipe).IsOK());

  auto out_of_bounds = workspace.recipe;
  out_of_bounds.v_offset_bytes = out_of_bounds.attention_workspace_bytes;
  out_of_bounds.v_bytes = 1;
  EXPECT_EQ(ValidatePackedAttentionWorkspaceRecipe(out_of_bounds).error,
            PackedAttentionWorkspaceError::InvalidArgument);

  out_of_bounds.v_offset_bytes = std::numeric_limits<size_t>::max();
  EXPECT_EQ(ValidatePackedAttentionWorkspaceRecipe(out_of_bounds).error,
            PackedAttentionWorkspaceError::Overflow);

  auto invalid_layout = workspace.recipe;
  invalid_layout.qkv_layout = static_cast<PackedAttentionQkvWorkspaceLayout>(999);
  EXPECT_EQ(ValidatePackedAttentionWorkspaceRecipe(invalid_layout).error,
            PackedAttentionWorkspaceError::InvalidArgument);

  auto missing_second_scratch = workspace.recipe;
  missing_second_scratch.has_second_scratch = false;
  EXPECT_EQ(ValidatePackedAttentionWorkspaceRecipe(missing_second_scratch).error,
            PackedAttentionWorkspaceError::InvalidArgument);

  problem.problem.backend = static_cast<PackedAttentionBackend>(999);
  EXPECT_EQ(GetPackedMultiHeadAttentionWorkspaceRecipe(problem.problem).status.error,
            PackedAttentionWorkspaceError::InvalidArgument);
}

TEST(PackedAttentionWorkspaceTest, PaRejectsFlashAndTrtRequiresExistingRunner) {
  auto problem_result = BuildPackedAttentionProblem(ValidPackedAttentionInputs());
  ASSERT_TRUE(problem_result.status.IsOK()) << problem_result.status.message;

  problem_result.problem.backend = PackedAttentionBackend::Flash;
  EXPECT_EQ(GetPackedAttentionWorkspaceRecipe(problem_result.problem).status.error,
            PackedAttentionWorkspaceError::InvalidArgument);

  problem_result.problem.backend = PackedAttentionBackend::Trt;
  problem_result.problem.trt_runner_available = false;
  EXPECT_EQ(GetPackedAttentionWorkspaceRecipe(problem_result.problem).status.error,
            PackedAttentionWorkspaceError::InvalidArgument);

  // A missing TRT prerequisite is represented by the actual selected fallback route.
  problem_result.problem.backend = PackedAttentionBackend::Unfused;
  auto fallback = GetPackedAttentionWorkspaceRecipe(problem_result.problem);
  ASSERT_TRUE(fallback.status.IsOK()) << fallback.status.message;
  EXPECT_EQ(fallback.recipe.attention_workspace_bytes, 832U);
}

TEST(PackedAttentionWorkspaceTest, ExactZeroWorkspaceIsValid) {
  PackedAttentionInputShapes inputs;
  inputs.input = Shape({0, 3});
  inputs.weights = Shape({3, 0});
  inputs.bias = Shape({0});
  inputs.token_offset = Shape({0, 0});
  inputs.cumulative_sequence_length = Shape({1});
  inputs.element_size = 2;
  inputs.num_heads = 2;
  inputs.qkv_hidden_sizes_count = 3;
  inputs.qkv_hidden_sizes = {0, 0, 0};

  auto problem_result = BuildPackedAttentionProblem(inputs);
  ASSERT_TRUE(problem_result.status.IsOK()) << problem_result.status.message;
  auto workspace_result = GetPackedAttentionWorkspaceRecipe(problem_result.problem);
  ASSERT_TRUE(workspace_result.status.IsOK()) << workspace_result.status.message;
  EXPECT_EQ(workspace_result.recipe.projection_bytes, 0U);
  EXPECT_EQ(workspace_result.recipe.attention_workspace_bytes, 0U);
  EXPECT_TRUE(workspace_result.recipe.has_second_scratch);
  EXPECT_EQ(workspace_result.recipe.second_scratch_offset_bytes, 0U);
  EXPECT_TRUE(ValidatePackedAttentionWorkspaceRecipe(workspace_result.recipe).IsOK());
}

}  // namespace test
}  // namespace onnxruntime
