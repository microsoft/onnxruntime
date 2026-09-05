// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"

#include <array>
#include <limits>
#include <string>
#include <utility>

#include "contrib_ops/cuda/bert/group_query_attention_workspace.h"

namespace onnxruntime {
namespace test {

using contrib::cuda::CheckedGQAWorkspaceAdd;
using contrib::cuda::CheckedGQAWorkspaceAlign;
using contrib::cuda::CheckedGQAWorkspaceMultiply;
using contrib::cuda::GetGQAPreparationRecipe;
using contrib::cuda::GQAKvQuantizationType;
using contrib::cuda::GQAPreparationRecipe;
using contrib::cuda::GQAPreparationRoute;
using contrib::cuda::GQAPreprocessMode;
using contrib::cuda::GQAWorkspaceError;
using contrib::cuda::GQAWorkspaceProblem;
using contrib::cuda::kGQAWorkspaceAlignment;
using contrib::cuda::ValidateGQAPreparationRecipe;

namespace {

GQAWorkspaceProblem ValidProblem() {
  GQAWorkspaceProblem problem;
  problem.qkv_element_size = 2;
  problem.cache_element_size = 2;
  problem.batch_size = 2;
  problem.sequence_length = 3;
  problem.num_heads = 4;
  problem.kv_num_heads = 2;
  problem.head_size = 8;
  problem.present_kv_cache_capacity = 5;
  return problem;
}

testing::AssertionResult BuildRecipe(
    const GQAWorkspaceProblem& problem,
    GQAPreparationRecipe& recipe,
    GQAPreprocessMode mode = GQAPreprocessMode::Unfused,
    bool fast_decode = false) {
  const auto result = GetGQAPreparationRecipe(
      problem, GQAPreparationRoute{mode, fast_decode});
  if (!result.status.IsOK()) {
    return testing::AssertionFailure() << result.status.message;
  }

  recipe = result.recipe;
  return testing::AssertionSuccess();
}

}  // namespace

TEST(GroupQueryAttentionWorkspaceTest, CheckedArithmeticRejectsOverflow) {
  size_t result = 0;
  EXPECT_TRUE(CheckedGQAWorkspaceAdd(7, 9, result).IsOK());
  EXPECT_EQ(result, 16U);
  EXPECT_EQ(
      CheckedGQAWorkspaceAdd(std::numeric_limits<size_t>::max(), 1, result).error,
      GQAWorkspaceError::Overflow);

  EXPECT_TRUE(CheckedGQAWorkspaceMultiply(7, 9, result).IsOK());
  EXPECT_EQ(result, 63U);
  EXPECT_EQ(
      CheckedGQAWorkspaceMultiply(std::numeric_limits<size_t>::max(), 2, result).error,
      GQAWorkspaceError::Overflow);

  EXPECT_TRUE(CheckedGQAWorkspaceAlign(257, 256, result).IsOK());
  EXPECT_EQ(result, 512U);
  EXPECT_EQ(CheckedGQAWorkspaceAlign(1, 0, result).error,
            GQAWorkspaceError::InvalidArgument);
}

TEST(GroupQueryAttentionWorkspaceTest, OrdinaryNonWindowedPreparationHasThreeVectorsAndQkv) {
  auto problem = ValidProblem();
  problem.is_packed_qkv = true;

  GQAPreparationRecipe recipe;
  ASSERT_TRUE(BuildRecipe(problem, recipe));

  // Sequence vectors: 3 * B * sizeof(int32_t) = 3*2*4 = 24.
  EXPECT_EQ(recipe.sequence_length_vector_count, 3U);
  EXPECT_EQ(recipe.sequence_lengths_offset_bytes, 0U);
  EXPECT_EQ(recipe.sequence_lengths_bytes, 24U);

  // Fallback packed preprocess: B*S*N*H*sizeof(T) = 2*3*4*8*2 = 384.
  EXPECT_EQ(recipe.qkv_preprocess_offset_bytes, 256U);
  EXPECT_EQ(recipe.qkv_preprocess_bytes, 384U);
  EXPECT_EQ(recipe.total_preparation_bytes, 640U);
  EXPECT_FALSE(recipe.uses_staging);
  EXPECT_FALSE(recipe.uses_compaction);
  EXPECT_EQ(recipe.effective_kv_cache_capacity, 5);
}

TEST(GroupQueryAttentionWorkspaceTest, WindowedSingleTokenUsesCompactionAndSixVectors) {
  auto problem = ValidProblem();
  problem.sequence_length = 1;
  problem.is_windowed_kv_cache = true;

  GQAPreparationRecipe recipe;
  ASSERT_TRUE(BuildRecipe(problem, recipe));

  // Dense row: H*sizeof(U) = 8*2 = 16.
  // Each compaction half: B*Nk*C*row = 2*2*5*16 = 320.
  EXPECT_TRUE(recipe.uses_compaction);
  EXPECT_FALSE(recipe.uses_staging);
  EXPECT_EQ(recipe.cache_row_bytes, 16U);
  EXPECT_EQ(recipe.compaction_offset_bytes, 0U);
  EXPECT_EQ(recipe.compaction_bytes, 640U);
  EXPECT_EQ(recipe.compaction_key_offset_bytes, 0U);
  EXPECT_EQ(recipe.compaction_key_bytes, 320U);
  EXPECT_EQ(recipe.compaction_value_offset_bytes, 320U);
  EXPECT_EQ(recipe.compaction_value_bytes, 320U);

  // Sequence vectors: 6 * B * sizeof(int32_t) = 6*2*4 = 48.
  EXPECT_EQ(recipe.sequence_length_vector_count, 6U);
  EXPECT_EQ(recipe.sequence_lengths_offset_bytes, 768U);
  EXPECT_EQ(recipe.sequence_lengths_bytes, 48U);
  EXPECT_EQ(recipe.total_preparation_bytes, 816U);
}

TEST(GroupQueryAttentionWorkspaceTest, WindowedMultiTokenUsesEffectiveCapacityForStaging) {
  auto problem = ValidProblem();
  problem.is_windowed_kv_cache = true;

  GQAPreparationRecipe recipe;
  ASSERT_TRUE(BuildRecipe(problem, recipe));

  // Effective staged capacity is C+S = 5+3 = 8 without changing the original
  // GroupQueryAttentionParameters::seqlen_present_kv_cache value.
  // Each staged cache: B*Nk*(C+S)*row = 2*2*8*16 = 512.
  EXPECT_EQ(problem.present_kv_cache_capacity, 5);
  EXPECT_EQ(recipe.effective_kv_cache_capacity, 8);
  EXPECT_TRUE(recipe.uses_staging);
  EXPECT_FALSE(recipe.uses_compaction);
  EXPECT_EQ(recipe.staged_key_offset_bytes, 0U);
  EXPECT_EQ(recipe.staged_key_bytes, 512U);
  EXPECT_EQ(recipe.staged_value_offset_bytes, 512U);
  EXPECT_EQ(recipe.staged_value_bytes, 512U);
  EXPECT_EQ(recipe.sequence_length_vector_count, 6U);
  EXPECT_EQ(recipe.sequence_lengths_offset_bytes, 1024U);
  EXPECT_EQ(recipe.sequence_lengths_bytes, 48U);
  EXPECT_EQ(recipe.total_preparation_bytes, 1072U);
}

TEST(GroupQueryAttentionWorkspaceTest, FlashFastDecodeSuppressesExactSingleTokenVectors) {
  auto problem = ValidProblem();
  problem.sequence_length = 1;

  GQAPreparationRecipe recipe;
  ASSERT_TRUE(BuildRecipe(problem, recipe, GQAPreprocessMode::Flash, true));
  EXPECT_EQ(recipe.sequence_length_vector_count, 0U);
  EXPECT_EQ(recipe.sequence_lengths_bytes, 0U);
  EXPECT_EQ(recipe.qkv_preprocess_bytes, 0U);
  EXPECT_EQ(recipe.total_preparation_bytes, 0U);
}

TEST(GroupQueryAttentionWorkspaceTest, FlashFastDecodeAllowsMultiTokenAndKeepsSequenceVectors) {
  auto problem = ValidProblem();
  problem.sequence_length = 2;

  GQAPreparationRecipe recipe;
  ASSERT_TRUE(BuildRecipe(problem, recipe, GQAPreprocessMode::Flash, true));
  EXPECT_EQ(recipe.sequence_length_vector_count, 3U);
  EXPECT_EQ(recipe.sequence_lengths_bytes, 24U);
  EXPECT_EQ(recipe.qkv_preprocess_bytes, 0U);
  EXPECT_EQ(recipe.total_preparation_bytes, 24U);
}

struct FastDecodeContradictionCase {
  const char* name;
  GQAPreprocessMode mode;
  bool is_first_prompt;
  bool is_windowed_kv_cache;
  GQAKvQuantizationType k_quantization;
  GQAKvQuantizationType v_quantization;
  bool use_qk_norm;
};

class GroupQueryAttentionFastDecodeValidationTest
    : public testing::TestWithParam<FastDecodeContradictionCase> {};

TEST_P(GroupQueryAttentionFastDecodeValidationTest, RejectsImpossibleRuntimeRouteFacts) {
  const auto& test_case = GetParam();
  auto problem = ValidProblem();
  problem.is_first_prompt = test_case.is_first_prompt;
  problem.is_windowed_kv_cache = test_case.is_windowed_kv_cache;
  problem.k_quantization = test_case.k_quantization;
  problem.v_quantization = test_case.v_quantization;
  problem.use_qk_norm = test_case.use_qk_norm;

  const auto result = GetGQAPreparationRecipe(
      problem, GQAPreparationRoute{test_case.mode, true});
  EXPECT_EQ(result.status.error, GQAWorkspaceError::InvalidArgument);
}

INSTANTIATE_TEST_SUITE_P(
    Contradictions,
    GroupQueryAttentionFastDecodeValidationTest,
    testing::Values(
        FastDecodeContradictionCase{"NonFlashMode", GQAPreprocessMode::Unfused,
                                    false, false, GQAKvQuantizationType::None,
                                    GQAKvQuantizationType::None, false},
        FastDecodeContradictionCase{"FirstPrompt", GQAPreprocessMode::Flash,
                                    true, false, GQAKvQuantizationType::None,
                                    GQAKvQuantizationType::None, false},
        FastDecodeContradictionCase{"WindowedCache", GQAPreprocessMode::Flash,
                                    false, true, GQAKvQuantizationType::None,
                                    GQAKvQuantizationType::None, false},
        FastDecodeContradictionCase{"QuantizedK", GQAPreprocessMode::Flash,
                                    false, false, GQAKvQuantizationType::PerTensor,
                                    GQAKvQuantizationType::None, false},
        FastDecodeContradictionCase{"QuantizedV", GQAPreprocessMode::Flash,
                                    false, false, GQAKvQuantizationType::None,
                                    GQAKvQuantizationType::PerChannel, false},
        FastDecodeContradictionCase{"QkNorm", GQAPreprocessMode::Flash,
                                    false, false, GQAKvQuantizationType::None,
                                    GQAKvQuantizationType::None, true}),
    [](const testing::TestParamInfo<FastDecodeContradictionCase>& info) {
      return std::string(info.param.name);
    });

struct QkvPreprocessCase {
  const char* name;
  GQAPreprocessMode mode;
  bool is_first_prompt;
  bool do_rotary;
  bool is_packed_qkv;
  bool use_qk_norm;
  GQAKvQuantizationType k_quantization;
  GQAKvQuantizationType v_quantization;
  size_t expected_bytes;
};

class GroupQueryAttentionQkvPreprocessTest
    : public testing::TestWithParam<QkvPreprocessCase> {};

TEST_P(GroupQueryAttentionQkvPreprocessTest, MatchesHandCalculatedRuntimeFormula) {
  const auto& test_case = GetParam();
  auto problem = ValidProblem();
  problem.is_first_prompt = test_case.is_first_prompt;
  problem.do_rotary = test_case.do_rotary;
  problem.is_packed_qkv = test_case.is_packed_qkv;
  problem.use_qk_norm = test_case.use_qk_norm;
  problem.k_quantization = test_case.k_quantization;
  problem.v_quantization = test_case.v_quantization;

  GQAPreparationRecipe recipe;
  ASSERT_TRUE(BuildRecipe(problem, recipe, test_case.mode));
  EXPECT_EQ(recipe.qkv_preprocess_bytes, test_case.expected_bytes);
  if (test_case.expected_bytes == 0) {
    EXPECT_EQ(recipe.qkv_preprocess_offset_bytes, 0U);
    EXPECT_EQ(recipe.total_preparation_bytes, 24U);
  } else {
    EXPECT_EQ(recipe.qkv_preprocess_offset_bytes, 256U);
    EXPECT_EQ(recipe.total_preparation_bytes, 256U + test_case.expected_bytes);
  }
}

INSTANTIATE_TEST_SUITE_P(
    RuntimeModes,
    GroupQueryAttentionQkvPreprocessTest,
    testing::Values(
        // B=2, S=3, N=4, Nk=2, H=8, sizeof(T)=2.
        // Q=384 bytes and K=V=192 bytes.
        QkvPreprocessCase{"XqaNoMaterialization", GQAPreprocessMode::Xqa,
                          false, false, false, false,
                          GQAKvQuantizationType::PerTensor, GQAKvQuantizationType::None, 0},
        QkvPreprocessCase{"XqaRotaryQ", GQAPreprocessMode::Xqa,
                          false, true, false, false,
                          GQAKvQuantizationType::None, GQAKvQuantizationType::None, 384},
        QkvPreprocessCase{"XqaPackedQ", GQAPreprocessMode::Xqa,
                          false, false, true, false,
                          GQAKvQuantizationType::None, GQAKvQuantizationType::None, 384},
        QkvPreprocessCase{"XqaQkNormQ", GQAPreprocessMode::Xqa,
                          false, false, false, true,
                          GQAKvQuantizationType::None, GQAKvQuantizationType::None, 384},
        QkvPreprocessCase{"XqaPerChannelKScaleQ", GQAPreprocessMode::Xqa,
                          false, false, false, false,
                          GQAKvQuantizationType::PerChannel, GQAKvQuantizationType::None, 384},
        QkvPreprocessCase{"XqaCombinedReasonsStillOneQ", GQAPreprocessMode::Xqa,
                          false, true, true, true,
                          GQAKvQuantizationType::PerChannel, GQAKvQuantizationType::None, 384},
        QkvPreprocessCase{"FlashQuantizedPromptQkv", GQAPreprocessMode::Flash,
                          true, false, false, false,
                          GQAKvQuantizationType::PerTensor, GQAKvQuantizationType::PerTensor, 768},
        // Decode: sizeof(T) * (Q elements + 2*B*C*Nk*H) + 256
        //       = 2 * (192 + 2*160) + 256 = 1280.
        QkvPreprocessCase{"FlashQuantizedDecodePresentCapacity", GQAPreprocessMode::Flash,
                          false, false, false, false,
                          GQAKvQuantizationType::PerTensor, GQAKvQuantizationType::PerTensor, 1280},
        QkvPreprocessCase{"MemoryEfficientPackedQkv", GQAPreprocessMode::MemoryEfficient,
                          false, false, true, false,
                          GQAKvQuantizationType::None, GQAKvQuantizationType::None, 768},
        QkvPreprocessCase{"MemoryEfficientRotaryQk", GQAPreprocessMode::MemoryEfficient,
                          false, true, false, false,
                          GQAKvQuantizationType::None, GQAKvQuantizationType::None, 576},
        QkvPreprocessCase{"UnfusedRotaryQ", GQAPreprocessMode::Unfused,
                          false, true, false, false,
                          GQAKvQuantizationType::None, GQAKvQuantizationType::None, 384},
        QkvPreprocessCase{"UnfusedQkNormQ", GQAPreprocessMode::Unfused,
                          false, false, false, true,
                          GQAKvQuantizationType::None, GQAKvQuantizationType::None, 384}),
    [](const testing::TestParamInfo<QkvPreprocessCase>& info) {
      return std::string(info.param.name);
    });

TEST(GroupQueryAttentionWorkspaceTest, QuantizedFlashWindowedDecodeUsesEffectiveCapacity) {
  GQAWorkspaceProblem problem;
  problem.qkv_element_size = 2;
  problem.cache_element_size = 1;
  problem.batch_size = 1;
  problem.sequence_length = 2;
  problem.num_heads = 2;
  problem.kv_num_heads = 1;
  problem.head_size = 16;
  problem.present_kv_cache_capacity = 4;
  problem.kv_cache_bit_width = 8;
  problem.k_quantization = GQAKvQuantizationType::PerTensor;
  problem.v_quantization = GQAKvQuantizationType::PerTensor;
  problem.is_windowed_kv_cache = true;

  GQAPreparationRecipe recipe;
  ASSERT_TRUE(BuildRecipe(problem, recipe, GQAPreprocessMode::Flash));

  EXPECT_EQ(recipe.effective_kv_cache_capacity, 6);
  EXPECT_EQ(recipe.cache_row_bytes, 16U);
  EXPECT_EQ(recipe.staged_key_bytes, 96U);
  EXPECT_EQ(recipe.staged_value_bytes, 96U);
  // Q=1*2*2*16=64 elements; full K=1*6*1*16=96 elements.
  // QKV preprocess = 2*(64 + 2*96) + 256 = 768 bytes.
  EXPECT_EQ(recipe.qkv_preprocess_bytes, 768U);
  EXPECT_EQ(recipe.qkv_preprocess_offset_bytes, 768U);
  EXPECT_EQ(recipe.total_preparation_bytes, 1536U);
}

TEST(GroupQueryAttentionWorkspaceTest, WindowedInt4UsesExactPackedRowBytes) {
  auto problem = ValidProblem();
  problem.cache_element_size = 1;
  problem.sequence_length = 1;
  problem.head_size = 32;
  problem.kv_cache_bit_width = 4;
  problem.is_windowed_kv_cache = true;

  GQAPreparationRecipe recipe;
  ASSERT_TRUE(BuildRecipe(problem, recipe));

  // Helper validation requires H divisible by 32. Physical INT4 row size is H*4/8 = 16 bytes.
  EXPECT_EQ(recipe.cache_row_bytes, 16U);
  EXPECT_EQ(recipe.compaction_key_bytes, 320U);
  EXPECT_EQ(recipe.compaction_value_bytes, 320U);
  EXPECT_EQ(recipe.compaction_bytes, 640U);
}

TEST(GroupQueryAttentionWorkspaceTest, WindowedDenseByteRowsMustBeSixteenByteAligned) {
  auto problem = ValidProblem();
  problem.is_windowed_kv_cache = true;
  problem.cache_element_size = 1;
  problem.head_size = 8;

  EXPECT_EQ(
      GetGQAPreparationRecipe(problem, {}).status.error,
      GQAWorkspaceError::InvalidArgument);
}

TEST(GroupQueryAttentionWorkspaceTest, RegionOffsetsAreAlignedContainedAndNonOverlapping) {
  auto problem = ValidProblem();
  problem.is_windowed_kv_cache = true;
  problem.is_packed_qkv = true;
  GQAPreparationRecipe recipe;
  ASSERT_TRUE(BuildRecipe(problem, recipe));

  const std::array<std::pair<size_t, size_t>, 5> ranges{{
      {recipe.staged_key_offset_bytes, recipe.staged_key_bytes},
      {recipe.staged_value_offset_bytes, recipe.staged_value_bytes},
      {recipe.compaction_offset_bytes, recipe.compaction_bytes},
      {recipe.sequence_lengths_offset_bytes, recipe.sequence_lengths_bytes},
      {recipe.qkv_preprocess_offset_bytes, recipe.qkv_preprocess_bytes},
  }};

  size_t previous_end = 0;
  for (const auto& [offset, bytes] : ranges) {
    if (bytes == 0) {
      continue;
    }
    EXPECT_EQ(offset % kGQAWorkspaceAlignment, 0U);
    EXPECT_GE(offset, previous_end);
    EXPECT_LE(offset + bytes, recipe.total_preparation_bytes);
    previous_end = offset + bytes;
  }
  EXPECT_EQ(previous_end, recipe.total_preparation_bytes);
  EXPECT_TRUE(ValidateGQAPreparationRecipe(recipe).IsOK());

  recipe.qkv_preprocess_offset_bytes = recipe.sequence_lengths_offset_bytes;
  EXPECT_EQ(ValidateGQAPreparationRecipe(recipe).error,
            GQAWorkspaceError::InvalidArgument);
}

TEST(GroupQueryAttentionWorkspaceTest, InvalidGeometryAndSizeOverflowFail) {
  auto problem = ValidProblem();
  problem.num_heads = 3;
  EXPECT_EQ(
      GetGQAPreparationRecipe(problem, {}).status.error,
      GQAWorkspaceError::InvalidArgument);

  problem = ValidProblem();
  problem.is_windowed_kv_cache = true;
  problem.cache_element_size = 1;
  problem.kv_cache_bit_width = 4;
  problem.head_size = 16;
  EXPECT_EQ(
      GetGQAPreparationRecipe(problem, {}).status.error,
      GQAWorkspaceError::InvalidArgument);

  problem = ValidProblem();
  problem.is_windowed_kv_cache = true;
  problem.batch_size = std::numeric_limits<int32_t>::max();
  problem.num_heads = std::numeric_limits<int32_t>::max();
  problem.kv_num_heads = std::numeric_limits<int32_t>::max();
  problem.present_kv_cache_capacity = std::numeric_limits<int32_t>::max();
  problem.sequence_length = 1;
  EXPECT_EQ(
      GetGQAPreparationRecipe(problem, {}).status.error,
      GQAWorkspaceError::Overflow);
}

TEST(GroupQueryAttentionWorkspaceTest, StagedCapacityMustFitRuntimeInt32Abi) {
  auto problem = ValidProblem();
  problem.is_windowed_kv_cache = true;
  problem.present_kv_cache_capacity = std::numeric_limits<int32_t>::max();
  problem.sequence_length = 2;

  EXPECT_EQ(
      GetGQAPreparationRecipe(problem, {}).status.error,
      GQAWorkspaceError::InvalidArgument);
}

}  // namespace test
}  // namespace onnxruntime
