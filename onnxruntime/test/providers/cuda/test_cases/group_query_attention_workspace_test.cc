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
using contrib::cuda::GetGQACachePreparationSizes;
using contrib::cuda::GetGQACacheRowBytes;
using contrib::cuda::GetGQAEffectiveWorkspaceKvLength;
using contrib::cuda::GetGQAPreparationRecipe;
using contrib::cuda::GetGQAQkvPreprocessBytes;
using contrib::cuda::GetGQASequenceLengthsSize;
using contrib::cuda::GQAKvQuantizationType;
using contrib::cuda::GQAPreparationRecipe;
using contrib::cuda::GQAPreparationRoute;
using contrib::cuda::GQAPreprocessMode;
using contrib::cuda::GQAWorkspaceError;
using contrib::cuda::GQAWorkspaceProblem;
using contrib::cuda::IsSupportedGQAXqaGroupSize;
using contrib::cuda::IsSupportedGQAXqaHeadSize;
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

GQAWorkspaceProblem ValidXqaProblem(bool is_quantized = false) {
  auto problem = ValidProblem();
  problem.sequence_length = 1;
  problem.num_heads = 4;
  problem.kv_num_heads = 1;
  problem.head_size = 64;
  if (is_quantized) {
    problem.cache_element_size = 1;
    problem.kv_cache_bit_width = 8;
  }
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

// Retain the former runtime calculation as an independent compatibility oracle.
size_t LegacyQkvPreprocessBytes(
    const GQAWorkspaceProblem& problem,
    const GQAPreparationRoute& route,
    int64_t effective_capacity) {
  if (route.use_flash_attention_fast_decode) return 0;
  const size_t q = static_cast<size_t>(problem.batch_size) * problem.sequence_length *
                   problem.num_heads * problem.head_size;
  const size_t k = static_cast<size_t>(problem.batch_size) * problem.sequence_length *
                   problem.kv_num_heads * problem.head_size;
  const size_t element_size = problem.qkv_element_size;
  size_t bytes = 0;
  if (route.preprocess_mode == GQAPreprocessMode::Xqa) {
    return problem.do_rotary || problem.is_packed_qkv || problem.use_qk_norm ||
                   problem.k_quantization == GQAKvQuantizationType::PerChannel
               ? element_size * q
               : 0;
  }
  if (route.preprocess_mode == GQAPreprocessMode::Flash) {
    const bool quantized = problem.k_quantization != GQAKvQuantizationType::None ||
                           problem.v_quantization != GQAKvQuantizationType::None;
    if (quantized) {
      const size_t full_k = static_cast<size_t>(problem.batch_size) * effective_capacity *
                            problem.kv_num_heads * problem.head_size;
      bytes = problem.is_first_prompt ? element_size * (q + 2 * k)
                                      : element_size * (q + 2 * full_k) + 256;
    } else if (problem.do_rotary || problem.is_packed_qkv) {
      bytes = element_size * q;
    }
  } else if (route.preprocess_mode == GQAPreprocessMode::MemoryEfficient) {
    if (problem.is_packed_qkv) {
      bytes = element_size * (q + 2 * k);
    } else if (problem.do_rotary) {
      bytes = element_size * (q + k);
    }
  }
  if (bytes == 0 && (problem.do_rotary || problem.is_packed_qkv || problem.use_qk_norm)) {
    bytes = element_size * q;
  }
  return bytes;
}

}  // namespace

TEST(GroupQueryAttentionWorkspaceTest, EffectiveKvLengthMatchesRuntimeCacheExtent) {
  EXPECT_EQ(GetGQAEffectiveWorkspaceKvLength(100, 8, false), 100);
  EXPECT_EQ(GetGQAEffectiveWorkspaceKvLength(5, 8, true), 5);
  EXPECT_EQ(GetGQAEffectiveWorkspaceKvLength(8, 8, true), 8);
  EXPECT_EQ(GetGQAEffectiveWorkspaceKvLength(100, 8, true), 8);

  // A multi-token windowed update changes the runtime cache extent from C to C + S.
  // The workspace must cover all staged rows, not only the final window capacity.
  constexpr int cache_capacity = 8;
  constexpr int sequence_length = 3;
  EXPECT_EQ(GetGQAEffectiveWorkspaceKvLength(
                100, cache_capacity + sequence_length, true),
            11);
}

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

TEST(GroupQueryAttentionPreparationSizingTest, CacheSizesMatchRuntimeAndRecipe) {
  for (size_t element_size : {size_t{1}, size_t{2}}) {
    for (int64_t bit_width : {int64_t{0}, int64_t{4}, int64_t{8}}) {
      if (bit_width != 0 && element_size != 1) continue;
      for (int64_t batch : {int64_t{1}, int64_t{2}}) {
        for (int64_t sequence : {int64_t{1}, int64_t{3}}) {
          for (int64_t capacity : {int64_t{1}, int64_t{5}, int64_t{256}}) {
            for (bool windowed : {false, true}) {
              for (bool preserve_past : {false, true}) {
                if (windowed && preserve_past) continue;
                SCOPED_TRACE(testing::Message()
                             << element_size << "/" << bit_width << " B=" << batch
                             << " S=" << sequence << " C=" << capacity
                             << " windowed=" << windowed << " preserve=" << preserve_past);
                auto problem = ValidProblem();
                problem.cache_element_size = element_size;
                problem.kv_cache_bit_width = bit_width;
                problem.head_size = 64;
                problem.batch_size = batch;
                problem.sequence_length = sequence;
                problem.present_kv_cache_capacity = capacity;
                problem.is_windowed_kv_cache = windowed;
                problem.requires_separate_past_buffer = preserve_past;
                problem.past_kv_cache_capacity = preserve_past ? capacity : 0;
                const size_t row_bytes = bit_width == 0 ? 64 * element_size : 64 * bit_width / 8;
                size_t shared_row_bytes = 0;
                ASSERT_TRUE(GetGQACacheRowBytes(problem, shared_row_bytes).IsOK());
                EXPECT_EQ(shared_row_bytes, row_bytes);
                const auto cache = GetGQACachePreparationSizes(problem, shared_row_bytes);
                ASSERT_TRUE(cache.status.IsOK()) << cache.status.message;
                const size_t cache_bytes = static_cast<size_t>(batch) * problem.kv_num_heads *
                                           capacity * row_bytes;
                EXPECT_EQ(cache.sizes.separate_past_bytes, preserve_past ? cache_bytes : 0);
                const int64_t staged_capacity = windowed && sequence > 1 ? capacity + sequence : capacity;
                EXPECT_EQ(cache.sizes.effective_kv_cache_capacity, staged_capacity);
                EXPECT_EQ(cache.sizes.staged_cache_bytes,
                          windowed && sequence > 1
                              ? static_cast<size_t>(batch) * problem.kv_num_heads * staged_capacity * row_bytes
                              : 0);
                EXPECT_EQ(cache.sizes.compaction_cache_bytes,
                          windowed && sequence == 1 ? cache_bytes : 0);
                EXPECT_EQ(cache.sizes.compaction_bytes,
                          windowed && sequence == 1 ? 2 * cache_bytes : 0);
                const auto recipe = GetGQAPreparationRecipe(problem, {});
                ASSERT_TRUE(recipe.status.IsOK()) << recipe.status.message;
                EXPECT_EQ(recipe.recipe.cache_row_bytes, cache.sizes.cache_row_bytes);
                EXPECT_EQ(recipe.recipe.effective_kv_cache_capacity, staged_capacity);
                EXPECT_EQ(recipe.recipe.separate_past_bytes, cache.sizes.separate_past_bytes);
                EXPECT_EQ(recipe.recipe.staged_key_bytes, cache.sizes.staged_cache_bytes);
                EXPECT_EQ(recipe.recipe.staged_value_bytes, cache.sizes.staged_cache_bytes);
                EXPECT_EQ(recipe.recipe.compaction_key_bytes, cache.sizes.compaction_cache_bytes);
                EXPECT_EQ(recipe.recipe.compaction_value_bytes, cache.sizes.compaction_cache_bytes);
                EXPECT_EQ(recipe.recipe.compaction_bytes, cache.sizes.compaction_bytes);
              }
            }
          }
        }
      }
    }
  }
}

TEST(GroupQueryAttentionPreparationSizingTest, PhysicalRowsAreNotPackedTwice) {
  auto problem = ValidProblem();
  problem.head_size = 64;
  problem.cache_element_size = 1;
  problem.kv_cache_bit_width = 4;
  problem.is_windowed_kv_cache = true;
  auto physical_row = problem;
  physical_row.head_size = 32;
  physical_row.kv_cache_bit_width = 0;
  size_t row_bytes = 0;
  ASSERT_TRUE(GetGQACacheRowBytes(physical_row, row_bytes).IsOK());
  EXPECT_EQ(row_bytes, 32U);
  const auto staged = GetGQACachePreparationSizes(problem, row_bytes);
  ASSERT_TRUE(staged.status.IsOK()) << staged.status.message;
  EXPECT_EQ(staged.sizes.staged_cache_bytes, 2U * 2U * 8U * 32U);

  problem.is_windowed_kv_cache = false;
  problem.requires_separate_past_buffer = true;
  problem.past_kv_cache_capacity = 5;
  const auto preserved = GetGQACachePreparationSizes(problem, row_bytes);
  ASSERT_TRUE(preserved.status.IsOK()) << preserved.status.message;
  EXPECT_EQ(preserved.sizes.separate_past_bytes, 2U * 2U * 5U * 32U);
}

TEST(GroupQueryAttentionPreparationSizingTest, SequenceSizesMatchRuntimeFormula) {
  for (int64_t batch : {int64_t{0}, int64_t{1}, int64_t{2}}) {
    for (int64_t sequence : {int64_t{1}, int64_t{3}}) {
      for (bool windowed : {false, true}) {
        for (bool fast_decode : {false, true}) {
          auto problem = ValidProblem();
          problem.batch_size = batch;
          problem.sequence_length = sequence;
          problem.is_windowed_kv_cache = windowed;
          size_t count = 0;
          size_t bytes = 0;
          const auto status = GetGQASequenceLengthsSize(problem, fast_decode, count, bytes);
          ASSERT_TRUE(status.IsOK()) << status.message;
          const size_t expected_count = fast_decode && sequence == 1 ? 0 : (windowed ? 6 : 3);
          EXPECT_EQ(count, expected_count);
          EXPECT_EQ(bytes, expected_count * static_cast<size_t>(batch) * sizeof(int32_t));
        }
      }
    }
  }
}

TEST(GroupQueryAttentionPreparationSizingTest, QkvSizesMatchLegacyFeatureMatrix) {
  for (int64_t batch : {int64_t{1}, int64_t{2}}) {
    for (int64_t sequence : {int64_t{1}, int64_t{3}}) {
      for (int64_t head : {int64_t{8}, int64_t{64}, int64_t{128}}) {
        for (auto mode : {GQAPreprocessMode::Xqa, GQAPreprocessMode::Flash,
                          GQAPreprocessMode::MemoryEfficient, GQAPreprocessMode::Unfused}) {
          for (bool first : {false, true}) {
            for (int features = 0; features < 8; ++features) {
              for (auto k_quant : {GQAKvQuantizationType::None, GQAKvQuantizationType::PerTensor,
                                   GQAKvQuantizationType::PerChannel}) {
                for (auto v_quant : {GQAKvQuantizationType::None, GQAKvQuantizationType::PerTensor,
                                     GQAKvQuantizationType::PerChannel}) {
                  for (bool windowed : {false, true}) {
                    for (bool fast_decode : {false, true}) {
                      if (fast_decode && mode != GQAPreprocessMode::Flash) continue;
                      SCOPED_TRACE(testing::Message()
                                   << "B=" << batch << " S=" << sequence << " H=" << head
                                   << " mode=" << static_cast<int>(mode) << " features=" << features
                                   << " K=" << static_cast<int>(k_quant) << " V=" << static_cast<int>(v_quant)
                                   << " first=" << first << " windowed=" << windowed << " fast=" << fast_decode);
                      auto problem = ValidProblem();
                      problem.batch_size = batch;
                      problem.sequence_length = sequence;
                      problem.head_size = head;
                      problem.is_first_prompt = first;
                      problem.do_rotary = (features & 1) != 0;
                      problem.is_packed_qkv = (features & 2) != 0;
                      problem.use_qk_norm = (features & 4) != 0;
                      problem.k_quantization = k_quant;
                      problem.v_quantization = v_quant;
                      problem.is_windowed_kv_cache = windowed;
                      const int64_t effective_capacity =
                          problem.present_kv_cache_capacity + (windowed && sequence > 1 ? sequence : 0);
                      const GQAPreparationRoute route{mode, fast_decode};
                      size_t bytes = 0;
                      const auto status = GetGQAQkvPreprocessBytes(problem, route, effective_capacity, bytes);
                      ASSERT_TRUE(status.IsOK()) << status.message;
                      EXPECT_EQ(bytes, LegacyQkvPreprocessBytes(problem, route, effective_capacity));
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
  }
}

TEST(GroupQueryAttentionPreparationSizingTest, SizingDoesNotApplyFullRecipeAdmissionRules) {
  auto problem = ValidProblem();
  problem.head_size = 6;
  problem.is_packed_qkv = true;
  size_t bytes = 0;
  ASSERT_TRUE(GetGQAQkvPreprocessBytes(problem, {}, 0, bytes).IsOK());
  EXPECT_EQ(bytes, 2U * 3U * 4U * 6U * 2U);
  EXPECT_EQ(GetGQAPreparationRecipe(problem, {}).status.error, GQAWorkspaceError::InvalidArgument);

  problem.requires_separate_past_buffer = true;
  problem.past_kv_cache_capacity = 0;
  size_t row_bytes = 0;
  ASSERT_TRUE(GetGQACacheRowBytes(problem, row_bytes).IsOK());
  const auto empty_past = GetGQACachePreparationSizes(problem, row_bytes);
  ASSERT_TRUE(empty_past.status.IsOK()) << empty_past.status.message;
  EXPECT_EQ(empty_past.sizes.separate_past_bytes, 0U);
}

TEST(GroupQueryAttentionPreparationSizingTest, RejectsInvalidDimensionsAndOverflow) {
  auto problem = ValidProblem();
  size_t bytes = 17;
  problem.batch_size = -1;
  EXPECT_EQ(GetGQACachePreparationSizes(problem, 16).status.error, GQAWorkspaceError::InvalidArgument);
  EXPECT_EQ(GetGQAQkvPreprocessBytes(problem, {}, 5, bytes).error, GQAWorkspaceError::InvalidArgument);
  EXPECT_EQ(bytes, 17U);
  size_t vectors = 19;
  EXPECT_EQ(GetGQASequenceLengthsSize(problem, false, vectors, bytes).error,
            GQAWorkspaceError::InvalidArgument);
  EXPECT_EQ(vectors, 19U);
  EXPECT_EQ(bytes, 17U);

  problem = ValidProblem();
  problem.kv_cache_bit_width = 3;
  EXPECT_EQ(GetGQACacheRowBytes(problem, bytes).error, GQAWorkspaceError::InvalidArgument);
  EXPECT_EQ(bytes, 17U);
  problem = ValidProblem();
  problem.batch_size = std::numeric_limits<int32_t>::max();
  problem.sequence_length = std::numeric_limits<int32_t>::max();
  problem.num_heads = std::numeric_limits<int32_t>::max();
  problem.kv_num_heads = std::numeric_limits<int32_t>::max();
  problem.head_size = std::numeric_limits<int32_t>::max();
  problem.present_kv_cache_capacity = std::numeric_limits<int32_t>::max();
  problem.is_windowed_kv_cache = true;
  problem.sequence_length = 1;
  EXPECT_EQ(GetGQACachePreparationSizes(problem, 16).status.error, GQAWorkspaceError::Overflow);
  problem.sequence_length = std::numeric_limits<int32_t>::max();
  EXPECT_EQ(GetGQAQkvPreprocessBytes(problem, {}, 5, bytes).error, GQAWorkspaceError::Overflow);
  EXPECT_EQ(bytes, 17U);

  problem = ValidProblem();
  problem.is_windowed_kv_cache = true;
  problem.sequence_length = 2;
  problem.present_kv_cache_capacity = std::numeric_limits<int32_t>::max();
  EXPECT_EQ(GetGQACachePreparationSizes(problem, 16).status.error, GQAWorkspaceError::InvalidArgument);
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

  // Unfused packed preprocess: B*S*N*H*sizeof(T) = 2*3*4*8*2 = 384.
  EXPECT_EQ(recipe.qkv_preprocess_offset_bytes, 256U);
  EXPECT_EQ(recipe.qkv_preprocess_bytes, 384U);
  EXPECT_EQ(recipe.total_preparation_bytes, 640U);
  EXPECT_FALSE(recipe.uses_staging);
  EXPECT_FALSE(recipe.uses_compaction);
  EXPECT_EQ(recipe.effective_kv_cache_capacity, 5);
}

TEST(GroupQueryAttentionWorkspaceTest, AsymmetricAliasPreservesOnePastCacheTensor) {
  auto problem = ValidProblem();
  problem.past_kv_cache_capacity = 5;
  problem.requires_separate_past_buffer = true;

  GQAPreparationRecipe recipe;
  ASSERT_TRUE(BuildRecipe(problem, recipe));

  // B*Nk*P*row = 2*2*5*16 = 320.
  EXPECT_TRUE(recipe.uses_separate_past_buffer);
  EXPECT_EQ(recipe.separate_past_offset_bytes, 0U);
  EXPECT_EQ(recipe.separate_past_bytes, 320U);
  EXPECT_EQ(recipe.sequence_lengths_offset_bytes, 512U);
  EXPECT_TRUE(ValidateGQAPreparationRecipe(recipe).IsOK());
}

TEST(GroupQueryAttentionWorkspaceTest, RejectsInvalidSeparatePastBufferFacts) {
  auto problem = ValidProblem();
  problem.requires_separate_past_buffer = true;
  EXPECT_EQ(GetGQAPreparationRecipe(problem, {}).status.error,
            GQAWorkspaceError::InvalidArgument);

  problem.past_kv_cache_capacity = 5;
  problem.is_windowed_kv_cache = true;
  EXPECT_EQ(GetGQAPreparationRecipe(problem, {}).status.error,
            GQAWorkspaceError::InvalidArgument);
}

TEST(GroupQueryAttentionWorkspaceTest, RejectsSeparatePastBufferWithWindowedRecipe) {
  auto problem = ValidProblem();
  problem.sequence_length = 1;
  problem.is_windowed_kv_cache = true;
  auto result = GetGQAPreparationRecipe(problem, {});
  ASSERT_TRUE(result.status.IsOK()) << result.status.message;

  result.recipe.uses_separate_past_buffer = true;
  result.recipe.separate_past_bytes = result.recipe.compaction_bytes;
  EXPECT_EQ(ValidateGQAPreparationRecipe(result.recipe).error,
            GQAWorkspaceError::InvalidArgument);
}

TEST(GroupQueryAttentionWorkspaceTest, RejectsSeparatePastBufferForSharedOnlyRoutes) {
  auto problem = ValidXqaProblem();
  problem.past_kv_cache_capacity = 5;
  problem.requires_separate_past_buffer = true;
  EXPECT_EQ(
      GetGQAPreparationRecipe(
          problem, GQAPreparationRoute{GQAPreprocessMode::Xqa, false})
          .status.error,
      GQAWorkspaceError::InvalidArgument);

  EXPECT_EQ(
      GetGQAPreparationRecipe(
          problem, GQAPreparationRoute{GQAPreprocessMode::Flash, true})
          .status.error,
      GQAWorkspaceError::InvalidArgument);
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
  const bool is_quantized_xqa =
      test_case.mode == GQAPreprocessMode::Xqa &&
      test_case.k_quantization != GQAKvQuantizationType::None;
  auto problem = test_case.mode == GQAPreprocessMode::Xqa
                     ? ValidXqaProblem(is_quantized_xqa)
                     : ValidProblem();
  problem.is_first_prompt = test_case.is_first_prompt;
  problem.do_rotary = test_case.do_rotary;
  problem.is_packed_qkv = test_case.is_packed_qkv;
  problem.use_qk_norm = test_case.use_qk_norm;
  problem.k_quantization = test_case.k_quantization;
  problem.v_quantization = test_case.v_quantization;

  GQAPreparationRecipe recipe;
  ASSERT_TRUE(BuildRecipe(problem, recipe, test_case.mode));
  size_t shared_bytes = 0;
  ASSERT_TRUE(GetGQAQkvPreprocessBytes(
                  problem, {test_case.mode, false}, recipe.effective_kv_cache_capacity, shared_bytes)
                  .IsOK());
  EXPECT_EQ(shared_bytes, test_case.expected_bytes);
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
        // XQA cases use B=2, S=1, N=4, Nk=1, H=64: Q=1024 bytes.
        // Other modes use B=2, S=3, N=4, Nk=2, H=8:
        // Q=384 bytes and K=V=192 bytes.
        QkvPreprocessCase{"XqaNoMaterialization", GQAPreprocessMode::Xqa,
                          false, false, false, false,
                          GQAKvQuantizationType::PerTensor, GQAKvQuantizationType::PerTensor, 0},
        QkvPreprocessCase{"XqaRotaryQ", GQAPreprocessMode::Xqa,
                          false, true, false, false,
                          GQAKvQuantizationType::None, GQAKvQuantizationType::None, 1024},
        QkvPreprocessCase{"XqaPackedQ", GQAPreprocessMode::Xqa,
                          false, false, true, false,
                          GQAKvQuantizationType::None, GQAKvQuantizationType::None, 1024},
        QkvPreprocessCase{"XqaQkNormQ", GQAPreprocessMode::Xqa,
                          false, false, false, true,
                          GQAKvQuantizationType::None, GQAKvQuantizationType::None, 1024},
        QkvPreprocessCase{"XqaPerChannelKScaleQ", GQAPreprocessMode::Xqa,
                          false, false, false, false,
                          GQAKvQuantizationType::PerChannel, GQAKvQuantizationType::PerTensor, 1024},
        QkvPreprocessCase{"XqaCombinedReasonsStillOneQ", GQAPreprocessMode::Xqa,
                          false, true, true, true,
                          GQAKvQuantizationType::None, GQAKvQuantizationType::None, 1024},
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

struct XqaContradictionCase {
  const char* name;
  int64_t sequence_length;
  bool is_first_prompt;
  GQAKvQuantizationType k_quantization;
  GQAKvQuantizationType v_quantization;
  int64_t kv_cache_bit_width;
  bool use_qk_norm;
};

class GroupQueryAttentionXqaValidationTest
    : public testing::TestWithParam<XqaContradictionCase> {};

TEST_P(GroupQueryAttentionXqaValidationTest, RejectsImpossibleRuntimeRouteFacts) {
  const auto& test_case = GetParam();
  auto problem = ValidXqaProblem();
  problem.sequence_length = test_case.sequence_length;
  problem.is_first_prompt = test_case.is_first_prompt;
  problem.k_quantization = test_case.k_quantization;
  problem.v_quantization = test_case.v_quantization;
  problem.kv_cache_bit_width = test_case.kv_cache_bit_width;
  problem.cache_element_size = test_case.kv_cache_bit_width == 0 ? 2 : 1;
  problem.use_qk_norm = test_case.use_qk_norm;

  EXPECT_EQ(
      GetGQAPreparationRecipe(
          problem, GQAPreparationRoute{GQAPreprocessMode::Xqa, false})
          .status.error,
      GQAWorkspaceError::InvalidArgument);
}

INSTANTIATE_TEST_SUITE_P(
    ImpossibleRuntimeRoutes,
    GroupQueryAttentionXqaValidationTest,
    testing::Values(
        XqaContradictionCase{"MultiToken", 2, false,
                             GQAKvQuantizationType::None, GQAKvQuantizationType::None, 0, false},
        XqaContradictionCase{"FirstPrompt", 1, true,
                             GQAKvQuantizationType::None, GQAKvQuantizationType::None, 0, false},
        XqaContradictionCase{"QuantizedKOnly", 1, false,
                             GQAKvQuantizationType::PerTensor, GQAKvQuantizationType::None, 8, false},
        XqaContradictionCase{"QuantizedVOnly", 1, false,
                             GQAKvQuantizationType::None, GQAKvQuantizationType::PerChannel, 8, false},
        XqaContradictionCase{"Int4Cache", 1, false,
                             GQAKvQuantizationType::PerTensor, GQAKvQuantizationType::PerTensor, 4, false},
        XqaContradictionCase{"PackedUnquantizedCache", 1, false,
                             GQAKvQuantizationType::None, GQAKvQuantizationType::None, 8, false},
        XqaContradictionCase{"QuantizedQkNorm", 1, false,
                             GQAKvQuantizationType::PerTensor, GQAKvQuantizationType::PerTensor, 8, true}),
    [](const testing::TestParamInfo<XqaContradictionCase>& info) {
      return std::string(info.param.name);
    });

TEST(GroupQueryAttentionWorkspaceTest, RejectsUnsupportedXqaGeometry) {
  const GQAPreparationRoute xqa_route{GQAPreprocessMode::Xqa, false};

  auto problem = ValidXqaProblem();
  problem.head_size = 8;
  EXPECT_EQ(GetGQAPreparationRecipe(problem, xqa_route).status.error,
            GQAWorkspaceError::InvalidArgument);

  problem = ValidXqaProblem();
  problem.cache_element_size = 1;
  EXPECT_EQ(GetGQAPreparationRecipe(problem, xqa_route).status.error,
            GQAWorkspaceError::InvalidArgument);

  problem = ValidXqaProblem();
  problem.num_heads = 6;
  problem.kv_num_heads = 2;
  EXPECT_EQ(GetGQAPreparationRecipe(problem, xqa_route).status.error,
            GQAWorkspaceError::InvalidArgument);

  problem = ValidXqaProblem(true);
  problem.num_heads = 4;
  problem.kv_num_heads = 2;
  problem.k_quantization = GQAKvQuantizationType::PerTensor;
  problem.v_quantization = GQAKvQuantizationType::PerTensor;
  EXPECT_EQ(GetGQAPreparationRecipe(problem, xqa_route).status.error,
            GQAWorkspaceError::InvalidArgument);
}

TEST(GroupQueryAttentionWorkspaceTest, XqaGeometryPredicatesCoverSupportedDomain) {
  for (int64_t head_size : {64, 128, 256}) {
    EXPECT_TRUE(IsSupportedGQAXqaHeadSize(head_size));
  }
  for (int64_t head_size : {8, 32, 96, 512}) {
    EXPECT_FALSE(IsSupportedGQAXqaHeadSize(head_size));
  }

  for (int64_t group_size : {1, 2, 4, 5, 8, 16, 32}) {
    EXPECT_TRUE(IsSupportedGQAXqaGroupSize(group_size, false));
  }
  for (int64_t group_size : {4, 8, 16, 32}) {
    EXPECT_TRUE(IsSupportedGQAXqaGroupSize(group_size, true));
  }
  for (int64_t group_size : {3, 6, 64}) {
    EXPECT_FALSE(IsSupportedGQAXqaGroupSize(group_size, false));
  }
  for (int64_t group_size : {1, 2, 3, 5, 64}) {
    EXPECT_FALSE(IsSupportedGQAXqaGroupSize(group_size, true));
  }
}

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
