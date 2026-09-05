// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"

#include <algorithm>
#include <cstddef>
#include <limits>
#include <tuple>

#include "contrib_ops/cuda/bert/group_query_attention_workspace.h"
#include "contrib_ops/cuda/bert/xqa/xqa_loader.h"

#if defined(USE_FLASH_ATTENTION)
namespace onnxruntime {
namespace flash {
std::tuple<size_t, size_t, size_t> get_num_splits_and_buffer_sizes(
    size_t batch_size, size_t seqlen_q, size_t seqlen_k, size_t num_heads,
    size_t head_size, size_t num_SMs);
size_t get_softmax_lse_accum_size(
    size_t num_splits, size_t batch_size, size_t num_heads, size_t seqlen_q);
size_t get_out_accum_size(
    size_t num_splits, size_t batch_size, size_t num_heads,
    size_t seqlen_q, size_t head_size_rounded);
}  // namespace flash
}  // namespace onnxruntime
#endif

namespace onnxruntime {
namespace test {

using contrib::cuda::GetGQAFlashWorkspaceRecipe;
using contrib::cuda::GetGQAXqaWorkspaceRecipe;
using contrib::cuda::GQAFlashConfig;
using contrib::cuda::GQAKvQuantizationType;
using contrib::cuda::GQAWorkspaceError;
using contrib::cuda::GQAWorkspaceProblem;
using contrib::cuda::GQAXqaConfig;
using contrib::cuda::GQAXqaHeadSinkStorage;
using contrib::cuda::GQAXqaKvType;
using contrib::cuda::ValidateGQAFlashWorkspaceRecipe;
using contrib::cuda::ValidateGQAXqaWorkspaceRecipe;
using contrib::cuda::XqaQuantType;

namespace {

GQAWorkspaceProblem XqaProblem() {
  GQAWorkspaceProblem problem;
  problem.qkv_element_size = 2;
  problem.cache_element_size = 2;
  problem.batch_size = 1;
  problem.sequence_length = 1;
  problem.num_heads = 8;
  problem.kv_num_heads = 2;
  problem.head_size = 64;
  problem.present_kv_cache_capacity = 512;
  return problem;
}

GQAXqaConfig XqaConfig() {
  GQAXqaConfig config;
  config.device_major = 8;
  config.device_minor = 0;
  config.multi_processor_count = 80;
  return config;
}

GQAWorkspaceProblem FlashProblem() {
  auto problem = XqaProblem();
  problem.batch_size = 1;
  problem.sequence_length = 1;
  problem.num_heads = 2;
  problem.kv_num_heads = 2;
  return problem;
}

}  // namespace

TEST(GroupQueryAttentionXqaWorkspaceTest, HandCalculatedMultiBlockLayoutAndExtras) {
  auto problem = XqaProblem();
  problem.do_rotary = true;
  auto config = XqaConfig();
  config.head_sink_storage = GQAXqaHeadSinkStorage::DynamicConversion;

  const auto result = GetGQAXqaWorkspaceRecipe(problem, config);
  ASSERT_TRUE(result.status.IsOK()) << result.status.message;
  const auto& recipe = result.recipe;

  // nbSeq=1*2=2, min(max(1,80/2),ceil(512/256))=2, nbSubSeq=4.
  EXPECT_EQ(recipe.sequence_count, 2U);
  EXPECT_EQ(recipe.subsequences_per_sequence, 2U);
  EXPECT_EQ(recipe.subsequence_count, 4U);
  EXPECT_EQ(recipe.m_tile_size, 8U);
  EXPECT_EQ(recipe.semaphore_bytes, 8U);
  EXPECT_EQ(recipe.semaphore_aligned_bytes, 128U);
  EXPECT_EQ(recipe.row_max_offset_bytes, 128U);
  EXPECT_EQ(recipe.row_max_bytes, 512U);
  EXPECT_EQ(recipe.row_sum_offset_bytes, 640U);
  EXPECT_EQ(recipe.row_sum_bytes, 512U);
  EXPECT_EQ(recipe.output_accumulator_offset_bytes, 1152U);
  EXPECT_EQ(recipe.output_accumulator_bytes, 4096U);
  EXPECT_EQ(recipe.internal_scratch_bytes, 5248U);

  // Runtime retains these in the XQA allocation even though the separate
  // GQABufferRequirements Q allocation is also requested for RoPE.
  EXPECT_EQ(recipe.rotary_q_offset_bytes, 5248U);
  EXPECT_EQ(recipe.rotary_q_bytes, 1024U);
  EXPECT_EQ(recipe.rotary_k_offset_bytes, 6272U);
  EXPECT_EQ(recipe.rotary_k_bytes, 256U);
  EXPECT_EQ(recipe.dynamic_head_sink_offset_bytes, 6528U);
  EXPECT_EQ(recipe.dynamic_head_sink_bytes, 256U);
  EXPECT_EQ(recipe.total_backend_bytes, 6784U);
  EXPECT_TRUE(ValidateGQAXqaWorkspaceRecipe(recipe).IsOK());
}

TEST(GroupQueryAttentionXqaWorkspaceTest, SingleBlockAndPersistentHeadSinkHaveNoExtraSinkScratch) {
  auto problem = XqaProblem();
  problem.present_kv_cache_capacity = 128;
  auto config = XqaConfig();
  config.multi_processor_count = 2;
  config.head_sink_storage = GQAXqaHeadSinkStorage::PrepackedFp32;

  const auto result = GetGQAXqaWorkspaceRecipe(problem, config);
  ASSERT_TRUE(result.status.IsOK()) << result.status.message;
  EXPECT_EQ(result.recipe.subsequences_per_sequence, 1U);
  EXPECT_EQ(result.recipe.subsequence_count, 2U);
  EXPECT_EQ(result.recipe.dynamic_head_sink_bytes, 0U);
  EXPECT_EQ(result.recipe.rotary_q_bytes, 0U);
  EXPECT_EQ(result.recipe.rotary_k_bytes, 0U);
  // Persistent xqa_head_sink_ is intentionally outside the transient recipe.
  EXPECT_EQ(result.recipe.total_backend_bytes, result.recipe.internal_scratch_bytes);
}

TEST(GroupQueryAttentionXqaWorkspaceTest, MatchesLegacyHelperAcrossSupportedFiniteDomain) {
  const int head_sizes[] = {64, 128, 256};
  const int groups[] = {1, 2, 4, 5, 8, 16, 32};
  const int capacities[] = {1, 255, 256, 257, 4096};
  const int sm_counts[] = {1, 16, 80, 132};
  const GQAXqaKvType kv_types[] = {
      GQAXqaKvType::None, GQAXqaKvType::Int8, GQAXqaKvType::Fp8};

  for (int head_size : head_sizes) {
    for (int group : groups) {
      for (int capacity : capacities) {
        for (int sm_count : sm_counts) {
          for (GQAXqaKvType kv_type : kv_types) {
            if (kv_type != GQAXqaKvType::None && (group == 1 || group == 2 || group == 5)) {
              continue;
            }
            auto problem = XqaProblem();
            problem.num_heads = 2 * group;
            problem.head_size = head_size;
            problem.present_kv_cache_capacity = capacity;
            auto config = XqaConfig();
            config.device_major = kv_type == GQAXqaKvType::Fp8 ? 9 : 8;
            config.multi_processor_count = sm_count;
            config.kv_type = kv_type;
            config.is_bf16 = (capacity & 1) != 0;
            if (kv_type != GQAXqaKvType::None) {
              problem.cache_element_size = 1;
              problem.kv_cache_bit_width = 8;
              problem.k_quantization =
                  (capacity & 1) == 0
                      ? GQAKvQuantizationType::PerTensor
                      : GQAKvQuantizationType::PerChannel;
              problem.v_quantization =
                  head_size == 128
                      ? GQAKvQuantizationType::PerChannel
                      : GQAKvQuantizationType::PerTensor;
            }

            const auto actual = GetGQAXqaWorkspaceRecipe(problem, config);
            ASSERT_TRUE(actual.status.IsOK()) << actual.status.message;

            cudaDeviceProp device{};
            device.major = static_cast<int>(config.device_major);
            device.minor = static_cast<int>(config.device_minor);
            device.multiProcessorCount = sm_count;
            const XqaQuantType legacy_type =
                kv_type == GQAXqaKvType::None
                    ? XqaQuantType::kNone
                    : (kv_type == GQAXqaKvType::Int8
                           ? XqaQuantType::kInt8
                           : XqaQuantType::kFp8);
            const size_t expected = contrib::cuda::GetXQAScratchSize(
                device, static_cast<int>(problem.batch_size),
                static_cast<int>(problem.num_heads),
                static_cast<int>(problem.kv_num_heads), head_size, capacity,
                legacy_type, config.is_bf16);
            EXPECT_EQ(actual.recipe.internal_scratch_bytes, expected)
                << "H=" << head_size << " group=" << group
                << " C=" << capacity << " SMs=" << sm_count;
          }
        }
      }
    }
  }
}

TEST(GroupQueryAttentionXqaWorkspaceTest, RejectsUnsupportedAndOverflowingConfigurations) {
  auto problem = XqaProblem();
  auto config = XqaConfig();

  problem.head_size = 96;
  EXPECT_EQ(GetGQAXqaWorkspaceRecipe(problem, config).status.error,
            GQAWorkspaceError::Unavailable);

  problem = XqaProblem();
  problem.num_heads = 6;
  EXPECT_EQ(GetGQAXqaWorkspaceRecipe(problem, config).status.error,
            GQAWorkspaceError::Unavailable);

  problem = XqaProblem();
  problem.cache_element_size = 1;
  problem.kv_cache_bit_width = 8;
  problem.k_quantization = GQAKvQuantizationType::PerTensor;
  problem.v_quantization = GQAKvQuantizationType::PerTensor;
  config.kv_type = GQAXqaKvType::Fp8;
  EXPECT_EQ(GetGQAXqaWorkspaceRecipe(problem, config).status.error,
            GQAWorkspaceError::Unavailable);

  problem.batch_size = std::numeric_limits<int32_t>::max();
  problem.kv_num_heads = std::numeric_limits<int32_t>::max();
  problem.num_heads = problem.kv_num_heads;
  config.kv_type = GQAXqaKvType::None;
  EXPECT_EQ(GetGQAXqaWorkspaceRecipe(problem, config).status.error,
            GQAWorkspaceError::InvalidArgument);
}

TEST(GroupQueryAttentionXqaWorkspaceTest, RejectsFactsOutsideSelectedRuntimeDomain) {
  auto problem = XqaProblem();
  auto config = XqaConfig();

  problem.is_first_prompt = true;
  EXPECT_EQ(GetGQAXqaWorkspaceRecipe(problem, config).status.error,
            GQAWorkspaceError::InvalidArgument);

  problem = XqaProblem();
  problem.cache_element_size = 1;
  problem.kv_cache_bit_width = 8;
  problem.k_quantization = GQAKvQuantizationType::PerTensor;
  problem.v_quantization = GQAKvQuantizationType::PerChannel;
  problem.use_qk_norm = true;
  config.kv_type = GQAXqaKvType::Int8;
  EXPECT_EQ(GetGQAXqaWorkspaceRecipe(problem, config).status.error,
            GQAWorkspaceError::InvalidArgument);

  problem.use_qk_norm = false;
  problem.kv_cache_bit_width = 4;
  EXPECT_EQ(GetGQAXqaWorkspaceRecipe(problem, config).status.error,
            GQAWorkspaceError::InvalidArgument);

  problem = XqaProblem();
  problem.kv_cache_bit_width = 8;
  config.kv_type = GQAXqaKvType::None;
  EXPECT_EQ(GetGQAXqaWorkspaceRecipe(problem, config).status.error,
            GQAWorkspaceError::InvalidArgument);

  problem = XqaProblem();
  problem.cache_element_size = 1;
  problem.kv_cache_bit_width = 0;
  problem.k_quantization = GQAKvQuantizationType::PerTensor;
  problem.v_quantization = GQAKvQuantizationType::PerTensor;
  config.kv_type = GQAXqaKvType::Int8;
  EXPECT_EQ(GetGQAXqaWorkspaceRecipe(problem, config).status.error,
            GQAWorkspaceError::InvalidArgument);
}

TEST(GroupQueryAttentionXqaWorkspaceTest, Fp8GateMatchesExactRuntimeArchitectures) {
  auto problem = XqaProblem();
  problem.cache_element_size = 1;
  problem.kv_cache_bit_width = 8;
  problem.k_quantization = GQAKvQuantizationType::PerTensor;
  problem.v_quantization = GQAKvQuantizationType::PerTensor;
  auto config = XqaConfig();
  config.kv_type = GQAXqaKvType::Fp8;
  config.device_minor = 9;

  EXPECT_TRUE(GetGQAXqaWorkspaceRecipe(problem, config).status.IsOK());

  config.device_minor = 10;
  EXPECT_EQ(GetGQAXqaWorkspaceRecipe(problem, config).status.error,
            GQAWorkspaceError::Unavailable);

  config.device_major = 9;
  config.device_minor = 0;
  EXPECT_TRUE(GetGQAXqaWorkspaceRecipe(problem, config).status.IsOK());
}

TEST(GroupQueryAttentionXqaWorkspaceTest, ValidatorRejectsMalformedInternalScratchOrderAndTerminal) {
  const auto result = GetGQAXqaWorkspaceRecipe(XqaProblem(), XqaConfig());
  ASSERT_TRUE(result.status.IsOK()) << result.status.message;

  auto malformed = result.recipe;
  malformed.row_sum_offset_bytes = malformed.row_max_offset_bytes;
  EXPECT_EQ(ValidateGQAXqaWorkspaceRecipe(malformed).error,
            GQAWorkspaceError::InvalidArgument);

  malformed = result.recipe;
  malformed.output_accumulator_offset_bytes += 128;
  EXPECT_EQ(ValidateGQAXqaWorkspaceRecipe(malformed).error,
            GQAWorkspaceError::InvalidArgument);

  malformed = result.recipe;
  malformed.output_accumulator_offset_bytes = malformed.row_sum_offset_bytes;
  EXPECT_EQ(ValidateGQAXqaWorkspaceRecipe(malformed).error,
            GQAWorkspaceError::InvalidArgument);

  malformed = result.recipe;
  ++malformed.internal_scratch_bytes;
  ++malformed.total_backend_bytes;
  EXPECT_EQ(ValidateGQAXqaWorkspaceRecipe(malformed).error,
            GQAWorkspaceError::InvalidArgument);
}

TEST(GroupQueryAttentionFlashWorkspaceTest, HandCalculatedOneAndMultipleSplitRecipes) {
  auto problem = FlashProblem();
  problem.batch_size = 2;
  problem.sequence_length = 3;
  problem.num_heads = 8;
  problem.kv_num_heads = 8;
  GQAFlashConfig config;
  config.total_sequence_length = 128;
  config.multi_processor_count = 108;

  auto result = GetGQAFlashWorkspaceRecipe(problem, config);
  ASSERT_TRUE(result.status.IsOK()) << result.status.message;
  EXPECT_EQ(result.recipe.selected_split_count, 1U);
  EXPECT_EQ(result.recipe.runtime_num_splits, 0U);
  EXPECT_EQ(result.recipe.softmax_lse_bytes, 192U);
  EXPECT_EQ(result.recipe.softmax_lse_accumulator_bytes, 0U);
  EXPECT_EQ(result.recipe.output_accumulator_bytes, 0U);
  EXPECT_EQ(result.recipe.total_backend_bytes, 192U);

  problem = FlashProblem();
  config.total_sequence_length = 4096;
  result = GetGQAFlashWorkspaceRecipe(problem, config);
  ASSERT_TRUE(result.status.IsOK()) << result.status.message;
  EXPECT_EQ(result.recipe.selected_split_count, 16U);
  EXPECT_EQ(result.recipe.runtime_num_splits, 16U);
  EXPECT_EQ(result.recipe.softmax_lse_bytes, 8U);
  EXPECT_EQ(result.recipe.softmax_lse_accumulator_offset_bytes, 256U);
  EXPECT_EQ(result.recipe.softmax_lse_accumulator_bytes, 128U);
  EXPECT_EQ(result.recipe.output_accumulator_offset_bytes, 512U);
  EXPECT_EQ(result.recipe.output_accumulator_bytes, 8192U);
  EXPECT_EQ(result.recipe.total_backend_bytes, 8704U);
  EXPECT_TRUE(ValidateGQAFlashWorkspaceRecipe(result.recipe).IsOK());
}

TEST(GroupQueryAttentionFlashWorkspaceTest, FastDecodeUsesKvHeadsAndWindowForSplitButQueryHeadsForBuffers) {
  auto problem = FlashProblem();
  problem.num_heads = 8;
  problem.kv_num_heads = 2;
  GQAFlashConfig config;
  config.total_sequence_length = 4096;
  config.multi_processor_count = 108;
  config.fast_decode = true;

  auto result = GetGQAFlashWorkspaceRecipe(problem, config);
  ASSERT_TRUE(result.status.IsOK()) << result.status.message;
  EXPECT_EQ(result.recipe.split_heuristic_head_count, 2U);
  EXPECT_EQ(result.recipe.split_heuristic_kv_length, 4096U);
  EXPECT_EQ(result.recipe.selected_split_count, 16U);
  EXPECT_EQ(result.recipe.softmax_lse_accumulator_bytes, 512U);
  EXPECT_EQ(result.recipe.output_accumulator_bytes, 32768U);

  config.local_window_size = 128;
  result = GetGQAFlashWorkspaceRecipe(problem, config);
  ASSERT_TRUE(result.status.IsOK()) << result.status.message;
  EXPECT_EQ(result.recipe.split_heuristic_kv_length, 128U);
  EXPECT_EQ(result.recipe.selected_split_count, 1U);
  EXPECT_EQ(result.recipe.softmax_lse_accumulator_bytes, 0U);
  EXPECT_EQ(result.recipe.output_accumulator_bytes, 0U);
}

TEST(GroupQueryAttentionFlashWorkspaceTest, RoundedHeadSizeControlsOutputAccumulator) {
  auto problem = FlashProblem();
  problem.head_size = 72;
  GQAFlashConfig config;
  config.total_sequence_length = 4096;
  config.multi_processor_count = 108;

  const auto result = GetGQAFlashWorkspaceRecipe(problem, config);
  ASSERT_TRUE(result.status.IsOK()) << result.status.message;
  EXPECT_EQ(result.recipe.rounded_head_size, 96U);
  ASSERT_GT(result.recipe.selected_split_count, 1U);
  EXPECT_EQ(result.recipe.output_accumulator_bytes,
            4U * result.recipe.selected_split_count * 2U * 96U);
}

TEST(GroupQueryAttentionFlashWorkspaceTest, MatchesLegacySplitAndBufferHelpers) {
#if defined(USE_FLASH_ATTENTION)
  for (int batch : {1, 2}) {
    for (int query_length : {1, 64, 65}) {
      for (int kv_length : {1, 128, 129, 255, 256, 257, 4096,
                            13824, 13825, 24576, 24577}) {
        for (int heads : {1, 2, 8}) {
          for (int head_size : {64, 72, 128, 136, 256}) {
            for (int sm_count : {24, 80, 108, 132}) {
              auto problem = FlashProblem();
              problem.batch_size = batch;
              problem.sequence_length = query_length;
              problem.num_heads = heads;
              problem.kv_num_heads = heads;
              problem.head_size = head_size;
              GQAFlashConfig config;
              config.total_sequence_length = kv_length;
              config.multi_processor_count = sm_count;

              const auto actual = GetGQAFlashWorkspaceRecipe(problem, config);
              ASSERT_TRUE(actual.status.IsOK()) << actual.status.message;
              const auto [legacy_splits, legacy_lse, legacy_out] =
                  flash::get_num_splits_and_buffer_sizes(
                      batch, query_length, kv_length, heads, head_size, sm_count);
              EXPECT_EQ(actual.recipe.runtime_num_splits, legacy_splits)
                  << "B=" << batch << " Sq=" << query_length
                  << " Skv=" << kv_length << " N=" << heads
                  << " H=" << head_size << " SM=" << sm_count;
              EXPECT_EQ(actual.recipe.softmax_lse_accumulator_bytes, legacy_lse);
              EXPECT_EQ(actual.recipe.output_accumulator_bytes, legacy_out);
            }
          }
        }
      }
    }
  }
#else
  GTEST_SKIP() << "Flash Attention is not enabled in this build.";
#endif
}

TEST(GroupQueryAttentionFlashWorkspaceTest, DoubleThresholdBoundaryMatchesRuntime) {
  auto problem = FlashProblem();
  problem.num_heads = 1;
  problem.kv_num_heads = 1;
  GQAFlashConfig config;
  config.total_sequence_length = 24577;
  config.multi_processor_count = 24;

  const auto result = GetGQAFlashWorkspaceRecipe(problem, config);
  ASSERT_TRUE(result.status.IsOK()) << result.status.message;
  // The runtime's double threshold selects 20; the prior float-literal copy
  // selected 17 and under-allocated both split accumulators.
  EXPECT_EQ(result.recipe.selected_split_count, 20U);
  EXPECT_EQ(result.recipe.runtime_num_splits, 20U);

#if defined(USE_FLASH_ATTENTION)
  const auto [runtime_splits, runtime_lse, runtime_out] =
      flash::get_num_splits_and_buffer_sizes(1, 1, 24577, 1, 64, 24);
  EXPECT_EQ(runtime_splits, 20U);
  EXPECT_EQ(result.recipe.runtime_num_splits, runtime_splits);
  EXPECT_EQ(result.recipe.softmax_lse_accumulator_bytes, runtime_lse);
  EXPECT_EQ(result.recipe.output_accumulator_bytes, runtime_out);
#endif
}

TEST(GroupQueryAttentionFlashWorkspaceTest, RejectsZeroLocalWindow) {
  GQAFlashConfig config;
  config.total_sequence_length = 128;
  config.local_window_size = 0;
  config.multi_processor_count = 24;
  EXPECT_EQ(GetGQAFlashWorkspaceRecipe(FlashProblem(), config).status.error,
            GQAWorkspaceError::InvalidArgument);
}

TEST(GroupQueryAttentionFlashWorkspaceTest, OriginalMaximumKvLengthIsNotMonotonic) {
  auto problem = FlashProblem();
  GQAFlashConfig config;
  config.multi_processor_count = 108;
  config.total_sequence_length = 13824;
  const auto lower = GetGQAFlashWorkspaceRecipe(problem, config);
  ASSERT_TRUE(lower.status.IsOK()) << lower.status.message;

  config.total_sequence_length = 13825;
  const auto upper = GetGQAFlashWorkspaceRecipe(problem, config);
  ASSERT_TRUE(upper.status.IsOK()) << upper.status.message;

  EXPECT_EQ(lower.recipe.selected_split_count, 54U);
  EXPECT_EQ(upper.recipe.selected_split_count, 28U);
  EXPECT_LT(upper.recipe.total_backend_bytes, lower.recipe.total_backend_bytes);

  size_t previous_bytes = 0;
  bool found_decrease = false;
  for (int kv_length = 1; kv_length <= 14000; ++kv_length) {
    config.total_sequence_length = kv_length;
    const auto current = GetGQAFlashWorkspaceRecipe(problem, config);
    ASSERT_TRUE(current.status.IsOK()) << current.status.message;
    found_decrease =
        found_decrease || current.recipe.total_backend_bytes < previous_bytes;
    previous_bytes = current.recipe.total_backend_bytes;
  }
  EXPECT_TRUE(found_decrease);
  // A future bound aggregate must use an envelope or report unavailable; sizing
  // only the componentwise maximum KV length is not conservative.
}

}  // namespace test
}  // namespace onnxruntime
