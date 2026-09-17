// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"

#include <cstddef>
#include <tuple>

#if defined(USE_FLASH_ATTENTION)
#include "contrib_ops/cuda/bert/group_query_attention_impl.h"

namespace onnxruntime {
namespace flash {
std::tuple<size_t, size_t, size_t> get_num_splits_and_buffer_sizes(size_t batch_size, size_t seqlen_q,
                                                                   size_t seqlen_k, size_t num_heads,
                                                                   size_t head_size, size_t num_SMs);
}  // namespace flash
}  // namespace onnxruntime
#endif

#if defined(USE_LEAN_ATTENTION)
namespace onnxruntime {
namespace lean {
std::tuple<size_t, size_t, size_t, size_t, size_t, size_t, size_t, size_t>
get_num_splits_and_buffer_sizes(size_t batch_size, size_t seqlen_q, size_t seqlen_k, size_t num_heads,
                                size_t num_heads_k, size_t head_size, size_t num_SMs, bool is_causal);
}  // namespace lean
}  // namespace onnxruntime
#endif

namespace onnxruntime {
namespace cuda {
namespace test {

TEST(FlashAttentionTest, GetNumSplitsHandlesZeroSmCount) {
#if defined(USE_FLASH_ATTENTION)
  const auto [num_splits, softmax_lse_accum_bytes, out_accum_bytes] =
      flash::get_num_splits_and_buffer_sizes(
          1,    // batch_size
          1,    // seqlen_q
          384,  // seqlen_k: 3 N-blocks when head_size is 128
          1,    // num_heads
          128,  // head_size
          0);   // num_SMs: regression coverage for divide-by-zero in PR #29550

  EXPECT_EQ(num_splits, 0U);
  EXPECT_EQ(softmax_lse_accum_bytes, 0U);
  EXPECT_EQ(out_accum_bytes, 0U);
#else
  GTEST_SKIP() << "Flash Attention is not enabled in this build.";
#endif
}

TEST(FlashAttentionTest, GetNumSplitsHandlesZeroKeyTiles) {
#if defined(USE_FLASH_ATTENTION)
  const auto [num_splits, softmax_lse_accum_bytes, out_accum_bytes] =
      flash::get_num_splits_and_buffer_sizes(
          1,    // batch_size
          1,    // seqlen_q
          0,    // seqlen_k: no N-blocks
          1,    // num_heads
          128,  // head_size
          2);   // num_SMs

  EXPECT_EQ(num_splits, 0U);
  EXPECT_EQ(softmax_lse_accum_bytes, 0U);
  EXPECT_EQ(out_accum_bytes, 0U);
#else
  GTEST_SKIP() << "Flash Attention is not enabled in this build.";
#endif
}

TEST(FlashAttentionTest, GetNumSplitsUsesLongContextParallelism) {
#if defined(USE_FLASH_ATTENTION)
  const auto [num_splits, softmax_lse_accum_bytes, out_accum_bytes] =
      flash::get_num_splits_and_buffer_sizes(
          1,     // batch_size
          1,     // seqlen_q
          4096,  // seqlen_k
          2,     // num_heads
          64,    // head_size
          108);  // num_SMs

  EXPECT_EQ(num_splits, 16U);
  EXPECT_GT(softmax_lse_accum_bytes, 0U);
  EXPECT_GT(out_accum_bytes, 0U);
#else
  GTEST_SKIP() << "Flash Attention is not enabled in this build.";
#endif
}

TEST(FlashAttentionTest, GetNumSplitsAvoidsShortContextOverhead) {
#if defined(USE_FLASH_ATTENTION)
  const auto [num_splits, softmax_lse_accum_bytes, out_accum_bytes] =
      flash::get_num_splits_and_buffer_sizes(
          1,     // batch_size
          1,     // seqlen_q
          128,   // KV sequence length
          2,     // num_heads
          64,    // head_size
          108);  // num_SMs

  EXPECT_EQ(num_splits, 0U);
  EXPECT_EQ(softmax_lse_accum_bytes, 0U);
  EXPECT_EQ(out_accum_bytes, 0U);
#else
  GTEST_SKIP() << "Flash Attention is not enabled in this build.";
#endif
}

TEST(FlashAttentionTest, GqaCaptureSplitPlanUsesCacheCapacity) {
#if defined(USE_FLASH_ATTENTION)
  constexpr int batch_size = 1;
  constexpr int sequence_length = 1;
  constexpr int total_sequence_length = 129;
  constexpr int cache_capacity = 4097;
  constexpr int num_heads = 32;
  constexpr int kv_num_heads = 4;
  constexpr int multi_processor_count = 108;

  for (int head_size : {64, 128, 256}) {
    const auto eager_plan = ::onnxruntime::contrib::cuda::GetFlashAttentionSplitPlan(
        batch_size, sequence_length, total_sequence_length, cache_capacity,
        num_heads, kv_num_heads, head_size, multi_processor_count,
        -1, true, true, false);
    const auto capture_plan = ::onnxruntime::contrib::cuda::GetFlashAttentionSplitPlan(
        batch_size, sequence_length, total_sequence_length, cache_capacity,
        num_heads, kv_num_heads, head_size, multi_processor_count,
        -1, true, true, true);
    const auto expected_eager_plan = flash::get_num_splits_and_buffer_sizes(
        batch_size, sequence_length, total_sequence_length, kv_num_heads,
        head_size, multi_processor_count);
    const auto expected_capture_plan = flash::get_num_splits_and_buffer_sizes(
        batch_size, sequence_length, cache_capacity, kv_num_heads,
        head_size, multi_processor_count);

    EXPECT_EQ(eager_plan.num_splits, std::get<0>(expected_eager_plan));
    EXPECT_EQ(capture_plan.num_splits, std::get<0>(expected_capture_plan));
    EXPECT_GT(capture_plan.num_splits, eager_plan.num_splits);
    EXPECT_EQ(eager_plan.softmax_lse_accum_bytes, capture_plan.softmax_lse_accum_bytes);
    EXPECT_EQ(eager_plan.out_accum_bytes, capture_plan.out_accum_bytes);
    EXPECT_GT(capture_plan.softmax_lse_accum_bytes, 0U);
    EXPECT_GT(capture_plan.out_accum_bytes, 0U);
  }
#else
  GTEST_SKIP() << "Flash Attention is not enabled in this build.";
#endif
}

TEST(FlashAttentionTest, GqaCaptureSplitPlanHonorsLocalWindowAndSequenceTail) {
#if defined(USE_FLASH_ATTENTION)
  constexpr int local_window_size = 257;
  const auto capture_plan = ::onnxruntime::contrib::cuda::GetFlashAttentionSplitPlan(
      1, 1, 4097, 8193, 32, 4, 128, 108,
      local_window_size, true, true, true);
  const auto expected_plan = flash::get_num_splits_and_buffer_sizes(
      1, 1, local_window_size, 4, 128, 108);

  EXPECT_EQ(capture_plan.num_splits, std::get<0>(expected_plan));
#else
  GTEST_SKIP() << "Flash Attention is not enabled in this build.";
#endif
}

TEST(FlashAttentionTest, NonDecodeSplitPlanKeepsLiveSequenceLength) {
#if defined(USE_FLASH_ATTENTION)
  constexpr int total_sequence_length = 257;
  const auto capture_plan = ::onnxruntime::contrib::cuda::GetFlashAttentionSplitPlan(
      1, 4, total_sequence_length, 8193, 32, 4, 64, 108,
      -1, false, true, true);
  const auto [expected_splits, expected_lse_bytes, expected_out_bytes] =
      flash::get_num_splits_and_buffer_sizes(
          1, 4, total_sequence_length, 32, 64, 108);

  EXPECT_EQ(capture_plan.num_splits, expected_splits);
  EXPECT_EQ(capture_plan.softmax_lse_accum_bytes, expected_lse_bytes);
  EXPECT_EQ(capture_plan.out_accum_bytes, expected_out_bytes);
#else
  GTEST_SKIP() << "Flash Attention is not enabled in this build.";
#endif
}

TEST(FlashAttentionTest, GqaEagerSplitPlanDoesNotReserveCaptureWorkspace) {
#if defined(USE_FLASH_ATTENTION)
  const auto eager_plan = ::onnxruntime::contrib::cuda::GetFlashAttentionSplitPlan(
      1, 1, 129, 4097, 32, 4, 64, 108,
      -1, true, false, false);

  EXPECT_EQ(eager_plan.num_splits, 0U);
  EXPECT_EQ(eager_plan.softmax_lse_accum_bytes, 0U);
  EXPECT_EQ(eager_plan.out_accum_bytes, 0U);
#else
  GTEST_SKIP() << "Flash Attention is not enabled in this build.";
#endif
}

TEST(LeanAttentionTest, GetNumSplitsHandlesZeroSmCount) {
#if defined(USE_LEAN_ATTENTION)
  const auto [num_splits, softmax_lse_accum_bytes, out_accum_bytes, sync_flag_bytes,
              grid_dim_z, max_tiles_per_tb, high_load_tbs, tiles_per_head] =
      lean::get_num_splits_and_buffer_sizes(
          1,    // batch_size
          1,    // seqlen_q
          384,  // seqlen_k: 3 N-blocks when head_size is 128
          1,    // num_heads
          1,    // num_heads_k
          128,  // head_size
          0,    // num_SMs: regression coverage for divide-by-zero in PR #29550
          true);

  EXPECT_EQ(num_splits, 3U);
  EXPECT_EQ(softmax_lse_accum_bytes, 12U);
  EXPECT_EQ(out_accum_bytes, 1536U);
  EXPECT_EQ(sync_flag_bytes, 4U);
  EXPECT_EQ(grid_dim_z, 2U);
  EXPECT_EQ(max_tiles_per_tb, 2U);
  EXPECT_EQ(high_load_tbs, 1U);
  EXPECT_EQ(tiles_per_head, 3U);
#else
  GTEST_SKIP() << "Lean Attention is not enabled in this build.";
#endif
}

TEST(LeanAttentionTest, GetNumSplitsHandlesZeroKeyTiles) {
#if defined(USE_LEAN_ATTENTION)
  const auto [num_splits, softmax_lse_accum_bytes, out_accum_bytes, sync_flag_bytes,
              grid_dim_z, max_tiles_per_tb, high_load_tbs, tiles_per_head] =
      lean::get_num_splits_and_buffer_sizes(
          1,    // batch_size
          1,    // seqlen_q
          0,    // seqlen_k: no N-blocks
          1,    // num_heads
          1,    // num_heads_k
          128,  // head_size
          2,    // num_SMs
          true);

  EXPECT_EQ(num_splits, 0U);
  EXPECT_EQ(softmax_lse_accum_bytes, 0U);
  EXPECT_EQ(out_accum_bytes, 0U);
  EXPECT_EQ(sync_flag_bytes, 0U);
  EXPECT_EQ(grid_dim_z, 1U);
  EXPECT_EQ(max_tiles_per_tb, 1U);
  EXPECT_EQ(high_load_tbs, 0U);
  EXPECT_EQ(tiles_per_head, 0U);
#else
  GTEST_SKIP() << "Lean Attention is not enabled in this build.";
#endif
}

}  // namespace test
}  // namespace cuda
}  // namespace onnxruntime
