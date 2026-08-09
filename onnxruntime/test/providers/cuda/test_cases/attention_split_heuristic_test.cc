// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"

#include <cstddef>
#include <tuple>

#if defined(USE_FLASH_ATTENTION)
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
