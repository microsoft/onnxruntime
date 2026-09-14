// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <cuda_runtime.h>

#include "core/framework/tensor_shape.h"
#include "contrib_ops/cpu/bert/paged_attention_helper.h"
#include "contrib_ops/cuda/bert/paged_attention_impl.h"

namespace onnxruntime {
namespace test {

namespace {

struct FakeTensor {
  explicit FakeTensor(std::initializer_list<int64_t> dims) : shape_(dims) {}

  const TensorShape& Shape() const { return shape_; }

 private:
  TensorShape shape_;
};

}  // namespace

TEST(PagedAttentionHelperTest, CheckSequenceLengthTensorsRejectsWrongSeqlensLength) {
  FakeTensor cumulative_sequence_length({65});
  FakeTensor seqlens({1});

  int batch_size = 0;
  const auto status = onnxruntime::contrib::paged_attention_helper::CheckSequenceLengthTensors(
      &cumulative_sequence_length, &seqlens, batch_size);

  EXPECT_FALSE(status.IsOK());
  EXPECT_THAT(status.ErrorMessage(), ::testing::HasSubstr("seqlens must be shape (batch_size)."));
}

TEST(PagedAttentionHelperTest, CheckSequenceLengthTensorsAcceptsMatchingSeqlensLength) {
  FakeTensor cumulative_sequence_length({65});
  FakeTensor seqlens({64});

  int batch_size = 0;
  const auto status = onnxruntime::contrib::paged_attention_helper::CheckSequenceLengthTensors(
      &cumulative_sequence_length, &seqlens, batch_size);

  EXPECT_TRUE(status.IsOK()) << status.ErrorMessage();
  EXPECT_EQ(batch_size, 64);
}

TEST(PagedAttentionHelperTest, CheckBlockTableAndPastSeqLensValuesRejectsOutOfRangeBlockId) {
  const int32_t cumulative_sequence_length[] = {0, 1};
  const int32_t past_seqlens[] = {0};
  const int32_t block_table[] = {4};

  const auto status = onnxruntime::contrib::paged_attention_helper::CheckBlockTableAndPastSeqLensValues(
      cumulative_sequence_length,
      past_seqlens,
      block_table,
      1,
      1,
      16,
      4,
      1);

  EXPECT_FALSE(status.IsOK());
  EXPECT_THAT(status.ErrorMessage(), ::testing::HasSubstr("block_table values must be in [-1, num_blocks)"));
}

TEST(PagedAttentionHelperTest, CheckBlockTableAndPastSeqLensValuesRejectsBlockIdBelowSentinel) {
  const int32_t cumulative_sequence_length[] = {0, 0};
  const int32_t past_seqlens[] = {0};
  const int32_t block_table[] = {-2};

  const auto status = onnxruntime::contrib::paged_attention_helper::CheckBlockTableAndPastSeqLensValues(
      cumulative_sequence_length, past_seqlens, block_table, 1, 1, 16, 1, 0);

  EXPECT_FALSE(status.IsOK());
  EXPECT_THAT(status.ErrorMessage(), ::testing::HasSubstr("block_table values must be in [-1, num_blocks)"));
}

TEST(PagedAttentionHelperTest, CheckBlockTableAndPastSeqLensValuesAllowsSentinelForEvictedHistory) {
  const int32_t cumulative_sequence_length[] = {0, 1};
  const int32_t past_seqlens[] = {16};
  const int32_t block_table[] = {-1, 0};

  const auto status = onnxruntime::contrib::paged_attention_helper::CheckBlockTableAndPastSeqLensValues(
      cumulative_sequence_length, past_seqlens, block_table, 1, 2, 16, 1, 1);

  EXPECT_TRUE(status.IsOK()) << status.ErrorMessage();
}

TEST(PagedAttentionHelperTest, CheckBlockTableAndPastSeqLensValuesRejectsOutOfRangePaddingBlock) {
  const int32_t cumulative_sequence_length[] = {0, 0};
  const int32_t past_seqlens[] = {0};
  const int32_t block_table[] = {1};

  const auto status = onnxruntime::contrib::paged_attention_helper::CheckBlockTableAndPastSeqLensValues(
      cumulative_sequence_length, past_seqlens, block_table, 1, 1, 16, 1, 0);

  EXPECT_FALSE(status.IsOK());
  EXPECT_THAT(status.ErrorMessage(), ::testing::HasSubstr("block_table values must be in [-1, num_blocks)"));
}

TEST(PagedAttentionHelperTest, CheckBlockTableAndPastSeqLensValuesRejectsPastSeqlensOverflow) {
  const int32_t cumulative_sequence_length[] = {0, 2};
  const int32_t past_seqlens[] = {15};
  const int32_t block_table[] = {0};

  const auto status = onnxruntime::contrib::paged_attention_helper::CheckBlockTableAndPastSeqLensValues(
      cumulative_sequence_length,
      past_seqlens,
      block_table,
      1,
      1,
      16,
      1,
      2);

  EXPECT_FALSE(status.IsOK());
  EXPECT_THAT(status.ErrorMessage(), ::testing::HasSubstr("past_seqlens + query_length exceeds block_table capacity"));
}

TEST(PagedAttentionHelperTest, CheckBlockTableAndPastSeqLensValuesAcceptsValidInputs) {
  const int32_t cumulative_sequence_length[] = {0, 2, 5};
  const int32_t past_seqlens[] = {10, 3};
  const int32_t block_table[] = {0, 1, 2, 3};

  const auto status = onnxruntime::contrib::paged_attention_helper::CheckBlockTableAndPastSeqLensValues(
      cumulative_sequence_length,
      past_seqlens,
      block_table,
      2,
      2,
      16,
      4,
      5);

  EXPECT_TRUE(status.IsOK()) << status.ErrorMessage();
}

TEST(PagedAttentionHelperTest, CheckBlockTableAndPastSeqLensValuesRejectsNegativeCumulativeSeqLen) {
  const int32_t cumulative_sequence_length[] = {-1, 2};
  const int32_t past_seqlens[] = {0};
  const int32_t block_table[] = {0};

  const auto status = onnxruntime::contrib::paged_attention_helper::CheckBlockTableAndPastSeqLensValues(
      cumulative_sequence_length,
      past_seqlens,
      block_table,
      1,
      1,
      16,
      4,
      2);

  EXPECT_FALSE(status.IsOK());
  EXPECT_THAT(status.ErrorMessage(), ::testing::HasSubstr("cumulative_sequence_length must start with 0"));
}

TEST(PagedAttentionHelperTest, CheckBlockTableAndPastSeqLensValuesRejectsCumulativeNotStartingAtZero) {
  const int32_t cumulative_sequence_length[] = {1, 2};
  const int32_t past_seqlens[] = {0};
  const int32_t block_table[] = {0};

  const auto status = onnxruntime::contrib::paged_attention_helper::CheckBlockTableAndPastSeqLensValues(
      cumulative_sequence_length,
      past_seqlens,
      block_table,
      1,
      1,
      16,
      4,
      2);

  EXPECT_FALSE(status.IsOK());
  EXPECT_THAT(status.ErrorMessage(), ::testing::HasSubstr("cumulative_sequence_length must start with 0"));
}

TEST(PagedAttentionHelperTest, CheckBlockTableAndPastSeqLensValuesRejectsNegativePastSeqlens) {
  const int32_t cumulative_sequence_length[] = {0, 2};
  const int32_t past_seqlens[] = {-1};
  const int32_t block_table[] = {0};

  const auto status = onnxruntime::contrib::paged_attention_helper::CheckBlockTableAndPastSeqLensValues(
      cumulative_sequence_length,
      past_seqlens,
      block_table,
      1,
      1,
      16,
      4,
      2);

  EXPECT_FALSE(status.IsOK());
  EXPECT_THAT(status.ErrorMessage(), ::testing::HasSubstr("past_seqlens values must be non-negative"));
}

TEST(PagedAttentionHelperTest, CheckBlockTableAndPastSeqLensValuesAllowsFullCacheWithZeroTokens) {
  // q_len == 0 (no new tokens) and past_length == max_cache_sequence_length (full cache)
  // This should be allowed because there's no write to cache
  const int32_t cumulative_sequence_length[] = {0, 0};  // q_len = 0
  const int32_t past_seqlens[] = {256};                 // max_cache_sequence_length = 16 * 16 = 256
  const int32_t block_table[] = {0};

  const auto status = onnxruntime::contrib::paged_attention_helper::CheckBlockTableAndPastSeqLensValues(
      cumulative_sequence_length,
      past_seqlens,
      block_table,
      1,
      1,
      256,
      1,
      0);

  EXPECT_TRUE(status.IsOK()) << status.ErrorMessage();
}

TEST(PagedAttentionHelperTest, CheckBlockTableAndPastSeqLensValuesRejectsFullCacheWithNewTokens) {
  // q_len > 0 (has new tokens) and past_length == max_cache_sequence_length (full cache)
  // This should be rejected because we need space for new tokens
  const int32_t cumulative_sequence_length[] = {0, 1};  // q_len = 1
  const int32_t past_seqlens[] = {256};                 // max_cache_sequence_length = 16 * 16 = 256
  const int32_t block_table[] = {0};

  const auto status = onnxruntime::contrib::paged_attention_helper::CheckBlockTableAndPastSeqLensValues(
      cumulative_sequence_length,
      past_seqlens,
      block_table,
      1,
      1,
      256,
      1,
      1);

  EXPECT_FALSE(status.IsOK());
  EXPECT_THAT(status.ErrorMessage(), ::testing::HasSubstr("past_seqlens must be less than max_num_blocks_per_seq * block_size when q_len > 0"));
}

TEST(PagedAttentionHelperTest, CheckBlockTableAndPastSeqLensValuesRejectsTokenCountMismatch) {
  const int32_t cumulative_sequence_length[] = {0, 1};
  const int32_t past_seqlens[] = {0};
  const int32_t block_table[] = {0};

  const auto status = onnxruntime::contrib::paged_attention_helper::CheckBlockTableAndPastSeqLensValues(
      cumulative_sequence_length, past_seqlens, block_table, 1, 1, 16, 1, 2);

  EXPECT_FALSE(status.IsOK());
  EXPECT_THAT(status.ErrorMessage(), ::testing::HasSubstr("must end with token_count"));
}

TEST(PagedAttentionHelperTest, CheckBlockTableAndPastSeqLensValuesAllowsUnmappedBlockSentinel) {
  const int32_t cumulative_sequence_length[] = {0, 1};
  const int32_t past_seqlens[] = {0};
  const int32_t block_table[] = {0, -1};

  const auto status = onnxruntime::contrib::paged_attention_helper::CheckBlockTableAndPastSeqLensValues(
      cumulative_sequence_length, past_seqlens, block_table, 1, 2, 16, 1, 1);

  EXPECT_TRUE(status.IsOK()) << status.ErrorMessage();
}

TEST(PagedAttentionHelperTest, SanitizeBlockTablePreservesSentinelAndBoundsInvalidIds) {
  const std::vector<int32_t> input{-2, -1, 0, 3, 4};
  const std::vector<int32_t> expected{0, -1, 0, 3, 0};
  int32_t* input_device = nullptr;
  int32_t* output_device = nullptr;
  ASSERT_EQ(cudaSuccess, cudaMalloc(&input_device, input.size() * sizeof(int32_t)));
  ASSERT_EQ(cudaSuccess, cudaMalloc(&output_device, input.size() * sizeof(int32_t)));
  auto cleanup = gsl::finally([&]() {
    cudaFree(input_device);
    cudaFree(output_device);
  });

  ASSERT_EQ(cudaSuccess,
            cudaMemcpy(input_device, input.data(), input.size() * sizeof(int32_t), cudaMemcpyHostToDevice));
  const auto status = onnxruntime::contrib::cuda::LaunchSanitizeBlockTable(
      input_device, output_device, static_cast<int>(input.size()), 4, nullptr);
  ASSERT_TRUE(status.IsOK()) << status.ErrorMessage();

  std::vector<int32_t> actual(input.size());
  ASSERT_EQ(cudaSuccess,
            cudaMemcpy(actual.data(), output_device, actual.size() * sizeof(int32_t), cudaMemcpyDeviceToHost));
  EXPECT_EQ(actual, expected);
}

}  // namespace test
}  // namespace onnxruntime
