// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <cuda_runtime.h>
#include <limits>

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

TEST(PagedAttentionHelperTest, CheckBlockTableRejectsZeroWidth) {
  FakeTensor block_table({2, 0});
  int max_num_blocks_per_seq = 0;

  const auto status = onnxruntime::contrib::paged_attention_helper::CheckBlockTable(
      &block_table, 2, max_num_blocks_per_seq);

  EXPECT_FALSE(status.IsOK());
  EXPECT_THAT(status.ErrorMessage(), ::testing::HasSubstr("dimension 1 must be positive"));
}

TEST(PagedAttentionHelperTest, CheckSequenceLengthValuesRejectsPastSeqlensOverflow) {
  const int32_t cumulative_seqlens_q[] = {0, 2};
  const int32_t cumulative_seqlens_kv[] = {0, 17};

  const auto status = onnxruntime::contrib::paged_attention_helper::CheckSequenceLengthValues(
      cumulative_seqlens_q, cumulative_seqlens_kv, 1, 1, 16, 2);

  EXPECT_FALSE(status.IsOK());
  EXPECT_THAT(status.ErrorMessage(), ::testing::HasSubstr("past_seqlens + query_length exceeds block_table capacity"));
}

TEST(PagedAttentionHelperTest, CheckSequenceLengthValuesAcceptsValidInputs) {
  const int32_t cumulative_seqlens_q[] = {0, 2, 5};
  const int32_t cumulative_seqlens_kv[] = {0, 12, 18};

  const auto status = onnxruntime::contrib::paged_attention_helper::CheckSequenceLengthValues(
      cumulative_seqlens_q, cumulative_seqlens_kv, 2, 2, 16, 5);

  EXPECT_TRUE(status.IsOK()) << status.ErrorMessage();
}

TEST(PagedAttentionHelperTest, CheckSequenceLengthValuesRejectsNegativeCumulativeSeqLen) {
  const int32_t cumulative_seqlens_q[] = {-1, 2};
  const int32_t cumulative_seqlens_kv[] = {0, 3};

  const auto status = onnxruntime::contrib::paged_attention_helper::CheckSequenceLengthValues(
      cumulative_seqlens_q, cumulative_seqlens_kv, 1, 1, 16, 2);

  EXPECT_FALSE(status.IsOK());
  EXPECT_THAT(status.ErrorMessage(), ::testing::HasSubstr("cumulative_seqlens_q must start with 0"));
}

TEST(PagedAttentionHelperTest, CheckSequenceLengthValuesRejectsCumulativeNotStartingAtZero) {
  const int32_t cumulative_seqlens_q[] = {1, 2};
  const int32_t cumulative_seqlens_kv[] = {0, 1};

  const auto status = onnxruntime::contrib::paged_attention_helper::CheckSequenceLengthValues(
      cumulative_seqlens_q, cumulative_seqlens_kv, 1, 1, 16, 2);

  EXPECT_FALSE(status.IsOK());
  EXPECT_THAT(status.ErrorMessage(), ::testing::HasSubstr("cumulative_seqlens_q must start with 0"));
}

TEST(PagedAttentionHelperTest, CheckSequenceLengthValuesRejectsKvCumulativeNotStartingAtZero) {
  const int32_t cumulative_seqlens_q[] = {0, 1};
  const int32_t cumulative_seqlens_kv[] = {1, 2};

  const auto status = onnxruntime::contrib::paged_attention_helper::CheckSequenceLengthValues(
      cumulative_seqlens_q, cumulative_seqlens_kv, 1, 1, 16, 1);

  EXPECT_FALSE(status.IsOK());
  EXPECT_THAT(status.ErrorMessage(), ::testing::HasSubstr("cumulative_seqlens_kv must start with 0"));
}

TEST(PagedAttentionHelperTest, CheckSequenceLengthValuesRejectsNegativePastSeqlens) {
  const int32_t cumulative_seqlens_q[] = {0, 2};
  const int32_t cumulative_seqlens_kv[] = {0, 1};

  const auto status = onnxruntime::contrib::paged_attention_helper::CheckSequenceLengthValues(
      cumulative_seqlens_q, cumulative_seqlens_kv, 1, 1, 16, 2);

  EXPECT_FALSE(status.IsOK());
  EXPECT_THAT(status.ErrorMessage(), ::testing::HasSubstr("past_seqlens values must be non-negative"));
}

TEST(PagedAttentionHelperTest, CheckSequenceLengthValuesAllowsFullCacheWithZeroTokens) {
  // q_len == 0 (no new tokens) and past_length == max_cache_sequence_length (full cache)
  // This should be allowed because there's no write to cache
  const int32_t cumulative_seqlens_q[] = {0, 0};     // q_len = 0
  const int32_t cumulative_seqlens_kv[] = {0, 256};  // max_cache_sequence_length = 1 * 256 = 256

  const auto status = onnxruntime::contrib::paged_attention_helper::CheckSequenceLengthValues(
      cumulative_seqlens_q, cumulative_seqlens_kv, 1, 1, 256, 0);

  EXPECT_TRUE(status.IsOK()) << status.ErrorMessage();
}

TEST(PagedAttentionHelperTest, CheckSequenceLengthValuesRejectsFullCacheWithNewTokens) {
  // q_len > 0 (has new tokens) and past_length == max_cache_sequence_length (full cache)
  // This should be rejected because we need space for new tokens
  const int32_t cumulative_seqlens_q[] = {0, 1};     // q_len = 1
  const int32_t cumulative_seqlens_kv[] = {0, 257};  // past_length = 256

  const auto status = onnxruntime::contrib::paged_attention_helper::CheckSequenceLengthValues(
      cumulative_seqlens_q, cumulative_seqlens_kv, 1, 1, 256, 1);

  EXPECT_FALSE(status.IsOK());
  EXPECT_THAT(status.ErrorMessage(), ::testing::HasSubstr("past_seqlens must be less than max_num_blocks_per_seq * block_size when q_len > 0"));
}

TEST(PagedAttentionHelperTest, CheckSequenceLengthValuesRejectsTokenCountMismatch) {
  const int32_t cumulative_seqlens_q[] = {0, 1};
  const int32_t cumulative_seqlens_kv[] = {0, 1};

  const auto status = onnxruntime::contrib::paged_attention_helper::CheckSequenceLengthValues(
      cumulative_seqlens_q, cumulative_seqlens_kv, 1, 1, 16, 2);

  EXPECT_FALSE(status.IsOK());
  EXPECT_THAT(status.ErrorMessage(), ::testing::HasSubstr("must end with token_count"));
}

TEST(PagedAttentionHelperTest, SanitizeBlockTablePreservesSentinelAndBoundsInvalidIds) {
  const std::vector<int32_t> input{-2, -1, 0, 3, 4};
  const std::vector<int32_t> expected{-1, -1, 0, 3, -1};
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

TEST(PagedAttentionHelperTest, SanitizeBlockTableAllowsZeroElements) {
  const auto status = onnxruntime::contrib::cuda::LaunchSanitizeBlockTable(
      nullptr, nullptr, 0, 0, nullptr);
  EXPECT_TRUE(status.IsOK()) << status.ErrorMessage();
}

TEST(PagedAttentionHelperTest, SanitizeBlockTableRejectsUnsupportedElementCount) {
  const auto status = onnxruntime::contrib::cuda::LaunchSanitizeBlockTable(
      nullptr, nullptr, static_cast<size_t>(std::numeric_limits<int32_t>::max()) + 1, 0, nullptr);
  EXPECT_FALSE(status.IsOK());
  EXPECT_THAT(status.ErrorMessage(), ::testing::HasSubstr("exceeds the CUDA kernel indexing limit"));
}

TEST(PagedAttentionHelperTest, SanitizeSequenceLengthsCanonicalizesUnsafeInputs) {
  const std::vector<int32_t> cumulative_seqlens_q{5, -2, 1};
  const std::vector<int32_t> past_seqlens{-4, 100};
  const std::vector<int32_t> block_table{0, -1, 1, 2};
  constexpr int batch_size = 2;

  int32_t* input_device = nullptr;
  int32_t* output_device = nullptr;
  ASSERT_EQ(cudaSuccess, cudaMalloc(&input_device, 9 * sizeof(int32_t)));
  ASSERT_EQ(cudaSuccess, cudaMalloc(&output_device, 8 * sizeof(int32_t)));
  auto cleanup = gsl::finally([&]() {
    cudaFree(input_device);
    cudaFree(output_device);
  });

  ASSERT_EQ(cudaSuccess,
            cudaMemcpy(input_device, cumulative_seqlens_q.data(), 3 * sizeof(int32_t), cudaMemcpyHostToDevice));
  ASSERT_EQ(cudaSuccess,
            cudaMemcpy(input_device + 3, past_seqlens.data(), 2 * sizeof(int32_t), cudaMemcpyHostToDevice));
  ASSERT_EQ(cudaSuccess,
            cudaMemcpy(input_device + 5, block_table.data(), 4 * sizeof(int32_t), cudaMemcpyHostToDevice));

  const auto status = onnxruntime::contrib::cuda::LaunchSanitizeSequenceLengths(
      output_device, output_device + 3, output_device + 5,
      input_device, input_device + 3, input_device + 5,
      batch_size, 2, 16, 3, nullptr);
  ASSERT_TRUE(status.IsOK()) << status.ErrorMessage();

  std::vector<int32_t> actual(8);
  ASSERT_EQ(cudaSuccess,
            cudaMemcpy(actual.data(), output_device, actual.size() * sizeof(int32_t), cudaMemcpyDeviceToHost));
  EXPECT_EQ(actual, (std::vector<int32_t>{0, 0, 3, 0, 29, 0, 0, 32}));
}

}  // namespace test
}  // namespace onnxruntime
