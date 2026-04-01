// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include "contrib_ops/cpu/bert/attention_parameters.h"
#include "contrib_ops/cpu/sparse/sparse_attention_helper.h"
#include "test/unittest_util/framework_test_utils.h"

namespace onnxruntime {
namespace test {

namespace {

struct SparseAttentionCheckInputsData {
  OrtValue query;
  OrtValue key;
  OrtValue value;
  OrtValue past_key;
  OrtValue past_value;
  OrtValue block_row_indices;
  OrtValue block_col_indices;
  OrtValue total_key_lengths;
  OrtValue total_seq_len;
};

SparseAttentionCheckInputsData CreateSparseAttentionCheckInputsData(const std::vector<int32_t>& total_key_lengths_data,
                                                                    const std::vector<int64_t>& total_key_lengths_dims) {
  SparseAttentionCheckInputsData data;
  auto allocator = CPUAllocator::DefaultInstance();

  CreateMLValue<float>(allocator, {1, 1, 16}, std::vector<float>(16, 0.0f), &data.query);
  CreateMLValue<float>(allocator, {1, 1, 16}, std::vector<float>(16, 0.0f), &data.key);
  CreateMLValue<float>(allocator, {1, 1, 16}, std::vector<float>(16, 0.0f), &data.value);
  CreateMLValue<float>(allocator, {1, 2, 4, 8}, std::vector<float>(64, 0.0f), &data.past_key);
  CreateMLValue<float>(allocator, {1, 2, 4, 8}, std::vector<float>(64, 0.0f), &data.past_value);
  CreateMLValue<int32_t>(allocator, {1, 5}, {0, 1, 2, 3, 4}, &data.block_row_indices);
  CreateMLValue<int32_t>(allocator, {1, 1}, {0}, &data.block_col_indices);
  CreateMLValue<int32_t>(allocator, total_key_lengths_dims, total_key_lengths_data, &data.total_key_lengths);
  CreateMLValue<int32_t>(allocator, {1}, {4}, &data.total_seq_len);

  return data;
}

Status CheckSparseAttentionInputs(const SparseAttentionCheckInputsData& data) {
  contrib::SparseAttentionParameters parameters{};
  parameters.sparse_block_size = 1;
  parameters.num_heads = 2;
  parameters.kv_num_heads = 2;
  parameters.scale = 1.0f;
  parameters.do_rotary = false;
  parameters.rotary_interleaved = false;

  return contrib::sparse_attention_helper::CheckInputs(&parameters,
                                                       &data.query.Get<Tensor>(),
                                                       &data.key.Get<Tensor>(),
                                                       &data.value.Get<Tensor>(),
                                                       &data.past_key.Get<Tensor>(),
                                                       &data.past_value.Get<Tensor>(),
                                                       nullptr,
                                                       nullptr,
                                                       &data.block_row_indices.Get<Tensor>(),
                                                       &data.block_col_indices.Get<Tensor>(),
                                                       &data.total_key_lengths.Get<Tensor>(),
                                                       &data.total_seq_len.Get<Tensor>());
}

}  // namespace

TEST(SparseAttentionTest, RejectsOutOfRangeKeyTotalSequenceLengths) {
  auto data = CreateSparseAttentionCheckInputsData({-5}, {1});
  const auto status = CheckSparseAttentionInputs(data);

  ASSERT_FALSE(status.IsOK());
  EXPECT_THAT(status.ErrorMessage(), testing::HasSubstr("key_total_sequence_lengths value -5"));
  EXPECT_THAT(status.ErrorMessage(), testing::HasSubstr("out of range [1, 4]"));
}

TEST(SparseAttentionTest, RejectsKeyTotalSequenceLengthsShapeMismatch) {
  auto data = CreateSparseAttentionCheckInputsData({4, 4}, {2});
  const auto status = CheckSparseAttentionInputs(data);

  ASSERT_FALSE(status.IsOK());
  EXPECT_THAT(status.ErrorMessage(), testing::HasSubstr("key_total_sequence_lengths must have shape (batch_size)"));
}

}  // namespace test
}  // namespace onnxruntime
