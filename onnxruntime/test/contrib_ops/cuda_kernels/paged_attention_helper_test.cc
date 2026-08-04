// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include "core/framework/tensor_shape.h"
#include "contrib_ops/cuda/bert/paged_attention_helper.h"

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

}  // namespace test
}  // namespace onnxruntime
