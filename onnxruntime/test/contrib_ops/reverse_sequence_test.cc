// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"
#include "test/providers/provider_test_utils.h"

namespace onnxruntime {
namespace test {

TEST(ReverseSequenceTest, BatchSeqInt) {
  OpTester test("ReverseSequence", 1, onnxruntime::kMSDomain);

  std::vector<int32_t> input = {
    1, 2, 3, 0, 0, 9,
    2, 2, 3, 4, 0, 9,
    3, 2, 0, 0, 0, 9,
    1, 0, 0, 0, 0, 9
  };
  
  std::vector<int32_t> expected_output = {
    3, 2, 1, 0, 0, 9,
    4, 3, 2, 2, 0, 9,
    2, 3, 0, 0, 0, 9,
    1, 0, 0, 0, 0, 9
  };
  std::vector<int32_t> seq_lengths = {3, 4, 2, 1};
  std::vector<int64_t> intput_shape = {4LL, 6LL, 1LL};
  std::vector<int64_t> seq_lengths_shape = { (int64_t)seq_lengths.size() };

  test.AddInput<int32_t>("input", intput_shape, input);
  test.AddInput<int32_t>("seq_lengths", seq_lengths_shape, seq_lengths);
  test.AddAttribute<int64_t>("batch_axis", 0LL);
  test.AddAttribute<int64_t>("seq_axis", 1LL);
  test.AddOutput<int32_t>("Y", intput_shape, expected_output);
  test.Run();
}

TEST(ReverseSequenceTest, SeqBatch2Float) {
  OpTester test("ReverseSequence", 1, onnxruntime::kMSDomain);

  std::vector<float> input= {
    1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
    2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f,
    3.0f, 3.0f, 3.0f, 3.0f, 0.0f, 0.0f,
    4.0f, 4.0f, 4.0f, 4.0f, 0.0f, 0.0f,
    9.0f, 9.0f, 5.0f, 5.0f, 9.0f, 9.0f
  };

  std::vector<float> expected_output = {
    4.0f, 4.0f, 5.0f, 5.0f, 2.0f, 2.0f,
    3.0f, 3.0f, 4.0f, 4.0f, 1.0f, 1.0f,
    2.0f, 2.0f, 3.0f, 3.0f, 0.0f, 0.0f,
    1.0f, 1.0f, 2.0f, 2.0f, 0.0f, 0.0f,
    9.0f, 9.0f, 1.0f, 1.0f, 9.0f, 9.0f
  };

  std::vector<int64_t> seq_lengths = {4LL, 5LL, 2LL};
  std::vector<int64_t> intput_shape = {5LL, 3LL, 2LL};
  std::vector<int64_t> seq_lengths_shape = { (int64_t)seq_lengths.size() };

  test.AddInput<float>("input", intput_shape, input);
  test.AddInput<int64_t>("seq_lengths", seq_lengths_shape, seq_lengths);
  test.AddAttribute<int64_t>("batch_axis", 1LL);
  test.AddAttribute<int64_t>("seq_axis", 0LL);
  test.AddOutput<float>("Y", intput_shape, expected_output);
  test.Run();
}

}  // namespace test
}  // namespace onnxruntime
