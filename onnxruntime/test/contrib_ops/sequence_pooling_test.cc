// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"
#include "test/providers/provider_test_utils.h"

namespace onnxruntime {
namespace test {

TEST(SequencePoolingTest, DummyTest) {
  OpTester test("SequencePooling", 1, onnxruntime::kMSDomain);
  std::vector<float> batch_input{1.0f, 2.0f, 3.0f, 3.0f, 5.0f, 5.0f, 4.0f, 3.0f, 6.0f,
                                 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f,
                                 1.0f, 2.0f, 3.0f, 3.0f, 5.0f, 5.0f, 4.0f, 3.0f, 6.0f,
                                 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f};
  std::vector<int32_t> batch_sequence_lengths{1, 2, 3, 1, 2, 3};
  std::vector<float> output{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f,
                            1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f};
  std::vector<float> masks{1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f};

  test.AddInput<float>("batch_input_tensor", {2, 6, 3}, batch_input);
  test.AddInput<int32_t>("batch_sentence_lengthes", {2, 3}, batch_sequence_lengths);
  test.AddOutput<float>("output", {2, 3, 3}, output);
  test.AddOutput<float>("masks", {2, 3}, masks);
  test.Run();
}

}  // namespace test
}  // namespace onnxruntime
