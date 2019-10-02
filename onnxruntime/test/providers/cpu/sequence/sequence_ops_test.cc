// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"
#include "test/providers/provider_test_utils.h"

namespace onnxruntime {
namespace test {

TEST(SequenceLengthOpTest, SequenceLengthPositive) {
  OpTester test("SequenceLength", 11);
  SeqTensors<float> input;
  input.tensors.push_back(std::make_pair<std::vector<int64_t>, std::vector<float>>({3, 2}, {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f}));
  input.tensors.push_back(std::make_pair<std::vector<int64_t>, std::vector<float>>({3, 3}, {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f}));
  test.AddSeqInput("S", input);
  test.AddOutput<int64_t>("I", {}, {2});
  test.Run();
}

}  // namespace test
}  // namespace onnxruntime