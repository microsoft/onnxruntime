// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"
#include "test/providers/provider_test_utils.h"

namespace onnxruntime {
namespace test {

TEST(SequenceOpsTest, ConcatFromSequence_Axis0) {
  OpTester test("ConcatFromSequence", 11);
  test.AddAttribute<int64_t>("axis", 0);
  test.AddAttribute<int64_t>("new_axis", 1);
  SeqTensors<float> input;
  input.AddTensor({1, 2}, {0.0f, 1.0f});
  input.AddTensor({1, 2}, {2.0f, 3.0f});
  test.AddSeqInput("S", input);
  test.AddOutput<float>("I", {2, 1, 2}, {0.0f, 1.0f, 2.0f, 3.0f});
  test.Run();
}

TEST(SequenceOpsTest, ConcatFromSequence_Axis1) {
  OpTester test("ConcatFromSequence", 11);
  test.AddAttribute<int64_t>("axis", 1);
  test.AddAttribute<int64_t>("new_axis", 1);
  SeqTensors<float> input;
  input.AddTensor({1, 2}, {0.0f, 1.0f});
  input.AddTensor({1, 2}, {2.0f, 3.0f});
  test.AddSeqInput("S", input);
  test.AddOutput<float>("I", {1, 2, 2}, {0.0f, 1.0f, 2.0f, 3.0f});
  test.Run();
}

TEST(SequenceOpsTest, ConcatFromSequence_Axis2) {
  OpTester test("ConcatFromSequence", 11);
  test.AddAttribute<int64_t>("axis", 2);
  test.AddAttribute<int64_t>("new_axis", 1);
  SeqTensors<float> input;
  input.AddTensor({1, 2}, {0.0f, 1.0f});
  input.AddTensor({1, 2}, {2.0f, 3.0f});
  test.AddSeqInput("S", input);
  test.AddOutput<float>("I", {1, 2, 2}, {0.0f, 2.0f, 1.0f, 3.0f});
  test.Run();
}

// TODO:
// 1. tests for scalar stacking
// 2. tests with empty input stacking
// 3. test for sequence concats (use existing tests for concat)

}  // namespace test
}  // namespace onnxruntime