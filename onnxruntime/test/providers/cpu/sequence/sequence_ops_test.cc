// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"
#include "test/providers/provider_test_utils.h"

namespace onnxruntime {
namespace test {

TEST(SequenceLengthOpTest, SequenceLengthPositive) {
  OpTester test("SequenceLength", 11);
  test.AddInput<float>("T1", {3, 2}, {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f});
  test.AddOutput<float>("T2", {3, 2}, {1.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f});
  test.Run();
}

}  // namespace test
}  // namespace onnxruntime