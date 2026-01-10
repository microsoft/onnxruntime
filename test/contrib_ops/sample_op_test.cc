// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"
#include "test/providers/provider_test_utils.h"

namespace onnxruntime {
namespace test {

TEST(MLOpTest, SampleOpFloat) {
  OpTester test("SampleOp", 1, onnxruntime::kMSDomain);
  std::vector<float> X = {0.8f, -0.5f, 0.0f, 1.f, 1.0f};
  std::vector<float> expected_output = X;
  const int64_t N = static_cast<int64_t>(X.size());
  test.AddInput<float>("X", {N}, X);
  test.AddOutput<float>("Y", {N}, expected_output);
  test.Run();
}

}  // namespace test
}  // namespace onnxruntime