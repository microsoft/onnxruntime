// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"
#include "test/providers/provider_test_utils.h"

namespace onnxruntime {
namespace test {

TEST(SelectTest, FloatSameDimension) {
  OpTester test("Select", 1, onnxruntime::kMSDomain);
  std::vector<float> X = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
  std::vector<float> Y = {2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f};
  std::vector<float> Z = {1.0f, 2.0f, 1.0f, 2.0f, 2.0f, 2.0f, 1.0f, 1.0f};

  test.AddInput<bool>("condition", {2LL, 4LL}, {true, false, true, false, false, false, true, true});
  test.AddInput<float>("X", {2LL, 4LL}, X);
  test.AddInput<float>("Y", {2LL, 4LL}, Y);
  test.AddOutput<float>("Z", {2LL, 4LL}, Z);
  test.Run();
}

TEST(SelectTest, Float1DCondition) {
  OpTester test("Select", 1, onnxruntime::kMSDomain);
  std::vector<float> X = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
  std::vector<float> Y = {2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f};
  std::vector<float> Z = {1.0f, 1.0f, 1.0f, 1.0f, 2.0f, 2.0f, 2.0f, 2.0f};

  test.AddInput<bool>("condition", {2LL}, {true, false});
  test.AddInput<float>("X", {2LL, 4LL}, X);
  test.AddInput<float>("Y", {2LL, 4LL}, Y);
  test.AddOutput<float>("Z", {2LL, 4LL}, Z);
  test.Run();
}

}  // namespace test
}  // namespace onnxruntime
