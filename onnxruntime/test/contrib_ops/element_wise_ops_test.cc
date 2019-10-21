// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"
#include "test/providers/provider_test_utils.h"
#include "core/util/math.h"
#include <algorithm>
#include <cmath>

using namespace onnxruntime::test;

namespace onnxruntime {
namespace test {

TEST(MathOpTest, AffineDefaultAttributes) {
  OpTester test("Affine", 7);
  std::vector<int64_t> dims{2, 2};
  test.AddInput<float>("A", dims, {0.0f, 1.0f, 2.0f, 3.0f});
  test.AddOutput<float>("B", dims, {0.0f, 1.0f, 2.0f, 3.0f});
  test.Run();
}

TEST(MathOpTest, Affine) {
  OpTester test("Affine", 7);
  std::vector<int64_t> dims{2, 2};
  test.AddAttribute("alpha", 2.0f);
  test.AddAttribute("beta", 1.0f);
  test.AddInput<float>("A", dims, {0.0f, 1.0f, 2.0f, 3.0f});
  test.AddOutput<float>("B", dims, {1.0f, 3.0f, 5.0f, 7.0f});
  test.Run();
}

TEST(MathOpTest, Scale) {
  OpTester test("Scale", 7);
  std::vector<int64_t> dims{2, 2};
  test.AddAttribute("scale", 2.0f);
  test.AddInput<float>("A", dims, {0.0f, 1.0f, 2.0f, 3.0f});
  test.AddOutput<float>("B", dims, {0.0f, 2.0f, 4.0f, 6.0f});
  test.Run();
}

TEST(MathOpTest, Scale_Default) {
  OpTester test("Scale", 7);
  std::vector<int64_t> dims{2, 2};
  test.AddInput<float>("A", dims, {0.0f, 1.0f, 2.0f, 3.0f});
  test.AddOutput<float>("B", dims, {0.0f, 1.0f, 2.0f, 3.0f});
  test.Run();
}

}  // namespace test
}  // namespace onnxruntime
