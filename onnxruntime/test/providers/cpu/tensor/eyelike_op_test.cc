// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"
#include "test/providers/provider_test_utils.h"

namespace onnxruntime {
namespace test {

TEST(EyeLikeOpTest, EyeLikeDefault) {
  OpTester test("EyeLike", 9);
  test.AddInput<float>("T1", {3, 2}, {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f});
  test.AddOutput<float>("T2", {3, 2}, {1.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f});
  test.Run();
}

TEST(EyeLikeOpTest, EyeLike_DifferentDtype) {
  OpTester test("EyeLike", 9);
  test.AddAttribute("dtype", int64_t(7));
  test.AddInput<float>("T1", {3, 3}, {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f});
  test.AddOutput<int64_t>("T2", {3, 3}, {1, 0, 0, 0, 1, 0, 0, 0, 1});
  test.Run();
}

TEST(EyeLikeOpTest, EyeLike_K_EgdeCase_1) {
  OpTester test("EyeLike", 9);
  test.AddInput<int64_t>("T1", {3, 2}, {0, 0, 0, 0, 0, 0});
  test.AddAttribute("k", int64_t(3));
  test.AddAttribute("dtype", int64_t(7));
  test.AddOutput<int64_t>("T2", {3, 2}, {0, 0, 0, 0, 0, 0});
  test.Run();
}

TEST(EyeLikeOpTest, EyeLike_K_EgdeCase_2) {
  OpTester test("EyeLike", 9);
  test.AddInput<int64_t>("T1", {3, 2}, {0, 0, 0, 0, 0, 0});
  test.AddAttribute("k", int64_t(-3));
  test.AddOutput<int64_t>("T2", {3, 2}, {0, 0, 0, 0, 0, 0});
  test.Run();
}

TEST(EyeLikeOpTest, EyeLike_UpperDiagonal) {
  OpTester test("EyeLike", 9);
  test.AddInput<float>("T1", {3, 4}, {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f});
  test.AddAttribute("k", int64_t(2));
  test.AddOutput<float>("T2", {3, 4}, {0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f});
  test.Run();
}

TEST(EyeLikeOpTest, EyeLike_UpperrDiagonal2) {
  OpTester test("EyeLike", 9);
  test.AddInput<float>("T1", {3, 2}, {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f});
  test.AddAttribute("k", int64_t(1));
  test.AddOutput<float>("T2", {3, 2}, {0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f});
  test.Run();
}

TEST(EyeLikeOpTest, EyeLike_LowerDiagonal) {
  OpTester test("EyeLike", 9);
  test.AddInput<float>("T1", {3, 2}, {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f});
  test.AddAttribute("k", int64_t(-1));
  test.AddAttribute("dtype", int64_t(1));
  test.AddOutput<float>("T2", {3, 2}, {0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 1.0f});
  test.Run();
}

TEST(EyeLikeOpTest, EyeLike_LowerDiagonal2) {
  OpTester test("EyeLike", 9);
  test.AddInput<float>("T1", {3, 4}, {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f});
  test.AddAttribute("k", int64_t(-2));
  test.AddAttribute("dtype", int64_t(1));
  test.AddOutput<float>("T2", {3, 4}, {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f});
  test.Run();
}

}  // namespace test
}  // namespace onnxruntime
