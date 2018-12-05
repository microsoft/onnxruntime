// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"
#include "test/providers/provider_test_utils.h"

namespace onnxruntime {
namespace test {

TEST(ConstantFillTest, ConstantFillDefault) {
  OpTester test("ConstantFill", 1);

  test.AddAttribute("input_as_shape", int64_t(0));
  test.AddAttribute("shape", std::vector<int64_t>({1, 2}));

  test.AddOutput<float>("T2", {1, 2}, {0.0f, 0.0f});

  test.Run();
}

TEST(ConstantFillTest, ConstantFillValue) {
  OpTester test("ConstantFill", 1);

  test.AddAttribute("input_as_shape", int64_t(0));
  test.AddAttribute("shape", std::vector<int64_t>({1, 2}));
  test.AddAttribute("value", 42.0f);

  test.AddOutput<float>("T2", {1, 2}, {42.0f, 42.0f});

  test.Run();
}

TEST(ConstantFillTest, ConstantFillInputAsFloatShape) {
  OpTester test("ConstantFill", 1);

  test.AddAttribute("input_as_shape", int64_t(1));
  test.AddAttribute("extra_shape", std::vector<int64_t>({3}));
  test.AddAttribute("value", 42.0f);

  test.AddInput<float>("T1", {2}, {1.0f, 2.0f});
  test.AddOutput<float>("T2", {1, 2, 3}, {42.0f, 42.0f, 42.0f, 42.0f, 42.0f, 42.0f});

  test.Run();
}

TEST(ConstantFillTest, ConstantFillInputAsInt32Shape) {
  OpTester test("ConstantFill", 1);

  test.AddAttribute("input_as_shape", int64_t(1));
  test.AddAttribute("extra_shape", std::vector<int64_t>({3}));
  test.AddAttribute("value", 42.0f);

  test.AddInput<int32_t>("T1", {2}, {1, 2});
  test.AddOutput<float>("T2", {1, 2, 3}, {42.0f, 42.0f, 42.0f, 42.0f, 42.0f, 42.0f});

  test.Run();
}

TEST(ConstantFillTest, ConstantFillInputAsInt64Shape) {
  OpTester test("ConstantFill", 1);

  test.AddAttribute("input_as_shape", int64_t(1));
  test.AddAttribute("extra_shape", std::vector<int64_t>({3}));
  test.AddAttribute("value", 42.0f);

  test.AddInput<int64_t>("T1", {2}, {1, 2});
  test.AddOutput<float>("T2", {1, 2, 3}, {42.0f, 42.0f, 42.0f, 42.0f, 42.0f, 42.0f});

  test.Run();
}

TEST(ConstantFillTest, ConstantFillFloat) {
  OpTester test("ConstantFill", 1);

  test.AddAttribute("dtype", int64_t(1));  // float
  test.AddAttribute("input_as_shape", int64_t(0));
  test.AddAttribute("shape", std::vector<int64_t>({1, 2}));
  test.AddAttribute("value", 42.0f);

  test.AddOutput<float>("T2", {1, 2}, {42.0f, 42.0f});

  test.Run();
}

TEST(ConstantFillTest, ConstantFillInt32) {
  OpTester test("ConstantFill", 1);

  test.AddAttribute("dtype", int64_t(6));  // int32
  test.AddAttribute("input_as_shape", int64_t(0));
  test.AddAttribute("shape", std::vector<int64_t>({1, 2}));
  test.AddAttribute("value", 42.0f);

  test.AddOutput<int32_t>("T2", {1, 2}, {42, 42});

  test.Run();
}

TEST(ConstantFillTest, ConstantFillInt64) {
  OpTester test("ConstantFill", 1);

  test.AddAttribute("dtype", int64_t(7));  // int64
  test.AddAttribute("input_as_shape", int64_t(0));
  test.AddAttribute("shape", std::vector<int64_t>({1, 2}));
  test.AddAttribute("value", 42.0f);

  test.AddOutput<int64_t>("T2", {1, 2}, {42, 42});

  test.Run();
}

TEST(ConstantFillTest, ConstantFillBool) {
  OpTester test("ConstantFill", 1);

  test.AddAttribute("dtype", int64_t(9));  // bool
  test.AddAttribute("input_as_shape", int64_t(0));
  test.AddAttribute("shape", std::vector<int64_t>({1, 2}));
  test.AddAttribute("value", 1.0f);

  test.AddOutput<bool>("T2", {1, 2}, {true, true});

  test.Run();
}

}  // namespace test
}  // namespace onnxruntime
