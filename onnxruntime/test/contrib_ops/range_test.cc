// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"
#include "test/providers/provider_test_utils.h"

namespace onnxruntime {
namespace test {

TEST(RangeTest, PositiveInt32DeltaDefault) {
  OpTester test("Range", 1, onnxruntime::kMSDomain);
  std::vector<int32_t> start = {0};
  std::vector<int32_t> limit = {5};
  std::vector<int32_t> expected_output = {0, 1, 2, 3, 4};

  test.AddInput<int32_t>("start", {1}, start);
  test.AddInput<int32_t>("limit", {1}, limit);
  test.AddOutput<int32_t>("Y", {5LL}, expected_output);
  test.Run();
}

TEST(RangeTest, PositiveInt32Delta_0) {
  OpTester test("Range", 1, onnxruntime::kMSDomain);
  std::vector<int32_t> start = {0};
  std::vector<int32_t> limit = {10};
  std::vector<int32_t> delta = {2};
  std::vector<int32_t> expected_output = {0, 2, 4, 6, 8};

  test.AddInput<int32_t>("start", {1}, start);
  test.AddInput<int32_t>("limit", {1}, limit);
  test.AddInput<int32_t>("delta", {1}, delta);
  test.AddOutput<int32_t>("Y", {5LL}, expected_output);
  test.Run();
}

TEST(RangeTest, PositiveInt32Delta_1) {
  OpTester test("Range", 1, onnxruntime::kMSDomain);
  std::vector<int32_t> start = {0};
  std::vector<int32_t> limit = {9};
  std::vector<int32_t> delta = {2};
  std::vector<int32_t> expected_output = {0, 2, 4, 6, 8};

  test.AddInput<int32_t>("start", {1}, start);
  test.AddInput<int32_t>("limit", {1}, limit);
  test.AddInput<int32_t>("delta", {1}, delta);
  test.AddOutput<int32_t>("Y", {5LL}, expected_output);
  test.Run();
}

TEST(RangeTest, Int32ScalarNegativeDelta_0) {
  OpTester test("Range", 1, onnxruntime::kMSDomain);
  std::vector<int32_t> start = {2};
  std::vector<int32_t> limit = {-9};
  std::vector<int32_t> delta = {-2};
  std::vector<int32_t> expected_output = {2, 0, -2, -4, -6, -8};

  test.AddInput<int32_t>("start", {}, start);
  test.AddInput<int32_t>("limit", {}, limit);
  test.AddInput<int32_t>("delta", {}, delta);
  test.AddOutput<int32_t>("Y", {6LL}, expected_output);
  test.Run();
}


TEST(RangeTest, Int32ScalarNegativeDelta_1) {
  OpTester test("Range", 1, onnxruntime::kMSDomain);
  std::vector<int32_t> start = {2};
  std::vector<int32_t> limit = {-9};
  std::vector<int32_t> delta = {-2};
  std::vector<int32_t> expected_output = {2, 0, -2, -4, -6, -8};

  test.AddInput<int32_t>("start", {}, start);
  test.AddInput<int32_t>("limit", {}, limit);
  test.AddInput<int32_t>("delta", {}, delta);
  test.AddOutput<int32_t>("Y", {6LL}, expected_output);
  test.Run();
}

TEST(RangeTest, ScalarFloatNegativeDelta) {
  OpTester test("Range", 1, onnxruntime::kMSDomain);
  std::vector<float> start = {2.0f};
  std::vector<float> limit = {-8.1f};
  std::vector<float> delta = {-2.0f};
  std::vector<float> expected_output = {2.0f, 0.0f, -2.0f, -4.0f, -6.0f, -8.0f};
  
  test.AddInput<float>("start", {}, start);
  test.AddInput<float>("limit", {}, limit);
  test.AddInput<float>("delta", {}, delta);
  test.AddOutput<float>("Y", {6LL}, expected_output);
  test.Run();
}

}  // namespace test
}  // namespace onnxruntime
