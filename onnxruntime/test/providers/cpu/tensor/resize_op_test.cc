// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cpu/tensor/resize.h"
#include "gtest/gtest.h"
#include "test/providers/provider_test_utils.h"

namespace onnxruntime {
namespace test {
TEST(ResizeOpTest, ResizeOpLineartDownSampleTest) {
  OpTester test("Resize", 10);
  std::vector<float> scales{1.0f, 1.0f, 0.6f, 0.6f};

  test.AddAttribute("mode", "linear");

  const int64_t N = 1, C = 1, H = 2, W = 4;
  std::vector<float> X = {
      1.0f, 2.0f, 3.0f, 4.0f,
      5.0f, 6.0f, 7.0f, 8.0f};

  test.AddInput<float>("X", {N, C, H, W}, X);
  test.AddInput<float>("scales", {4}, scales);

  std::vector<float> Y = {1.0f, 2.66666651f};

  test.AddOutput<float>("Y", {N, C, (int64_t)(H * scales[2]), (int64_t)(W * scales[3])}, Y);
  test.Run();
}

TEST(ResizeOpTest, ResizeOpUpsampleNearestTest) {
  OpTester test("Resize", 10);
  std::vector<float> scales{1.0f, 1.0f, 2.0f, 3.0f};

  test.AddAttribute("mode", "nearest");

  const int64_t N = 1, C = 1, H = 2, W = 2;
  std::vector<float> X = {1.0f, 2.0f, 3.0f, 4.0f};

  test.AddInput<float>("X", {N, C, H, W}, X);
  test.AddInput<float>("scales", {4}, scales);

  std::vector<float> Y = {1.0f, 1.0f, 1.0f, 2.0f, 2.0f, 2.0f,
                          1.0f, 1.0f, 1.0f, 2.0f, 2.0f, 2.0f,
                          3.0f, 3.0f, 3.0f, 4.0f, 4.0f, 4.0f,
                          3.0f, 3.0f, 3.0f, 4.0f, 4.0f, 4.0f};

  test.AddOutput<float>("Y", {N, C, (int64_t)(H * scales[2]), (int64_t)(W * scales[3])}, Y);
  test.Run();
}

TEST(ResizeOpTest, ResizeOpNearestTest) {
  OpTester test("Resize", 10);
  std::vector<float> scales{1.0f, 1.0f, 0.6f, 0.6f};

  test.AddAttribute("mode", "nearest");

  const int64_t N = 1, C = 1, H = 2, W = 4;
  std::vector<float> X = {
      1.0f, 2.0f, 3.0f, 4.0f,
      5.0f, 6.0f, 7.0f, 8.0f};

  test.AddInput<float>("X", {N, C, H, W}, X);
  test.AddInput<float>("scales", {4}, scales);

  std::vector<float> Y = {1.0f, 3.0f};

  test.AddOutput<float>("Y", {N, C, (int64_t)(H * scales[2]), (int64_t)(W * scales[3])}, Y);
  test.Run();
}

TEST(ResizeOpTest, ResizeOpBilinearTest) {
  OpTester test("Resize", 10);
  std::vector<float> scales{1.0f, 1.0f, 0.5f, 0.5f};

  test.AddAttribute("mode", "linear");

  const int64_t N = 2, C = 1, H = 4, W = 8;
  std::vector<float> X = {
      1.0f, 1.5f, 2.0f, 2.5f, 3.0f, 3.0f, 3.0f, 3.0f,
      2.0f, 2.5f, 3.0f, 3.5f, 4.0f, 4.0f, 4.0f, 4.0f,
      3.0f, 3.5f, 4.0f, 4.5f, 5.0f, 5.0f, 5.0f, 5.0f,
      3.0f, 3.5f, 4.0f, 4.5f, 5.0f, 5.0f, 5.0f, 5.0f,

      3.0f, 3.5f, 4.0f, 4.5f, 5.0f, 5.0f, 5.0f, 5.0f,
      5.0f, 5.5f, 6.0f, 6.5f, 7.0f, 7.0f, 7.0f, 7.0f,
      7.0f, 7.5f, 8.0f, 8.5f, 9.0f, 9.0f, 9.0f, 9.0f,
      7.0f, 7.5f, 8.0f, 8.5f, 9.0f, 9.0f, 9.0f, 9.0f};

  test.AddInput<float>("X", {N, C, H, W}, X);
  test.AddInput<float>("scales", {4}, scales);

  std::vector<float> Y = {
      1.0f, 2.0f, 3.0f, 3.0f,
      3.0f, 4.0f, 5.0f, 5.0f,

      3.0f, 4.0f, 5.0f, 5.0f,
      7.0f, 8.0f, 9.0f, 9.0f};

  test.AddOutput<float>("Y", {N, C, (int64_t)(H * scales[2]), (int64_t)(W * scales[3])}, Y);
  test.Run();
}

}  // namespace test
}  // namespace onnxruntime
