// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"
#include "test/providers/provider_test_utils.h"
#include "contrib_ops/cpu/crop.h"
#include "core/util/math.h"

using namespace ONNX_NAMESPACE;
namespace onnxruntime {
namespace test {

TEST(TensorOpTest, CropBorderOnly) {
  const int N = 2, C = 1, H = 3, W = 4;
  std::vector<float> X = {1.0f, 2.0f, 3.0f, 4.0f,
                          2.0f, 3.0f, 4.0f, 5.0f,
                          3.0f, 4.0f, 5.0f, 6.0f,

                          4.0f, 5.0f, 6.0f, 7.0f,
                          5.0f, 6.0f, 7.0f, 8.0f,
                          6.0f, 7.0f, 8.0f, 9.0f};

  const std::vector<int64_t> border{0, 1, 2, 1};
  std::vector<float> output = {
      2.0f, 3.0f,

      5.0f, 6.0f};

  OpTester test("Crop");
  test.AddAttribute("border", border);
  test.AddInput<float>("input", {N, C, H, W}, X);
  test.AddOutput<float>("output", {N, C, (H - border[2] - border[0]), (W - border[3] - border[1])}, output);
  test.Run();
}

TEST(TensorOpTest, CropBorderAndScale) {
  const int N = 2, C = 1, H = 3, W = 4;
  std::vector<float> X = {1.0f, 2.0f, 3.0f, 4.0f,
                          2.0f, 3.0f, 4.0f, 5.0f,
                          3.0f, 4.0f, 5.0f, 6.0f,

                          4.0f, 5.0f, 6.0f, 7.0f,
                          5.0f, 6.0f, 7.0f, 8.0f,
                          6.0f, 7.0f, 8.0f, 9.0f};

  const std::vector<int64_t> border = {0, 0, 0, 0};
  const std::vector<int64_t> scale = {2, 2};

  std::vector<float> output = {
      1.0f, 2.0f,
      2.0f, 3.0f,

      4.0f, 5.0f,
      5.0f, 6.0f};

  OpTester test("Crop");
  test.AddAttribute("border", border);
  test.AddAttribute("scale", scale);
  test.AddInput<float>("input", {N, C, H, W}, X);
  test.AddOutput<float>("output", {N, C, scale[0], scale[1]}, output);
  test.Run();
}

TEST(TensorOpTest, ImageScalerTest) {
  const int64_t N = 1, C = 2, H = 2, W = 2;
  std::vector<float> X = {
      1.0f, 3.0f,
      3.0f, 5.0f,

      3.0f, 5.0f,
      7.0f, 9.0f};

  float scale = 2.0f;
  std::vector<float> bias = {1.0f, 2.0f};

  std::vector<float> result = {
      3.0f, 7.0f,
      7.0f, 11.0f,

      8.0f, 12.0f,
      16.0f, 20.0f};

  OpTester test("ImageScaler");
  test.AddAttribute("scale", scale);
  test.AddAttribute("bias", bias);
  test.AddInput<float>("input", {N, C, H, W}, X);
  test.AddOutput<float>("output", {N, C, H, W}, result);
  test.Run();
}

}  // namespace test
}  // namespace onnxruntime
