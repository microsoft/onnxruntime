// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cpu/tensor/resize.h"
#include "gtest/gtest.h"
#include "test/providers/provider_test_utils.h"

namespace onnxruntime {
namespace test {
TEST(ResizeOpTest, ResizeOpLineartDownSampleTest_tf_crop_and_resize) {
  OpTester test("Resize", 11);
  std::vector<float> roi{0.4f, 0.6f, 0.6f, 0.8f};
  std::vector<float> scales{};
  std::vector<int64_t> sizes{3, 3};

  test.AddAttribute("mode", "linear");
  test.AddAttribute("coordinate_transformation_mode", "tf_crop_and_resize");

  const int64_t H = 4, W = 4;

  std::vector<float> X = {
      1.0f, 2.0f, 3.0f, 4.0f,
      5.0f, 6.0f, 7.0f, 8.0f,
      9.0f, 10.0f, 11.0f, 12.0f,
      13.0f, 14.0f, 15.0f, 16.0f};

  test.AddInput<float>("X", {H, W}, X);
  test.AddInput<float>("roi", {4}, roi);
  test.AddInput<float>("scales", {0}, scales);
  test.AddInput<int64_t>("sizes", {2}, sizes);

  std::vector<float> Y = {7.600004f, 7.9f, 8.2f,
                          8.8f, 9.1f, 9.4f,
                          10.0f, 10.3f, 10.6f};

  test.AddOutput<float>("Y", {sizes[0], sizes[1]}, Y);
  test.Run();
}

TEST(ResizeOpTest, ResizeOpLinearDownSampleTest_tf_crop_and_resize_with_extrapolation) {
  OpTester test("Resize", 11);
  std::vector<float> scales{1.0f, 1.0f, 0.8f, 0.8f};
  std::vector<float> roi{0.0f, 0.0f, 0.4f, 0.6f, 1.0f, 1.0f, 1.2f, 1.7f};

  test.AddAttribute("mode", "linear");
  test.AddAttribute("coordinate_transformation_mode", "tf_crop_and_resize");
  test.AddAttribute("extrapolation_value", 10.0f);

  const int64_t N = 1, C = 1, H = 4, W = 4;
  std::vector<float> X = {
      1.0f, 2.0f, 3.0f, 4.0f,
      5.0f, 6.0f, 7.0f, 8.0f,
      9.0f, 10.0f, 11.0f, 12.0f,
      13.0f, 14.0f, 15.0f, 16.0f};

  test.AddInput<float>("X", {N, C, H, W}, X);
  test.AddInput<float>("roi", {8}, roi);
  test.AddInput<float>("scales", {4}, scales);

  std::vector<float> Y = {7.6f, 10.0f, 10.0f,
                          12.4f, 10.f, 10.0f,
                          10.0f, 10.0f, 10.0f};

  test.AddOutput<float>("Y", {N, C, static_cast<int64_t>(H * scales[2]), static_cast<int64_t>(W * scales[3])}, Y);
  test.Run();
}

TEST(ResizeOpTest, ResizeOpLineartDownSampleTest_4DBilinear) {
  OpTester test("Resize", 11);
  std::vector<float> roi{};  
  std::vector<float> scales{1.0f, 1.0f, 0.6f, 0.6f};

  test.AddAttribute("mode", "linear");

  const int64_t N = 1, C = 1, H = 2, W = 4;
  std::vector<float> X = {
      1.0f, 2.0f, 3.0f, 4.0f,
      5.0f, 6.0f, 7.0f, 8.0f};

  test.AddInput<float>("X", {N, C, H, W}, X);
  test.AddInput<float>("roi", {0}, roi);
  test.AddInput<float>("scales", {4}, scales);

  std::vector<float> Y = {2.66666651f, 4.3333331f};

  test.AddOutput<float>("Y", {N, C, static_cast<int64_t>(H * scales[2]), static_cast<int64_t>(W * scales[3])}, Y);
  test.Run();
}

TEST(ResizeOpTest, ResizeOpLineartDownSampleTest_4DBilinear_align_corners) {
  OpTester test("Resize", 11);
  std::vector<float> roi{};
  std::vector<float> scales{1.0f, 1.0f, 0.6f, 0.6f};

  test.AddAttribute("mode", "linear");
  test.AddAttribute("coordinate_transformation_mode", "align_corners");

  const int64_t N = 1, C = 1, H = 2, W = 4;
  std::vector<float> X = {
      1.0f, 2.0f, 3.0f, 4.0f,
      5.0f, 6.0f, 7.0f, 8.0f};

  test.AddInput<float>("X", {N, C, H, W}, X);
  test.AddInput<float>("roi", {0}, roi);
  test.AddInput<float>("scales", {4}, scales);

  std::vector<float> Y = {1.0f, 4.0f};

  test.AddOutput<float>("Y", {N, C, static_cast<int64_t>(H * scales[2]), static_cast<int64_t>(W * scales[3])}, Y);
  test.Run();
}

TEST(ResizeOpTest, ResizeOpLineartDownSampleTest_2DBilinear_pytorch_half_pixel) {
  OpTester test("Resize", 11);
  std::vector<float> roi{};
  std::vector<float> scales{};
  std::vector<int64_t> sizes{3, 1};

  test.AddAttribute("mode", "linear");
  test.AddAttribute("coordinate_transformation_mode", "pytorch_half_pixel");

  const int64_t H = 4, W = 4;

  std::vector<float> X = {
      1.0f, 2.0f, 3.0f, 4.0f,
      5.0f, 6.0f, 7.0f, 8.0f,
      9.0f, 10.0f, 11.0f, 12.0f,
      13.0f, 14.0f, 15.0f, 16.0f};
  

  test.AddInput<float>("X", {H, W}, X);
  test.AddInput<float>("roi", {0}, roi);
  test.AddInput<float>("scales", {0}, scales);
  test.AddInput<int64_t>("sizes", {2}, sizes);

  std::vector<float> Y = {1.6666666f, 7.0f, 12.333333f};

  test.AddOutput<float>("Y", {sizes[0], sizes[1]}, Y);
  test.Run();
}

TEST(ResizeOpTest, ResizeOpLineartUpSampleTest_4DBilinear_asymmetric) {
  OpTester test("Resize", 11);
  std::vector<float> roi{};
  std::vector<float> scales{1.0f, 1.0f, 2.0f, 4.0f};

  test.AddAttribute("mode", "linear");
  test.AddAttribute("coordinate_transformation_mode", "asymmetric");

  const int64_t N = 2, C = 1, H = 2, W = 2;
  std::vector<float> X = {1.0f, 3.0f,
                          4.0f, 8.0f,

                          6.0f, 2.0f,
                          7.0f, 11.0f};

  test.AddInput<float>("X", {N, C, H, W}, X);
  test.AddInput<float>("roi", {0}, roi);
  test.AddInput<float>("scales", {4}, scales);

  std::vector<float> Y = {
      1.0f, 1.5f, 2.0f, 2.5f, 3.0f, 3.0f, 3.0f, 3.0f,
      2.5f, 3.25f, 4.0f, 4.75f, 5.5f, 5.5f, 5.5f, 5.5f,
      4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 8.0f, 8.0f, 8.0f,
      4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 8.0f, 8.0f, 8.0f,

      6.0f, 5.0f, 4.0f, 3.0f, 2.0f, 2.0f, 2.0f, 2.0f,
      6.5f, 6.5f, 6.5f, 6.5f, 6.5f, 6.5f, 6.5f, 6.5f,
      7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 11.0f, 11.0f, 11.0f,
      7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 11.0f, 11.0f, 11.0f};

  test.AddOutput<float>("Y", {N, C, static_cast<int64_t>(H * scales[2]), static_cast<int64_t>(W * scales[3])}, Y);
  test.Run();
}

TEST(ResizeOpTest, ResizeOpLineartUpSampleTest_2DBilinear_align_corners) {
  OpTester test("Resize", 11);
  std::vector<float> roi{};
  std::vector<float> scales{2.0f, 4.0f};
  test.AddAttribute("mode", "linear");
  test.AddAttribute("coordinate_transformation_mode", "align_corners");

  const int64_t H = 2, W = 2;
  std::vector<float> X = {1.0f, 3.0f,
                          4.0f, 8.0f};

  test.AddInput<float>("X", {H, W}, X);
  test.AddInput<float>("roi", {0}, roi);
  test.AddInput<float>("scales", {2}, scales);

  std::vector<float> Y = {
      1.0f, 1.2857143f, 1.571428f, 1.857142f, 2.142857f, 2.4285715f, 2.714285f, 3.0f,
      2.0f, 2.3809524f, 2.761904f, 3.142857f, 3.523809f, 3.9047618f, 4.285714f, 4.66666f,
      3.0f, 3.4761906f, 3.952381f, 4.428571f, 4.904762f, 5.3809524f, 5.857143f, 6.33333f,
      4.0f, 4.5714290f, 5.142857f, 5.714286f, 6.285714f, 6.8571430f, 7.428571f, 8.0f};

  test.AddOutput<float>("Y", {static_cast<int64_t>(H * scales[0]), static_cast<int64_t>(W * scales[1])}, Y);
  test.Run();
}

TEST(ResizeOpTest, ResizeOpLineartScalesNoOpTest) {
  OpTester test("Resize", 11);
  std::vector<float> roi{};
  std::vector<float> scales{1.0f, 1.0f, 1.0f, 1.0f};
  test.AddAttribute("mode", "linear");

  const int64_t N = 2, C = 1, H = 2, W = 2;
  std::vector<float> X = {1.0f, 3.0f,
                          4.0f, 8.0f,

                          6.0f, 2.0f,
                          7.0f, 11.0f};

  test.AddInput<float>("X", {N, C, H, W}, X);
  test.AddInput<float>("roi", {0}, roi);
  test.AddInput<float>("scales", {4}, scales);

  std::vector<float> Y = {1.0f, 3.0f,
                          4.0f, 8.0f,

                          6.0f, 2.0f,
                          7.0f, 11.0f};

  test.AddOutput<float>("Y", {N, C, H, W}, Y);
  test.Run();
}

TEST(ResizeOpTest, ResizeOpNearestDownSampleTest) {
  OpTester test("Resize", 11);
  std::vector<float> scales{1.0f, 1.0f, 0.6f, 0.6f};
  std::vector<float> roi{};
 
  test.AddAttribute("mode", "nearest");

  const int64_t N = 1, C = 1, H = 2, W = 4;
  std::vector<float> X = {
      1.0f, 2.0f, 3.0f, 4.0f,
      5.0f, 6.0f, 7.0f, 8.0f};

  test.AddInput<float>("X", {N, C, H, W}, X);
  test.AddInput<float>("roi", {0}, roi);
  test.AddInput<float>("scales", {4}, scales);

  std::vector<float> Y = {1.0f, 3.0f};

  test.AddOutput<float>("Y", {N, C, static_cast<int64_t>(H * scales[2]), static_cast<int64_t>(W * scales[3])}, Y);
  test.Run();
}

TEST(ResizeOpTest, ResizeOpNearestDownSampleTest_WithSizes) {
  OpTester test("Resize", 11);
  std::vector<float> scales{};
  std::vector<float> roi{};
  std::vector<int64_t> sizes{1, 1, 1, 3};

  test.AddAttribute("mode", "nearest");

  const int64_t N = 1, C = 1, H = 2, W = 4;
  std::vector<float> X = {
      1.0f, 2.0f, 3.0f, 4.0f,
      5.0f, 6.0f, 7.0f, 8.0f};

  test.AddInput<float>("X", {N, C, H, W}, X);
  test.AddInput<float>("roi", {0}, roi);
  test.AddInput<float>("scales", {0}, scales);
  test.AddInput<int64_t>("sizes", {4}, sizes);

  std::vector<float> Y = {1.0f, 2.0f, 4.0f};

  test.AddOutput<float>("Y", {N, C, sizes[2], sizes[3]}, Y);
  test.Run();
}

TEST(ResizeOpTest, ResizeOpNearestDownSampleTest_tf_half_pixel) {
  OpTester test("Resize", 11);
  std::vector<float> scales{};
  std::vector<float> roi{};
  std::vector<int64_t> sizes{1, 1, 3, 2};

  test.AddAttribute("coordinate_transformation_mode", "tf_half_pixel_for_nn");
  test.AddAttribute("mode", "nearest");

  const int64_t N = 1, C = 1, H = 4, W = 4;
  std::vector<float> X = {
      1.0f, 2.0f, 3.0f, 4.0f,
      5.0f, 6.0f, 7.0f, 8.0f,
      9.0f, 10.0f, 11.0f, 12.0f,
      13.0f, 14.0f, 15.0f, 16.0f
};

  test.AddInput<float>("X", {N, C, H, W}, X);
  test.AddInput<float>("roi", {0}, roi);
  test.AddInput<float>("scales", {0}, scales);
  test.AddInput<int64_t>("sizes", {4}, sizes);

  std::vector<float> Y = {6.0f, 8.0f,
                          10.0f, 12.0f,
                          14.0f, 16.0f};

  test.AddOutput<float>("Y", {N, C, sizes[2], sizes[3]}, Y);
  test.Run();
}

TEST(ResizeOpTest, ResizeOpNearestDownSampleTest_tf_crop_and_resize_with_extrapolation) {
  OpTester test("Resize", 11);
  std::vector<float> scales{1.0f, 1.0f, 0.8f, 0.8f};
  std::vector<float> roi{0.0f, 0.0f, 0.4f, 0.6f, 1.0f, 1.0f, 1.2f, 1.7f};

  test.AddAttribute("mode", "nearest");
  test.AddAttribute("coordinate_transformation_mode", "tf_crop_and_resize");
  test.AddAttribute("extrapolation_value", 10.0f);

  const int64_t N = 1, C = 1, H = 4, W = 4;
  std::vector<float> X = {
      1.0f, 2.0f, 3.0f, 4.0f,
      5.0f, 6.0f, 7.0f, 8.0f,
      9.0f, 10.0f, 11.0f, 12.0f,
      13.0f, 14.0f, 15.0f, 16.0f};

  test.AddInput<float>("X", {N, C, H, W}, X);
  test.AddInput<float>("roi", {8}, roi);
  test.AddInput<float>("scales", {4}, scales);

  std::vector<float> Y = {7.0f, 10.0f, 10.0f,
                          11.0f, 10.f, 10.0f,
                          10.0f, 10.0f, 10.0f};

  test.AddOutput<float>("Y", {N, C, static_cast<int64_t>(H * scales[2]), static_cast<int64_t>(W * scales[3])}, Y);
  test.Run();
}

TEST(ResizeOpTest, ResizeOpNearestUpSampleTest) {
  OpTester test("Resize", 11);
  std::vector<float> roi{};
  std::vector<float> scales{1.0f, 1.0f, 2.0f, 3.0f};

  test.AddAttribute("mode", "nearest");

  const int64_t N = 1, C = 1, H = 2, W = 2;
  std::vector<float> X = {1.0f, 2.0f, 3.0f, 4.0f};

  test.AddInput<float>("X", {N, C, H, W}, X);
  test.AddInput<float>("roi", {0}, roi);
  test.AddInput<float>("scales", {4}, scales);

  std::vector<float> Y = {1.0f, 1.0f, 1.0f, 2.0f, 2.0f, 2.0f,
                          1.0f, 1.0f, 1.0f, 2.0f, 2.0f, 2.0f,
                          3.0f, 3.0f, 3.0f, 4.0f, 4.0f, 4.0f,
                          3.0f, 3.0f, 3.0f, 4.0f, 4.0f, 4.0f};

  test.AddOutput<float>("Y", {N, C, static_cast<int64_t>(H * scales[2]), static_cast<int64_t>(W * scales[3])}, Y);
  test.Run();
}

TEST(ResizeOpTest, ResizeOpNearestUpSampleTest_WithSizes_CeilMode) {
  OpTester test("Resize", 11);
  std::vector<float> roi{};
  std::vector<float> scales{};
  std::vector<int64_t> sizes{1, 1, 7, 8};

  test.AddAttribute("mode", "nearest");
  test.AddAttribute("nearest_mode", "ceil");

  const int64_t N = 1, C = 1, H = 2, W = 2;
  std::vector<float> X = {1.0f, 2.0f, 3.0f, 4.0f};

  test.AddInput<float>("X", {N, C, H, W}, X);
  test.AddInput<float>("roi", {0}, roi);
  test.AddInput<float>("scales", {0}, scales);
  test.AddInput<int64_t>("sizes", {4}, sizes);

  std::vector<float> Y = {1.0f, 1.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f,
                          1.0f, 1.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f,
                          3.0f, 3.0f, 4.0f, 4.0f, 4.0f, 4.0f, 4.0f, 4.0f,
                          3.0f, 3.0f, 4.0f, 4.0f, 4.0f, 4.0f, 4.0f, 4.0f,
                          3.0f, 3.0f, 4.0f, 4.0f, 4.0f, 4.0f, 4.0f, 4.0f,
                          3.0f, 3.0f, 4.0f, 4.0f, 4.0f, 4.0f, 4.0f, 4.0f,
                          3.0f, 3.0f, 4.0f, 4.0f, 4.0f, 4.0f, 4.0f, 4.0f};
  
  test.AddOutput<float>("Y", {N, C, sizes[2], sizes[3]}, Y);
  test.Run();
}

TEST(ResizeOpTest, ResizeOpNearestUpSample_Floor_Align_Corners) {
  OpTester test("Resize", 11);

  std::vector<float> roi{};
  std::vector<float> scales{1.0f, 1.0f, 2.0f, 2.0f};

  test.AddAttribute("mode", "nearest");
  test.AddAttribute("coordinate_transformation_mode", "align_corners");
  test.AddAttribute("nearest_mode", "floor");

  const int64_t N = 1, C = 1, H = 4, W = 4;
  std::vector<float> X = {
      1.0f, 2.0f, 3.0f, 4.0f,
      5.0f, 6.0f, 7.0f, 8.0f,
      9.0f, 10.0f, 11.0f, 12.0f,
      13.0f, 14.0f, 15.0f, 16.0f};

  test.AddInput<float>("X", {N, C, H, W}, X);
  test.AddInput<float>("roi", {0}, roi);
  test.AddInput<float>("scales", {4}, scales);

  std::vector<float> Y = {1.0f, 1.0f, 1.0f, 2.0f, 2.0f, 3.0f, 3.0f, 4.0f,
                          1.0f, 1.0f, 1.0f, 2.0f, 2.0f, 3.0f, 3.0f, 4.0f,
                          1.0f, 1.0f, 1.0f, 2.0f, 2.0f, 3.0f, 3.0f, 4.0f,
                          5.0f, 5.0f, 5.0f, 6.0f, 6.0f, 7.0f, 7.0f, 8.0f,
                          5.0f, 5.0f, 5.0f, 6.0f, 6.0f, 7.0f, 7.0f, 8.0f,
                          9.0f, 9.0f, 9.0f, 10.0f, 10.0f, 11.0f, 11.0f, 12.0f,
                          9.0f, 9.0f, 9.0f, 10.0f, 10.0f, 11.0f, 11.0f, 12.0f,
                          13.0f, 13.0f, 13.0f, 14.0f, 14.0f, 15.0f, 15.0f, 16.0f};

  test.AddOutput<float>("Y", {N, C, static_cast<int64_t>(H * scales[2]), static_cast<int64_t>(W * scales[3])}, Y);
  test.Run();
}

TEST(ResizeOpTest, ResizeOpCubicDownSampleTest) {
  OpTester test("Resize", 11);
  std::vector<float> scales{1.0f, 1.0f, 0.8f, 0.8f};
  std::vector<float> roi{};

  test.AddAttribute("mode", "cubic");  

  const int64_t N = 1, C = 1, H = 4, W = 4;
  std::vector<float> X = {
      1.0f, 2.0f, 3.0f, 4.0f,
      5.0f, 6.0f, 7.0f, 8.0f,
      9.0f, 10.0f, 11.0f, 12.0f,
      13.0f, 14.0f, 15.0f, 16.0f};

  test.AddInput<float>("X", {N, C, H, W}, X);
  test.AddInput<float>("roi", {0}, roi);
  test.AddInput<float>("scales", {4}, scales);

  std::vector<float> Y = {1.47119f, 2.78125f, 4.08252f,
                          6.71143f, 8.02148f, 9.32275f,
                          11.9165f, 13.2266f, 14.5278f};

  test.AddOutput<float>("Y", {N, C, static_cast<int64_t>(H * scales[2]), static_cast<int64_t>(W * scales[3])}, Y);
  test.Run();
}

TEST(ResizeOpTest, ResizeOpLineartDownSampleTest_exclude_outside) {
  OpTester test("Resize", 11);
  std::vector<float> roi{};
  std::vector<float> scales{0.8f, 0.8f};
  std::vector<int64_t> sizes{};

  test.AddAttribute("mode", "cubic");
  test.AddAttribute("exclude_outside", static_cast<int64_t>(1));
  test.AddAttribute("cubic_coeff_a", -0.5f);

  const int64_t H = 4, W = 4;

  std::vector<float> X = {
      1.0f, 2.0f, 3.0f, 4.0f,
      5.0f, 6.0f, 7.0f, 8.0f,
      9.0f, 10.0f, 11.0f, 12.0f,
      13.0f, 14.0f, 15.0f, 16.0f};

  test.AddInput<float>("X", {H, W}, X);
  test.AddInput<float>("roi", {0}, roi);
  test.AddInput<float>("scales", {2}, scales);

  std::vector<float> Y = {1.36813f, 2.6695f, 4.01334f,
                          6.57363f, 7.875f, 9.21884f,
                          11.949f, 13.2503f, 14.5942f};

  test.AddOutput<float>("Y", {static_cast<int64_t>(H * scales[0]), static_cast<int64_t>(W * scales[1])}, Y);
  test.Run();
}

TEST(ResizeOpTest, ResizeOpCubicDownSampleTest_coeff) {
  OpTester test("Resize", 11);
  std::vector<float> scales{1.0f, 1.0f, 0.8f, 0.8f};
  std::vector<float> roi{};

  test.AddAttribute("mode", "cubic");
  test.AddAttribute("cubic_coeff_a", -0.5f);

  const int64_t N = 1, C = 1, H = 4, W = 4;
  std::vector<float> X = {
      1.0f, 2.0f, 3.0f, 4.0f,
      5.0f, 6.0f, 7.0f, 8.0f,
      9.0f, 10.0f, 11.0f, 12.0f,
      13.0f, 14.0f, 15.0f, 16.0f};

  test.AddInput<float>("X", {N, C, H, W}, X);
  test.AddInput<float>("roi", {0}, roi);
  test.AddInput<float>("scales", {4}, scales);

  std::vector<float> Y = {1.38574f, 2.68359f, 4.00684f,
                          6.57715f, 7.875f, 9.19824f,
                          11.8701f, 13.168f, 14.4912f,};

  test.AddOutput<float>("Y", {N, C, static_cast<int64_t>(H * scales[2]), static_cast<int64_t>(W * scales[3])}, Y);
  test.Run();
}

TEST(ResizeOpTest, ResizeOpCubicDownSampleTest_with_roi) {
  OpTester test("Resize", 11);
  std::vector<float> scales{1.0f, 1.0f, 0.8f, 0.8f};
  std::vector<float> roi{0.0f, 0.0f, 0.4f, 0.6f, 1.0f, 1.0f, 0.6f, 0.8f};

  test.AddAttribute("mode", "cubic");
  test.AddAttribute("coordinate_transformation_mode", "tf_crop_and_resize");

  const int64_t N = 1, C = 1, H = 4, W = 4;
  std::vector<float> X = {
      1.0f, 2.0f, 3.0f, 4.0f,
      5.0f, 6.0f, 7.0f, 8.0f,
      9.0f, 10.0f, 11.0f, 12.0f,
      13.0f, 14.0f, 15.0f, 16.0f};

  test.AddInput<float>("X", {N, C, H, W}, X);
  test.AddInput<float>("roi", {8}, roi);
  test.AddInput<float>("scales", {4}, scales);

  std::vector<float> Y = {7.744f, 8.13475f, 8.488f,
                          8.752f, 9.14275f, 9.496f,
                          9.76f, 10.1507f, 10.504f};

  test.AddOutput<float>("Y", {N, C, static_cast<int64_t>(H * scales[2]), static_cast<int64_t>(W * scales[3])}, Y);
  test.Run();
}

TEST(ResizeOpTest, ResizeOpCubicDownSampleTest_asymmetric) {
  OpTester test("Resize", 11);
  std::vector<float> scales{1.0f, 1.0f, 0.8f, 0.8f};
  std::vector<float> roi{};

  test.AddAttribute("mode", "cubic");
  test.AddAttribute("coordinate_transformation_mode", "asymmetric");

  const int64_t N = 1, C = 1, H = 4, W = 4;
  std::vector<float> X = {
      1.0f, 2.0f, 3.0f, 4.0f,
      5.0f, 6.0f, 7.0f, 8.0f,
      9.0f, 10.0f, 11.0f, 12.0f,
      13.0f, 14.0f, 15.0f, 16.0f};

  test.AddInput<float>("X", {N, C, H, W}, X);
  test.AddInput<float>("roi", {0}, roi);
  test.AddInput<float>("scales", {4}, scales);

  std::vector<float> Y = {1.0f, 2.29688f, 3.59375f,
                          6.1875f, 7.48438f, 8.78125f,
                          11.375f, 12.6719f, 13.9688f};

  test.AddOutput<float>("Y", {N, C, static_cast<int64_t>(H * scales[2]), static_cast<int64_t>(W * scales[3])}, Y);
  test.Run();
}

TEST(ResizeOpTest, ResizeOpCubicUpSampleTest) {
  OpTester test("Resize", 11);
  std::vector<float> scales{1.0f, 1.0f, 2.0f, 2.0f};
  std::vector<float> roi{};

  test.AddAttribute("mode", "cubic");
  test.AddAttribute("coordinate_transformation_mode", "asymmetric");

  const int64_t N = 1, C = 1, H = 4, W = 4;
  std::vector<float> X = {
      1.0f, 2.0f, 3.0f, 4.0f,
      5.0f, 6.0f, 7.0f, 8.0f,
      9.0f, 10.0f, 11.0f, 12.0f,
      13.0f, 14.0f, 15.0f, 16.0f};

  test.AddInput<float>("X", {N, C, H, W}, X);
  test.AddInput<float>("roi", {0}, roi);
  test.AddInput<float>("scales", {4}, scales);

  std::vector<float> Y = {1.0f, 1.40625f, 2.0f, 2.5f, 3.0f, 3.59375f, 4.0f, 4.09375f,
                          2.625f, 3.03125f, 3.625f, 4.125f, 4.625f, 5.21875f, 5.625f, 5.71875f,
                          5.0f, 5.40625f, 6.0f, 6.5f, 7.0f, 7.59375f, 8.0f, 8.09375f,
                          7.0f, 7.40625f, 8.0f, 8.5f, 9.0f, 9.59375f, 10.0f, 10.0938f,
                          9.0f, 9.40625f, 10.0f, 10.5f, 11.0f, 11.5938f, 12.0f, 12.0938f,
                          11.375f, 11.7813f, 12.375f, 12.875f, 13.375f, 13.9688f, 14.375f, 14.4688f,
                          13.0f, 13.4063f, 14.0f, 14.5f, 15.0f, 15.5938f, 16.0f, 16.0938f,
                          13.375f, 13.7813f, 14.375f, 14.875f, 15.375f, 15.9688f, 16.375f, 16.4688f,};

  test.AddOutput<float>("Y", {N, C, static_cast<int64_t>(H * scales[2]), static_cast<int64_t>(W * scales[3])}, Y);
  test.Run();
}

TEST(ResizeOpTest, ResizeOpCubicUpSampleTest_MultiChannel) {
  OpTester test("Resize", 11);
  std::vector<float> scales{};
  std::vector<int64_t> sizes{1, 2, 9, 9};
  std::vector<float> roi{};

  test.AddAttribute("mode", "cubic");

  const int64_t N = 1, C = 2, H = 4, W = 4;
  std::vector<float> X = {
      0.0f, 1.0f, 2.0f, 3.0f,
      4.0f, 5.0f, 6.0f, 7.0f,
      8.0f, 9.0f, 10.0f, 11.0f,
      12.0f, 13.0f, 14.0f, 15.0f,

      16.0f, 17.0f, 18.0f, 19.0f,
      20.0f, 21.0f, 22.0f, 23.0f,
      24.0f, 25.0f, 26.0f, 27.0f,
      28.0f, 29.0f, 30.0f, 31.0f,
  };

  test.AddInput<float>("X", {N, C, H, W}, X);
  test.AddInput<float>("roi", {0}, roi);
  test.AddInput<float>("scales", {0}, scales);
  test.AddInput<int64_t>("sizes", {4}, sizes);

  std::vector<float> Y = {-0.543341f, -0.308515f, 0.0807175f, 0.644203f, 1.06533f, 1.48645f, 2.04994f, 2.43917f, 2.674f,
                           0.395961f, 0.630787f, 1.02002f, 1.5835f, 2.00463f, 2.42575f, 2.98924f, 3.37847f, 3.6133f,
                           1.95289f, 2.18772f, 2.57695f, 3.14043f, 3.56156f, 3.98268f, 4.54617f, 4.9354f, 5.17023f,
                           4.20683f, 4.44166f, 4.83089f, 5.39437f, 5.8155f, 6.23662f, 6.80011f, 7.18934f, 7.42417f,
                           5.89133f, 6.12616f, 6.51539f, 7.07887f, 7.5f, 7.92112f, 8.48461f, 8.87384f, 9.10867f,
                           7.57583f, 7.81066f, 8.19989f, 8.76337f, 9.1845f, 9.60562f, 10.1691f, 10.5583f, 10.7932f,
                           9.82977f, 10.0646f, 10.4538f, 11.0173f, 11.4384f, 11.8596f, 12.423f, 12.8123f, 13.0471f,
                           11.3867f, 11.6215f, 12.0108f, 12.5742f, 12.9954f, 13.4165f, 13.98f, 14.3692f, 14.604f,
                           12.326f, 12.5608f, 12.9501f, 13.5135f, 13.9347f, 14.3558f, 14.9193f, 15.3085f, 15.5433f,

                           15.4567f, 15.6915f, 16.0807f, 16.6442f, 17.0653f, 17.4865f, 18.0499f, 18.4392f, 18.674f,
                           16.396f, 16.6308f, 17.02f, 17.5835f, 18.0046f, 18.4258f, 18.9892f, 19.3785f, 19.6133f,
                           17.9529f, 18.1877f, 18.5769f, 19.1404f, 19.5616f, 19.9827f, 20.5462f, 20.9354f, 21.1702f,
                           20.2068f, 20.4417f, 20.8309f, 21.3944f, 21.8155f, 22.2366f, 22.8001f, 23.1893f, 23.4242f,
                           21.8913f, 22.1262f, 22.5154f, 23.0789f, 23.5f, 23.9211f, 24.4846f, 24.8738f, 25.1087f,
                           23.5758f, 23.8107f, 24.1999f, 24.7634f, 25.1845f, 25.6056f, 26.1691f, 26.5583f, 26.7932f,
                           25.8298f, 26.0646f, 26.4538f, 27.0173f, 27.4384f, 27.8596f, 28.423f, 28.8123f, 29.0471f,
                           27.3867f, 27.6215f, 28.0108f, 28.5742f, 28.9954f, 29.4165f, 29.98f, 30.3692f, 30.604f,
                           28.326f, 28.5608f, 28.9501f, 29.5135f, 29.9347f, 30.3558f, 30.9193f, 31.3085f, 31.5433f,};

  test.AddOutput<float>("Y", {N, C, sizes[2], sizes[3]}, Y);
  test.Run();
}
TEST(ResizeOpTest, ResizeOpCubicUpSampleTest_tf_half_pixel_for_nn) {
  OpTester test("Resize", 11);
  std::vector<float> scales{1.0f, 1.0f, 2.0f, 2.0f};
  std::vector<float> roi{};

  test.AddAttribute("mode", "cubic");
  test.AddAttribute("coordinate_transformation_mode", "tf_half_pixel_for_nn");

  const int64_t N = 1, C = 1, H = 4, W = 4;
  std::vector<float> X = {
      1.0f, 2.0f, 3.0f, 4.0f,
      5.0f, 6.0f, 7.0f, 8.0f,
      9.0f, 10.0f, 11.0f, 12.0f,
      13.0f, 14.0f, 15.0f, 16.0f};

  test.AddInput<float>("X", {N, C, H, W}, X);
  test.AddInput<float>("roi", {0}, roi);
  test.AddInput<float>("scales", {4}, scales);

  std::vector<float> Y = {1.95703f, 2.43359f, 3.0625f, 3.46875f, 4.09766f, 4.57422f, 4.87109f, 4.80078f,
                          3.86328f, 4.33984f, 4.96875f, 5.375f, 6.00391f, 6.48047f, 6.77734f, 6.70703f,
                          6.37891f, 6.85547f, 7.48438f, 7.89063f, 8.51953f, 8.99609f, 9.29297f, 9.22266f,
                          8.00391f, 8.48047f, 9.10938f, 9.51563f, 10.1445f, 10.6211f, 10.918f, 10.8477f,
                          10.5195f, 10.9961f, 11.625f, 12.0313f, 12.6602f, 13.1367f, 13.4336f, 13.3633f,
                          12.4258f, 12.9023f, 13.5313f, 13.9375f, 14.5664f, 15.043f, 15.3398f, 15.2695f,
                          13.6133f, 14.0898f, 14.7188f, 15.125f, 15.7539f, 16.2305f, 16.5273f, 16.457f,
                          13.332f, 13.8086f, 14.4375f, 14.8438f, 15.4727f, 15.9492f, 16.2461f, 16.1758f,};

  test.AddOutput<float>("Y", {N, C, static_cast<int64_t>(H * scales[2]), static_cast<int64_t>(W * scales[3])}, Y);
  test.Run();
}

TEST(ResizeOpTest, ResizeOpLineartDownSampleTest_4DBilinear_Ver10) {
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

  test.AddOutput<float>("Y", {N, C, static_cast<int64_t>(H * scales[2]), static_cast<int64_t>(W * scales[3])}, Y);
  test.Run();
}

TEST(ResizeOpTest, ResizeOpLineartDownSampleTest_2DBilinear_Ver10) {
  OpTester test("Resize", 10);
  std::vector<float> scales{0.6f, 0.6f};

  test.AddAttribute("mode", "linear");

  const int64_t H = 2, W = 4;
  std::vector<float> X = {
      1.0f, 2.0f, 3.0f, 4.0f,
      5.0f, 6.0f, 7.0f, 8.0f};

  test.AddInput<float>("X", {H, W}, X);
  test.AddInput<float>("scales", {2}, scales);

  std::vector<float> Y = {1.0f, 2.66666651f};

  test.AddOutput<float>("Y", {static_cast<int64_t>(H * scales[0]), static_cast<int64_t>(W * scales[1])}, Y);
  test.Run();
}

TEST(ResizeOpTest, ResizeOpLineartUpSampleTest_4DBilinear_Ver10) {
  OpTester test("Resize", 10);
  std::vector<float> scales{1.0f, 1.0f, 2.0f, 4.0f};
  test.AddAttribute("mode", "linear");

  const int64_t N = 2, C = 1, H = 2, W = 2;
  std::vector<float> X = {1.0f, 3.0f,
                          4.0f, 8.0f,

                          6.0f, 2.0f,
                          7.0f, 11.0f};

  test.AddInput<float>("X", {N, C, H, W}, X);
  test.AddInput<float>("scales", {4}, scales);

  std::vector<float> Y = {
      1.0f, 1.5f, 2.0f, 2.5f, 3.0f, 3.0f, 3.0f, 3.0f,
      2.5f, 3.25f, 4.0f, 4.75f, 5.5f, 5.5f, 5.5f, 5.5f,
      4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 8.0f, 8.0f, 8.0f,
      4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 8.0f, 8.0f, 8.0f,

      6.0f, 5.0f, 4.0f, 3.0f, 2.0f, 2.0f, 2.0f, 2.0f,
      6.5f, 6.5f, 6.5f, 6.5f, 6.5f, 6.5f, 6.5f, 6.5f,
      7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 11.0f, 11.0f, 11.0f,
      7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 11.0f, 11.0f, 11.0f};

  test.AddOutput<float>("Y", {N, C, static_cast<int64_t>(H * scales[2]), static_cast<int64_t>(W * scales[3])}, Y);
  test.Run();
}

TEST(ResizeOpTest, ResizeOpLineartUpSampleTest_2DBilinear_Ver10) {
  OpTester test("Resize", 10);
  std::vector<float> scales{2.0f, 4.0f};
  test.AddAttribute("mode", "linear");

  const int64_t H = 2, W = 2;
  std::vector<float> X = {1.0f, 3.0f,
                          4.0f, 8.0f};

  test.AddInput<float>("X", {H, W}, X);
  test.AddInput<float>("scales", {2}, scales);

  std::vector<float> Y = {
      1.0f, 1.5f, 2.0f, 2.5f, 3.0f, 3.0f, 3.0f, 3.0f,
      2.5f, 3.25f, 4.0f, 4.75f, 5.5f, 5.5f, 5.5f, 5.5f,
      4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 8.0f, 8.0f, 8.0f,
      4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 8.0f, 8.0f, 8.0f};

  test.AddOutput<float>("Y", {static_cast<int64_t>(H * scales[0]), static_cast<int64_t>(W * scales[1])}, Y);
  test.Run();
}

TEST(ResizeOpTest, ResizeOpLineartScalesNoOpTest_Ver10) {
  OpTester test("Resize", 10);
  std::vector<float> scales{1.0f, 1.0f, 1.0f, 1.0f};
  test.AddAttribute("mode", "linear");

  const int64_t N = 2, C = 1, H = 2, W = 2;
  std::vector<float> X = {1.0f, 3.0f,
                          4.0f, 8.0f,

                          6.0f, 2.0f,
                          7.0f, 11.0f};

  test.AddInput<float>("X", {N, C, H, W}, X);
  test.AddInput<float>("scales", {4}, scales);

  std::vector<float> Y = {1.0f, 3.0f,
                          4.0f, 8.0f,

                          6.0f, 2.0f,
                          7.0f, 11.0f};

  test.AddOutput<float>("Y", {N, C, H, W}, Y);
  test.Run();
}

TEST(ResizeOpTest, ResizeOpNearestDownSampleTest_Ver10) {
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

  test.AddOutput<float>("Y", {N, C, static_cast<int64_t>(H * scales[2]), static_cast<int64_t>(W * scales[3])}, Y);
  test.Run();
}

TEST(ResizeOpTest, ResizeOpNearestUpSampleTest_Ver10) {
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

  test.AddOutput<float>("Y", {N, C, static_cast<int64_t>(H * scales[2]), static_cast<int64_t>(W * scales[3])}, Y);
  test.Run();
}

TEST(UpsampleOpTest, ResizeOpNearestNoScaleTest_Ver10) {
  OpTester test("Resize", 10);
  std::vector<float> scales{1.0f, 1.0f, 1.0f, 1.0f};

  test.AddAttribute("mode", "nearest");

  const int64_t N = 1, C = 1, H = 2, W = 2;
  std::vector<float> X = {1.0f, 2.0f, 3.0f, 4.0f};

  test.AddInput<float>("X", {N, C, H, W}, X);
  test.AddInput<float>("scales", {4}, scales);

  std::vector<float> Y = {1.0f, 2.0f, 3.0f, 4.0f};

  test.AddOutput<float>("Y", {N, C, H, W}, Y);
  test.Run();
}

}  // namespace test
}  // namespace onnxruntime
