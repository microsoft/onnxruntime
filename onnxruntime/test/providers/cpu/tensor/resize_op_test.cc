// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <exception>
#include "gtest/gtest.h"
#include "test/providers/provider_test_utils.h"
#include "test/util/include/default_providers.h"

namespace onnxruntime {
namespace test {
TEST(ResizeOpTest, ResizeOpLinearDownSampleTest_tf_crop_and_resize) {
  // TODO: Unskip when fixed #41968513
  if (DefaultDmlExecutionProvider().get() != nullptr) {
    GTEST_SKIP() << "Skipping because of the following error: The difference between expected[i] and output[i] is 0.20000028610229492, which exceeds threshold";
  }

  OpTester test("Resize", 13);
  std::vector<float> roi{0.4f, 0.6f, 0.6f, 0.8f};
  std::vector<float> scales{};
  std::vector<int64_t> sizes{3, 3};

  test.AddAttribute("mode", "linear");
  test.AddAttribute("coordinate_transformation_mode", "tf_crop_and_resize");

  constexpr int64_t H = 4, W = 4;

  std::vector<float> X = {
      1.0f, 2.0f, 3.0f, 4.0f,
      5.0f, 6.0f, 7.0f, 8.0f,
      9.0f, 10.0f, 11.0f, 12.0f,
      13.0f, 14.0f, 15.0f, 16.0f};

  test.AddInput<float>("X", {H, W}, X);
  test.AddInput<float>("roi", {4}, roi);
  test.AddInput<float>("", {0}, scales);  // opset13 requires either 'sizes' or 'scales' must be provided, but not both of them
  test.AddInput<int64_t>("sizes", {2}, sizes);

  std::vector<float> Y = {7.600004f, 7.9f, 8.2f,
                          8.8f, 9.1f, 9.4f,
                          10.0f, 10.3f, 10.6f};

  test.AddOutput<float>("Y", {sizes[0], sizes[1]}, Y);
  test.Run();
}

TEST(ResizeOpTest, ResizeOpLinearDownSampleTest_tf_crop_and_resize_with_extrapolation) {
  OpTester test("Resize", 13);
  std::vector<float> scales{1.0f, 1.0f, 0.8f, 0.8f};
  std::vector<float> roi{0.0f, 0.0f, 0.4f, 0.6f, 1.0f, 1.0f, 1.2f, 1.7f};

  test.AddAttribute("mode", "linear");
  test.AddAttribute("coordinate_transformation_mode", "tf_crop_and_resize");
  test.AddAttribute("extrapolation_value", 10.0f);

  constexpr int64_t N = 1, C = 1, H = 4, W = 4;
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

TEST(ResizeOpTest, NhwcResizeOpLinearDownSampleTest_tf_crop_and_resize_with_extrapolation) {
  OpTester test("Resize", 13);
  std::vector<float> scales{1.0f, 0.8f, 0.8f, 1.0f};
  std::vector<float> roi{0.0f, 0.4f, 0.6f, 0.0f, 1.0f, 1.2f, 1.7f, 1.0f};

  test.AddAttribute("mode", "linear");
  test.AddAttribute("coordinate_transformation_mode", "tf_crop_and_resize");
  test.AddAttribute("extrapolation_value", 10.0f);

  constexpr int64_t N = 1, H = 4, W = 4, C = 1;
  std::vector<float> X = {
      1.0f, 2.0f, 3.0f, 4.0f,
      5.0f, 6.0f, 7.0f, 8.0f,
      9.0f, 10.0f, 11.0f, 12.0f,
      13.0f, 14.0f, 15.0f, 16.0f};

  test.AddInput<float>("X", {N, H, W, C}, X);
  test.AddInput<float>("roi", {8}, roi);
  test.AddInput<float>("scales", {4}, scales);

  std::vector<float> Y = {7.6000004f, 10.0f, 10.0f,
                          12.400001f, 10.f, 10.0f,
                          10.0f, 10.0f, 10.0f};

  test.AddOutput<float>("Y", {N, static_cast<int64_t>(H * scales[1]), static_cast<int64_t>(W * scales[2]), C}, Y);
  // CUDA: result mismatch due to not implementing NHWC support
  // TensorRT: results mismatch
  // ROCm: results mismatch
  test.Run(OpTester::ExpectResult::kExpectSuccess, "",
           {kCudaExecutionProvider, kTensorrtExecutionProvider, kRocmExecutionProvider});
}

TEST(ResizeOpTest, NhwcResizeOpLinearDownSampleTest_tf_crop_and_resize_with_extrapolation_uint8) {
  OpTester test("Resize", 13);
  std::vector<float> scales{1.0f, 0.8f, 0.8f, 1.0f};
  std::vector<float> roi{0.0f, 0.4f, 0.6f, 0.0f, 1.0f, 1.2f, 1.7f, 1.0f};

  test.AddAttribute("mode", "linear");
  test.AddAttribute("coordinate_transformation_mode", "tf_crop_and_resize");
  test.AddAttribute("extrapolation_value", 10.0f);

  constexpr int64_t N = 1, H = 4, W = 4, C = 1;
  std::vector<uint8_t> X = {
      1, 2, 3, 4,
      5, 6, 7, 8,
      9, 10, 11, 12,
      13, 14, 15, 16};

  test.AddInput<uint8_t>("X", {N, H, W, C}, X);
  test.AddInput<float>("roi", {8}, roi);
  test.AddInput<float>("scales", {4}, scales);

  std::vector<uint8_t> Y = {7, 10, 10,
                            12, 10, 10,
                            10, 10, 10};

  test.AddOutput<uint8_t>("Y", {N, static_cast<int64_t>(H * scales[1]), static_cast<int64_t>(W * scales[2]), C}, Y);
  // CUDA: result mismatch due to not implementing NHWC support
  // ROCm: results mismatch
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kCudaExecutionProvider, kRocmExecutionProvider});
}

TEST(ResizeOpTest, NhwcResizeOpLinearDownSampleTest_tf_crop_and_resize_with_extrapolation_int8) {
  OpTester test("Resize", 13);
  std::vector<float> scales{1.0f, 0.8f, 0.8f, 1.0f};
  std::vector<float> roi{0.0f, 0.4f, 0.6f, 0.0f, 1.0f, 1.2f, 1.7f, 1.0f};

  test.AddAttribute("mode", "linear");
  test.AddAttribute("coordinate_transformation_mode", "tf_crop_and_resize");
  test.AddAttribute("extrapolation_value", 10.0f);

  constexpr int64_t N = 1, H = 4, W = 4, C = 1;
  std::vector<int8_t> X = {
      1, -2, 3, -4,
      -5, 6, -7, 8,
      9, -10, 11, -12,
      -13, 14, -15, 16};

  test.AddInput<int8_t>("X", {N, H, W, C}, X);
  test.AddInput<float>("roi", {8}, roi);
  test.AddInput<float>("scales", {4}, scales);

  std::vector<int8_t> Y = {-2, 10, 10,
                           0, 10, 10,
                           10, 10, 10};

  test.AddOutput<int8_t>("Y", {N, static_cast<int64_t>(H * scales[1]), static_cast<int64_t>(W * scales[2]), C}, Y);
  test.Run();
}

TEST(ResizeOpTest, NhwcResizeOpLinearDownSampleTest_tf_crop_and_resize_without_extrapolation_uint8) {
  OpTester test("Resize", 13);
  std::vector<float> scales{1.0f, 0.8f, 0.8f, 1.0f};
  std::vector<float> roi{0.0f, 0.4f, 0.6f, 0.0f, 1.0f, 1.2f, 1.7f, 1.0f};

  test.AddAttribute("mode", "linear");
  test.AddAttribute("coordinate_transformation_mode", "tf_crop_and_resize");

  constexpr int64_t N = 1, H = 4, W = 4, C = 1;
  std::vector<uint8_t> X = {
      1, 2, 3, 4,
      5, 6, 7, 8,
      9, 10, 11, 12,
      13, 14, 15, 16};

  test.AddInput<uint8_t>("X", {N, H, W, C}, X);
  test.AddInput<float>("roi", {8}, roi);
  test.AddInput<float>("scales", {4}, scales);

  std::vector<uint8_t> Y = {7, 0, 0,
                            12, 0, 0,
                            0, 0, 0};

  test.AddOutput<uint8_t>("Y", {N, static_cast<int64_t>(H * scales[1]), static_cast<int64_t>(W * scales[2]), C}, Y);
  // CUDA: result mismatch due to not implementing NHWC support
  // ROCm: results mismatch
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kCudaExecutionProvider, kRocmExecutionProvider});
}

TEST(ResizeOpTest, NhwcResizeOpLinearDownSampleTest_tf_crop_and_resize_without_extrapolation_int8) {
  OpTester test("Resize", 13);
  std::vector<float> scales{1.0f, 0.8f, 0.8f, 1.0f};
  std::vector<float> roi{0.0f, 0.4f, 0.6f, 0.0f, 1.0f, 1.2f, 1.7f, 1.0f};

  test.AddAttribute("mode", "linear");
  test.AddAttribute("coordinate_transformation_mode", "tf_crop_and_resize");

  constexpr int64_t N = 1, H = 4, W = 4, C = 1;
  std::vector<int8_t> X = {
      1, -2, 3, -4,
      -5, 6, -7, 8,
      9, -10, 11, -12,
      -13, 14, -15, 16};

  test.AddInput<int8_t>("X", {N, H, W, C}, X);
  test.AddInput<float>("roi", {8}, roi);
  test.AddInput<float>("scales", {4}, scales);

  std::vector<int8_t> Y = {-2, 0, 0,
                           0, 0, 0,
                           0, 0, 0};

  test.AddOutput<int8_t>("Y", {N, static_cast<int64_t>(H * scales[1]), static_cast<int64_t>(W * scales[2]), C}, Y);
  test.Run();
}

TEST(ResizeOpTest, ResizeOpLinearDownSampleTest_4DBilinear) {
  OpTester test("Resize", 13);
  std::vector<float> roi{};
  std::vector<float> scales{1.0f, 1.0f, 0.6f, 0.6f};

  test.AddAttribute("mode", "linear");

  constexpr int64_t N = 1, C = 1, H = 2, W = 4;
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

TEST(ResizeOpTest, NhwcResizeOpLinearDownSampleTest_4DBilinear) {
  OpTester test("Resize", 13);
  std::vector<float> roi{};
  std::vector<float> scales{1.0f, 0.6f, 0.6f, 1.0f};

  test.AddAttribute("mode", "linear");

  constexpr int64_t N = 1, H = 2, W = 4, C = 1;
  std::vector<float> X = {
      1.0f, 2.0f, 3.0f, 4.0f,
      5.0f, 6.0f, 7.0f, 8.0f};

  test.AddInput<float>("X", {N, H, W, C}, X);
  test.AddInput<float>("roi", {0}, roi);
  test.AddInput<float>("scales", {4}, scales);

  std::vector<float> Y = {2.66666651f, 4.3333331f};

  test.AddOutput<float>("Y", {N, static_cast<int64_t>(H * scales[1]), static_cast<int64_t>(W * scales[2]), C}, Y);
  // CUDA: result mismatch due to not implementing NHWC support
  // ROCm: results mismatch
  test.Run(OpTester::ExpectResult::kExpectSuccess, "",
           {kCudaExecutionProvider, kRocmExecutionProvider});
}

TEST(ResizeOpTest, NhwcResizeOpLinearDownSampleTest_4DBilinear_uint8) {
  OpTester test("Resize", 13);
  std::vector<float> roi{};
  std::vector<float> scales{1.0f, 0.6f, 0.6f, 1.0f};

  test.AddAttribute("mode", "linear");

  constexpr int64_t N = 1, H = 2, W = 4, C = 1;
  std::vector<uint8_t> X = {
      1, 2, 3, 4,
      5, 6, 7, 8};

  test.AddInput<uint8_t>("X", {N, H, W, C}, X);
  test.AddInput<float>("roi", {0}, roi);
  test.AddInput<float>("scales", {4}, scales);

  std::vector<uint8_t> Y = {2, 4};

  test.AddOutput<uint8_t>("Y", {N, static_cast<int64_t>(H * scales[1]), static_cast<int64_t>(W * scales[2]), C}, Y);
  // CUDA: result mismatch due to not implementing NHWC support
  // ROCm: results mismatch
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kCudaExecutionProvider, kRocmExecutionProvider});
}

TEST(ResizeOpTest, NhwcResizeOpLinearDownSampleTest_4DBilinear_int8) {
  OpTester test("Resize", 13);
  std::vector<float> roi{};
  std::vector<float> scales{1.0f, 0.6f, 0.6f, 1.0f};

  test.AddAttribute("mode", "linear");

  constexpr int64_t N = 1, H = 2, W = 4, C = 1;
  std::vector<int8_t> X = {
      1, -2, 3, -4,
      -5, 6, -7, 8};

  test.AddInput<int8_t>("X", {N, H, W, C}, X);
  test.AddInput<float>("roi", {0}, roi);
  test.AddInput<float>("scales", {4}, scales);

  std::vector<int8_t> Y = {0, 0};

  test.AddOutput<int8_t>("Y", {N, static_cast<int64_t>(H * scales[1]), static_cast<int64_t>(W * scales[2]), C}, Y);
  test.Run();
}

// Since NNAPI(TFLite) only using the scale calculate using the input/output size
// For the above test (ResizeOpLinearDownSampleTest_4DBilinear)
// The output size is [1,1,2,4].*[1,1,0.6,0.6]=[1,1,1,2]
// NNAPI will recaluclate the scales as the output size divided by input size
// scales = [1,1,1,2]./[1,1,2,4] = [1,1,0.5,0.5]
// See, https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/kernels/internal/reference/reference_ops.h
// So the result of the above example will be different than CPU EP
// Add the following 2 tests to test with scales valid to NNAPI
TEST(ResizeOpTest, ResizeOpLinearDownSampleTest_4DBilinear1) {
  // To test NNAPI EP, we need the sclaes/sizes to be in initializers
  auto run_test = [](bool scales_in_initializer) {
    OpTester test("Resize", 13);
    std::vector<float> roi{};
    std::vector<float> scales{1.0f, 1.0f, 0.5f, 0.5f};

    test.AddAttribute("mode", "linear");

    constexpr int64_t N = 1, C = 1, H = 2, W = 4;
    std::vector<float> X = {
        1.0f, 2.0f, 3.0f, 4.0f,
        5.0f, 6.0f, 7.0f, 8.0f};

    test.AddInput<float>("X", {N, C, H, W}, X);
    test.AddInput<float>("roi", {0}, roi);
    test.AddInput<float>("scales", {4}, scales, scales_in_initializer);

    std::vector<float> Y = {3.5f, 5.5f};

    test.AddOutput<float>("Y", {N, C, static_cast<int64_t>(H * scales[2]), static_cast<int64_t>(W * scales[3])}, Y);
    test.Run();
  };

  run_test(false);
  run_test(true);
}

TEST(ResizeOpTest, ResizeOpLinearDownSampleTest_4DBilinear1_WithSizes) {
  // To test NNAPI EP, we need the sclaes/sizes to be in initializers
  auto run_test = [](bool scales_and_sizes_in_initializer) {
    OpTester test("Resize", 13);
    std::vector<float> roi{};
    std::vector<float> scales{};
    constexpr int64_t N = 1, C = 1, H = 2, W = 4;
    std::vector<int64_t> sizes{N, C, 1, 2};
    test.AddAttribute("mode", "linear");

    std::vector<float> X = {
        1.0f, 2.0f, 3.0f, 4.0f,
        5.0f, 6.0f, 7.0f, 8.0f};

    test.AddInput<float>("X", {N, C, H, W}, X);
    test.AddInput<float>("roi", {0}, roi);
    test.AddInput<float>("", {0}, scales);
    test.AddInput<int64_t>("sizes", {4}, sizes, scales_and_sizes_in_initializer);

    std::vector<float> Y = {3.5f, 5.5f};

    test.AddOutput<float>("Y", sizes, Y);
    test.Run();
  };

  run_test(false);
  run_test(true);
}

TEST(ResizeOpTest, ResizeOpLinearDownSampleTest_4DBilinear_align_corners) {
  // To test NNAPI EP, we need the sclaes/sizes to be in initializers
  auto run_test = [](bool scales_in_initializer) {
    OpTester test("Resize", 13);
    std::vector<float> roi{};
    std::vector<float> scales{1.0f, 1.0f, 0.6f, 0.6f};

    test.AddAttribute("mode", "linear");
    test.AddAttribute("coordinate_transformation_mode", "align_corners");

    constexpr int64_t N = 1, C = 1, H = 2, W = 4;
    std::vector<float> X = {
        1.0f, 2.0f, 3.0f, 4.0f,
        5.0f, 6.0f, 7.0f, 8.0f};

    test.AddInput<float>("X", {N, C, H, W}, X);
    test.AddInput<float>("roi", {0}, roi);
    test.AddInput<float>("scales", {4}, scales, scales_in_initializer);

    std::vector<float> Y = {1.0f, 4.0f};

    test.AddOutput<float>("Y", {N, C, static_cast<int64_t>(H * scales[2]), static_cast<int64_t>(W * scales[3])}, Y);
    test.Run();
  };

  run_test(false);

#ifdef USE_NNAPI
  // NNAPI will need the scales as an initializer
  // Also tensor RT EP will fail if scales is an initializer but will pass if it is not
  run_test(true);
#endif
}

TEST(ResizeOpTest, NhwcResizeOpLinearDownSampleTest_4DBilinear_align_corners_uint8) {
  // To test NNAPI EP, we need the sclaes/sizes to be in initializers
  auto run_test = [](bool scales_in_initializer) {
    OpTester test("Resize", 13);
    std::vector<float> roi{};
    std::vector<float> scales{1.0f, 0.6f, 0.6f, 1.0f};

    test.AddAttribute("mode", "linear");
    test.AddAttribute("coordinate_transformation_mode", "align_corners");

    constexpr int64_t N = 1, H = 2, W = 4, C = 1;
    std::vector<uint8_t> X = {
        1, 2, 3, 4,
        5, 6, 7, 8};

    test.AddInput<uint8_t>("X", {N, H, W, C}, X);
    test.AddInput<float>("roi", {0}, roi);
    test.AddInput<float>("scales", {4}, scales, scales_in_initializer);

    std::vector<uint8_t> Y = {1, 4};

    test.AddOutput<uint8_t>("Y", {N, static_cast<int64_t>(H * scales[1]), static_cast<int64_t>(W * scales[2]), C}, Y);
    // CUDA: result mismatch due to not implementing NHWC support
    // ROCm: results mismatch
    test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kCudaExecutionProvider, kRocmExecutionProvider});
  };

  run_test(false);
  run_test(true);
}

TEST(ResizeOpTest, NhwcResizeOpLinearDownSampleTest_4DBilinear_align_corners_int8) {
  // To test NNAPI EP, we need the sclaes/sizes to be in initializers
  auto run_test = [](bool scales_in_initializer) {
    OpTester test("Resize", 13);
    std::vector<float> roi{};
    std::vector<float> scales{1.0f, 0.6f, 0.6f, 1.0f};

    test.AddAttribute("mode", "linear");
    test.AddAttribute("coordinate_transformation_mode", "align_corners");

    constexpr int64_t N = 1, H = 2, W = 4, C = 1;
    std::vector<int8_t> X = {
        1, -2, 3, -4,
        -5, 6, -7, 8};

    test.AddInput<int8_t>("X", {N, H, W, C}, X);
    test.AddInput<float>("roi", {0}, roi);
    test.AddInput<float>("scales", {4}, scales, scales_in_initializer);

    std::vector<int8_t> Y = {1, -4};

    test.AddOutput<int8_t>("Y", {N, static_cast<int64_t>(H * scales[1]), static_cast<int64_t>(W * scales[2]), C}, Y);
    // TensorRT: results mismatch
    test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});
  };

  run_test(false);
  run_test(true);
}

TEST(ResizeOpTest, ResizeOpLinearDownSampleTest_2DBilinear_pytorch_half_pixel) {
  // TODO: Unskip when fixed #41968513
  if (DefaultDmlExecutionProvider().get() != nullptr) {
    GTEST_SKIP() << "Skipping because of the following error: The difference between expected[i] and output[i] is 1.5000001192092896, which exceeds threshold";
  }

  OpTester test("Resize", 13);
  std::vector<float> roi{};
  std::vector<float> scales{};
  std::vector<int64_t> sizes{3, 1};

  test.AddAttribute("mode", "linear");
  test.AddAttribute("coordinate_transformation_mode", "pytorch_half_pixel");

  constexpr int64_t H = 4, W = 4;

  std::vector<float> X = {
      1.0f, 2.0f, 3.0f, 4.0f,
      5.0f, 6.0f, 7.0f, 8.0f,
      9.0f, 10.0f, 11.0f, 12.0f,
      13.0f, 14.0f, 15.0f, 16.0f};

  test.AddInput<float>("X", {H, W}, X);
  test.AddInput<float>("roi", {0}, roi);
  test.AddInput<float>("", {0}, scales);
  test.AddInput<int64_t>("sizes", {2}, sizes);

  std::vector<float> Y = {1.6666666f, 7.0f, 12.333333f};

  test.AddOutput<float>("Y", {sizes[0], sizes[1]}, Y);
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});  // TensorRT: results mismatch
}

TEST(ResizeOpTest, NhwcResizeOpLinearDownSampleTest_4DBilinear_pytorch_half_pixel_uint8) {
  OpTester test("Resize", 13);
  std::vector<float> roi{};
  std::vector<float> scales{};
  std::vector<int64_t> sizes{1, 3, 1, 1};

  test.AddAttribute("mode", "linear");
  test.AddAttribute("coordinate_transformation_mode", "pytorch_half_pixel");

  constexpr int64_t N = 1, H = 4, W = 4, C = 1;

  std::vector<uint8_t> X = {
      1, 2, 3, 4,
      5, 6, 7, 8,
      9, 10, 11, 12,
      13, 14, 15, 16};

  test.AddInput<uint8_t>("X", {N, H, W, C}, X);
  test.AddInput<float>("roi", {0}, roi);
  test.AddInput<float>("", {0}, scales);
  test.AddInput<int64_t>("sizes", {4}, sizes);

  std::vector<uint8_t> Y = {1, 7, 12};

  test.AddOutput<uint8_t>("Y", {N, sizes[1], sizes[2], C}, Y);
  // CUDA: result mismatch due to not implementing NHWC support
  // ROCm: results mismatch
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kCudaExecutionProvider, kRocmExecutionProvider});
}

TEST(ResizeOpTest, NhwcResizeOpLinearDownSampleTest_4DBilinear_pytorch_half_pixel_int8) {
  OpTester test("Resize", 13);
  std::vector<float> roi{};
  std::vector<float> scales{};
  std::vector<int64_t> sizes{1, 3, 1, 1};

  test.AddAttribute("mode", "linear");
  test.AddAttribute("coordinate_transformation_mode", "pytorch_half_pixel");

  constexpr int64_t N = 1, H = 4, W = 4, C = 1;

  std::vector<int8_t> X = {
      1, -2, 3, -4,
      -5, 6, -7, 8,
      9, -10, 11, -12,
      -13, 14, -15, 16};

  test.AddInput<int8_t>("X", {N, H, W, C}, X);
  test.AddInput<float>("roi", {0}, roi);
  test.AddInput<float>("", {0}, scales);
  test.AddInput<int64_t>("sizes", {4}, sizes);

  std::vector<int8_t> Y = {0, 2, -9};

  test.AddOutput<int8_t>("Y", {N, sizes[1], sizes[2], C}, Y);
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});  // TensorRT: results mismatch
}

TEST(ResizeOpTest, ResizeOpLinearUpSampleTest_4DBilinear_asymmetric) {
  // To test NNAPI EP, we need the sclaes/sizes to be in initializers
  auto run_test = [](bool scales_in_initializer) {
    OpTester test("Resize", 13);
    std::vector<float> roi{};
    std::vector<float> scales{1.0f, 1.0f, 2.0f, 4.0f};

    test.AddAttribute("mode", "linear");
    test.AddAttribute("coordinate_transformation_mode", "asymmetric");

    constexpr int64_t N = 2, C = 1, H = 2, W = 2;
    std::vector<float> X = {1.0f, 3.0f,
                            4.0f, 8.0f,

                            6.0f, 2.0f,
                            7.0f, 11.0f};

    test.AddInput<float>("X", {N, C, H, W}, X);
    test.AddInput<float>("roi", {0}, roi);
    test.AddInput<float>("scales", {4}, scales, scales_in_initializer);

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
  };

  run_test(false);
  run_test(true);
}

TEST(ResizeOpTest, NhwcResizeOpLinearUpSampleTest_4DBilinear_asymmetric_uint8) {
  // To test NNAPI EP, we need the sclaes/sizes to be in initializers
  auto run_test = [](bool scales_in_initializer) {
    OpTester test("Resize", 13);
    std::vector<float> roi{};
    std::vector<float> scales{1.0f, 2.0f, 4.0f, 1.0f};

    test.AddAttribute("mode", "linear");
    test.AddAttribute("coordinate_transformation_mode", "asymmetric");

    constexpr int64_t N = 2, H = 2, W = 2, C = 1;
    std::vector<uint8_t> X = {1, 3,
                              4, 8,

                              6, 2,
                              7, 11};

    test.AddInput<uint8_t>("X", {N, H, W, C}, X);
    test.AddInput<float>("roi", {0}, roi);
    test.AddInput<float>("scales", {4}, scales, scales_in_initializer);

    std::vector<uint8_t> Y = {
        1, 1, 2, 2, 3, 3, 3, 3,
        2, 3, 4, 4, 5, 5, 5, 5,
        4, 5, 6, 7, 8, 8, 8, 8,
        4, 5, 6, 7, 8, 8, 8, 8,

        6, 5, 4, 3, 2, 2, 2, 2,
        6, 6, 6, 6, 6, 6, 6, 6,
        7, 8, 9, 10, 11, 11, 11, 11,
        7, 8, 9, 10, 11, 11, 11, 11};

    // Due to Xnnpack EP has a different rounding behavior, we need to allow a tolerance of 1
    // The tolerance only works for Xnnpack EP
    test.AddOutput<uint8_t>("Y", {N, static_cast<int64_t>(H * scales[1]), static_cast<int64_t>(W * scales[2]), C},
                            Y, false, .0f, 1.0f);
    // CUDA: result mismatch due to not implementing NHWC support
    // ROCm: results mismatch
    test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kCudaExecutionProvider, kRocmExecutionProvider});
  };

  run_test(false);
  run_test(true);
}

TEST(ResizeOpTest, NhwcResizeOpLinearUpSampleTest_4DBilinear_asymmetric_int8) {
  // To test NNAPI EP, we need the sclaes/sizes to be in initializers
  auto run_test = [](bool scales_in_initializer) {
    OpTester test("Resize", 13);
    std::vector<float> roi{};
    std::vector<float> scales{1.0f, 2.0f, 4.0f, 1.0f};

    test.AddAttribute("mode", "linear");
    test.AddAttribute("coordinate_transformation_mode", "asymmetric");

    constexpr int64_t N = 2, H = 2, W = 2, C = 1;
    std::vector<int8_t> X = {1, -3,
                             -4, 8,

                             6, -2,
                             -7, 11};

    test.AddInput<int8_t>("X", {N, H, W, C}, X);
    test.AddInput<float>("roi", {0}, roi);
    test.AddInput<float>("scales", {4}, scales, scales_in_initializer);

    std::vector<int8_t> Y = {
        1, 0, -1, -2, -3, -3, -3, -3,
        -1, 0, 0, 1, 2, 2, 2, 2,
        -4, -1, 2, 5, 8, 8, 8, 8,
        -4, -1, 2, 5, 8, 8, 8, 8,

        6, 4, 2, 0, -2, -2, -2, -2,
        0, 0, 2, 3, 4, 4, 4, 4,
        -7, -2, 2, 6, 11, 11, 11, 11,
        -7, -2, 2, 6, 11, 11, 11, 11};

    test.AddOutput<int8_t>("Y", {N, static_cast<int64_t>(H * scales[1]), static_cast<int64_t>(W * scales[2]), C},
                           Y, false, .0f, 1.0f);
    // TensorRT: results mismatch
    test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});
  };

  run_test(false);
  run_test(true);
}

TEST(ResizeOpTest, ResizeOpLinearUpSampleTest_2DBilinear_align_corners) {
  OpTester test("Resize", 13);
  std::vector<float> roi{};
  std::vector<float> scales{2.0f, 4.0f};
  test.AddAttribute("mode", "linear");
  test.AddAttribute("coordinate_transformation_mode", "align_corners");

  constexpr int64_t H = 2, W = 2;
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

TEST(ResizeOpTest, ResizeOpLinearDownSampleTest_3DTrilinear_pytorch_half_pixel) {
  // TODO: Unskip when fixed #41968513
  if (DefaultDmlExecutionProvider().get() != nullptr) {
    GTEST_SKIP() << "Skipping because of the following error: The difference between expected[i] and output[i] is 1.5000001192092896, which exceeds threshold";
  }

  OpTester test("Resize", 13);
  std::vector<float> roi{};
  std::vector<float> scales{};
  std::vector<int64_t> sizes{1, 3, 1};

  test.AddAttribute("mode", "linear");
  test.AddAttribute("coordinate_transformation_mode", "pytorch_half_pixel");

  constexpr int64_t D = 2, H = 4, W = 4;

  std::vector<float> X = {
      1.0f, 2.0f, 3.0f, 4.0f,
      5.0f, 6.0f, 7.0f, 8.0f,
      9.0f, 10.0f, 11.0f, 12.0f,
      13.0f, 14.0f, 15.0f, 16.0f,

      1.0f, 2.0f, 3.0f, 4.0f,
      5.0f, 6.0f, 7.0f, 8.0f,
      9.0f, 10.0f, 11.0f, 12.0f,
      13.0f, 14.0f, 15.0f, 16.0f};

  test.AddInput<float>("X", {D, H, W}, X);
  test.AddInput<float>("roi", {0}, roi);
  test.AddInput<float>("", {0}, scales);
  test.AddInput<int64_t>("sizes", {3}, sizes);

  std::vector<float> Y = {1.6666666f, 7.0f, 12.333333f};

  test.AddOutput<float>("Y", {sizes[0], sizes[1], sizes[2]}, Y);
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});  // TensorRT: results mismatch
}

TEST(ResizeOpTest, ResizeOpLinearUpSampleTest_5DTrilinear_pytorch_half_pixel) {
  OpTester test("Resize", 13);
  std::vector<float> roi{};
  std::vector<float> scales{1.0f, 1.0f, 2.0f, 2.0f, 1.0f};

  test.AddAttribute("mode", "linear");
  test.AddAttribute("coordinate_transformation_mode", "pytorch_half_pixel");

  constexpr int64_t N = 1, C = 2, D = 2, H = 1, W = 2;

  std::vector<float> X = {
      1.0f, 2.0f, 1.0f, 2.0f, 1.0f, 2.0f, 1.0f, 2.0f};

  test.AddInput<float>("X", {N, C, D, H, W}, X);
  test.AddInput<float>("roi", {0}, roi);
  test.AddInput<float>("scales", {5}, scales);

  std::vector<float> Y = {1.0f, 2.0f, 1.0f, 2.0f, 1.0f, 2.0f, 1.0f, 2.0f,
                          1.0f, 2.0f, 1.0f, 2.0f, 1.0f, 2.0f, 1.0f, 2.0f,
                          1.0f, 2.0f, 1.0f, 2.0f, 1.0f, 2.0f, 1.0f, 2.0f,
                          1.0f, 2.0f, 1.0f, 2.0f, 1.0f, 2.0f, 1.0f, 2.0f};

  test.AddOutput<float>("Y", {1, 2, 4, 2, 2}, Y);
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});  // TensorRT: results mismatch
}

TEST(ResizeOpTest, ResizeOpLinearScalesNoOpTest) {
  // To test NNAPI EP, we need the scales/sizes to be in initializers
  auto run_test = [](bool scales_in_initializer) {
    OpTester test("Resize", 13);
    std::vector<float> roi{};
    std::vector<float> scales{1.0f, 1.0f, 1.0f, 1.0f};
    test.AddAttribute("mode", "linear");

    constexpr int64_t N = 2, C = 1, H = 2, W = 2;
    std::vector<float> X = {1.0f, 3.0f,
                            4.0f, 8.0f,

                            6.0f, 2.0f,
                            7.0f, 11.0f};

    test.AddInput<float>("X", {N, C, H, W}, X);
    test.AddInput<float>("roi", {0}, roi);
    test.AddInput<float>("scales", {4}, scales, scales_in_initializer);

    std::vector<float> Y = {1.0f, 3.0f,
                            4.0f, 8.0f,

                            6.0f, 2.0f,
                            7.0f, 11.0f};

    test.AddOutput<float>("Y", {N, C, H, W}, Y);
    test.Run();
  };

  run_test(false);
  run_test(true);
}

TEST(ResizeOpTest, ResizeOpNearestDownSampleTest) {
  OpTester test("Resize", 13);
  std::vector<float> scales{1.0f, 1.0f, 0.6f, 0.6f};
  std::vector<float> roi{};

  test.AddAttribute("mode", "nearest");

  constexpr int64_t N = 1, C = 1, H = 2, W = 4;
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

TEST(ResizeOpTest, ResizeOpNearestDownSampleTest_Opset12) {
  OpTester test("Resize", 12);
  std::vector<float> scales{1.0f, 1.0f, 0.6f, 0.6f};
  std::vector<float> roi{};

  test.AddAttribute("mode", "nearest");

  constexpr int64_t N = 1, C = 1, H = 2, W = 4;
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
  OpTester test("Resize", 13);
  std::vector<float> scales{};
  std::vector<float> roi{};
  std::vector<int64_t> sizes{1, 1, 1, 3};

  test.AddAttribute("mode", "nearest");

  constexpr int64_t N = 1, C = 1, H = 2, W = 4;
  std::vector<float> X = {
      1.0f, 2.0f, 3.0f, 4.0f,
      5.0f, 6.0f, 7.0f, 8.0f};

  test.AddInput<float>("X", {N, C, H, W}, X);
  test.AddInput<float>("roi", {0}, roi);
  test.AddInput<float>("", {0}, scales);
  test.AddInput<int64_t>("sizes", {4}, sizes);

  std::vector<float> Y = {1.0f, 2.0f, 4.0f};

  test.AddOutput<float>("Y", {N, C, sizes[2], sizes[3]}, Y);
  test.Run();
}

TEST(ResizeOpTest, ResizeOpNearestDownSampleTest_tf_half_pixel) {
  OpTester test("Resize", 12);  // tf_half_pixel_for_nn is deprecated since opset 13
  std::vector<float> scales{};
  std::vector<float> roi{};
  std::vector<int64_t> sizes{1, 1, 3, 2};

  test.AddAttribute("coordinate_transformation_mode", "tf_half_pixel_for_nn");
  test.AddAttribute("mode", "nearest");

  constexpr int64_t N = 1, C = 1, H = 4, W = 4;
  std::vector<float> X = {
      1.0f, 2.0f, 3.0f, 4.0f,
      5.0f, 6.0f, 7.0f, 8.0f,
      9.0f, 10.0f, 11.0f, 12.0f,
      13.0f, 14.0f, 15.0f, 16.0f};

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
  OpTester test("Resize", 13);
  std::vector<float> scales{1.0f, 1.0f, 0.8f, 0.8f};
  std::vector<float> roi{0.0f, 0.0f, 0.4f, 0.6f, 1.0f, 1.0f, 1.2f, 1.7f};

  test.AddAttribute("mode", "nearest");
  test.AddAttribute("coordinate_transformation_mode", "tf_crop_and_resize");
  test.AddAttribute("extrapolation_value", 10.0f);

  constexpr int64_t N = 1, C = 1, H = 4, W = 4;
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

TEST(ResizeOpTest, ResizeOpNearestDownSample5dTest_tf_crop_and_resize_with_extrapolation) {
  OpTester test("Resize", 13);
  std::vector<float> scales{1.0f, 1.0f, 1.0f, 0.8f, 0.8f};
  std::vector<float> roi{0.0f, 0.0f, 0.0f, 0.4f, 0.6f, 1.0f, 1.0f, 1.0f, 1.2f, 1.7f};

  test.AddAttribute("mode", "nearest");
  test.AddAttribute("coordinate_transformation_mode", "tf_crop_and_resize");
  test.AddAttribute("extrapolation_value", 10.0f);

  constexpr int64_t N = 1, C = 1, H = 4, W = 4;
  std::vector<float> X = {
      1.0f, 2.0f, 3.0f, 4.0f,
      5.0f, 6.0f, 7.0f, 8.0f,
      9.0f, 10.0f, 11.0f, 12.0f,
      13.0f, 14.0f, 15.0f, 16.0f};

  test.AddInput<float>("X", {1, N, C, H, W}, X);
  test.AddInput<float>("roi", {10}, roi);
  test.AddInput<float>("scales", {5}, scales);

  std::vector<float> Y = {7.0f, 10.0f, 10.0f,
                          11.0f, 10.f, 10.0f,
                          10.0f, 10.0f, 10.0f};

  test.AddOutput<float>("Y", {1, N, C, static_cast<int64_t>(H * scales[3]), static_cast<int64_t>(W * scales[4])}, Y);
  // Current cuda provider do not support more than 4d
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kCudaExecutionProvider});
}

TEST(ResizeOpTest, ResizeOpNearestUpSampleTest) {
  OpTester test("Resize", 13);
  std::vector<float> roi{};
  std::vector<float> scales{1.0f, 1.0f, 2.0f, 3.0f};

  test.AddAttribute("mode", "nearest");

  constexpr int64_t N = 1, C = 1, H = 2, W = 2;
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
  OpTester test("Resize", 13);
  std::vector<float> roi{};
  std::vector<float> scales{};
  std::vector<int64_t> sizes{1, 1, 7, 8};

  test.AddAttribute("mode", "nearest");
  test.AddAttribute("nearest_mode", "ceil");

  constexpr int64_t N = 1, C = 1, H = 2, W = 2;
  std::vector<float> X = {1.0f, 2.0f, 3.0f, 4.0f};

  test.AddInput<float>("X", {N, C, H, W}, X);
  test.AddInput<float>("roi", {0}, roi);
  test.AddInput<float>("", {0}, scales);
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

TEST(ResizeOpTest, ResizeOpNearestUpSample5dTest_WithSizes_CeilMode) {
  OpTester test("Resize", 13);
  std::vector<float> roi{};
  std::vector<float> scales{};
  std::vector<int64_t> sizes{1, 1, 1, 7, 8};

  test.AddAttribute("mode", "nearest");
  test.AddAttribute("nearest_mode", "ceil");

  constexpr int64_t N = 1, C = 1, H = 2, W = 2;
  std::vector<float> X = {1.0f, 2.0f, 3.0f, 4.0f};

  test.AddInput<float>("X", {1, N, C, H, W}, X);
  test.AddInput<float>("roi", {0}, roi);
  test.AddInput<float>("", {0}, scales);
  test.AddInput<int64_t>("sizes", {5}, sizes);

  std::vector<float> Y = {1.0f, 1.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f,
                          1.0f, 1.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f,
                          3.0f, 3.0f, 4.0f, 4.0f, 4.0f, 4.0f, 4.0f, 4.0f,
                          3.0f, 3.0f, 4.0f, 4.0f, 4.0f, 4.0f, 4.0f, 4.0f,
                          3.0f, 3.0f, 4.0f, 4.0f, 4.0f, 4.0f, 4.0f, 4.0f,
                          3.0f, 3.0f, 4.0f, 4.0f, 4.0f, 4.0f, 4.0f, 4.0f,
                          3.0f, 3.0f, 4.0f, 4.0f, 4.0f, 4.0f, 4.0f, 4.0f};

  test.AddOutput<float>("Y", {1, N, C, sizes[3], sizes[4]}, Y);
  // Current cuda provider do not support more than 4d
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kCudaExecutionProvider});
}

TEST(ResizeOpTest, ResizeOpNearestUpSample_Floor_Align_Corners) {
  OpTester test("Resize", 13);

  std::vector<float> roi{};
  std::vector<float> scales{1.0f, 1.0f, 2.0f, 2.0f};

  test.AddAttribute("mode", "nearest");
  test.AddAttribute("coordinate_transformation_mode", "align_corners");
  test.AddAttribute("nearest_mode", "floor");

  constexpr int64_t N = 1, C = 1, H = 4, W = 4;
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

TEST(ResizeOpTest, ResizeOpNearest_OneToOneMappingBetweenInputAndOutputDataDims) {
  // TODO: Unskip when fixed #41968513
  if (DefaultDmlExecutionProvider().get() != nullptr) {
    GTEST_SKIP() << "Skipping because of the following error: The difference between expected[i] and output[i] is 3, which exceeds threshold";
  }

  OpTester test("Resize", 12);  // tf_half_pixel_for_nn is deprecated since opset 13

  std::vector<float> roi{};
  // There is one-to-one mapping in the outermost dim.
  // This test is to ensure that the co-ordinate transformation is not applied to the
  // outermost dim as there is no "resizing".
  // If it were applied using the provided attributes ,it would result in result mismatch
  std::vector<float> scales{1.0f, 0.5f};

  test.AddAttribute("mode", "nearest");
  test.AddAttribute("coordinate_transformation_mode", "tf_half_pixel_for_nn");
  test.AddAttribute("nearest_mode", "ceil");

  constexpr int64_t C = 2, D = 3;
  std::vector<float> X = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};

  test.AddInput<float>("X", {C, D}, X);
  test.AddInput<float>("roi", {0}, roi);
  test.AddInput<float>("scales", {2}, scales);

  // would produce {5.0f, 5.0f} if co-ordinate transformation was applied
  // to the outermost dim
  std::vector<float> Y = {2.0f, 5.0f};
  test.AddOutput<float>("Y", {2, 1}, Y);
  test.Run();
}

// custom OpTester to make 'scales' or 'sizes' a constant initializer if needed
class ResizeOpTester : public OpTester {
 public:
  ResizeOpTester(bool scales_in_initializer, bool sizes_in_initializer)
      : OpTester("Resize", 13),
        scales_in_initializer_(scales_in_initializer),
        sizes_in_initializer_(sizes_in_initializer) {
  }

 protected:
  void AddNodes(onnxruntime::Graph& graph,
                std::vector<onnxruntime::NodeArg*>& graph_input_defs,
                std::vector<onnxruntime::NodeArg*>& graph_output_defs,
                std::vector<std::function<void(onnxruntime::Node& node)>>& add_attribute_funcs) override {
    // adding the node will result in a copy of the NodeArg (as it currently only exists in the OpTester)
    // so we need to do that first before calling Graph::SetInputs so the address of the 'real' NodeArg is in that
    // list
    OpTester::AddNodes(graph, graph_input_defs, graph_output_defs, add_attribute_funcs);

    // set the Graph inputs to just X and roi (exclude 'scales') so the 'scales' are a constant initializer
    if (scales_in_initializer_) {
      graph.SetInputs({graph.GetNodeArg(graph_input_defs[0]->Name()),
                       graph.GetNodeArg(graph_input_defs[1]->Name())});
      if (sizes_in_initializer_) {
        ASSERT_TRUE(graph_input_defs.size() == 4);
      } else {
        ASSERT_TRUE(graph_input_defs.size() == 3);
      }
    } else if (sizes_in_initializer_) {
      ASSERT_TRUE(graph_input_defs.size() == 4);  // 'sizes' is 4th input
      graph.SetInputs({graph.GetNodeArg(graph_input_defs[0]->Name()),
                       graph.GetNodeArg(graph_input_defs[1]->Name()),
                       graph.GetNodeArg(graph_input_defs[2]->Name())});
    }
  }

 private:
  bool scales_in_initializer_;
  bool sizes_in_initializer_;
};

TEST(ResizeOpTest, ResizeOpNearestUpSample_Nearest2xOptimization_Scales) {
  auto run_test = [](bool scales_in_initializer) {
    ResizeOpTester test(scales_in_initializer, false);

    std::vector<float> roi{};
    std::vector<float> scales{1.0f, 1.0f, 2.0f, 2.0f};

    test.AddAttribute("mode", "nearest");
    test.AddAttribute("coordinate_transformation_mode", "asymmetric");
    test.AddAttribute("nearest_mode", "floor");

    constexpr int64_t N = 1, C = 1, H = 2, W = 2;
    std::vector<float> X = {
        1.0f, 2.0f,
        3.0f, 4.0f};

    test.AddInput<float>("X", {N, C, H, W}, X);
    test.AddInput<float>("roi", {0}, roi);
    test.AddInput<float>("scales", {4}, scales, scales_in_initializer);

    std::vector<float> Y = {1.0f, 1.0f, 2.0f, 2.0f,
                            1.0f, 1.0f, 2.0f, 2.0f,
                            3.0f, 3.0f, 4.0f, 4.0f,
                            3.0f, 3.0f, 4.0f, 4.0f};

    test.AddOutput<float>("Y", {N, C, static_cast<int64_t>(H * scales[2]), static_cast<int64_t>(W * scales[3])}, Y);
    test.Run();
  };

  run_test(false);
  run_test(true);
}

TEST(ResizeOpTest, ResizeOpNearestUpSample_Nearest2xOptimization_Sizes) {
  auto run_test = [](bool sizes_in_initializer) {
    ResizeOpTester test(sizes_in_initializer, sizes_in_initializer);

    std::vector<float> roi{};
    std::vector<float> scales{};
    std::vector<int64_t> sizes{1, 1, 4, 4};

    test.AddAttribute("mode", "nearest");
    test.AddAttribute("coordinate_transformation_mode", "asymmetric");
    test.AddAttribute("nearest_mode", "floor");

    constexpr int64_t N = 1, C = 1, H = 2, W = 2;
    std::vector<float> X = {
        1.0f, 2.0f,
        3.0f, 4.0f};

    test.AddInput<float>("X", {N, C, H, W}, X);
    test.AddInput<float>("roi", {0}, roi);
    test.AddInput<float>("", {0}, scales);
    test.AddInput<int64_t>("sizes", {4}, sizes, sizes_in_initializer);

    std::vector<float> Y = {1.0f, 1.0f, 2.0f, 2.0f,
                            1.0f, 1.0f, 2.0f, 2.0f,
                            3.0f, 3.0f, 4.0f, 4.0f,
                            3.0f, 3.0f, 4.0f, 4.0f};

    test.AddOutput<float>("Y", {N, C, sizes[2], sizes[3]}, Y);
    test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});  // TensorRT: results mismatch
  };

  run_test(false);
  run_test(true);
}

TEST(ResizeOpTest, ResizeOpCubicDownSampleTest) {
  OpTester test("Resize", 13);
  std::vector<float> scales{1.0f, 1.0f, 0.8f, 0.8f};
  std::vector<float> roi{};

  test.AddAttribute("mode", "cubic");

  constexpr int64_t N = 1, C = 1, H = 4, W = 4;
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

TEST(ResizeOpTest, ResizeOpCubicDownSampleTest_exclude_outside) {
  OpTester test("Resize", 13);
  std::vector<float> roi{};
  std::vector<float> scales{0.8f, 0.8f};
  std::vector<int64_t> sizes{};

  test.AddAttribute("mode", "cubic");
  test.AddAttribute("exclude_outside", static_cast<int64_t>(1));
  test.AddAttribute("cubic_coeff_a", -0.5f);

  constexpr int64_t H = 4, W = 4;

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
  OpTester test("Resize", 13);
  std::vector<float> scales{1.0f, 1.0f, 0.8f, 0.8f};
  std::vector<float> roi{};

  test.AddAttribute("mode", "cubic");
  test.AddAttribute("cubic_coeff_a", -0.5f);

  constexpr int64_t N = 1, C = 1, H = 4, W = 4;
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
                          11.8701f, 13.168f, 14.4912f};

  test.AddOutput<float>("Y", {N, C, static_cast<int64_t>(H * scales[2]), static_cast<int64_t>(W * scales[3])}, Y);
  test.Run();
}

TEST(ResizeOpTest, ResizeOpCubicDownSampleTest_with_roi) {
  OpTester test("Resize", 13);
  std::vector<float> scales{1.0f, 1.0f, 0.8f, 0.8f};
  std::vector<float> roi{0.0f, 0.0f, 0.4f, 0.6f, 1.0f, 1.0f, 0.6f, 0.8f};

  test.AddAttribute("mode", "cubic");
  test.AddAttribute("coordinate_transformation_mode", "tf_crop_and_resize");

  constexpr int64_t N = 1, C = 1, H = 4, W = 4;
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
  OpTester test("Resize", 13);
  std::vector<float> scales{1.0f, 1.0f, 0.8f, 0.8f};
  std::vector<float> roi{};

  test.AddAttribute("mode", "cubic");
  test.AddAttribute("coordinate_transformation_mode", "asymmetric");

  constexpr int64_t N = 1, C = 1, H = 4, W = 4;
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
  OpTester test("Resize", 13);
  std::vector<float> scales{1.0f, 1.0f, 2.0f, 2.0f};
  std::vector<float> roi{};

  test.AddAttribute("mode", "cubic");
  test.AddAttribute("coordinate_transformation_mode", "asymmetric");

  constexpr int64_t N = 1, C = 1, H = 4, W = 4;
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
                          13.375f, 13.7813f, 14.375f, 14.875f, 15.375f, 15.9688f, 16.375f, 16.4688f};

  test.AddOutput<float>("Y", {N, C, static_cast<int64_t>(H * scales[2]), static_cast<int64_t>(W * scales[3])}, Y);
  test.Run();
}

TEST(ResizeOpTest, ResizeOpCubicUpSampleTest_MultiChannel) {
  OpTester test("Resize", 13);
  std::vector<float> scales{};
  std::vector<int64_t> sizes{1, 2, 9, 9};
  std::vector<float> roi{};

  test.AddAttribute("mode", "cubic");

  constexpr int64_t N = 1, C = 2, H = 4, W = 4;
  std::vector<float> X = {
      0.0f, 1.0f, 2.0f, 3.0f,
      4.0f, 5.0f, 6.0f, 7.0f,
      8.0f, 9.0f, 10.0f, 11.0f,
      12.0f, 13.0f, 14.0f, 15.0f,

      16.0f, 17.0f, 18.0f, 19.0f,
      20.0f, 21.0f, 22.0f, 23.0f,
      24.0f, 25.0f, 26.0f, 27.0f,
      28.0f, 29.0f, 30.0f, 31.0f};

  test.AddInput<float>("X", {N, C, H, W}, X);
  test.AddInput<float>("roi", {0}, roi);
  test.AddInput<float>("", {0}, scales);
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
                          28.326f, 28.5608f, 28.9501f, 29.5135f, 29.9347f, 30.3558f, 30.9193f, 31.3085f, 31.5433f};

  test.AddOutput<float>("Y", {N, C, sizes[2], sizes[3]}, Y);
  test.Run();
}
TEST(ResizeOpTest, ResizeOpCubicUpSampleTest_tf_half_pixel_for_nn) {
  // tf_half_pixel_for_nn has been deprecated since opset 13
  OpTester test("Resize", 12);
  std::vector<float> scales{1.0f, 1.0f, 2.0f, 2.0f};
  std::vector<float> roi{};

  test.AddAttribute("mode", "cubic");
  test.AddAttribute("coordinate_transformation_mode", "tf_half_pixel_for_nn");

  constexpr int64_t N = 1, C = 1, H = 4, W = 4;
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
                          13.332f, 13.8086f, 14.4375f, 14.8438f, 15.4727f, 15.9492f, 16.2461f, 16.1758f};

  test.AddOutput<float>("Y", {N, C, static_cast<int64_t>(H * scales[2]), static_cast<int64_t>(W * scales[3])}, Y);
  test.Run();
}

TEST(ResizeOpTest, ResizeOpLinearDownSampleTest_4DBilinear_Ver10) {
  // TODO: Unskip when fixed #41968513
  if (DefaultDmlExecutionProvider().get() != nullptr) {
    GTEST_SKIP() << "Skipping because of the following error: The difference between expected[i] and output[i] is 1.6666665077209473, which exceeds threshold";
  }

  OpTester test("Resize", 10);
  std::vector<float> scales{1.0f, 1.0f, 0.6f, 0.6f};

  test.AddAttribute("mode", "linear");

  constexpr int64_t N = 1, C = 1, H = 2, W = 4;
  std::vector<float> X = {
      1.0f, 2.0f, 3.0f, 4.0f,
      5.0f, 6.0f, 7.0f, 8.0f};

  test.AddInput<float>("X", {N, C, H, W}, X);
  test.AddInput<float>("scales", {4}, scales);

  std::vector<float> Y = {1.0f, 2.66666651f};

  test.AddOutput<float>("Y", {N, C, static_cast<int64_t>(H * scales[2]), static_cast<int64_t>(W * scales[3])}, Y);
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kQnnExecutionProvider});  // QNN: result diff
}

TEST(ResizeOpTest, ResizeOpLinearDownSampleTest_2DBilinear_Ver10) {
  // TODO: Unskip when fixed #41968513
  if (DefaultDmlExecutionProvider().get() != nullptr) {
    GTEST_SKIP() << "Skipping because of the following error: The difference between expected[i] and output[i] is 1.6666665077209473, which exceeds threshold";
  }

  OpTester test("Resize", 10);
  std::vector<float> scales{0.6f, 0.6f};

  test.AddAttribute("mode", "linear");

  constexpr int64_t H = 2, W = 4;
  std::vector<float> X = {
      1.0f, 2.0f, 3.0f, 4.0f,
      5.0f, 6.0f, 7.0f, 8.0f};

  test.AddInput<float>("X", {H, W}, X);
  test.AddInput<float>("scales", {2}, scales);

  std::vector<float> Y = {1.0f, 2.66666651f};

  test.AddOutput<float>("Y", {static_cast<int64_t>(H * scales[0]), static_cast<int64_t>(W * scales[1])}, Y);
  test.Run();
}

TEST(ResizeOpTest, ResizeOpLinearUpSampleTest_4DBilinear_Ver10) {
  // TODO: Unskip when fixed #41968513
  if (DefaultDmlExecutionProvider().get() != nullptr) {
    GTEST_SKIP() << "Skipping because of the following error: The difference between expected[i] and output[i] is 0.5, which exceeds threshold";
  }

  OpTester test("Resize", 10);
  std::vector<float> scales{1.0f, 1.0f, 2.0f, 4.0f};
  test.AddAttribute("mode", "linear");

  constexpr int64_t N = 2, C = 1, H = 2, W = 2;
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
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kQnnExecutionProvider});  // QNN: result diff
}

TEST(ResizeOpTest, ResizeOpLinearUpSampleTest_2DBilinear_Ver10) {
  // TODO: Unskip when fixed #41968513
  if (DefaultDmlExecutionProvider().get() != nullptr) {
    GTEST_SKIP() << "Skipping because of the following error: The difference between expected[i] and output[i] is 0.5, which exceeds threshold";
  }

  OpTester test("Resize", 10);
  std::vector<float> scales{2.0f, 4.0f};
  test.AddAttribute("mode", "linear");

  constexpr int64_t H = 2, W = 2;
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

TEST(ResizeOpTest, ResizeOpLinearScalesNoOpTest_Ver10) {
  OpTester test("Resize", 10);
  std::vector<float> scales{1.0f, 1.0f, 1.0f, 1.0f};
  test.AddAttribute("mode", "linear");

  constexpr int64_t N = 2, C = 1, H = 2, W = 2;
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

  constexpr int64_t N = 1, C = 1, H = 2, W = 4;
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

  constexpr int64_t N = 1, C = 1, H = 2, W = 2;
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

  constexpr int64_t N = 1, C = 1, H = 2, W = 2;
  std::vector<float> X = {1.0f, 2.0f, 3.0f, 4.0f};

  test.AddInput<float>("X", {N, C, H, W}, X);
  test.AddInput<float>("scales", {4}, scales);

  std::vector<float> Y = {1.0f, 2.0f, 3.0f, 4.0f};

  test.AddOutput<float>("Y", {N, C, H, W}, Y);
  test.Run();
}

TEST(ResizeOpTest, ResizeOp_MissingRoiAndMissingScalesOptionalInputs) {
  // TODO: Unskip when fixed #41968513
  if (DefaultDmlExecutionProvider().get() != nullptr) {
    GTEST_SKIP() << "Skipping because of the following error: MLOperatorAuthorImpl.cpp(1876): The parameter is incorrect.";
  }

  OpTester test("Resize", 13);

  test.AddAttribute("mode", "linear");
  test.AddAttribute("coordinate_transformation_mode", "tf_crop_and_resize");

  constexpr int64_t H = 4, W = 4;

  std::vector<float> X = {
      1.0f, 1.0f, 1.0f, 1.0f,
      1.0f, 6.0f, 1.0f, 1.0f,
      1.0f, 1.0f, 1.0f, 11.0f,
      1.0f, 1.0f, 1.0f, 1.0f};

  test.AddInput<float>("X", {H, W}, X);
  test.AddOptionalInputEdge<float>();
  test.AddOptionalInputEdge<float>();
  test.AddInput<int64_t>("sizes", {2}, {4, 4});

  test.AddOutput<float>("Y", {H, W}, X);
  test.Run();
}

template <typename T>
void ResizeOpTypeCheck_Ver_10() {
  OpTester test("Resize", 10);
  std::vector<float> scales{1.0f, 1.0f, 2.0f, 3.0f};

  test.AddAttribute("mode", "nearest");

  constexpr int64_t N = 1, C = 1, H = 2, W = 2;
  std::vector<T> X = {1, 2, 3, 4};

  test.AddInput<T>("X", {N, C, H, W}, X);
  test.AddInput<float>("scales", {4}, scales);

  std::vector<T> Y = {1, 1, 1, 2, 2, 2,
                      1, 1, 1, 2, 2, 2,
                      3, 3, 3, 4, 4, 4,
                      3, 3, 3, 4, 4, 4};

  test.AddOutput<T>("Y", {N, C, static_cast<int64_t>(H * scales[2]), static_cast<int64_t>(W * scales[3])}, Y);
  test.Run();
}

TEST(ResizeOpTest, ResizeOpTypeCheck_Ver_10) {
  ResizeOpTypeCheck_Ver_10<float>();
  ResizeOpTypeCheck_Ver_10<int32_t>();
  ResizeOpTypeCheck_Ver_10<int8_t>();
  ResizeOpTypeCheck_Ver_10<uint8_t>();
}

template <typename T>
void ResizeOpTypeCheck_Ver_11_13_18(int opset_version) {
  OpTester test("Resize", opset_version);
  std::vector<float> roi{};
  std::vector<float> scales{1.0f, 1.0f, 2.0f, 3.0f};

  test.AddAttribute("mode", "nearest");

  constexpr int64_t N = 1, C = 1, H = 2, W = 2;
  std::vector<T> X = {1, 2, 3, 4};

  test.AddInput<T>("X", {N, C, H, W}, X);
  test.AddInput<float>("roi", {0}, roi);
  test.AddInput<float>("scales", {4}, scales);

  std::vector<T> Y = {1, 1, 1, 2, 2, 2,
                      1, 1, 1, 2, 2, 2,
                      3, 3, 3, 4, 4, 4,
                      3, 3, 3, 4, 4, 4};

  test.AddOutput<T>("Y", {N, C, static_cast<int64_t>(H * scales[2]), static_cast<int64_t>(W * scales[3])}, Y);
  test.Run();
}

TEST(ResizeOpTest, ResizeOpTypeCheck_Ver11) {
  ResizeOpTypeCheck_Ver_11_13_18<float>(11);
  ResizeOpTypeCheck_Ver_11_13_18<int32_t>(11);
  ResizeOpTypeCheck_Ver_11_13_18<int8_t>(11);
  ResizeOpTypeCheck_Ver_11_13_18<uint8_t>(11);
}

TEST(ResizeOpTest, ResizeOpTypeCheck_Ver13) {
  ResizeOpTypeCheck_Ver_11_13_18<float>(13);
  ResizeOpTypeCheck_Ver_11_13_18<int32_t>(13);
  ResizeOpTypeCheck_Ver_11_13_18<int8_t>(13);
  ResizeOpTypeCheck_Ver_11_13_18<uint8_t>(13);
}

TEST(ResizeOpTest, ResizeOpTypeCheck_Ver18) {
  ResizeOpTypeCheck_Ver_11_13_18<float>(18);
  ResizeOpTypeCheck_Ver_11_13_18<int32_t>(18);
  ResizeOpTypeCheck_Ver_11_13_18<int8_t>(18);
  ResizeOpTypeCheck_Ver_11_13_18<uint8_t>(18);
}

TEST(ResizeOpTest, ResizeOpHalfPixelSymmetricDownSample_ver19) {
  OpTester test("Resize", 19);
  test.AddAttribute("mode", "linear");
  test.AddAttribute("coordinate_transformation_mode", "half_pixel_symmetric");

  constexpr int64_t N = 1, C = 1, H = 1, W = 4;
  std::vector<float> X = {1.0f, 2.0f, 3.0f, 4.0f};
  test.AddInput<float>("X", {N, C, H, W}, X);
  std::vector<float> roi{};
  test.AddInput<float>("roi", {0}, roi);
  std::vector<float> scales{1.0f, 1.0f, 1.0f, 0.6f};
  test.AddInput<float>("scales", {4}, scales);

  std::vector<float> Y = {1.666667f, 3.333333f};

  test.AddOutput<float>("Y", {N, C, static_cast<int64_t>(H * scales[2]), static_cast<int64_t>(W * scales[3])}, Y);
  test.Run();
}

TEST(ResizeOpTest, ResizeOpHalfPixelSymmetricUpSample_ver19) {
  OpTester test("Resize", 19);
  test.AddAttribute("mode", "linear");
  test.AddAttribute("coordinate_transformation_mode", "half_pixel_symmetric");

  constexpr int64_t N = 1, C = 1, H = 2, W = 2;
  std::vector<float> X = {1.0f, 2.0f, 3.0f, 4.0f};

  test.AddInput<float>("X", {N, C, H, W}, X);
  std::vector<float> roi{};
  test.AddInput<float>("roi", {0}, roi);
  std::vector<float> scales{1.0f, 1.0f, 2.3f, 2.94f};
  test.AddInput<float>("scales", {4}, scales);

  std::vector<float> Y = {1.0f, 1.159864f, 1.5f, 1.840136f, 2.0f,
                          1.565217f, 1.725081f, 2.065217f, 2.405353f, 2.565217f,
                          2.434782f, 2.594647f, 2.934783f, 3.274919f, 3.434783f,
                          3.0f, 3.159864f, 3.5f, 3.840136f, 4.0f};

  test.AddOutput<float>("Y", {N, C, static_cast<int64_t>(H * scales[2]), static_cast<int64_t>(W * scales[3])}, Y);
  test.Run();
}

/*
 * Most of TestCase against Anti-aliasing will have the attribute of "exclude_outside" as 1.
 * It's as Pillow 's Resize is corresponding to ONNX op with exclude_outside equaling 1.
 * Besides, for cubic mode, PIllow's one has a default value of 0.5 for "cubic_coeff_a",
 * while ONNX op has a default value of 0.75.
 */
template <typename T, typename T1 = int64_t>
void TestAntialiasing(std::map<std::string, std::string> attributes,
                      std::vector<int64_t> input_shape,
                      std::vector<T> input_data,
                      std::vector<T1> output_shape_or_scale, std::vector<T> output_data) {
  auto parse_attr = [](const std::string& str, auto typed_v) {
    using Tdata = decltype(typed_v);
    std::vector<Tdata> vect;

    std::stringstream ss(str.substr(1, str.size() - 2));

    for (Tdata i; ss >> i;) {
      vect.push_back(i);
      if (ss.peek() == ',')
        ss.ignore();
    }
    return vect;
  };

  OpTester test("Resize", 18);
  test.AddAttribute<int64_t>("antialias", 1LL);

  std::vector<float> roi{};
  std::vector<float> scales{};
  std::vector<int64_t> output_shape;

  for (auto& [k, v] : attributes) {
    if (k == "mode") {
      test.AddAttribute("mode", v);
    } else if (k == "exclude_outside") {
      test.AddAttribute<int64_t>("exclude_outside", std::stoll(v));
    } else if (k == "cubic_coeff_a") {
      test.AddAttribute<float>("cubic_coeff_a", std::stof(v));
    } else if (k == "axes") {
      int64_t type = 0;
      test.AddAttribute<std::vector<int64_t>>("axes", parse_attr(v, type));
    } else if (k == "output_shape") {
      int64_t type = 0;
      output_shape = parse_attr(v, type);
    } else if (k == "coordinate_transformation_mode") {
      test.AddAttribute("coordinate_transformation_mode", v);
    } else if (k == "policy") {
      test.AddAttribute("keep_aspect_ratio_policy", v);
    } else if (k == "extrapolation_value") {
      test.AddAttribute<float>("extrapolation_value", std::stof(v));
    } else if (k == "roi") {
      roi = parse_attr(v, 0.0f);
    } else {
      throw std::invalid_argument("Unknown attribute");
    }
  }

  test.AddInput<T>("X", input_shape, input_data);
  test.AddInput<float>("roi", {int64_t(roi.size())}, roi);

  if constexpr (std::is_same_v<T1, float>) {
    test.AddInput<float>("scales", {int64_t(output_shape_or_scale.size())}, output_shape_or_scale, true);
  } else {
    test.AddInput<float>("", {0}, scales);
    test.AddInput<int64_t>("sizes", {int64_t(output_shape_or_scale.size())}, output_shape_or_scale, true);
    if (output_shape.empty()) {
      output_shape = output_shape_or_scale;
    }
  }

  test.AddOutput<T>("Y", output_shape, output_data);
  // TensorRT 8.5 supports operators up to Opset 17. Temporarily exclude TensorRT EP due to accurarcy issue.
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});
}

TEST(ResizeOpTest, Antialias_Bilinear_No_ExcludeOutside) {
  std::vector<float> X(16);
  std::iota(X.begin(), X.end(), 1.f);

  std::vector<float> Y = {2.3636363f, 3.590909f, 4.818182f,
                          7.2727275f, 8.5f, 9.727273f,
                          12.181818f, 13.409091f, 14.636364f};
  TestAntialiasing({{"mode", "linear"}, {"exclude_outside", "0"}}, {1, 1, 4, 4}, X, {1, 1, 3, 3}, Y);
}

// match pillow
TEST(ResizeOpTest, Antialias_Bilinear_ExcludeOutside) {
  std::vector<float> X(16);
  std::iota(X.begin(), X.end(), 1.f);
  std::vector<float> Y = {2.5f, 3.7f, 4.9f,
                          7.3f, 8.5f, 9.7f,
                          12.1f, 13.3f, 14.5f};
  TestAntialiasing({{"mode", "linear"}, {"exclude_outside", "1"}}, {1, 1, 4, 4}, X, {1, 1, 3, 3}, Y);
}

TEST(ResizeOpTest, Antialias_Bilinear_Scale_Is_All_1) {
  std::vector<float> X(3 * 4 * 5 * 6);
  std::iota(X.begin(), X.end(), 1.f);
  std::vector<float> Y = X;
  TestAntialiasing({{"mode", "linear"}, {"exclude_outside", "1"}}, {3, 4, 5, 6}, X, {3, 4, 5, 6}, Y);
}

TEST(ResizeOpTest, Antialias_Bilinear_dtype) {
  {
    std::vector<uint8_t> X(16);
    std::iota(X.begin(), X.end(), uint8_t(0));
    std::vector<uint8_t> Y = {1, 3, 4,
                              6, 8, 9,
                              11, 13, 14};
    TestAntialiasing({{"mode", "linear"}, {"exclude_outside", "1"}}, {1, 1, 4, 4}, X, {1, 1, 3, 3}, Y);
  }
  {
    std::vector<int8_t> X(16);
    std::iota(X.begin(), X.end(), int8_t(0));
    std::vector<int8_t> Y = {1, 3, 4,
                             6, 8, 9,
                             11, 13, 14};
    TestAntialiasing({{"mode", "linear"}, {"exclude_outside", "1"}}, {1, 1, 4, 4}, X, {1, 1, 3, 3}, Y);
  }
  {
    std::vector<int32_t> X(16);
    std::iota(X.begin(), X.end(), 0);
    std::vector<int32_t> Y = {1, 3, 4,
                              6, 8, 9,
                              11, 13, 14};
    TestAntialiasing({{"mode", "linear"}, {"exclude_outside", "1"}}, {1, 1, 4, 4}, X, {1, 1, 3, 3}, Y);
  }
}

TEST(ResizeOpTest, Antialias_NhwcBilinear) {
  std::vector<float> X(3 * 5 * 8);
  std::vector<float> X1(3 * 5 * 8);
  std::iota(X1.begin(), X1.end(), 0.f);
  for (size_t x = 0; x < 5; x++) {
    for (size_t y = 0; y < 8; y++) {
      for (size_t c = 0; c < 3; c++) {
        X[x * 8 * 3 + y * 3 + c] = X1[c * 5 * 8 + x * 8 + y];
      }
    }
  }
  std::vector<float> Y = {2.409091f, 42.409092f, 82.40909f,
                          3.925926f, 43.925926f, 83.92593f,
                          5.5f, 45.5f, 85.5f,
                          7.0740743f, 47.074074f, 87.07407f,
                          8.590909f, 48.590908f, 88.59091f,
                          11.742424f, 51.742424f, 91.742424f,
                          13.259259f, 53.25926f, 93.25926f,
                          14.833333f, 54.833332f, 94.833336f,
                          16.407408f, 56.407406f, 96.40741f,
                          17.924242f, 57.924244f, 97.92424f,
                          21.075758f, 61.075756f, 101.07576f,
                          22.592592f, 62.592594f, 102.59259f,
                          24.166666f, 64.166664f, 104.166664f,
                          25.74074f, 65.74074f, 105.74074f,
                          27.257576f, 67.257576f, 107.257576f,
                          30.40909f, 70.40909f, 110.40909f,
                          31.925926f, 71.92593f, 111.92593f,
                          33.5f, 73.5f, 113.5f,
                          35.074074f, 75.07407f, 115.07407f,
                          36.590908f, 76.59091f, 116.59091f};
  TestAntialiasing({{"mode", "linear"}, {"exclude_outside", "1"}}, {1, 5, 8, 3}, X, {1, 4, 5, 3}, Y);
}

TEST(ResizeOpTest, Antialias_NhwcBilinear_dtype) {
  {
    std::vector<uint8_t> X(16);
    std::iota(X.begin(), X.end(), uint8_t(0));
    std::vector<uint8_t> Y = {1, 3, 4,
                              6, 8, 9,
                              11, 13, 14};
    TestAntialiasing({{"mode", "linear"}, {"exclude_outside", "1"}}, {1, 4, 4, 1}, X, {1, 3, 3, 1}, Y);
  }
  {
    std::vector<int8_t> X(16);
    std::iota(X.begin(), X.end(), int8_t(0));
    std::vector<int8_t> Y = {1, 3, 4,
                             6, 8, 9,
                             11, 13, 14};
    TestAntialiasing({{"mode", "linear"}, {"exclude_outside", "1"}}, {1, 4, 4, 1}, X, {1, 3, 3, 1}, Y);
  }
  {
    std::vector<int32_t> X(16);
    std::iota(X.begin(), X.end(), 0);
    std::vector<int32_t> Y = {1, 3, 4,
                              6, 8, 9,
                              11, 13, 14};
    TestAntialiasing({{"mode", "linear"}, {"exclude_outside", "1"}}, {1, 4, 4, 1}, X, {1, 3, 3, 1}, Y);
  }
}

TEST(ResizeOpTest, Antialias_Trilinear_No_ExcludeOutside) {
  std::vector<float> X(16 * 4);
  std::iota(X.begin(), X.end(), 0.f);
  std::vector<float> Y = {5.7272725f, 6.9545455f, 8.181818f, 10.636364f, 11.863636f,
                          13.090909f, 15.545455f, 16.772728f, 18.f, 25.363636f,
                          26.59091f, 27.818182f, 30.272728f, 31.5f, 32.727272f,
                          35.18182f, 36.409092f, 37.636364f, 45.f, 46.227272f,
                          47.454544f, 49.909092f, 51.136364f, 52.363636f, 54.81818f,
                          56.045456f, 57.272728f};
  TestAntialiasing({{"mode", "linear"}, {"exclude_outside", "0"}}, {4, 4, 4}, X, {3, 3, 3}, Y);
}

TEST(ResizeOpTest, Antialias_Trilinear_ExcludeOutside) {
  std::vector<float> X(16 * 4);
  std::iota(X.begin(), X.end(), 0.f);
  std::vector<float> Y = {6.3f, 7.5f, 8.7f, 11.1f, 12.3f, 13.5f, 15.9f, 17.1f, 18.3f, 25.5f, 26.7f,
                          27.9f, 30.3f, 31.5f, 32.7f, 35.1f, 36.3f, 37.5f, 44.7f, 45.9f, 47.1f, 49.5f,
                          50.7f, 51.9f, 54.3f, 55.5f, 56.7f};
  TestAntialiasing({{"mode", "linear"}, {"exclude_outside", "1"}}, {4, 4, 4}, X, {3, 3, 3}, Y);
}

TEST(ResizeOpTest, Antialias_Trilinear_Scale_Is_11s_and_1s1) {
  std::vector<float> X(16 * 4 * 4);
  std::iota(X.begin(), X.end(), 0.f);
  {
    std::vector<float> Y = X;
    TestAntialiasing({{"mode", "linear"}, {"exclude_outside", "1"}}, {4, 1, 4, 4, 4}, X, {4, 1, 4, 4, 4}, Y);
  }
  {
    std::vector<float> Y = {0.625f, 2.375f, 4.625f, 6.375f, 8.625f, 10.375f, 12.625f,
                            14.375f, 16.625f, 18.375f, 20.625f, 22.375f, 24.625f, 26.375f,
                            28.625f, 30.375f, 32.625f, 34.375f, 36.625f, 38.375f, 40.625f,
                            42.375f, 44.625f, 46.375f, 48.625f, 50.375f, 52.625f, 54.375f,
                            56.625f, 58.375f, 60.625f, 62.375f, 64.625f, 66.375f, 68.625f,
                            70.375f, 72.625f, 74.375f, 76.625f, 78.375f, 80.625f, 82.375f,
                            84.625f, 86.375f, 88.625f, 90.375f, 92.625f, 94.375f, 96.625f,
                            98.375f, 100.625f, 102.375f, 104.625f, 106.375f, 108.625f, 110.375f,
                            112.625f, 114.375f, 116.625f, 118.375f, 120.625f, 122.375f, 124.625f,
                            126.375f, 128.625f, 130.375f, 132.625f, 134.375f, 136.625f, 138.375f,
                            140.625f, 142.375f, 144.625f, 146.375f, 148.625f, 150.375f, 152.625f,
                            154.375f, 156.625f, 158.375f, 160.625f, 162.375f, 164.625f, 166.375f,
                            168.625f, 170.375f, 172.625f, 174.375f, 176.625f, 178.375f, 180.625f,
                            182.375f, 184.625f, 186.375f, 188.625f, 190.375f, 192.625f, 194.375f,
                            196.625f, 198.375f, 200.625f, 202.375f, 204.625f, 206.375f, 208.625f,
                            210.375f, 212.625f, 214.375f, 216.625f, 218.375f, 220.625f, 222.375f,
                            224.625f, 226.375f, 228.625f, 230.375f, 232.625f, 234.375f, 236.625f,
                            238.375f, 240.625f, 242.375f, 244.625f, 246.375f, 248.625f, 250.375f,
                            252.625f, 254.375f};
    TestAntialiasing({{"mode", "linear"}, {"exclude_outside", "0"}}, {4, 1, 4, 4, 4}, X, {4, 1, 4, 4, 2}, Y);
  }
  {
    std::vector<float> Y = {2.5f, 3.5f, 4.5f, 5.5f, 9.5f, 10.5f, 11.5f, 12.5f, 18.5f,
                            19.5f, 20.5f, 21.5f, 25.5f, 26.5f, 27.5f, 28.5f, 34.5f, 35.5f,
                            36.5f, 37.5f, 41.5f, 42.5f, 43.5f, 44.5f, 50.5f, 51.5f, 52.5f,
                            53.5f, 57.5f, 58.5f, 59.5f, 60.5f, 66.5f, 67.5f, 68.5f, 69.5f,
                            73.5f, 74.5f, 75.5f, 76.5f, 82.5f, 83.5f, 84.5f, 85.5f, 89.5f,
                            90.5f, 91.5f, 92.5f, 98.5f, 99.5f, 100.5f, 101.5f, 105.5f, 106.5f,
                            107.5f, 108.5f, 114.5f, 115.5f, 116.5f, 117.5f, 121.5f, 122.5f, 123.5f,
                            124.5f, 130.5f, 131.5f, 132.5f, 133.5f, 137.5f, 138.5f, 139.5f, 140.5f,
                            146.5f, 147.5f, 148.5f, 149.5f, 153.5f, 154.5f, 155.5f, 156.5f, 162.5f,
                            163.5f, 164.5f, 165.5f, 169.5f, 170.5f, 171.5f, 172.5f, 178.5f, 179.5f,
                            180.5f, 181.5f, 185.5f, 186.5f, 187.5f, 188.5f, 194.5f, 195.5f, 196.5f,
                            197.5f, 201.5f, 202.5f, 203.5f, 204.5f, 210.5f, 211.5f, 212.5f, 213.5f,
                            217.5f, 218.5f, 219.5f, 220.5f, 226.5f, 227.5f, 228.5f, 229.5f, 233.5f,
                            234.5f, 235.5f, 236.5f, 242.5f, 243.5f, 244.5f, 245.5f, 249.5f, 250.5f,
                            251.5f, 252.5f};
    TestAntialiasing({{"mode", "linear"}, {"exclude_outside", "0"}}, {4, 1, 4, 4, 4}, X, {4, 1, 4, 2, 4}, Y);
  }
}

TEST(ResizeOpTest, Antialias_Bicubic_No_ExcludeOutside) {
  std::vector<float> X(48);
  std::iota(X.begin(), X.end(), 1.0f);
  std::vector<float> Y = {2.175381f, 3.655320f, 5.204702f, 6.684640f, 10.245370f,
                          11.725308f, 13.274692f, 14.754630f, 18.315359f, 19.795298f,
                          21.344681f, 22.824619f, 26.175381f, 27.655319f, 29.204702f,
                          30.684641f, 34.245369f, 35.725307f, 37.274693f, 38.754631f,
                          42.315361f, 43.795300f, 45.344681f, 46.824619f};
  TestAntialiasing({{"mode", "cubic"}, {"exclude_outside", "0"}}, {1, 2, 4, 6}, X, {1, 2, 3, 4}, Y);
}

TEST(ResizeOpTest, Antialias_NHWCBicubic_ExcludeOutside) {
  std::vector<float> X(48);
  std::vector<float> X1(48);
  std::iota(X1.begin(), X1.end(), 1.0f);
  for (size_t x = 0; x < 4; x++) {
    for (size_t y = 0; y < 6; y++) {
      for (size_t c = 0; c < 2; c++) {
        X[x * 6 * 2 + y * 2 + c] = X1[c * 4 * 6 + x * 6 + y];
      }
    }
  }
  std::vector<float> Y = {
      0.6125579f, 24.612558f, 2.0924962f, 26.092497f, 3.6418788f,
      27.641878f, 5.121817f, 29.121817f, 2.393808f, 26.393808f,
      3.8737462f, 27.873747f, 5.423129f, 29.423128f, 6.903067f,
      30.903067f, 5.253183f, 29.253183f, 6.733121f, 30.733122f,
      8.282504f, 32.282505f, 9.762443f, 33.762444f, 9.02662f,
      33.02662f, 10.506558f, 34.506557f, 12.055942f, 36.055943f,
      13.53588f, 37.53588f, 11.46412f, 35.46412f, 12.944058f,
      36.944057f, 14.493442f, 38.493443f, 15.97338f, 39.97338f,
      15.237557f, 39.237556f, 16.717497f, 40.717495f, 18.266878f,
      42.26688f, 19.746817f, 43.74682f, 18.096933f, 42.09693f,
      19.576872f, 43.57687f, 21.126253f, 45.126255f, 22.606192f,
      46.606194f, 19.878183f, 43.87818f, 21.358122f, 45.35812f,
      22.907503f, 46.907505f, 24.387442f, 48.387444f};
  TestAntialiasing({{"mode", "cubic"}, {"exclude_outside", "0"}}, {1, 4, 6, 2}, X, {1, 8, 4, 2}, Y);
}

TEST(ResizeOpTest, Antialias_Linear_AlignCorners) {
  std::vector<float> X(256);
  std::iota(X.begin(), X.end(), 0.0f);

  std::vector<float> Y = {
      3.9166667f, 6.4166665f, 13.916667f, 16.416666f, 25.25f,
      27.75f, 35.25f, 37.75f, 46.583332f, 49.083332f,
      56.583332f, 59.083332f, 67.916664f, 70.416664f, 77.916664f,
      80.416664f, 89.25f, 91.75f, 99.25f, 101.75f,
      110.583336f, 113.083336f, 120.583336f, 123.083336f, 131.91667f,
      134.41667f, 141.91667f, 144.41667f, 153.25f, 155.75f,
      163.25f, 165.75f, 174.58333f, 177.08333f, 184.58333f,
      187.08333f, 195.91667f, 198.41667f, 205.91667f, 208.41667f,
      217.25f, 219.75f, 227.25f, 229.75f, 238.58333f,
      241.08333f, 248.58333f, 251.08333f};
  TestAntialiasing(
      {{"mode", "linear"}, {"exclude_outside", "0"}, {"coordinate_transformation_mode", "align_corners"}},
      {4, 1, 4, 4, 4}, X, {4, 1, 3, 2, 2}, Y);
}

TEST(ResizeOpTest, Antialias_Bicubic_ExcludeOutside) {
  std::vector<float> X(48);
  std::iota(X.begin(), X.end(), 1.0f);
  std::vector<float> Y = {2.222252f, 3.670954f, 5.259818f, 6.708520f, 10.256866f, 11.705568f,
                          13.294432f, 14.743134f, 18.291479f, 19.740183f, 21.329046f,
                          22.777748f, 26.222252f, 27.670954f, 29.259817f,
                          30.708521f, 34.256866f, 35.705566f, 37.294434f, 38.743134f, 42.291481f,
                          43.740181f, 45.329044f, 46.777748f};
  TestAntialiasing({{"mode", "cubic"}, {"exclude_outside", "1"}}, {1, 2, 4, 6}, X, {1, 2, 3, 4}, Y);
}

TEST(ResizeOpTest, Antialias_Bicubic_Dtype) {
  {
    std::vector<uint8_t> X(36);
    std::iota(X.begin(), X.end(), uint8_t(0));
    std::vector<uint8_t> Y = {4, 6, 7, 16, 18, 19, 28, 30, 31};
    TestAntialiasing({{"mode", "cubic"}, {"cubic_coeff_a", "-0.5f"}, {"exclude_outside", "1"}}, {1, 1, 6, 6}, X, {1, 1, 3, 3}, Y);
  }
  {
    std::vector<int8_t> X(36);
    std::iota(X.begin(), X.end(), int8_t(0));
    std::vector<int8_t> Y = {4, 6, 7, 16, 18, 19, 28, 30, 31};
    TestAntialiasing({{"mode", "cubic"}, {"cubic_coeff_a", "-0.5f"}, {"exclude_outside", "1"}}, {1, 1, 6, 6}, X, {1, 1, 3, 3}, Y);
  }
  {
    std::vector<int32_t> X(36);
    std::iota(X.begin(), X.end(), 0);
    std::vector<int32_t> Y = {4, 6, 7, 16, 18, 19, 28, 30, 31};
    TestAntialiasing({{"mode", "cubic"}, {"cubic_coeff_a", "-0.5f"}, {"exclude_outside", "1"}}, {1, 1, 6, 6}, X, {1, 1, 3, 3}, Y);
  }
}

// test new attributes
TEST(ResizeOpTest, Antialias_Axes_and_Scale) {
  std::vector<float> X(16 * 4);
  std::iota(X.begin(), X.end(), 0.f);
  std::vector<float> Y = {6.3f, 7.5f, 8.7f, 11.1f, 12.3f, 13.5f, 15.9f, 17.1f, 18.3f, 25.5f, 26.7f,
                          27.9f, 30.3f, 31.5f, 32.7f, 35.1f, 36.3f, 37.5f, 44.7f, 45.9f, 47.1f, 49.5f,
                          50.7f, 51.9f, 54.3f, 55.5f, 56.7f};
  TestAntialiasing({{"mode", "linear"}, {"exclude_outside", "1"}, {"axes", "{2,3,4}"}, {"output_shape", "{1,1,3,3,3}"}}, {1, 1, 4, 4, 4}, X,
                   std::vector<float>{3 / 4.0f, 3 / 4.0f, 3 / 4.0f}, Y);
}

TEST(ResizeOpTest, Antialias_Axes_and_Size) {
  std::vector<float> X(16 * 4);
  std::iota(X.begin(), X.end(), 0.f);
  std::vector<float> Y = {6.3f, 7.5f, 8.7f, 11.1f, 12.3f, 13.5f, 15.9f, 17.1f, 18.3f, 25.5f, 26.7f,
                          27.9f, 30.3f, 31.5f, 32.7f, 35.1f, 36.3f, 37.5f, 44.7f, 45.9f, 47.1f, 49.5f,
                          50.7f, 51.9f, 54.3f, 55.5f, 56.7f};
  TestAntialiasing({{"mode", "linear"}, {"exclude_outside", "1"}, {"axes", "{2,3,4}"}, {"output_shape", "{1,1,3,3,3}"}}, {1, 1, 4, 4, 4}, X,
                   {3, 3, 3}, Y);
}

TEST(ResizeOpTest, Antialias_Axes_and_PolicyNoLarger) {
  std::vector<float> X(16 * 4);
  std::iota(X.begin(), X.end(), 0.f);
  std::vector<float> Y = {6.3f, 7.5f, 8.7f, 11.1f, 12.3f, 13.5f, 15.9f, 17.1f, 18.3f, 25.5f, 26.7f,
                          27.9f, 30.3f, 31.5f, 32.7f, 35.1f, 36.3f, 37.5f, 44.7f, 45.9f, 47.1f, 49.5f,
                          50.7f, 51.9f, 54.3f, 55.5f, 56.7f};
  TestAntialiasing({{"mode", "linear"}, {"exclude_outside", "1"}, {"axes", "{2,3,4}"}, {"output_shape", "{1,1,3,3,3}"}, {"policy", "not_larger"}},
                   {1, 1, 4, 4, 4}, X,
                   {3, 4, 5}, Y);
}

TEST(ResizeOpTest, Antialias_Axes_and_PolicyNoSmaller) {
  std::vector<float> X(16 * 4);
  std::iota(X.begin(), X.end(), 0.f);
  std::vector<float> Y = {6.3f, 7.5f, 8.7f, 11.1f, 12.3f, 13.5f, 15.9f, 17.1f, 18.3f, 25.5f, 26.7f,
                          27.9f, 30.3f, 31.5f, 32.7f, 35.1f, 36.3f, 37.5f, 44.7f, 45.9f, 47.1f, 49.5f,
                          50.7f, 51.9f, 54.3f, 55.5f, 56.7f};
  TestAntialiasing({{"mode", "linear"}, {"exclude_outside", "1"}, {"axes", "{2,3,4}"}, {"output_shape", "{1,1,3,3,3}"}, {"policy", "not_smaller"}},
                   {1, 1, 4, 4, 4}, X,
                   {1, 2, 3}, Y);
}

TEST(ResizeOpTest, Antialias_Use_Extrapolation) {
  std::vector<float> X(16 * 4);
  std::iota(X.begin(), X.end(), 0.f);
  std::vector<float> Y = {4.5555553f, 5.4385967f, 6.1666665f, 9.888889f, 10.77193f,
                          11.5f, 15.222222f, 16.105263f, 16.833334f, 16.20468f,
                          17.08772f, 17.81579f, 21.538012f, 22.421053f, 23.149124f,
                          26.871346f, 27.754387f, 28.482456f, 30.333334f, 31.216375f,
                          31.944447f, 35.666668f, 36.54971f, 37.27778f, 41.,
                          41.88304f, 42.61111f};
  TestAntialiasing(
      {{"mode", "linear"}, {"exclude_outside", "0"}, {"extrapolation_value", "1.1f"},

       {"coordinate_transformation_mode", "tf_crop_and_resize"},
       {"roi", "{0, 0, 0.4, 0.6, 1, 1}"},
       {"axes", "{0,1,2}"}

      },
      {4, 4, 4}, X, {3, 3, 3}, Y);
}
}  // namespace test
}  // namespace onnxruntime
