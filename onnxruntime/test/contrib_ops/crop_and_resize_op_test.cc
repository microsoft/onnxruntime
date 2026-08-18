// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"
#include "test/providers/provider_test_utils.h"

#include <limits>

namespace onnxruntime {
namespace test {

// TensorRT EP is disabled in all tests because the operator is not supported (Attribute not found: crop_width)

TEST(CropAndResizeTest, CropAndResize_1122) {
  OpTester test1("CropAndResize", 1, onnxruntime::kMSDomain);
  test1.AddInput<float>("X", {1, 1, 2, 2}, {1.1f, 2.2f, 3.3f, 4.4f});
  test1.AddInput<float>("rois", {3, 4}, {0.0f, 0.0f, 1.0f, 1.0f, 0.0f, 0.0f, 0.5f, 0.5f, 0.0f, 0.0f, 0.5f, 1.0f});
  test1.AddInput<int32_t>("batch_indices", {3}, {0, 0, 0});
  test1.AddInput<int32_t>("crop_size", {2}, {1, 1});
  test1.AddOutput<float>("output", {3, 1, 1, 1}, {2.75f, 1.925f, 2.2f});
  test1.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});

  OpTester test2("CropAndResize", 1, onnxruntime::kMSDomain);
  test2.AddInput<float>("X", {1, 1, 2, 2}, {1.1f, 2.2f, 3.3f, 4.4f});
  test2.AddInput<float>("rois", {3, 4}, {0.0f, 0.0f, 1.0f, 1.0f, 0.0f, 0.0f, 0.5f, 0.5f, 0.0f, 0.0f, 0.5f, 1.0f});
  test2.AddInput<int32_t>("batch_indices", {3}, {0, 0, 0});
  test2.AddInput<int32_t>("crop_size", {2}, {2, 2});
  test2.AddOutput<float>("output", {3, 1, 2, 2}, {1.1f, 2.2f, 3.3f, 4.4f, 1.1f, 1.65f, 2.2f, 2.75f, 1.1f, 2.2f, 2.2f, 3.3f});
  test2.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});

  OpTester test3("CropAndResize", 1, onnxruntime::kMSDomain);
  test3.AddInput<float>("X", {1, 1, 2, 2}, {1.1f, 2.2f, 3.3f, 4.4f});
  test3.AddInput<float>("rois", {2, 4}, {0.0f, 0.0f, 1.5f, 1.5f, 0.25f, 0.25f, 0.75f, 0.5f});
  test3.AddInput<int32_t>("batch_indices", {2}, {0, 0});
  test3.AddInput<int32_t>("crop_size", {2}, {2, 2});
  test3.AddAttribute("extrapolation_value", (float)5.5);
  test3.AddOutput<float>("output", {2, 1, 2, 2}, {1.1f, 5.5f, 5.5f, 5.5f, 1.925f, 2.2f, 3.025f, 3.3f});
  test3.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});
}

TEST(CropAndResizeTest, CropAndResize_2122) {
  OpTester test1("CropAndResize", 1, onnxruntime::kMSDomain);
  test1.AddInput<float>("X", {2, 1, 2, 2}, {1.1f, 2.2f, 3.3f, 4.4f, 5.5f, 6.6f, 7.7f, 8.8f});
  test1.AddInput<float>("rois", {3, 4}, {0.0f, 0.0f, 1.0f, 1.0f, 0.0f, 0.0f, 0.5f, 0.5f, 0.0f, 0.0f, 0.5f, 1.0f});
  test1.AddInput<int32_t>("batch_indices", {3}, {0, 1, 1});
  test1.AddInput<int32_t>("crop_size", {2}, {1, 1});
  test1.AddOutput<float>("output", {3, 1, 1, 1}, {2.75f, 6.325f, 6.6f});
  test1.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});

  OpTester test2("CropAndResize", 1, onnxruntime::kMSDomain);
  test2.AddInput<float>("X", {2, 1, 2, 2}, {1.1f, 2.2f, 3.3f, 4.4f, 5.5f, 6.6f, 7.7f, 8.8f});
  test2.AddInput<float>("rois", {3, 4}, {0.0f, 0.0f, 1.0f, 1.0f, 0.0f, 0.0f, 0.5f, 0.5f, 0.0f, 0.0f, 0.5f, 1.0f});
  test2.AddInput<int32_t>("batch_indices", {3}, {0, 1, 1});
  test2.AddInput<int32_t>("crop_size", {2}, {2, 2});
  test2.AddOutput<float>("output", {3, 1, 2, 2}, {1.1f, 2.2f, 3.3f, 4.4f, 5.5f, 6.05f, 6.6f, 7.15f, 5.5f, 6.6f, 6.6f, 7.7f});
  test2.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});
}

TEST(CropAndResizeTest, CropAndResize_1222) {
  OpTester test1("CropAndResize", 1, onnxruntime::kMSDomain);
  test1.AddInput<float>("X", {1, 2, 2, 2}, {1.1f, 2.2f, 3.3f, 4.4f, 5.5f, 6.6f, 7.7f, 8.8f});
  test1.AddInput<float>("rois", {3, 4}, {0.0f, 0.0f, 1.0f, 1.0f, 0.0f, 0.0f, 0.5f, 0.5f, 0.0f, 0.0f, 0.5f, 1.0f});
  test1.AddInput<int32_t>("batch_indices", {3}, {0, 0, 0});
  test1.AddInput<int32_t>("crop_size", {2}, {1, 1});
  test1.AddOutput<float>("output", {3, 2, 1, 1}, {2.75f, 7.15f, 1.925f, 6.325f, 2.2f, 6.6f});
  test1.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});

  OpTester test2("CropAndResize", 1, onnxruntime::kMSDomain);
  test2.AddInput<float>("X", {1, 2, 2, 2}, {1.1f, 2.2f, 3.3f, 4.4f, 5.5f, 6.6f, 7.7f, 8.8f});
  test2.AddInput<float>("rois", {3, 4}, {0.0f, 0.0f, 1.0f, 1.0f, 0.0f, 0.0f, 0.5f, 0.5f, 0.0f, 0.0f, 0.5f, 1.0f});
  test2.AddInput<int32_t>("batch_indices", {3}, {0, 0, 0});
  test2.AddInput<int32_t>("crop_size", {2}, {2, 2});
  test2.AddOutput<float>("output", {3, 2, 2, 2}, {1.1f, 2.2f, 3.3f, 4.4f, 5.5f, 6.6f, 7.7f, 8.8f, 1.1f, 1.65f, 2.2f, 2.75f, 5.5f, 6.05f, 6.6f, 7.15f, 1.1f, 2.2f, 2.2f, 3.3f, 5.5f, 6.6f, 6.6f, 7.7f});
  test2.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});
}

TEST(CropAndResizeTest, CropAndResizeRejectsMalformedCropSize) {
  OpTester test("CropAndResize", 1, onnxruntime::kMSDomain);
  test.AddInput<float>("X", {1, 1, 2, 2}, {1.1f, 2.2f, 3.3f, 4.4f});
  test.AddInput<float>("rois", {1, 4}, {0.0f, 0.0f, 1.0f, 1.0f});
  test.AddInput<int32_t>("batch_indices", {1}, {0});
  test.AddInput<int32_t>("crop_size", {1}, {2});
  test.AddOutput<float>("output", {1, 1, 1, 1}, {0.0f});

  test.Run(OpTester::ExpectResult::kExpectFailure,
           "[ShapeInferenceError] crop_size input tensor must have exactly 2 elements; got 1",
           {kTensorrtExecutionProvider});

  OpTester test_with_three_elements("CropAndResize", 1, onnxruntime::kMSDomain);
  test_with_three_elements.AddInput<float>("X", {1, 1, 2, 2}, {1.1f, 2.2f, 3.3f, 4.4f});
  test_with_three_elements.AddInput<float>("rois", {1, 4}, {0.0f, 0.0f, 1.0f, 1.0f});
  test_with_three_elements.AddInput<int32_t>("batch_indices", {1}, {0});
  test_with_three_elements.AddInput<int32_t>("crop_size", {3}, {2, 2, 2});
  test_with_three_elements.AddOutput<float>("output", {1, 1, 1, 1}, {0.0f});

  test_with_three_elements.Run(OpTester::ExpectResult::kExpectFailure,
                               "[ShapeInferenceError] crop_size input tensor must have exactly 2 elements; got 3",
                               {kTensorrtExecutionProvider});
}

TEST(CropAndResizeTest, CropAndResize_1133) {
  OpTester test1("CropAndResize", 1, onnxruntime::kMSDomain);
  test1.AddInput<float>("X", {1, 1, 3, 3}, {1.1f, 2.2f, 3.3f, 4.4f, 5.5f, 6.6f, 7.7f, 8.8f, 9.9f});
  test1.AddInput<float>("rois", {3, 4}, {0.0f, 0.0f, 1.0f, 1.0f, 0.0f, 0.0f, 0.5f, 0.5f, 0.0f, 0.0f, 0.5f, 1.0f});
  test1.AddInput<int32_t>("batch_indices", {3}, {0, 0, 0});
  test1.AddInput<int32_t>("crop_size", {2}, {1, 1});
  test1.AddOutput<float>("output", {3, 1, 1, 1}, {5.5f, 3.3f, 3.85f});
  test1.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});

  OpTester test2("CropAndResize", 1, onnxruntime::kMSDomain);
  test2.AddInput<float>("X", {1, 1, 3, 3}, {1.1f, 2.2f, 3.3f, 4.4f, 5.5f, 6.6f, 7.7f, 8.8f, 9.9f});
  test2.AddInput<float>("rois", {3, 4}, {0.0f, 0.0f, 1.0f, 1.0f, 0.0f, 0.0f, 0.5f, 0.5f, 0.0f, 0.0f, 0.5f, 1.0f});
  test2.AddInput<int32_t>("batch_indices", {3}, {0, 0, 0});
  test2.AddInput<int32_t>("crop_size", {2}, {2, 2});
  test2.AddOutput<float>("output", {3, 1, 2, 2}, {1.1f, 3.3f, 7.7f, 9.9f, 1.1f, 2.2f, 4.4f, 5.5f, 1.1f, 3.3f, 4.4f, 6.6f});
  test2.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});

  OpTester test3("CropAndResize", 1, onnxruntime::kMSDomain);
  test3.AddInput<float>("X", {1, 1, 3, 3}, {1.1f, 2.2f, 3.3f, 4.4f, 5.5f, 6.6f, 7.7f, 8.8f, 9.9f});
  test3.AddInput<float>("rois", {3, 4}, {0.0f, 0.0f, 1.0f, 1.0f, 0.0f, 0.0f, 0.5f, 0.5f, 0.0f, 0.0f, 0.5f, 1.0f});
  test3.AddInput<int32_t>("batch_indices", {3}, {0, 0, 0});
  test3.AddInput<int32_t>("crop_size", {2}, {2, 2});
  test3.AddAttribute("mode", "nearest");
  test3.AddOutput<float>("output", {3, 1, 2, 2}, {1.1f, 3.3f, 7.7f, 9.9f, 1.1f, 2.2f, 4.4f, 5.5f, 1.1f, 3.3f, 4.4f, 6.6f});
  test3.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});
}

// Non-finite ROI coordinates (NaN) must route to the extrapolation branch instead of
// reaching the integer index math, which would otherwise compute an invalid image index.
TEST(CropAndResizeTest, CropAndResize_NaN_roi_extrapolates) {
  const float nan = std::numeric_limits<float>::quiet_NaN();

  // NaN start/end height -> in_y is NaN for every row -> whole output extrapolates.
  OpTester test_y("CropAndResize", 1, onnxruntime::kMSDomain);
  test_y.AddInput<float>("X", {1, 1, 2, 2}, {1.1f, 2.2f, 3.3f, 4.4f});
  test_y.AddInput<float>("rois", {1, 4}, {nan, 0.0f, nan, 1.0f});
  test_y.AddInput<int32_t>("batch_indices", {1}, {0});
  test_y.AddInput<int32_t>("crop_size", {2}, {2, 2});
  test_y.AddAttribute("extrapolation_value", 5.5f);
  test_y.AddOutput<float>("output", {1, 1, 2, 2}, {5.5f, 5.5f, 5.5f, 5.5f});
  test_y.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});

  // Finite height but NaN width -> in_y is in range, in_x is NaN -> every pixel extrapolates.
  OpTester test_x("CropAndResize", 1, onnxruntime::kMSDomain);
  test_x.AddInput<float>("X", {1, 1, 2, 2}, {1.1f, 2.2f, 3.3f, 4.4f});
  test_x.AddInput<float>("rois", {1, 4}, {0.0f, nan, 1.0f, nan});
  test_x.AddInput<int32_t>("batch_indices", {1}, {0});
  test_x.AddInput<int32_t>("crop_size", {2}, {2, 2});
  test_x.AddAttribute("extrapolation_value", 5.5f);
  test_x.AddOutput<float>("output", {1, 1, 2, 2}, {5.5f, 5.5f, 5.5f, 5.5f});
  test_x.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});

  // Same NaN-height case in nearest mode.
  OpTester test_nearest("CropAndResize", 1, onnxruntime::kMSDomain);
  test_nearest.AddInput<float>("X", {1, 1, 2, 2}, {1.1f, 2.2f, 3.3f, 4.4f});
  test_nearest.AddInput<float>("rois", {1, 4}, {nan, 0.0f, nan, 1.0f});
  test_nearest.AddInput<int32_t>("batch_indices", {1}, {0});
  test_nearest.AddInput<int32_t>("crop_size", {2}, {2, 2});
  test_nearest.AddAttribute("mode", "nearest");
  test_nearest.AddAttribute("extrapolation_value", 5.5f);
  test_nearest.AddOutput<float>("output", {1, 1, 2, 2}, {5.5f, 5.5f, 5.5f, 5.5f});
  test_nearest.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});
}

// Infinite ROI coordinates must also land in the extrapolation branch.
TEST(CropAndResizeTest, CropAndResize_Inf_roi_extrapolates) {
  const float inf = std::numeric_limits<float>::infinity();

  OpTester test_pos("CropAndResize", 1, onnxruntime::kMSDomain);
  test_pos.AddInput<float>("X", {1, 1, 2, 2}, {1.1f, 2.2f, 3.3f, 4.4f});
  test_pos.AddInput<float>("rois", {1, 4}, {inf, 0.0f, inf, 1.0f});
  test_pos.AddInput<int32_t>("batch_indices", {1}, {0});
  test_pos.AddInput<int32_t>("crop_size", {2}, {2, 2});
  test_pos.AddAttribute("extrapolation_value", 5.5f);
  test_pos.AddOutput<float>("output", {1, 1, 2, 2}, {5.5f, 5.5f, 5.5f, 5.5f});
  test_pos.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});

  OpTester test_neg("CropAndResize", 1, onnxruntime::kMSDomain);
  test_neg.AddInput<float>("X", {1, 1, 2, 2}, {1.1f, 2.2f, 3.3f, 4.4f});
  test_neg.AddInput<float>("rois", {1, 4}, {-inf, 0.0f, -inf, 1.0f});
  test_neg.AddInput<int32_t>("batch_indices", {1}, {0});
  test_neg.AddInput<int32_t>("crop_size", {2}, {2, 2});
  test_neg.AddAttribute("extrapolation_value", 5.5f);
  test_neg.AddOutput<float>("output", {1, 1, 2, 2}, {5.5f, 5.5f, 5.5f, 5.5f});
  test_neg.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});
}

// Finite coordinates on and just outside the image boundary must behave exactly as the
// original guard: coordinates in [0, dim-1] interpolate, coordinates just past dim-1 extrapolate.
TEST(CropAndResizeTest, CropAndResize_finite_boundary_no_regression) {
  // Exact boundary [0, 1] maps to the identity crop (no extrapolation at 0 or height-1).
  OpTester test_exact("CropAndResize", 1, onnxruntime::kMSDomain);
  test_exact.AddInput<float>("X", {1, 1, 2, 2}, {1.1f, 2.2f, 3.3f, 4.4f});
  test_exact.AddInput<float>("rois", {1, 4}, {0.0f, 0.0f, 1.0f, 1.0f});
  test_exact.AddInput<int32_t>("batch_indices", {1}, {0});
  test_exact.AddInput<int32_t>("crop_size", {2}, {2, 2});
  test_exact.AddAttribute("extrapolation_value", 9.0f);
  test_exact.AddOutput<float>("output", {1, 1, 2, 2}, {1.1f, 2.2f, 3.3f, 4.4f});
  test_exact.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});

  // End just past the boundary (1.0001) extrapolates only the out-of-range row/column.
  OpTester test_outside("CropAndResize", 1, onnxruntime::kMSDomain);
  test_outside.AddInput<float>("X", {1, 1, 2, 2}, {1.1f, 2.2f, 3.3f, 4.4f});
  test_outside.AddInput<float>("rois", {1, 4}, {0.0f, 0.0f, 1.0001f, 1.0001f});
  test_outside.AddInput<int32_t>("batch_indices", {1}, {0});
  test_outside.AddInput<int32_t>("crop_size", {2}, {2, 2});
  test_outside.AddAttribute("extrapolation_value", 9.0f);
  test_outside.AddOutput<float>("output", {1, 1, 2, 2}, {1.1f, 9.0f, 9.0f, 9.0f});
  test_outside.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});
}

// crop_size values must be positive; non-positive crop dimensions are rejected via Status.
TEST(CropAndResizeTest, CropAndResize_rejects_nonpositive_crop_size) {
  OpTester test_zero("CropAndResize", 1, onnxruntime::kMSDomain);
  test_zero.AddInput<float>("X", {1, 1, 2, 2}, {1.1f, 2.2f, 3.3f, 4.4f});
  test_zero.AddInput<float>("rois", {1, 4}, {0.0f, 0.0f, 1.0f, 1.0f});
  test_zero.AddInput<int32_t>("batch_indices", {1}, {0});
  test_zero.AddInput<int32_t>("crop_size", {2}, {0, 2});
  test_zero.AddOutput<float>("output", {1, 1, 1, 1}, {0.0f});
  test_zero.Run(OpTester::ExpectResult::kExpectFailure,
                "crop_size values must be positive", {kTensorrtExecutionProvider});

  OpTester test_negative("CropAndResize", 1, onnxruntime::kMSDomain);
  test_negative.AddInput<float>("X", {1, 1, 2, 2}, {1.1f, 2.2f, 3.3f, 4.4f});
  test_negative.AddInput<float>("rois", {1, 4}, {0.0f, 0.0f, 1.0f, 1.0f});
  test_negative.AddInput<int32_t>("batch_indices", {1}, {0});
  test_negative.AddInput<int32_t>("crop_size", {2}, {-1, 2});
  test_negative.AddOutput<float>("output", {1, 1, 1, 1}, {0.0f});
  test_negative.Run(OpTester::ExpectResult::kExpectFailure,
                    "crop_size values must be positive", {kTensorrtExecutionProvider});
}

// A batch index outside [0, batch_size) must be rejected rather than computing an invalid image index.
TEST(CropAndResizeTest, CropAndResize_rejects_out_of_range_batch_index) {
  OpTester test("CropAndResize", 1, onnxruntime::kMSDomain);
  test.AddInput<float>("X", {1, 1, 2, 2}, {1.1f, 2.2f, 3.3f, 4.4f});
  test.AddInput<float>("rois", {1, 4}, {0.0f, 0.0f, 1.0f, 1.0f});
  test.AddInput<int32_t>("batch_indices", {1}, {5});
  test.AddInput<int32_t>("crop_size", {2}, {2, 2});
  test.AddOutput<float>("output", {1, 1, 2, 2}, {0.0f, 0.0f, 0.0f, 0.0f});
  test.Run(OpTester::ExpectResult::kExpectFailure,
           "is out of range [0, 1)", {kTensorrtExecutionProvider});
}

}  // namespace test
}  // namespace onnxruntime
