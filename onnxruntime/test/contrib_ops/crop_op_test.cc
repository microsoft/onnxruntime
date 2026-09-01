// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"
#include "test/providers/provider_test_utils.h"

#include <limits>

namespace onnxruntime {
namespace test {

namespace {

void RunInvalidCropTest(const std::vector<int64_t>& input_shape,
                        const std::vector<int64_t>& border,
                        const std::vector<int64_t>* scale) {
  OpTester test("Crop", 7, onnxruntime::kOnnxDomain);
  test.AddInput<float>("x", input_shape, std::vector<float>(TensorShape(input_shape).Size(), 0.0f));
  test.AddAttribute("border", border);
  if (scale != nullptr) {
    test.AddAttribute("scale", *scale);
  }
  // Expected output values and shape are not checked when failure is expected.
  test.AddOutput<float>("y", {1, 1, 1, 1}, {0.0f});
  test.Run(OpTester::ExpectResult::kExpectFailure);
}

}  // namespace

TEST(CropOpTest, Crop_Border) {
  OpTester test("Crop", 1, onnxruntime::kOnnxDomain);
  test.AddInput<float>("x", {1, 1, 4, 4}, {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0});
  std::vector<int64_t> border{1, 1, 1, 1};
  test.AddAttribute("border", border);
  test.AddOutput<float>("y", {1, 1, 2, 2}, {6.0, 7.0, 10.0, 11.0});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});
}

TEST(CropOpTest, Crop_Scale) {
  OpTester test("Crop", 1, onnxruntime::kOnnxDomain);
  test.AddInput<float>("x", {1, 1, 4, 4}, {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0});

  std::vector<int64_t> border{1, 1, 1, 1};
  test.AddAttribute("border", border);

  std::vector<int64_t> scale{2, 2};
  test.AddAttribute("scale", scale);

  test.AddOutput<float>("y", {1, 1, 2, 2}, {6.0, 7.0, 10.0, 11.0});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});
}

TEST(CropOpTest, Crop_Invalid_Scale_Size) {
  const std::vector<int64_t> scale{2};
  RunInvalidCropTest({1, 1, 4, 4}, {1, 1, 1, 1}, &scale);
}

TEST(CropOpTest, Crop_Empty_Scale) {
  const std::vector<int64_t> scale;
  RunInvalidCropTest({1, 1, 4, 4}, {1, 1, 1, 1}, &scale);
}

TEST(CropOpTest, Crop_Negative_Border) {
  RunInvalidCropTest({1, 1, 4, 4}, {0, -1, 0, 0}, nullptr);
}

TEST(CropOpTest, Crop_Negative_Scale) {
  const std::vector<int64_t> scale{-1, 1};
  RunInvalidCropTest({1, 1, 4, 4}, {0, 0, 0, 0}, &scale);
}

TEST(CropOpTest, Crop_Border_Exceeds_Input) {
  RunInvalidCropTest({1, 1, 4, 4}, {5, 0, 0, 0}, nullptr);
  RunInvalidCropTest({1, 1, 4, 4}, {0, 5, 0, 0}, nullptr);
}

TEST(CropOpTest, Crop_Scale_Exceeds_Input) {
  const std::vector<int64_t> scale{2, 2};
  RunInvalidCropTest({1, 1, 4, 4}, {3, 3, 0, 0}, &scale);
}

TEST(CropOpTest, Crop_Large_Border_Does_Not_Overflow) {
  RunInvalidCropTest({1, 1, 4, 4}, {0, std::numeric_limits<int64_t>::max(), 0, 1}, nullptr);
}

TEST(CropOpTest, Crop_Large_Scale_Does_Not_Overflow) {
  const std::vector<int64_t> scale{std::numeric_limits<int64_t>::max(), 1};
  RunInvalidCropTest({1, 1, 4, 4}, {0, 1, 0, 0}, &scale);
}

TEST(CropOpTest, Crop_Scale_Overrides_Right_And_Bottom_Borders) {
  OpTester test("Crop", 7, onnxruntime::kOnnxDomain);
  test.AddInput<float>("x", {1, 1, 4, 4},
                       {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f,
                        9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f});
  test.AddAttribute("border", std::vector<int64_t>{1, 1, 5, 5});
  test.AddAttribute("scale", std::vector<int64_t>{2, 2});
  test.AddOutput<float>("y", {1, 1, 2, 2}, {6.0f, 7.0f, 10.0f, 11.0f});
  test.Run();
}

TEST(CropOpTest, Crop_Zero_Sized_Output) {
  OpTester test("Crop", 7, onnxruntime::kOnnxDomain);
  test.AddInput<float>("x", {1, 1, 4, 4}, std::vector<float>(16, 0.0f));
  test.AddAttribute("border", std::vector<int64_t>{4, 4, 0, 0});
  test.AddAttribute("scale", std::vector<int64_t>{0, 0});
  test.AddOutput<float>("y", {1, 1, 0, 0}, {});
  test.Run();
}

}  // namespace test
}  // namespace onnxruntime
