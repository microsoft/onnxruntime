// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"
#include "contrib_ops/cpu/crop.h"
#include "test/common/tensor_op_test_utils.h"
#include "test/providers/provider_test_utils.h"
#include "test/util/include/default_providers.h"

using namespace ONNX_NAMESPACE;
using namespace onnxruntime::test;
namespace onnxruntime {
namespace test {

using ExpectResult = OpTester::ExpectResult;

TEST(CropContribOpTest, CropBorderOnly) {
  constexpr int N = 2, C = 1, H = 3, W = 4;
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

TEST(CropContribOpTest, CropBorderAndScale) {
  constexpr int N = 2, C = 1, H = 3, W = 4;
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

TEST(ImageScalerContribOpTest, ImageScalerTest) {
  // Won't fix for DML which only accepts 1 or 3 channels, as the op was only experimental and since removed.
  if (DefaultDmlExecutionProvider().get() != nullptr) {
    GTEST_SKIP() << "Skipping because of the following error: AbiCustomRegistry.cpp(507): The parameter is incorrect.";
  }

  constexpr int64_t N = 1, C = 2, H = 2, W = 2;
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

void MeanVarianceNormalizationAcrossChannels(bool across_channels, bool normalize_variance) {
  constexpr int64_t N = 2, C = 2, H = 2, W = 3;
  constexpr int64_t one = 1;
  constexpr int64_t zero = 0;

  std::vector<float> X = {3.0f, -3.0f, -1.0f,
                          1.0f, 2.0f, -1.0f,
                          -2.0f, -2.0f, -2.0f,
                          4.0f, 1.0f, 4.0f,
                          0.0f, -2.0f, -2.0f,
                          -4.0f, 5.0f, 7.0f,
                          5.0f, -5.0f, -5.0f,
                          3.0f, 4.0f, 4.0f};
  auto mean_stdev = MeanStdev(X);

  std::vector<float> result(X);
  Normalize(result, mean_stdev, normalize_variance);

  OpTester test("MeanVarianceNormalization", 7);
  test.AddAttribute("across_channels", across_channels ? one : zero);
  test.AddAttribute("normalize_variance", normalize_variance ? one : zero);
  test.AddInput<float>("input", {N, C, H, W}, X);
  test.AddOutput<float>("output", {N, C, H, W}, result);
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kOpenVINOExecutionProvider, kTensorrtExecutionProvider});  // OpenVINO doesn't support MVN operator below opset 9. TensorRT doesn't support opset 8 of MVN operator.
}

void MeanVarianceNormalizationPerChannel(bool across_channels, bool normalize_variance) {
  constexpr int64_t N = 2, C = 2, H = 2, W = 3;
  constexpr int64_t one = 1;
  constexpr int64_t zero = 0;

  std::vector<float> N1C1 = {3.0f, -3.0f, -1.0f,
                             1.0f, 2.0f, -1.0f};
  std::vector<float> N1C2 = {-2.0f, -2.0f, -2.0f,
                             4.0f, 1.0f, 4.0f};
  std::vector<float> N2C1 = {
      0.0f,
      -2.0f,
      -2.0f,
      -4.0f,
      5.0f,
      7.0f,
  };
  std::vector<float> N2C2 = {
      5.0f,
      -5.0f,
      -5.0f,
      3.0f,
      4.0f,
      4.0f,
  };

  std::vector<float> X;
  X.reserve(N * C * H * W);
  X.insert(X.end(), N1C1.begin(), N1C1.end());
  X.insert(X.end(), N1C2.begin(), N1C2.end());
  X.insert(X.end(), N2C1.begin(), N2C1.end());
  X.insert(X.end(), N2C2.begin(), N2C2.end());

  std::vector<float> C1;
  C1.reserve(N * H * W);
  C1.insert(C1.end(), N1C1.begin(), N1C1.end());
  C1.insert(C1.end(), N2C1.begin(), N2C1.end());
  auto C1_meam_stdev = MeanStdev(C1);

  std::vector<float> C2;
  C2.reserve(N * H * W);
  C2.insert(C2.end(), N1C2.begin(), N1C2.end());
  C2.insert(C2.end(), N2C2.begin(), N2C2.end());
  auto C2_meam_stdev = MeanStdev(C2);

  std::vector<float> N1C1_result(N1C1), N1C2_result(N1C2),
      N2C1_result(N2C1), N2C2_result(N2C2);
  Normalize(N1C1_result, C1_meam_stdev, normalize_variance);
  Normalize(N2C1_result, C1_meam_stdev, normalize_variance);
  Normalize(N1C2_result, C2_meam_stdev, normalize_variance);
  Normalize(N2C2_result, C2_meam_stdev, normalize_variance);

  std::vector<float> result;
  result.reserve(N * C * H * W);
  result.insert(result.end(), N1C1_result.begin(), N1C1_result.end());
  result.insert(result.end(), N1C2_result.begin(), N1C2_result.end());
  result.insert(result.end(), N2C1_result.begin(), N2C1_result.end());
  result.insert(result.end(), N2C2_result.begin(), N2C2_result.end());

  OpTester test("MeanVarianceNormalization", 7);
  test.AddAttribute("across_channels", across_channels ? one : zero);
  test.AddAttribute("normalize_variance", normalize_variance ? one : zero);
  test.AddInput<float>("input", {N, C, H, W}, X);
  test.AddOutput<float>("output", {N, C, H, W}, result);
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kOpenVINOExecutionProvider, kTensorrtExecutionProvider});  // OpenVINO doesn't support MVN operator below opset 9. TensorRT doesn't support opset 8 of MVN operator.
}

TEST(MVNContribOpTest, MeanVarianceNormalizationCPUTest_Version1_TO_8) {
  // across_channels: true, normalize_variance: true
  MeanVarianceNormalizationAcrossChannels(true, true);

  // across_channels: true, normalize_variance: false
  MeanVarianceNormalizationAcrossChannels(true, false);

  // across_channels: false, normalize_variance: false
  MeanVarianceNormalizationPerChannel(false, false);

  // across_channels: false, normalize_variance: true
  MeanVarianceNormalizationPerChannel(false, true);
}

}  // namespace test
}  // namespace onnxruntime
