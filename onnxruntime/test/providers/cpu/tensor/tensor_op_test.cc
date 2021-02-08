// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"
#include "test/providers/provider_test_utils.h"
#include "test/common/tensor_op_test_utils.h"

using namespace ONNX_NAMESPACE;
namespace onnxruntime {
namespace test {

using ExpectResult = OpTester::ExpectResult;

// Some of the tests can't run on TensorrtExecutionProvider because of unsupported data types.
// Those tests will fallback to other EPs.

TEST(TensorOpTest, Reshape) {
  OpTester test("Reshape");

  test.AddInput<float>("data", {2, 3}, std::vector<float>(6, 1.0f));
  test.AddInput<int64_t>("shape", {3}, {-1, 0, 2});
  test.AddOutput<float>("reshaped", {1, 3, 2}, std::vector<float>(6, 1.0f));
  //TensorRT doesn't support dynamic shape tensor for now
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kNupharExecutionProvider, kTensorrtExecutionProvider});  // Nuphar only supports reshape shape from initializer
}

TEST(TensorOpTest, ReshapeWithEmptyDim) {
  OpTester test("Reshape");

  test.AddInput<float>("data", {1, 1, 1}, std::vector<float>(1, 1.0f));
  test.AddInput<int64_t>("shape", {0}, {}, true);
  test.AddOutput<float>("reshaped", {}, std::vector<float>(1, 1.0f));
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});  // TensorRT doesn't support empty dimension
}

TEST(TensorOpTest, ReshapeWithEmptyInput) {
  OpTester test("Reshape");
  test.AddInput<float>("data", {0, 10}, std::vector<float>());
  test.AddInput<int64_t>("shape", {3}, {0, 10, 1}, false);
  test.AddOutput<float>("reshaped", {0, 10, 1}, std::vector<float>());
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});  // TensorRT doesn't support empty dimension
}

TEST(TensorOpTest, ReshapeWithEmptyInputAndDynamicShape) {
  {
    OpTester test("Reshape");
    test.AddInput<float>("data", {1, 0}, std::vector<float>());
    test.AddInput<int64_t>("shape", {3}, {1, 0, -1}, false);
    test.AddOutput<float>("reshaped", {1, 0, 1}, {});
    test.Run(OpTester::ExpectResult::kExpectFailure, "The input tensor cannot be reshaped to the requested shape", {kTensorrtExecutionProvider});  // TensorRT doesn't support empty dimension
  }

  {
    OpTester test("Reshape");
    test.AddInput<float>("data", {1, 0}, std::vector<float>());
    test.AddInput<int64_t>("shape", {3}, {1, 1, -1}, false);
    test.AddOutput<float>("reshaped", {1, 1, 0}, {});
    test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});  // TensorRT doesn't support empty dimension
  }
}

TEST(TensorOpTest, ReshapeWithInitializer) {
  OpTester test("Reshape");

  test.AddInput<float>("data", {2, 3}, std::vector<float>(6, 1.0f));
  test.AddInput<int64_t>("shape", {3}, {-1, 0, 2}, true);
  test.AddOutput<float>("reshaped", {1, 3, 2}, std::vector<float>(6, 1.0f));
  test.Run();
}

TEST(TensorOpTest, ShapeTest2D) {
  OpTester test("Shape");

  test.AddInput<float>("data", {2, 3}, std::vector<float>(6, 1.0f));
  test.AddOutput<int64_t>("shape", {2}, {2, 3});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});  //TensorRT: volume of dimensions is not consistent with weights size
}

TEST(TensorOpTest, ShapeTest3D) {
  OpTester test("Shape");

  test.AddInput<float>("data", {2, 3, 4}, std::vector<float>(24, 1.0f));
  test.AddOutput<int64_t>("shape", {3}, {2, 3, 4});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});  //TensorRT: volume of dimensions is not consistent with weights size
}

void MeanVarianceNormalizationFunctionDefaultPerChannel() {
  const int64_t N = 2, C = 2, H = 2, W = 3;

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
  Normalize(N1C1_result, C1_meam_stdev, 1);
  Normalize(N2C1_result, C1_meam_stdev, 1);
  Normalize(N1C2_result, C2_meam_stdev, 1);
  Normalize(N2C2_result, C2_meam_stdev, 1);

  std::vector<float> result;
  result.reserve(N * C * H * W);
  result.insert(result.end(), N1C1_result.begin(), N1C1_result.end());
  result.insert(result.end(), N1C2_result.begin(), N1C2_result.end());
  result.insert(result.end(), N2C1_result.begin(), N2C1_result.end());
  result.insert(result.end(), N2C2_result.begin(), N2C2_result.end());

  OpTester test("MeanVarianceNormalization", 9);
  test.AddInput<float>("input", {N, C, H, W}, X);
  test.AddOutput<float>("output", {N, C, H, W}, result);
  test.Run();
}

void MeanVarianceNormalizationFunctionAcrossChannels(std::vector<int64_t> axes) {
  const int64_t N = 2, C = 2, H = 2, W = 3;

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
  Normalize(result, mean_stdev, 1);

  OpTester test("MeanVarianceNormalization", 9);
  test.AddAttribute("axes", axes);
  test.AddInput<float>("input", {N, C, H, W}, X);
  test.AddOutput<float>("output", {N, C, H, W}, result);
  test.Run();
}

TEST(TensorOpTest, MeanVarianceNormalizationCPUTest) {
  // axes: {0, 1, 2, 3} for across_channels
  MeanVarianceNormalizationFunctionAcrossChannels({0, 1, 2, 3});

  // Default (axes: {0, 2, 3}) for non across_channels
  MeanVarianceNormalizationFunctionDefaultPerChannel();
}

}  // namespace test
}  // namespace onnxruntime
