// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"
#include "test/providers/provider_test_utils.h"
#include "test/common/dnnl_op_test_utils.h"
#include "test/common/tensor_op_test_utils.h"
#include "test/util/include/default_providers.h"
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
  // TensorRT doesn't support dynamic shape tensor for now
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});
}

TEST(TensorOpTest, ReshapeWithEmptyDim) {
  OpTester test("Reshape");

  test.AddInput<float>("data", {1, 1, 1}, std::vector<float>(1, 1.0f));
  test.AddInput<int64_t>("shape", {0}, {}, true);
  test.AddOutput<float>("reshaped", {}, std::vector<float>(1, 1.0f));
  // TensorRT doesn't support empty dimension
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});
}

TEST(TensorOpTest, ReshapeWithEmptyInput) {
  OpTester test("Reshape");
  test.AddInput<float>("data", {0, 10}, std::vector<float>());
  test.AddInput<int64_t>("shape", {3}, {0, 10, 1}, false);
  test.AddOutput<float>("reshaped", {0, 10, 1}, std::vector<float>());
  // TensorRT, QNN don't support empty dimension
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider, kQnnExecutionProvider});
}

TEST(TensorOpTest, ReshapeWithEmptyInputAndDynamicShape) {
  // TODO: Unskip when fixed #41968513
  if (DefaultDmlExecutionProvider().get() != nullptr) {
    GTEST_SKIP() << "Skipping because of the following error: The input tensor cannot be reshaped to the requested shape. Input shape:{1,0}, requested shape:{1,0,-1}";
  }

  {
    OpTester test("Reshape");
    test.AddInput<float>("data", {1, 0}, std::vector<float>());
    test.AddInput<int64_t>("shape", {3}, {1, 0, -1}, false);
    test.AddOutput<float>("reshaped", {1, 0, 1}, {});
    // TensorRT, QNN don't support empty dimension
    test.Run(OpTester::ExpectResult::kExpectFailure,
             "The input tensor cannot be reshaped to the requested shape",
             {kTensorrtExecutionProvider, kQnnExecutionProvider});
  }

  {
    OpTester test("Reshape");
    test.AddInput<float>("data", {1, 0}, std::vector<float>());
    test.AddInput<int64_t>("shape", {3}, {1, 1, -1}, false);
    test.AddOutput<float>("reshaped", {1, 1, 0}, {});
    // TensorRT, QNN don't support empty dimension
    test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider, kQnnExecutionProvider});
  }
}

TEST(TensorOpTest, ReshapeWithInitializer) {
  OpTester test("Reshape");

  test.AddInput<float>("data", {2, 3}, std::vector<float>(6, 1.0f));
  test.AddInput<int64_t>("shape", {3}, {-1, 0, 2}, true);
  test.AddOutput<float>("reshaped", {1, 3, 2}, std::vector<float>(6, 1.0f));
  test.Run();
}

TEST(TensorOpTest, Reshape_WithOutAllowZero) {
  OpTester test("Reshape", 14);

  test.AddInput<float>("data", {2, 3}, std::vector<float>(6, 1.0f));
  test.AddInput<int64_t>("shape", {2}, {0, 3});
  test.AddAttribute<int64_t>("allowzero", 0);
  test.AddOutput<float>("reshaped", {2, 3}, std::vector<float>(6, 1.0f));
  // TensorRT doesn't support dynamic shape tensor for now
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});
}

TEST(TensorOpTest, Reshape_WithAllowZero) {
  // TODO: Unskip when fixed #41968513
  if (DefaultDmlExecutionProvider().get() != nullptr) {
    GTEST_SKIP() << "Skipping because of the following error: MLOperatorAuthorImpl.cpp(2100): The parameter is incorrect.";
  }

  OpTester test("Reshape", 14);

  test.AddInput<float>("data", {2, 3}, std::vector<float>(6, 1.0f));
  test.AddInput<int64_t>("shape", {2}, {0, 3});
  test.AddAttribute<int64_t>("allowzero", 1);
  test.AddOutput<float>("reshaped", {2, 3}, std::vector<float>(6, 1.0f));
  // TensorRT doesn't support dynamic shape tensor for now
  test.Run(OpTester::ExpectResult::kExpectFailure,
           "The input tensor cannot be reshaped to the requested shape",
           {kTensorrtExecutionProvider});
}

TEST(TensorOpTest, Reshape_EmptyInputWithoutAllowZero) {
  // TODO: Unskip when fixed #41968513
  if (DefaultDmlExecutionProvider().get() != nullptr) {
    GTEST_SKIP() << "Skipping because of the following error: MLOperatorAuthorImpl.cpp(2100): The parameter is incorrect.";
  }

  OpTester test("Reshape");

  test.AddInput<float>("data", {0, 3, 4}, std::vector<float>());
  test.AddInput<int64_t>("shape", {3}, {3, 4, 0});
  test.AddOutput<float>("reshaped", {3, 4, 0}, std::vector<float>());
  // TensorRT, QNN don't support dynamic shape tensor for now
  test.Run(OpTester::ExpectResult::kExpectFailure,
           "The input tensor cannot be reshaped to the requested shape",
           {kTensorrtExecutionProvider, kQnnExecutionProvider});
}

TEST(TensorOpTest, Reshape_EmptyInputWithAllowZero) {
  OpTester test("Reshape", 14);

  test.AddInput<float>("data", {0, 3, 4}, std::vector<float>());
  test.AddInput<int64_t>("shape", {3}, {3, 4, 0});
  test.AddAttribute<int64_t>("allowzero", 1);
  test.AddOutput<float>("reshaped", {3, 4, 0}, std::vector<float>());

  // TensorRT doesn't support dynamic shape tensor for now
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});
}

TEST(TensorOpTest, Reshape_UnknownDimWithoutAllowZero) {
  OpTester test("Reshape");

  test.AddInput<float>("data", {2, 3}, std::vector<float>(6, 1.0f));
  test.AddInput<int64_t>("shape", {2}, {-1, 6});
  test.AddOutput<float>("reshaped", {1, 6}, std::vector<float>(6, 1.0f));
  test.Run();
}

TEST(TensorOpTest, Reshape_UnknownDimWithAllowZero) {
  // TODO: Unskip when fixed #41968513
  if (DefaultDmlExecutionProvider().get() != nullptr) {
    GTEST_SKIP() << "Skipping because of the following error: MLOperatorAuthorImpl.cpp(2100): The parameter is incorrect.";
  }

  OpTester test("Reshape", 14);

  test.AddInput<float>("data", {2, 3}, std::vector<float>(6, 1.0f));
  test.AddInput<int64_t>("shape", {2}, {-1, 6});
  test.AddAttribute<int64_t>("allowzero", 1);
  test.AddOutput<float>("reshaped", {1, 6}, std::vector<float>(6, 1.0f));
  test.Run();
}

#if defined(USE_DNNL)
TEST(TensorOpTest, Reshape_bfloat16) {
#ifdef USE_DNNL
  if (!DnnlHasBF16Support()) {
    LOGS_DEFAULT(WARNING) << "Hardware does NOT support BF16";
    return;
  }
#endif
  OpTester test("Reshape", 14);

  test.AddInput<BFloat16>("data", {2, 3}, MakeBFloat16({1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f}));
  test.AddInput<int64_t>("shape", {3}, {-1, 0, 2});
  test.AddOutput<BFloat16>("reshaped", {1, 3, 2}, MakeBFloat16({1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f}));
  // TensorRT doesn't support dynamic shape tensor for now
  // Nuphar only supports reshape shape from initializer
  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
#if defined(USE_DNNL)
  execution_providers.push_back(DefaultDnnlExecutionProvider());
#endif  //  USE_DNNL
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider}, nullptr, &execution_providers);
}

TEST(TensorOpTest, ReshapeWithEmptyDim_bfloat16) {
#ifdef USE_DNNL
  if (!DnnlHasBF16Support()) {
    LOGS_DEFAULT(WARNING) << "Hardware does NOT support BF16";
    return;
  }
#endif
  OpTester test("Reshape", 14);

  test.AddInput<BFloat16>("data", {1, 1}, MakeBFloat16({1.0f}));
  test.AddInput<int64_t>("shape", {0}, {});
  test.AddOutput<BFloat16>("reshaped", {}, MakeBFloat16({1.0f}));
  // TensorRT doesn't support dynamic shape tensor for now
  // Nuphar only supports reshape shape from initializer
  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
#if defined(USE_DNNL)
  execution_providers.push_back(DefaultDnnlExecutionProvider());
#endif  //  USE_DNNL
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider}, nullptr, &execution_providers);
}

TEST(TensorOpTest, ReshapeWithEmptyInput_bfloat16) {
#ifdef USE_DNNL
  if (!DnnlHasBF16Support()) {
    LOGS_DEFAULT(WARNING) << "Hardware does NOT support BF16";
    return;
  }
#endif
  OpTester test("Reshape", 14);
  test.AddInput<BFloat16>("data", {0, 10}, std::vector<BFloat16>());
  test.AddInput<int64_t>("shape", {3}, {0, 10, 1}, false);
  test.AddOutput<BFloat16>("reshaped", {0, 10, 1}, std::vector<BFloat16>());
  // TensorRT doesn't support empty dimension
  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
#if defined(USE_DNNL)
  execution_providers.push_back(DefaultDnnlExecutionProvider());
#endif  //  USE_DNNL
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider}, nullptr, &execution_providers);
}

TEST(TensorOpTest, Reshape_WithOutAllowZero_bfloat16) {
#ifdef USE_DNNL
  if (!DnnlHasBF16Support()) {
    LOGS_DEFAULT(WARNING) << "Hardware does NOT support BF16";
    return;
  }
#endif
  OpTester test("Reshape", 14);

  test.AddInput<BFloat16>("data", {2, 3}, MakeBFloat16({1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f}));
  test.AddInput<int64_t>("shape", {2}, {0, 3});
  test.AddAttribute<int64_t>("allowzero", 0);
  test.AddOutput<BFloat16>("reshaped", {2, 3}, MakeBFloat16({1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f}));
  // TensorRT doesn't support dynamic shape tensor for now
  // Nuphar only supports reshape shape from initializer
  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
#if defined(USE_DNNL)
  execution_providers.push_back(DefaultDnnlExecutionProvider());
#endif  //  USE_DNNL
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider}, nullptr, &execution_providers);
}

TEST(TensorOpTest, Reshape_WithAllowZero_bfloat16) {
#ifdef USE_DNNL
  if (!DnnlHasBF16Support()) {
    LOGS_DEFAULT(WARNING) << "Hardware does NOT support BF16";
    return;
  }
#endif
  OpTester test("Reshape", 14);

  test.AddInput<BFloat16>("data", {2, 3}, MakeBFloat16({1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f}));
  test.AddInput<int64_t>("shape", {2}, {0, 3});
  test.AddAttribute<int64_t>("allowzero", 1);
  test.AddOutput<BFloat16>("reshaped", {2, 3}, MakeBFloat16({1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f}));
  // TensorRT doesn't support dynamic shape tensor for now
  // Nuphar only supports reshape shape from initializer
  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
#if defined(USE_DNNL)
  execution_providers.push_back(DefaultDnnlExecutionProvider());
#endif  //  USE_DNNL
  test.Run(OpTester::ExpectResult::kExpectFailure,
           "The input tensor cannot be reshaped to the requested shape", {kTensorrtExecutionProvider}, nullptr, &execution_providers);
}

TEST(TensorOpTest, ReshapeWithInitializer_bfloat16) {
#ifdef USE_DNNL
  if (!DnnlHasBF16Support()) {
    LOGS_DEFAULT(WARNING) << "Hardware does NOT support BF16";
    return;
  }
#endif
  OpTester test("Reshape", 14);

  test.AddInput<BFloat16>("data", {2, 3}, MakeBFloat16({1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f}));
  test.AddInput<int64_t>("shape", {3}, {-1, 0, 2}, true);
  test.AddOutput<BFloat16>("reshaped", {1, 3, 2}, MakeBFloat16({1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f}));
  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
#if defined(USE_DNNL)
  execution_providers.push_back(DefaultDnnlExecutionProvider());
#endif  //  USE_DNNL
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider}, nullptr, &execution_providers);
}

TEST(TensorOpTest, ReshapeWithEmptyInputAndDynamicShape_bfloat16) {
#ifdef USE_DNNL
  if (!DnnlHasBF16Support()) {
    LOGS_DEFAULT(WARNING) << "Hardware does NOT support BF16";
    return;
  }
#endif
  {
    OpTester test("Reshape", 14);
    test.AddInput<BFloat16>("data", {1, 0}, std::vector<BFloat16>());
    test.AddInput<int64_t>("shape", {3}, {1, 0, -1}, false);
    test.AddOutput<BFloat16>("reshaped", {1, 0, 1}, {});
    // TensorRT doesn't support empty dimension
    std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
#if defined(USE_DNNL)
    execution_providers.push_back(DefaultDnnlExecutionProvider());
#endif  //  USE_DNNL
    test.Run(OpTester::ExpectResult::kExpectFailure,
             "The input tensor cannot be reshaped to the requested shape", {kTensorrtExecutionProvider}, nullptr, &execution_providers);
  }

  {
    OpTester test("Reshape", 14);
    test.AddInput<BFloat16>("data", {1, 0}, std::vector<BFloat16>());
    test.AddInput<int64_t>("shape", {3}, {1, 1, -1}, false);
    test.AddOutput<BFloat16>("reshaped", {1, 1, 0}, {});
    // TensorRT doesn't support empty dimension
    std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
#if defined(USE_DNNL)
    execution_providers.push_back(DefaultDnnlExecutionProvider());
#endif  //  USE_DNNL
    test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider}, nullptr, &execution_providers);
  }
}

TEST(TensorOpTest, Reshape_EmptyInputWithoutAllowZero_bfloat16) {
#ifdef USE_DNNL
  if (!DnnlHasBF16Support()) {
    LOGS_DEFAULT(WARNING) << "Hardware does NOT support BF16";
    return;
  }
#endif
  OpTester test("Reshape", 14);

  test.AddInput<BFloat16>("data", {0, 3, 4}, std::vector<BFloat16>());
  test.AddInput<int64_t>("shape", {3}, {3, 4, 0});
  test.AddOutput<BFloat16>("reshaped", {3, 4, 0}, std::vector<BFloat16>());
  // TensorRT doesn't support dynamic shape tensor for now
  // Nuphar only supports reshape shape from initializer
  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
#if defined(USE_DNNL)
  execution_providers.push_back(DefaultDnnlExecutionProvider());
#endif  //  USE_DNNL
  test.Run(OpTester::ExpectResult::kExpectFailure,
           "The input tensor cannot be reshaped to the requested shape", {kTensorrtExecutionProvider}, nullptr, &execution_providers);
}

TEST(TensorOpTest, Reshape_EmptyInputWithAllowZero_bfloat16) {
#ifdef USE_DNNL
  if (!DnnlHasBF16Support()) {
    LOGS_DEFAULT(WARNING) << "Hardware does NOT support BF16";
    return;
  }
#endif
  OpTester test("Reshape", 14);

  test.AddInput<BFloat16>("data", {0, 3, 4}, std::vector<BFloat16>());
  test.AddInput<int64_t>("shape", {3}, {3, 4, 0});
  test.AddAttribute<int64_t>("allowzero", 1);
  test.AddOutput<BFloat16>("reshaped", {3, 4, 0}, std::vector<BFloat16>());

  // TensorRT doesn't support dynamic shape tensor for now
  // Nuphar only supports reshape shape from initializer
  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
#if defined(USE_DNNL)
  execution_providers.push_back(DefaultDnnlExecutionProvider());
#endif  //  USE_DNNL
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider}, nullptr, &execution_providers);
}

TEST(TensorOpTest, Reshape_UnknownDimWithoutAllowZero_bfloat16) {
#ifdef USE_DNNL
  if (!DnnlHasBF16Support()) {
    LOGS_DEFAULT(WARNING) << "Hardware does NOT support BF16";
    return;
  }
#endif
  OpTester test("Reshape", 14);

  test.AddInput<BFloat16>("data", {2, 3}, MakeBFloat16({1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f}));
  test.AddInput<int64_t>("shape", {2}, {-1, 6});
  test.AddOutput<BFloat16>("reshaped", {1, 6}, MakeBFloat16({1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f}));
  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
#if defined(USE_DNNL)
  execution_providers.push_back(DefaultDnnlExecutionProvider());
#endif  //  USE_DNNL
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
}

TEST(TensorOpTest, Reshape_UnknownDimWithAllowZero_bfloat16) {
#ifdef USE_DNNL
  if (!DnnlHasBF16Support()) {
    LOGS_DEFAULT(WARNING) << "Hardware does NOT support BF16";
    return;
  }
#endif
  OpTester test("Reshape", 14);

  test.AddInput<BFloat16>("data", {2, 3}, MakeBFloat16({1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f}));
  test.AddInput<int64_t>("shape", {2}, {-1, 6});
  test.AddAttribute<int64_t>("allowzero", 1);
  test.AddOutput<BFloat16>("reshaped", {1, 6}, MakeBFloat16({1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f}));
  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
#if defined(USE_DNNL)
  execution_providers.push_back(DefaultDnnlExecutionProvider());
#endif  //  USE_DNNL
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
}
#endif  //  USE_DNNL

TEST(TensorOpTest, ShapeTest2D) {
  OpTester test("Shape");

  test.AddInput<float>("data", {2, 3}, std::vector<float>(6, 1.0f));
  test.AddOutput<int64_t>("shape", {2}, {2, 3});
  // TensorRT: volume of dimensions is not consistent with weights size
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});
}

TEST(TensorOpTest, ShapeTest3D) {
  OpTester test("Shape");

  test.AddInput<float>("data", {2, 3, 4}, std::vector<float>(24, 1.0f));
  test.AddOutput<int64_t>("shape", {3}, {2, 3, 4});
  // TensorRT: volume of dimensions is not consistent with weights size
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});
}

void MeanVarianceNormalizationFunctionDefaultPerChannel() {
  constexpr int64_t N = 2, C = 2, H = 2, W = 3;

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
#if defined(OPENVINO_CONFIG_MYRIAD)
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kOpenVINOExecutionProvider});  // OpenVINO: Unsupported combination of indices in layer "output"
#else
  test.Run();
#endif
}

void MeanVarianceNormalizationFunctionAcrossChannels(std::vector<int64_t> axes) {
  constexpr int64_t N = 2, C = 2, H = 2, W = 3;

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
#if defined(OPENVINO_CONFIG_MYRIAD)
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kOpenVINOExecutionProvider});  // OpenVINO: Unsupported combination of indices in layer "output"
#else
  test.Run();
#endif
}

TEST(TensorOpTest, MeanVarianceNormalizationCPUTest) {
  // axes: {0, 1, 2, 3} for across_channels
  MeanVarianceNormalizationFunctionAcrossChannels({0, 1, 2, 3});

  // Default (axes: {0, 2, 3}) for non across_channels
  MeanVarianceNormalizationFunctionDefaultPerChannel();
}

}  // namespace test
}  // namespace onnxruntime
