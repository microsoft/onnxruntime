// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"
#include "test/common/cuda_op_test_utils.h"
#include "test/providers/provider_test_utils.h"
#include "test/util/include/default_providers.h"

namespace onnxruntime {
namespace test {
// scalar zero & scale with uint8
TEST(DequantizeLinearOpTest, Uint8) {
  OpTester test("DequantizeLinear", 10);
  std::vector<int64_t> dims{4};
  test.AddInput<uint8_t>("x", dims, {0, 3, 128, 255});
  test.AddInput<float>("x_scale", {}, {2.0f});
  test.AddInput<uint8_t>("x_zero_point", {}, {128});
  test.AddOutput<float>("y", dims, {-256.0f, -250.0f, 0.0f, 254.0f});
  // Disable Tensorrt EP due to error:node1_quantize_scale_node: out of bounds channel axis 1. Number of input dimensions is 1.
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});
}

// scalar zero & scale with int8
TEST(DequantizeLinearOpTest, Int8) {
  OpTester test("DequantizeLinear", 10);
  std::vector<int64_t> dims{4};
  test.AddInput<int8_t>("x", dims, {-30, -3, 100, 127});
  test.AddInput<float>("x_scale", {}, {2.0f});
  test.AddInput<int8_t>("x_zero_point", {}, {-10});
  test.AddOutput<float>("y", dims, {-40.0f, 14.0f, 220.0f, 274.0f});
  // Disable Tensorrt EP due to error:node1_quantize_scale_node: out of bounds channel axis 1. Number of input dimensions is 1.
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});
}

// scalar zero & scale with int8
TEST(DequantizeLinearOpTest, Int32) {
  // TODO: Unskip when fixed #41968513
  if (DefaultDmlExecutionProvider().get() != nullptr) {
    GTEST_SKIP() << "Skipping because of the following error: AbiCustomRegistry.cpp(507): The parameter is incorrect";
  }

  OpTester test("DequantizeLinear", 10);
  std::vector<int64_t> dims{4};
  test.AddInput<int32_t>("x", dims, {-30, -3, 100, 127});
  test.AddInput<float>("x_scale", {}, {2.0f});
  test.AddOutput<float>("y", dims, {-60.f, -6.f, 200.f, 254.f});
  test.Run();
}

TEST(DequantizeLinearOpTest_1D, Int32) {
  OpTester test("DequantizeLinear", 13);
  std::vector<int64_t> dims{4};
  test.AddInput<int32_t>("x", dims, {-30, -3, 100, 127});
  test.AddInput<float>("x_scale", {1}, {2.0f});
  test.AddInput<int32_t>("zero_point", {1}, {0});
  test.AddOutput<float>("y", dims, {-60.f, -6.f, 200.f, 254.f});
  test.Run();
}

// 2d inputs
TEST(DequantizeLinearOpTest, 2D) {
  OpTester test("DequantizeLinear", 10);
  std::vector<int64_t> dims{3, 4};
  test.AddInput<uint8_t>("X", dims,
                         {0, 1, 2, 3,
                          0, 1, 2, 3,
                          0, 10, 20, 30});
  test.AddInput<float>("scale", {}, {1.0f});
  test.AddInput<uint8_t>("zero_point", {}, {0});
  test.AddOutput<float>("Y", dims,
                        {0, 1, 2, 3,
                         0, 1, 2, 3,
                         0, 10, 20, 30});
  test.Run();
}

// dequantize with scalar data
TEST(DequantizeLinearOpTest, Scalar) {
  OpTester test("DequantizeLinear", 10);
  test.AddInput<int8_t>("x", {}, {100});
  test.AddInput<float>("x_scale", {}, {2.0f});
  test.AddInput<int8_t>("x_zero_point", {}, {-10});
  test.AddOutput<float>("y", {}, {220.0f});
  // Disable Tensorrt EP due to error:node1_quantize_scale_node: out of bounds channel axis 1. Number of input dimensions is 0.
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});
}

// dequantize with scalar data
TEST(DequantizeLinearOpMLFloat16Test, Scalar) {
  OpTester test("DequantizeLinear", 19);
  test.AddInput<int8_t>("x", {}, {100});
  test.AddInput<MLFloat16>("x_scale", {}, {MLFloat16(2.0f)});
  test.AddInput<int8_t>("x_zero_point", {}, {-10});
  test.AddOutput<MLFloat16>("y", {}, {MLFloat16(220.0f)});
  // Disable Tensorrt EP due to error:node1_quantize_scale_node: out of bounds channel axis 1. Number of input dimensions is 0.
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});
}

// dequantize without zero point
TEST(DequantizeLinearOpTest, Without_Zero_Point) {
  // TODO: Unskip when fixed #41968513
  if (DefaultDmlExecutionProvider().get() != nullptr) {
    GTEST_SKIP() << "Skipping because of the following error: AbiCustomRegistry.cpp(507): The parameter is incorrect";
  }

  OpTester test("DequantizeLinear", 10);
  test.AddInput<int8_t>("x", {}, {100});
  test.AddInput<float>("x_scale", {}, {2.0f});
  test.AddOutput<float>("y", {}, {200.0f});
  test.Run();
}

// 1d zero & scale with default axis
TEST(DequantizeLinearOpTest, Per_Channel_Axis_Default) {
  OpTester test("DequantizeLinear", 13);
  std::vector<int64_t> dims{2, 3, 2, 4};
  test.AddInput<int8_t>("X", dims,
                        {7, 9, 10, 10,
                         5, 8, 9, 1,

                         8, 6, 7, 9,
                         10, 0, 7, 10,

                         8, 2, 6, 0,
                         5, 9, 8, 1,

                         2, 7, 5, 3,
                         2, 4, 1, 3,

                         8, 7, 4, 8,
                         10, 1, 5, 5,

                         7, 7, 0, 2,
                         4, 4, 0, 5});
  test.AddInput<float>("scale", {3}, {1, 10, 7});
  test.AddInput<int8_t>("zero_point", {3}, {10, 2, 1});
  test.AddOutput<float>("Y", dims,
                        {-3, -1, 0, 0,
                         -5, -2, -1, -9,

                         60, 40, 50, 70,
                         80, -20, 50, 80,

                         49, 7, 35, -7,
                         28, 56, 49, 0,

                         -8, -3, -5, -7,
                         -8, -6, -9, -7,

                         60, 50, 20, 60,
                         80, -10, 30, 30,

                         42, 42, -7, 7,
                         21, 21, -7, 28});
  // Disable Tensorrt EP due to the non-zero zero_point.
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});
}

// 1d zero & scale with uint8 broadcast axis 0
TEST(DequantizeLinearOpTest, Per_Channel_Axis_0) {
  OpTester test("DequantizeLinear", 13);
  std::vector<int64_t> dims{3, 4};
  test.AddInput<uint8_t>("X", dims,
                         {0, 1, 2, 3,
                          0, 1, 2, 3,
                          0, 10, 20, 30});
  test.AddAttribute<int64_t>("axis", 0);
  test.AddInput<float>("scale", {3},
                       {1.0f,
                        2.0f,
                        4.0f});
  test.AddInput<uint8_t>("zero_point", {3},
                         {0,
                          0,
                          0});
  test.AddOutput<float>("Y", dims,
                        {0, 1, 2, 3,
                         0, 2, 4, 6,
                         0, 40, 80, 120});
  test.Run();
}

// 1d zero & scale with int8 broadcast axis 1
TEST(DequantizeLinearOpTest, Per_Channel_Axis_1_int8) {
  OpTester test("DequantizeLinear", 13);
  std::vector<int64_t> dims{3, 4};
  test.AddInput<int8_t>("X", dims,
                        {0, 1, 2, 3,
                         0, 2, 4, 6,
                         0, 10, 20, 30});
  test.AddAttribute<int64_t>("axis", 1);
  test.AddInput<float>("scale", {4}, {1, 2, 4, 8});
  test.AddInput<int8_t>("zero_point", {4}, {0, -10, -20, -30});
  test.AddOutput<float>("Y", dims,
                        {0, 22, 88, 264,
                         0, 24, 96, 288,
                         0, 40, 160, 480});
  // Disable Tensorrt EP due to the non-zero zero_point.
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});
}

// 1d zero & scale with int32 broadcast axis 1
TEST(DequantizeLinearOpTest, Per_Channel_Axis_1_int32) {
  OpTester test("DequantizeLinear", 13);
  std::vector<int64_t> dims{3, 4};
  test.AddInput<int32_t>("X", dims,
                         {0, 1, 2, 3,
                          0, 2, 4, 6,
                          0, 10, 20, 30});
  test.AddAttribute<int64_t>("axis", 1);
  test.AddInput<float>("scale", {4}, {1, 2, 4, 8});
  test.AddInput<int32_t>("zero_point", {4}, {0, 0, 0, 0});
  test.AddOutput<float>("Y", dims,
                        {0, 2, 8, 24,
                         0, 4, 16, 48,
                         0, 20, 80, 240});
  // Disable Tensorrt EP due to error, only activation types allowed as input to this layer.
  // Disable CUDA, ROCm EP, there is no implementation for int32_t.
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider, kCudaExecutionProvider, kRocmExecutionProvider});
}

// 1d zero & scale with uint8 broadcast axis -2 (-2 resolves to axis 0)
TEST(DequantizeLinearOpTest, Per_Channel_Neg_2) {
  OpTester test("DequantizeLinear", 13);
  std::vector<int64_t> dims{3, 4};
  test.AddInput<uint8_t>("X", dims,
                         {0, 1, 2, 3,
                          0, 1, 2, 3,
                          0, 10, 20, 30});
  test.AddAttribute<int64_t>("axis", -2);
  test.AddInput<float>("scale", {3},
                       {1.0f,
                        2.0f,
                        4.0f});
  test.AddInput<uint8_t>("zero_point", {3},
                         {0,
                          0,
                          0});
  test.AddOutput<float>("Y", dims,
                        {0, 1, 2, 3,
                         0, 2, 4, 6,
                         0, 40, 80, 120});
  test.Run();
}

// quantize with scalar zero point and scale
TEST(QuantizeLinearOpTest, Uint8) {
  OpTester test("QuantizeLinear", 10);
  std::vector<int64_t> dims{6};
  test.AddInput<float>("x", dims, {0, 2, 3, 1000, -254, -1000});
  test.AddInput<float>("y_scale", {}, {2.0f});
  test.AddInput<uint8_t>("y_zero_point", {}, {128});
  test.AddOutput<uint8_t>("y", dims, {128, 129, 130, 255, 1, 0});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});  // TensorRT doesn't support support UINT8 for quantization
}

TEST(QuantizeLinearOpMLFloat16Test, Uint8) {
  OpTester test("QuantizeLinear", 19);
  std::vector<int64_t> dims{6};
  test.AddInput<MLFloat16>("x", dims, {MLFloat16(0.0f), MLFloat16(2.0f), MLFloat16(4.0f), MLFloat16(1000.0f), MLFloat16(-254.0f), MLFloat16(-1000.0f)});
  test.AddInput<MLFloat16>("y_scale", {}, {MLFloat16(2.0f)});
  test.AddInput<uint8_t>("y_zero_point", {}, {128});
  test.AddOutput<uint8_t>("y", dims, {128, 129, 130, 255, 1, 0});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});  // TensorRT doesn't support support UINT8 for quantization
}

// quantize with scalar zero point and scale
TEST(QuantizeLinearOpTest, Int8) {
  // TODO: Unskip when fixed #41968513
  if (DefaultDmlExecutionProvider().get() != nullptr) {
    GTEST_SKIP() << "Skipping because of the following error: Expected equality of these values: -127 and -128";
  }

  OpTester test("QuantizeLinear", 10);
  std::vector<int64_t> dims{6};
  test.AddInput<float>("x", dims, {0, 2, 3, 5, -2, -5});
  test.AddInput<float>("y_scale", {}, {.039215686f});
  test.AddInput<int8_t>("y_zero_point", {}, {0});
  test.AddOutput<int8_t>("y", dims, {0, 51, 76, 127, -51, -127});
  // Disable Tensorrt EP due to the error, out of bounds channel axis 1. Number of input dimensions is 1.
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});
}

// quantize with scalar zero point and scale
TEST(QuantizeLinearOpTest, Int8_NegativeZeroPoint) {
  // TODO: Unskip when fixed #41968513
  if (DefaultDmlExecutionProvider().get() != nullptr) {
    GTEST_SKIP() << "Skipping because of the following error: Expected equality of these values: 104 and 105";
  }

  OpTester test("QuantizeLinear", 10);
  std::vector<int64_t> dims{8};
  test.AddInput<float>("x", dims, {0, 2, 3, 5, 6, -2, -5, -6});
  test.AddInput<float>("y_scale", {}, {.039215686f});
  test.AddInput<int8_t>("y_zero_point", {}, {-23});
  test.AddOutput<int8_t>("y", dims, {-23, 28, 53, 104, 127, -74, -128, -128});
  // Disable Tensorrt EP due to the error, node1_quantize_scale_node: out of bounds channel axis 1. Number of input dimensions is 1.
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});
}

// quantize with scalar zero point and scale
TEST(QuantizeLinearOpTest, Int8_PositiveZeroPoint) {
  // TODO: Unskip when fixed #41968513
  if (DefaultDmlExecutionProvider().get() != nullptr) {
    GTEST_SKIP() << "Skipping because of the following error: Expected equality of these values: -104 and -105";
  }

  OpTester test("QuantizeLinear", 10);
  std::vector<int64_t> dims{8};
  test.AddInput<float>("x", dims, {0, 2, 3, 5, 6, -2, -5, -6});
  test.AddInput<float>("y_scale", {}, {.039215686f});
  test.AddInput<int8_t>("y_zero_point", {}, {23});
  test.AddOutput<int8_t>("y", dims, {23, 74, 99, 127, 127, -28, -104, -128});
  // Disable Tensorrt EP due to error:node1_quantize_scale_node: out of bounds channel axis 1. Number of input dimensions is 1.
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});
}

// quantize with 2D data
TEST(QuantizeLinearOpTest, 2D) {
  OpTester test("QuantizeLinear", 10);
  std::vector<int64_t> dims{3, 4};
  test.AddInput<float>("X", dims,
                       {0, 2, 3, 1000,
                        0, 2, 3, 1000,
                        0, 2, 3, 1000});
  test.AddInput<float>("scale", {}, {4});
  test.AddInput<uint8_t>("zero_point", {}, {0});
  test.AddOutput<uint8_t>("Y", dims,
                          {0, 0, 1, 250,
                           0, 0, 1, 250,
                           0, 0, 1, 250});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});  // TensorRT doesn't support support UINT8 for quantization
}

// quantize with scalar data
TEST(QuantizeLinearOpTest, Scalar) {
  OpTester test("QuantizeLinear", 10);
  test.AddInput<float>("x", {}, {3});
  test.AddInput<float>("y_scale", {}, {2.0f});
  test.AddInput<uint8_t>("y_zero_point", {}, {128});
  test.AddOutput<uint8_t>("y", {}, {130});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});  // TensorRT doesn't support support UINT8 for quantization
}

// quantize with scalar data
TEST(QuantizeLinearOpTest, QuantizeLinear_Without_Zero_Point_Opset10) {
  // TODO: Unskip when fixed #41968513
  if (DefaultDmlExecutionProvider().get() != nullptr) {
    GTEST_SKIP() << "Skipping because of the following error: AbiCustomRegistry.cpp(507): The parameter is incorrect";
  }

  OpTester test("QuantizeLinear", 10);
  test.AddInput<float>("x", {}, {3});
  test.AddInput<float>("y_scale", {}, {2.0f});
  test.AddOutput<uint8_t>("y", {}, {2});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});  // TensorRT doesn't support support UINT8 for quantization
}

TEST(QuantizeLinearOpTest, QuantizeLinear_Without_Zero_Point_Opset13) {
  // TODO: Unskip when fixed #41968513
  if (DefaultDmlExecutionProvider().get() != nullptr) {
    GTEST_SKIP() << "Skipping because of the following error: AbiCustomRegistry.cpp(507): The parameter is incorrect";
  }

  OpTester test("QuantizeLinear", 13);
  test.AddInput<float>("x", {}, {3});
  test.AddInput<float>("y_scale", {}, {2.0f});
  test.AddOutput<uint8_t>("y", {}, {2});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});  // TensorRT doesn't support support UINT8 for quantization
}

TEST(QuantizeLinearOpTest, QuantizeLinear_With_Zero_Point0) {
  OpTester test("QuantizeLinear", 10);
  test.AddInput<float>("x", {}, {3});
  test.AddInput<float>("y_scale", {}, {2.0f});
  test.AddInput<uint8_t>("y_zero_point", {}, {0});
  test.AddOutput<uint8_t>("y", {}, {2});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});  // TensorRT doesn't support support UINT8 for quantization
}

TEST(QuantizeLinearOpTest, QuantizeLinear_With_Zero_Dim1) {
  // TODO: Unskip when fixed #41968513
  if (DefaultDmlExecutionProvider().get() != nullptr) {
    GTEST_SKIP() << "Skipping because of the following error: AbiCustomRegistry.cpp(507): The parameter is incorrect";
  }

  OpTester test("QuantizeLinear", 10);
  test.AddInput<float>("x", {1}, {3});
  test.AddInput<float>("y_scale", {1}, {2.0f});
  test.AddOutput<uint8_t>("y", {1}, {2});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});  // TensorRT doesn't support support UINT8 for quantization
}

TEST(QuantizeLinearOpTest, Per_Channel_Axis_Default) {
  OpTester test("QuantizeLinear", 13);
  std::vector<int64_t> dims{3, 4};
  test.AddInput<float>("X", dims,
                       {0, 2, 1, 1001,
                        1, 1, 2, 1100,
                        2, 4.2f, 3, 1200});
  test.AddInput<float>("scale", {4}, {1, 2, 3, 20});
  test.AddInput<uint8_t>("zero_point", {4}, {64, 100, 127, 127});
  test.AddOutput<uint8_t>("Y", dims,
                          {64, 101, 127, 177,
                           65, 100, 128, 182,
                           66, 102, 128, 187});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});  // TensorRT doesn't support support UINT8 for quantization
}

TEST(QuantizeLinearOpTest, Per_Channel_Axis_0) {
  OpTester test("QuantizeLinear", 13);
  std::vector<int64_t> dims{3, 4};
  test.AddInput<float>("X", dims,
                       {0, 2, 3, 1000,
                        0, 2, 3, 1000,
                        0, 2, 3, 1000});
  test.AddAttribute<int64_t>("axis", 0);
  test.AddInput<float>("scale", {3}, {1, 2, 4});
  test.AddInput<uint8_t>("zero_point", {3}, {0, 0, 0});
  test.AddOutput<uint8_t>("Y", dims,
                          {0, 2, 3, 255,
                           0, 1, 2, 255,
                           0, 0, 1, 250});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});  // TensorRT doesn't support support UINT8 for quantization
}

// quantize with per-channel and negative axis (-2 resolves to axis 0)
TEST(QuantizeLinearOpTest, Per_Channel_Axis_neg) {
  OpTester test("QuantizeLinear", 13);
  std::vector<int64_t> dims{3, 4};
  test.AddInput<float>("X", dims,
                       {0, 2, 3, 1000,
                        0, 2, 3, 1000,
                        0, 2, 3, 1000});
  test.AddAttribute<int64_t>("axis", -2);
  test.AddInput<float>("scale", {3}, {1, 2, 4});
  test.AddInput<uint8_t>("zero_point", {3}, {0, 0, 0});
  test.AddOutput<uint8_t>("Y", dims,
                          {0, 2, 3, 255,
                           0, 1, 2, 255,
                           0, 0, 1, 250});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});  // TensorRT doesn't support support UINT8 for quantization
}

#if !defined(DISABLE_FLOAT8_TYPES)

template <typename InT, typename OutT>
void DequantizeLinearOp19Test() {
  OpTester test("DequantizeLinear", 19);
  std::vector<int64_t> dims{4};
  std::vector<InT> x;
  x.push_back(InT(0.0f, true));
  x.push_back(InT(1.0f, true));
  x.push_back(InT(2.0f, true));
  x.push_back(InT(3.0f, true));
  test.AddInput<InT>("x", dims, x);
  test.AddInput<OutT>("x_scale", {}, {static_cast<OutT>(1.0f)});
  test.AddInput<InT>("x_zero_point", {}, {InT(0.0f, true)});
  std::vector<OutT> y;
  for (auto it : x) {
    y.push_back(static_cast<OutT>(it.ToFloat()));
  }
  test.AddOutput<OutT>("y", dims, y);
  // Disable Tensorrt EP due to error:node1_quantize_scale_node: out of bounds channel axis 1. Number of input dimensions is 1.
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});
}

TEST(DequantizeLinearOpTest, Float8) {
  constexpr int min_cuda_architecture = 11080;
  bool enable_cuda = (nullptr != DefaultCpuExecutionProvider().get()) && HasCudaEnvironment(min_cuda_architecture);
  bool enable_cpu = (nullptr != DefaultCpuExecutionProvider().get());

  if (enable_cpu || enable_cuda)
    DequantizeLinearOp19Test<Float8E4M3FN, float>();
  if (enable_cpu)
    DequantizeLinearOp19Test<Float8E4M3FNUZ, float>();
  if (enable_cpu || enable_cuda)
    DequantizeLinearOp19Test<Float8E5M2, float>();
  if (enable_cpu)
    DequantizeLinearOp19Test<Float8E5M2FNUZ, float>();
}

TEST(DequantizeLinearOpMLFloat16Test, Float8) {
  constexpr int min_cuda_architecture = 11080;
  bool enable_cuda = (nullptr != DefaultCpuExecutionProvider().get()) && HasCudaEnvironment(min_cuda_architecture);
  bool enable_cpu = (nullptr != DefaultCpuExecutionProvider().get());

  if (enable_cpu || enable_cuda)
    DequantizeLinearOp19Test<Float8E4M3FN, MLFloat16>();
  if (enable_cpu)
    DequantizeLinearOp19Test<Float8E4M3FNUZ, MLFloat16>();
  if (enable_cpu || enable_cuda)
    DequantizeLinearOp19Test<Float8E5M2, MLFloat16>();
  if (enable_cpu)
    DequantizeLinearOp19Test<Float8E5M2FNUZ, MLFloat16>();
}

template <typename InT, typename OutT>
void QuantizeLinearOp19Test(bool saturate) {
  OpTester test("QuantizeLinear", 19);
  if (!saturate) {
    test.AddAttribute<int64_t>("saturate", 0);
  }
  std::vector<int64_t> dims{6};
  std::vector<InT> x{0, 2, 3, 1000, -254, -1000};
  test.AddInput<InT>("x", dims, x);
  test.AddInput<InT>("y_scale", {}, {1.0f});
  test.AddInput<OutT>("y_zero_point", {}, {OutT(0.0f, true)});
  std::vector<OutT> y;
  for (auto it : x) {
    y.push_back(OutT(it, saturate));
  }
  test.AddOutput<OutT>("y", dims, y);
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});
}

TEST(QuantizeLinearOpTest, Float8) {
  constexpr int min_cuda_architecture = 11080;
  bool enable_cuda = (nullptr != DefaultCpuExecutionProvider().get()) && HasCudaEnvironment(min_cuda_architecture);
  bool enable_cpu = (nullptr != DefaultCpuExecutionProvider().get());

  if (enable_cpu || enable_cuda)
    QuantizeLinearOp19Test<float, Float8E4M3FN>(true);
  if (enable_cpu)
    QuantizeLinearOp19Test<float, Float8E4M3FNUZ>(true);
  if (enable_cpu || enable_cuda)
    QuantizeLinearOp19Test<float, Float8E5M2>(true);
  if (enable_cpu)
    QuantizeLinearOp19Test<float, Float8E5M2FNUZ>(true);
  if (enable_cpu || enable_cuda)
    QuantizeLinearOp19Test<float, Float8E4M3FN>(false);
  if (enable_cpu)
    QuantizeLinearOp19Test<float, Float8E4M3FNUZ>(false);
  if (enable_cpu || enable_cuda)
    QuantizeLinearOp19Test<float, Float8E5M2>(false);
  if (enable_cpu)
    QuantizeLinearOp19Test<float, Float8E5M2FNUZ>(false);
}

template <typename InT, typename OutT>
void QuantizeLinearOp19F16Test(bool saturate) {
  OpTester test("QuantizeLinear", 19);
  if (!saturate) {
    test.AddAttribute<int64_t>("saturate", 0);
  }
  std::vector<int64_t> dims{6};
  std::vector<InT> x{InT(0.0f), InT(2.0f), InT(3.0f), InT(1000.0f), InT(-254.0f), InT(-1000.0f)};
  test.AddInput<InT>("x", dims, x);
  test.AddInput<InT>("y_scale", {}, {InT(1.0f)});
  test.AddInput<OutT>("y_zero_point", {}, {OutT(0.0f, true)});
  std::vector<OutT> y;
  for (auto it : x) {
    y.push_back(OutT(it, saturate));
  }
  test.AddOutput<OutT>("y", dims, y);
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});
}

TEST(QuantizeLinearOpMLFloat16Test, Float8) {
  constexpr int min_cuda_architecture = 11080;
  bool enable_cuda = (nullptr != DefaultCpuExecutionProvider().get()) && HasCudaEnvironment(min_cuda_architecture);
  bool enable_cpu = (nullptr != DefaultCpuExecutionProvider().get());

  if (enable_cpu || enable_cuda)
    QuantizeLinearOp19F16Test<MLFloat16, Float8E4M3FN>(true);
  if (enable_cpu)
    QuantizeLinearOp19F16Test<MLFloat16, Float8E4M3FNUZ>(true);
  if (enable_cpu || enable_cuda)
    QuantizeLinearOp19F16Test<MLFloat16, Float8E5M2>(true);
  if (enable_cpu)
    QuantizeLinearOp19F16Test<MLFloat16, Float8E5M2FNUZ>(true);
  if (enable_cpu || enable_cuda)
    QuantizeLinearOp19F16Test<MLFloat16, Float8E4M3FN>(false);
  if (enable_cpu)
    QuantizeLinearOp19F16Test<MLFloat16, Float8E4M3FNUZ>(false);
  if (enable_cpu || enable_cuda)
    QuantizeLinearOp19F16Test<MLFloat16, Float8E5M2>(false);
  if (enable_cpu)
    QuantizeLinearOp19F16Test<MLFloat16, Float8E5M2FNUZ>(false);
}

#endif

}  // namespace test
}  // namespace onnxruntime
