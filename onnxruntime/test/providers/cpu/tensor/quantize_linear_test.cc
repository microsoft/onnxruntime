// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"
#include "test/common/cuda_op_test_utils.h"
#include "test/providers/provider_test_utils.h"
#include "test/util/include/default_providers.h"
#include "core/framework/int4.h"

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

// scalar zero & scale with int4
TEST(DequantizeLinearOpTest, Int4) {
  OpTester test("DequantizeLinear", 21);
  std::vector<int64_t> dims{5};
  constexpr int unused_val = 0;

  // Odd number of int4 values to test packing/unpacking
  test.AddInput<Int4x2>("x", dims, {Int4x2(-8, -3), Int4x2(1, 7), Int4x2(2, unused_val)});
  test.AddInput<float>("x_scale", {}, {2.0f});
  test.AddInput<Int4x2>("x_zero_point", {}, {Int4x2(-1, unused_val)});
  test.AddOutput<float>("y", dims, {-14.0f, -4.0f, 4.0f, 16.0f, 6.0f});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});
}

// scalar scale with int4
TEST(DequantizeLinearOpTest, Int4NoZeroPoint) {
  OpTester test("DequantizeLinear", 21);
  std::vector<int64_t> dims{5};
  constexpr int unused_val = 0;

  // Odd number of int4 values to test packing/unpacking
  test.AddInput<Int4x2>("x", dims, {Int4x2(-8, -3), Int4x2(1, 7), Int4x2(2, unused_val)});
  test.AddInput<float>("x_scale", {}, {2.0f});
  test.AddOutput<float>("y", dims, {-16.0f, -6.0f, 2.0f, 14.0f, 4.0f});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});
}

// scalar zero & scale with uint4
TEST(DequantizeLinearOpTest, UInt4) {
  OpTester test("DequantizeLinear", 21);
  std::vector<int64_t> dims{5};
  constexpr int unused_val = 0;

  // Odd number of uint4 values to test packing/unpacking
  test.AddInput<UInt4x2>("x", dims, {UInt4x2(0, 1), UInt4x2(3, 15), UInt4x2(2, unused_val)});
  test.AddInput<float>("x_scale", {}, {2.0f});
  test.AddInput<UInt4x2>("x_zero_point", {}, {UInt4x2(1, unused_val)});
  test.AddOutput<float>("y", dims, {-2.0f, 0.0f, 4.0f, 28.0f, 2.0f});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});
}

// scalar scale with uint4
TEST(DequantizeLinearOpTest, UInt4NoZeroPoint) {
  OpTester test("DequantizeLinear", 21);
  std::vector<int64_t> dims{5};
  constexpr int unused_val = 0;

  // Odd number of uint4 values to test packing/unpacking
  test.AddInput<UInt4x2>("x", dims, {UInt4x2(0, 1), UInt4x2(3, 15), UInt4x2(2, unused_val)});
  test.AddInput<float>("x_scale", {}, {2.0f});
  test.AddOutput<float>("y", dims, {0.0f, 2.0f, 6.0f, 30.0f, 4.0f});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});
}

// Test int16 DequantizeLinear (per tensor)
TEST(DequantizeLinearOpTest, Int16) {
  OpTester test("DequantizeLinear", 21);
  std::vector<int64_t> dims{4};
  test.AddInput<int16_t>("x", dims, {-300, -30, -1025, 1270});
  test.AddInput<float>("scale", {}, {2.0f}, true);
  test.AddInput<int16_t>("zero_point", {}, {-1024}, true);
  test.AddOutput<float>("y", dims, {1448.0f, 1988.0f, -2.0f, 4588.0f});
  // Disable Tensorrt EP due to error: unsupported data type
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});
}

// Test uint16 DequantizeLinear (per tensor)
TEST(DequantizeLinearOpTest, Uint16) {
  OpTester test("DequantizeLinear", 21);
  std::vector<int64_t> dims{4};
  test.AddInput<uint16_t>("x", dims, {30000, 31000, 32768, 33000});
  test.AddInput<float>("scale", {}, {2.0f}, true);
  test.AddInput<uint16_t>("zero_point", {}, {32767}, true);
  test.AddOutput<float>("y", dims, {-5534.0f, -3534.0f, 2.0f, 466.0f});
  // Disable Tensorrt EP due to error: unsupported data type
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});
}

// scalar zero & scale with int8
TEST(DequantizeLinearOpTest, Int32) {
  OpTester test("DequantizeLinear", 10);
  std::vector<int64_t> dims{4};
  test.AddInput<int32_t>("x", dims, {-30, -3, 100, 127});
  test.AddInput<float>("x_scale", {}, {2.0f});
  test.AddOutput<float>("y", dims, {-60.f, -6.f, 200.f, 254.f});
  test.Run();
}

TEST(DequantizeLinearOpTest_BroadcastTensor, Int32) {
  OpTester test("DequantizeLinear", 13);
  test.AddInput<int32_t>("x", {4}, {-30, -3, 100, 127});
  test.AddAttribute<int64_t>("axis", 0);
  test.AddInput<float>("x_scale", {1}, {2.0f});
  test.AddInput<int32_t>("x_zero_point", {1}, {0});
  test.AddOutput<float>("y", {4}, {-60.f, -6.f, 200.f, 254.f});
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
  OpTester test("DequantizeLinear", 10);
  test.AddInput<int8_t>("x", {}, {100});
  test.AddInput<float>("x_scale", {}, {2.0f});
  test.AddOutput<float>("y", {}, {200.0f});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});  // No DQ allowed without corresponding Q. Skip since TRT10
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

// Test uint16 QuantizeLinear (per tensor)
TEST(QuantizeLinearOpTest, Uint16) {
  OpTester test("QuantizeLinear", 21);
  std::vector<int64_t> dims{12};
  test.AddInput<float>("x", dims, {
                                      0.f, -128.f, 3.f, -3.f,  // rounding half to even
                                      2.9f, -2.9f,             // round < .5
                                      3.1f, -3.1f,             // round > .5
                                      65536.f, -65534.f,       // critical point
                                      70000.f, -70000.f        // saturate case
                                  });
  test.AddInput<float>("scale", {}, {2.0f}, true);
  test.AddInput<uint16_t>("zero_point", {}, {32767}, true);
  test.AddOutput<uint16_t>("y", dims,
                           {32767, 32703,
                            32769, 32765,
                            32768, 32766,
                            32769, 32765,
                            65535, 0,
                            65535, 0});

  // Disable Tensorrt EP due to error: unsupported data type
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});
}

// Test int16 QuantizeLinear (per tensor)
TEST(QuantizeLinearOpTest, Int16) {
  OpTester test("QuantizeLinear", 21);
  std::vector<int64_t> dims{16};
  test.AddInput<float>("x", dims, {
                                      0.f, -514.f, 3.f, -3.f,  // rounding half to even
                                      2.9f, -2.9f,             // round < .5
                                      3.1f, -3.1f,             // round > .5
                                      65022.f, -66046.f,       // critical point
                                      65023.f, -66047.f,       // critical point
                                      65024.f, -66048.f,       // critical point
                                      70000.f, -70000.f        // saturate case
                                  });
  test.AddInput<float>("scale", {}, {2.0f}, true);
  test.AddInput<int16_t>("zero_point", {}, {256}, true);
  test.AddOutput<int16_t>("y", dims,
                          {256, -1,
                           258, 254,
                           257, 255,
                           258, 254,
                           32767, -32767,
                           32767, -32768,
                           32767, -32768,
                           32767, -32768});

  // Disable Tensorrt EP due to error: unsupported data type
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});
}

// Test int4 QuantizeLinear (per tensor)
TEST(QuantizeLinearOpTest, Int4) {
  OpTester test("QuantizeLinear", 21);
  std::vector<int64_t> dims{7};
  constexpr int8_t unused_val = 0;
  test.AddInput<float>("x", dims, {
                                      -20.0f,  // Clamp to qmin
                                      -16.0f,  // Close to qmin
                                      -3.0f,   // round
                                      0.0f,    // Zero-point
                                      2.9f,    // round
                                      12.0f,   // qmax
                                      20.0f,   // Clamp to qmax
                                  });
  test.AddInput<float>("scale", {}, {2.0f}, true);
  test.AddInput<Int4x2>("zero_point", {}, {Int4x2(1, unused_val)}, true);
  test.AddOutput<Int4x2>("y", dims,
                         {Int4x2(-8, -7), Int4x2(-1, 1), Int4x2(2, 7),
                          Int4x2(7, unused_val)});

  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});
}

// Test uint4 QuantizeLinear (per tensor)
TEST(QuantizeLinearOpTest, UInt4) {
  OpTester test("QuantizeLinear", 21);
  std::vector<int64_t> dims{7};
  constexpr uint8_t unused_val = 0;
  test.AddInput<float>("x", dims, {
                                      -20.0f,  // Clamp to qmin
                                      -8.0f,   // qmin
                                      -3.0f,   // round
                                      0.0f,    // Zero-point
                                      2.9f,    // round
                                      22.0f,   // qmax
                                      30.0f,   // Clamp to qmax
                                  });
  test.AddInput<float>("scale", {}, {2.0f}, true);
  test.AddInput<UInt4x2>("zero_point", {}, {UInt4x2(4, unused_val)}, true);
  test.AddOutput<UInt4x2>("y", dims,
                          {UInt4x2(0, 0), UInt4x2(2, 4), UInt4x2(5, 15),
                           UInt4x2(15, unused_val)});

  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});
}

template <bool Signed>
static void GetExpectedInt4Quant(const float* input, Int4x2Base<Signed>* output, size_t num_elems, float scale,
                                 int8_t zero_point) {
  using UnpackedType = typename Int4x2Base<Signed>::UnpackedType;

  for (size_t n = 0; n < num_elems; n++) {
    float float_val = std::nearbyintf(input[n] / scale) + static_cast<float>(zero_point);
    float_val = std::max(float_val, static_cast<float>(Int4x2Base<Signed>::min_val));
    float_val = std::min(float_val, static_cast<float>(Int4x2Base<Signed>::max_val));

    UnpackedType int_val = static_cast<UnpackedType>(float_val);

    size_t i = n >> 1;
    size_t j = n & 0x1;
    output[i].SetElem(j, int_val);
  }
}

// Test int4 QuantizeLinear (per tensor) with a "large" and odd number of input elements.
// This exercises the TryParallelFor call which splits the input into blocks of even size.
TEST(QuantizeLinearOpTest, OddLarge_Int4) {
  OpTester test("QuantizeLinear", 21);
  std::vector<int64_t> dims{1017};
  constexpr int8_t unused_val = 0;
  constexpr std::array<float, 6> pattern = {-20.0f, -14.0f, -4.1f, -0.0f, 3.0f, 3.3f};
  std::vector<float> input_f32s(static_cast<size_t>(dims[0]));
  std::vector<Int4x2> output(Int4x2::CalcNumInt4Pairs(input_f32s.size()));

  for (size_t i = 0; i < input_f32s.size(); ++i) {
    input_f32s[i] = pattern[i % pattern.size()];
  }

  float scale = 2.0f;
  int8_t zp = 1;
  GetExpectedInt4Quant(input_f32s.data(), &output[0], input_f32s.size(), scale, zp);

  test.AddInput<float>("x", dims, input_f32s);
  test.AddInput<float>("scale", {}, {scale}, true);
  test.AddInput<Int4x2>("zero_point", {}, {Int4x2(zp, unused_val)}, true);
  test.AddOutput<Int4x2>("y", dims, output);

  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});
}

// Test uint4 QuantizeLinear (per tensor) with a "large" and odd number of input elements.
// This exercises the TryParallelFor call which splits the input into blocks of even size.
TEST(QuantizeLinearOpTest, OddLarge_UInt4) {
  OpTester test("QuantizeLinear", 21);
  std::vector<int64_t> dims{1017};
  constexpr uint8_t unused_val = 0;
  constexpr std::array<float, 6> pattern = {-20.0f, -14.0f, -4.1f, -0.0f, 3.0f, 3.3f};
  std::vector<float> input_f32s(static_cast<size_t>(dims[0]));
  std::vector<UInt4x2> output(UInt4x2::CalcNumInt4Pairs(input_f32s.size()));

  for (size_t i = 0; i < input_f32s.size(); ++i) {
    input_f32s[i] = pattern[i % pattern.size()];
  }

  float scale = 2.0f;
  uint8_t zp = 1;
  GetExpectedInt4Quant(input_f32s.data(), &output[0], input_f32s.size(), scale, zp);

  test.AddInput<float>("x", dims, input_f32s);
  test.AddInput<float>("scale", {}, {scale}, true);
  test.AddInput<UInt4x2>("zero_point", {}, {UInt4x2(zp, unused_val)}, true);
  test.AddOutput<UInt4x2>("y", dims, output);

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

namespace blocked_dequantization {

template <typename Tin, typename Tout>
void DequantizeLinearOp21BlockedTest_InvalidBlockSize_Int(int64_t block_size,
                                                          int64_t scale_block_count,
                                                          int64_t zero_point_block_count) {
  OpTester test("DequantizeLinear", 21);
  std::vector<int64_t> dims{2, 4};
  std::vector<Tout> x_scale, y;
  std::vector<Tin> x, x_zero_point;
  SessionOptions so;
  std::vector<std::string> log_msgs;  // redirect error messages
  std::vector<std::unique_ptr<IExecutionProvider>> eps;
  eps.push_back(DefaultCpuExecutionProvider());
  so.user_logging_function = [](void* param, OrtLoggingLevel severity, const char* category,
                                const char* logid, const char* code_location, const char* message) {
    ORT_UNUSED_PARAMETER(severity);
    ORT_UNUSED_PARAMETER(category);
    ORT_UNUSED_PARAMETER(logid);
    ORT_UNUSED_PARAMETER(code_location);
    std::vector<std::string>* v_ptr = reinterpret_cast<std::vector<std::string>*>(param);
    std::vector<std::string>& msg_vector = *v_ptr;
    msg_vector.push_back(std::string(message));
  };
  so.user_logging_param = &log_msgs;
  so.session_logid = "DequantizeLinear";
  so.use_per_session_threads = false;
  so.session_log_verbosity_level = 1;
  so.graph_optimization_level = TransformerLevel::Default;

  for (int64_t i = 0, n = 2 * zero_point_block_count; i < n; ++i) x_zero_point.push_back(0);
  for (int64_t i = 0, n = 2 * scale_block_count; i < n; i++) x_scale.push_back(Tout(2.0f));
  for (int i = 0; i < 8; ++i) {
    x.push_back(i);
    y.push_back(Tout(static_cast<float>(i) * 2.0f));
  }

  test.AddInput<Tin>("x", dims, x);
  test.AddAttribute<int64_t>("axis", 1);
  test.AddAttribute<int64_t>("block_size", block_size);
  test.AddInput<Tout>("x_scale", {2, scale_block_count}, x_scale);
  test.AddInput<Tin>("x_zero_point", {2, zero_point_block_count}, x_zero_point);
  test.AddOutput<Tout>("y", dims, y);
  test.Run(so, OpTester::ExpectResult::kExpectFailure, "", {}, nullptr, &eps);
}

template <typename Tin, typename Tout>
void DequantizeLinearOp21BlockedTest_InvalidBlockSize_Int4(int64_t block_size,
                                                           int64_t scale_block_count,
                                                           int64_t zero_point_block_count,
                                                           std::unique_ptr<IExecutionProvider> ep = nullptr) {
  OpTester test("DequantizeLinear", 21);
  std::vector<int64_t> dims{2, 4};
  std::vector<Tout> x_scale, y;
  std::vector<Tin> x, x_zero_point;
  SessionOptions so;
  std::vector<std::string> log_msgs;  // redirect error messages
  std::vector<std::unique_ptr<IExecutionProvider>> eps;
  eps.push_back(ep ? std::move(ep) : DefaultCpuExecutionProvider());
  so.user_logging_function = [](void* param, OrtLoggingLevel severity, const char* category,
                                const char* logid, const char* code_location, const char* message) {
    ORT_UNUSED_PARAMETER(severity);
    ORT_UNUSED_PARAMETER(category);
    ORT_UNUSED_PARAMETER(logid);
    ORT_UNUSED_PARAMETER(code_location);
    std::vector<std::string>* v_ptr = reinterpret_cast<std::vector<std::string>*>(param);
    std::vector<std::string>& msg_vector = *v_ptr;
    msg_vector.push_back(std::string(message));
  };
  so.user_logging_param = &log_msgs;
  so.session_logid = "DequantizeLinear";
  so.use_per_session_threads = false;
  so.session_log_verbosity_level = 1;
  so.graph_optimization_level = TransformerLevel::Default;

  for (int64_t i = 0, n = zero_point_block_count; i < n; ++i) x_zero_point.push_back(Tin(0, 0));
  for (int64_t i = 0, n = 2 * scale_block_count; i < n; i++) x_scale.push_back(Tout(2.0f));
  for (int i = 0; i < 8; ++i) {
    if (i & 1) x.push_back(Tin(i - 1, i));
    y.push_back(Tout(static_cast<float>(i) * 2.0f));
  }

  test.AddInput<Tin>("x", dims, x);
  test.AddAttribute<int64_t>("axis", 1);
  test.AddAttribute<int64_t>("block_size", block_size);
  test.AddInput<Tout>("x_scale", {2, scale_block_count}, x_scale);
  test.AddInput<Tin>("x_zero_point", {2, zero_point_block_count}, x_zero_point);
  test.AddOutput<Tout>("y", dims, y);
  test.Run(so, OpTester::ExpectResult::kExpectFailure, "", {}, nullptr, &eps);
}

template <typename Tin, typename Tout>
void DequantizeLinearOp21BlockedTest_InvalidBlockSize_Float8(int64_t block_size,
                                                             int64_t scale_block_count,
                                                             int64_t zero_point_block_count) {
  OpTester test("DequantizeLinear", 21);
  std::vector<int64_t> dims{2, 4};
  std::vector<Tout> x_scale, y;
  std::vector<Tin> x, x_zero_point;
  SessionOptions so;
  std::vector<std::string> log_msgs;  // redirect error messages
  std::vector<std::unique_ptr<IExecutionProvider>> eps;
  eps.push_back(DefaultCpuExecutionProvider());
  so.user_logging_function = [](void* param, OrtLoggingLevel severity, const char* category,
                                const char* logid, const char* code_location, const char* message) {
    ORT_UNUSED_PARAMETER(severity);
    ORT_UNUSED_PARAMETER(category);
    ORT_UNUSED_PARAMETER(logid);
    ORT_UNUSED_PARAMETER(code_location);
    std::vector<std::string>* v_ptr = reinterpret_cast<std::vector<std::string>*>(param);
    std::vector<std::string>& msg_vector = *v_ptr;
    msg_vector.push_back(std::string(message));
  };
  so.user_logging_param = &log_msgs;
  so.session_logid = "DequantizeLinear";
  so.use_per_session_threads = false;
  so.session_log_verbosity_level = 1;
  so.graph_optimization_level = TransformerLevel::Default;

  for (int64_t i = 0, n = 2 * zero_point_block_count; i < n; i++) x_zero_point.push_back(Tin(0.0f));
  for (int64_t i = 0, n = 2 * scale_block_count; i < n; i++) x_scale.push_back(Tout(2.0f));
  for (int i = 0; i < 8; ++i) x.push_back(Tin(static_cast<float>(i)));
  for (int i = 0; i < 8; ++i) y.push_back(Tout(static_cast<float>(i) * 2.0f));

  test.AddInput<Tin>("x", dims, x);
  test.AddAttribute<int64_t>("axis", 1);
  test.AddAttribute<int64_t>("block_size", block_size);
  test.AddInput<Tout>("x_scale", {2, scale_block_count}, x_scale);
  test.AddInput<Tin>("x_zero_point", {2, zero_point_block_count}, x_zero_point);
  test.AddOutput<Tout>("y", dims, y);
  test.Run(so, OpTester::ExpectResult::kExpectFailure, "", {}, nullptr, &eps);
}

// test negative block size fail
TEST(DequantizeLinearOp21BlockedTest, NagativeBlockSize_Int) {
  DequantizeLinearOp21BlockedTest_InvalidBlockSize_Int4<Int4x2, float>(-1, 2, 2);
  DequantizeLinearOp21BlockedTest_InvalidBlockSize_Int4<Int4x2, MLFloat16>(-1, 2, 2);
  DequantizeLinearOp21BlockedTest_InvalidBlockSize_Int4<UInt4x2, float>(-2, 2, 2);
  DequantizeLinearOp21BlockedTest_InvalidBlockSize_Int4<UInt4x2, MLFloat16>(-2, 2, 2);
  DequantizeLinearOp21BlockedTest_InvalidBlockSize_Int<int8_t, float>(-3, 2, 2);
  DequantizeLinearOp21BlockedTest_InvalidBlockSize_Int<int8_t, MLFloat16>(-3, 2, 2);
  DequantizeLinearOp21BlockedTest_InvalidBlockSize_Int<uint8_t, float>(-4, 2, 2);
  DequantizeLinearOp21BlockedTest_InvalidBlockSize_Int<uint8_t, MLFloat16>(-4, 2, 2);
  DequantizeLinearOp21BlockedTest_InvalidBlockSize_Int<int16_t, float>(-5, 2, 2);
  DequantizeLinearOp21BlockedTest_InvalidBlockSize_Int<int16_t, MLFloat16>(-5, 2, 2);
  DequantizeLinearOp21BlockedTest_InvalidBlockSize_Int<uint16_t, float>(-6, 2, 2);
  DequantizeLinearOp21BlockedTest_InvalidBlockSize_Int<uint16_t, MLFloat16>(-1, 2, 2);
  DequantizeLinearOp21BlockedTest_InvalidBlockSize_Int<int32_t, float>(-1, 2, 2);
  DequantizeLinearOp21BlockedTest_InvalidBlockSize_Int<int32_t, MLFloat16>(-1, 2, 2);
}

TEST(DequantizeLinearOp21BlockedTest, NagativeBlockSize_Int_Cuda) {
  DequantizeLinearOp21BlockedTest_InvalidBlockSize_Int4<Int4x2, float>(-1, 2, 2, DefaultCudaExecutionProvider());
  DequantizeLinearOp21BlockedTest_InvalidBlockSize_Int4<Int4x2, MLFloat16>(-1, 2, 2, DefaultCudaExecutionProvider());
  DequantizeLinearOp21BlockedTest_InvalidBlockSize_Int4<UInt4x2, float>(-2, 2, 2, DefaultCudaExecutionProvider());
  DequantizeLinearOp21BlockedTest_InvalidBlockSize_Int4<UInt4x2, MLFloat16>(-2, 2, 2, DefaultCudaExecutionProvider());
}

#if !defined(DISABLE_FLOAT8_TYPES)
TEST(DequantizeLinearOp21BlockedTest, NagativeBlockSize_Float8) {
  constexpr int min_cuda_architecture = 11080;
  bool enable_cuda = (nullptr != DefaultCpuExecutionProvider().get()) && HasCudaEnvironment(min_cuda_architecture);
  bool enable_cpu = (nullptr != DefaultCpuExecutionProvider().get());

  if (enable_cpu || enable_cuda) {
    DequantizeLinearOp21BlockedTest_InvalidBlockSize_Float8<Float8E4M3FN, float>(-1, 2, 2);
    DequantizeLinearOp21BlockedTest_InvalidBlockSize_Float8<Float8E4M3FN, MLFloat16>(-2, 2, 2);
  }
  if (enable_cpu) {
    DequantizeLinearOp21BlockedTest_InvalidBlockSize_Float8<Float8E4M3FNUZ, float>(-3, 2, 2);
    DequantizeLinearOp21BlockedTest_InvalidBlockSize_Float8<Float8E4M3FNUZ, MLFloat16>(-4, 2, 2);
  }
  if (enable_cpu || enable_cuda) {
    DequantizeLinearOp21BlockedTest_InvalidBlockSize_Float8<Float8E5M2, float>(-5, 2, 2);
    DequantizeLinearOp21BlockedTest_InvalidBlockSize_Float8<Float8E5M2, MLFloat16>(-6, 2, 2);
  }
  if (enable_cpu) {
    DequantizeLinearOp21BlockedTest_InvalidBlockSize_Float8<Float8E5M2FNUZ, float>(-1, 2, 2);
    DequantizeLinearOp21BlockedTest_InvalidBlockSize_Float8<Float8E5M2FNUZ, MLFloat16>(-1, 2, 2);
  }
}
#endif

// test block size incompatible with x_scale shape fail
TEST(DequantizeLinearOp21BlockedTest, IncompatibleBlockSizeWithX_Int) {
  DequantizeLinearOp21BlockedTest_InvalidBlockSize_Int4<Int4x2, float>(3, 1, 1);
  DequantizeLinearOp21BlockedTest_InvalidBlockSize_Int4<Int4x2, MLFloat16>(3, 3, 3);
  DequantizeLinearOp21BlockedTest_InvalidBlockSize_Int4<UInt4x2, float>(3, 3, 3);
  DequantizeLinearOp21BlockedTest_InvalidBlockSize_Int4<UInt4x2, MLFloat16>(3, 1, 1);
  DequantizeLinearOp21BlockedTest_InvalidBlockSize_Int<int8_t, float>(3, 1, 1);
  DequantizeLinearOp21BlockedTest_InvalidBlockSize_Int<int8_t, MLFloat16>(3, 3, 3);
  DequantizeLinearOp21BlockedTest_InvalidBlockSize_Int<uint8_t, float>(3, 3, 3);
  DequantizeLinearOp21BlockedTest_InvalidBlockSize_Int<uint8_t, MLFloat16>(3, 1, 1);
  DequantizeLinearOp21BlockedTest_InvalidBlockSize_Int<int16_t, float>(3, 1, 1);
  DequantizeLinearOp21BlockedTest_InvalidBlockSize_Int<int16_t, MLFloat16>(3, 3, 3);
  DequantizeLinearOp21BlockedTest_InvalidBlockSize_Int<uint16_t, float>(3, 3, 3);
  DequantizeLinearOp21BlockedTest_InvalidBlockSize_Int<uint16_t, MLFloat16>(3, 1, 1);
  DequantizeLinearOp21BlockedTest_InvalidBlockSize_Int<int32_t, float>(3, 3, 3);
  DequantizeLinearOp21BlockedTest_InvalidBlockSize_Int<int32_t, MLFloat16>(3, 1, 1);
}

TEST(DequantizeLinearOp21BlockedTest, IncompatibleBlockSizeWithX_Int_Cuda) {
  DequantizeLinearOp21BlockedTest_InvalidBlockSize_Int4<Int4x2, float>(3, 1, 1, DefaultCudaExecutionProvider());
  DequantizeLinearOp21BlockedTest_InvalidBlockSize_Int4<Int4x2, MLFloat16>(3, 3, 3, DefaultCudaExecutionProvider());
  DequantizeLinearOp21BlockedTest_InvalidBlockSize_Int4<UInt4x2, float>(3, 3, 3, DefaultCudaExecutionProvider());
  DequantizeLinearOp21BlockedTest_InvalidBlockSize_Int4<UInt4x2, MLFloat16>(3, 1, 1, DefaultCudaExecutionProvider());
}

#if !defined(DISABLE_FLOAT8_TYPES)
TEST(DequantizeLinearOp21BlockedTest, IncompatibleBlockSizeWithX_Float8) {
  constexpr int min_cuda_architecture = 11080;
  bool enable_cuda = (nullptr != DefaultCpuExecutionProvider().get()) && HasCudaEnvironment(min_cuda_architecture);
  bool enable_cpu = (nullptr != DefaultCpuExecutionProvider().get());

  if (enable_cpu || enable_cuda) {
    DequantizeLinearOp21BlockedTest_InvalidBlockSize_Float8<Float8E4M3FN, float>(3, 1, 1);
    DequantizeLinearOp21BlockedTest_InvalidBlockSize_Float8<Float8E4M3FN, MLFloat16>(3, 3, 3);
    DequantizeLinearOp21BlockedTest_InvalidBlockSize_Float8<Float8E5M2, float>(3, 1, 1);
    DequantizeLinearOp21BlockedTest_InvalidBlockSize_Float8<Float8E5M2, MLFloat16>(3, 3, 3);
  }
  if (enable_cpu) {
    DequantizeLinearOp21BlockedTest_InvalidBlockSize_Float8<Float8E4M3FNUZ, float>(3, 1, 1);
    DequantizeLinearOp21BlockedTest_InvalidBlockSize_Float8<Float8E4M3FNUZ, MLFloat16>(3, 3, 3);
    DequantizeLinearOp21BlockedTest_InvalidBlockSize_Float8<Float8E5M2FNUZ, float>(3, 1, 1);
    DequantizeLinearOp21BlockedTest_InvalidBlockSize_Float8<Float8E5M2FNUZ, MLFloat16>(3, 3, 3);
  }
}
#endif

// test x_scale vs. x_zero_point shape incompatible fail
TEST(DequantizeLinearOp21BlockedTest, ScaleShapeUnmatchZeroPoint_Int) {
  DequantizeLinearOp21BlockedTest_InvalidBlockSize_Int4<Int4x2, float>(3, 2, 1);
  DequantizeLinearOp21BlockedTest_InvalidBlockSize_Int4<Int4x2, MLFloat16>(3, 2, 3);
  DequantizeLinearOp21BlockedTest_InvalidBlockSize_Int4<UInt4x2, float>(3, 2, 3);
  DequantizeLinearOp21BlockedTest_InvalidBlockSize_Int4<UInt4x2, MLFloat16>(3, 2, 1);
  DequantizeLinearOp21BlockedTest_InvalidBlockSize_Int<int8_t, float>(3, 2, 1);
  DequantizeLinearOp21BlockedTest_InvalidBlockSize_Int<int8_t, MLFloat16>(3, 2, 3);
  DequantizeLinearOp21BlockedTest_InvalidBlockSize_Int<uint8_t, float>(3, 2, 3);
  DequantizeLinearOp21BlockedTest_InvalidBlockSize_Int<uint8_t, MLFloat16>(3, 2, 1);
  DequantizeLinearOp21BlockedTest_InvalidBlockSize_Int<int16_t, float>(3, 2, 1);
  DequantizeLinearOp21BlockedTest_InvalidBlockSize_Int<int16_t, MLFloat16>(3, 2, 3);
  DequantizeLinearOp21BlockedTest_InvalidBlockSize_Int<uint16_t, float>(3, 2, 3);
  DequantizeLinearOp21BlockedTest_InvalidBlockSize_Int<uint16_t, MLFloat16>(3, 2, 1);
  DequantizeLinearOp21BlockedTest_InvalidBlockSize_Int<int32_t, float>(3, 2, 3);
  DequantizeLinearOp21BlockedTest_InvalidBlockSize_Int<int32_t, MLFloat16>(3, 2, 1);
}

TEST(DequantizeLinearOp21BlockedTest, ScaleShapeUnmatchZeroPoint_Int_Cuda) {
  DequantizeLinearOp21BlockedTest_InvalidBlockSize_Int4<Int4x2, float>(3, 2, 1, DefaultCudaExecutionProvider());
  DequantizeLinearOp21BlockedTest_InvalidBlockSize_Int4<Int4x2, MLFloat16>(3, 2, 3, DefaultCudaExecutionProvider());
  DequantizeLinearOp21BlockedTest_InvalidBlockSize_Int4<UInt4x2, float>(3, 2, 3, DefaultCudaExecutionProvider());
  DequantizeLinearOp21BlockedTest_InvalidBlockSize_Int4<UInt4x2, MLFloat16>(3, 2, 1, DefaultCudaExecutionProvider());
}

#if !defined(DISABLE_FLOAT8_TYPES)
TEST(DequantizeLinearOp21BlockedTest, ScaleShapeUnmatchZeroPoint_Float8) {
  constexpr int min_cuda_architecture = 11080;
  bool enable_cuda = (nullptr != DefaultCpuExecutionProvider().get()) && HasCudaEnvironment(min_cuda_architecture);
  bool enable_cpu = (nullptr != DefaultCpuExecutionProvider().get());

  if (enable_cpu || enable_cuda) {
    DequantizeLinearOp21BlockedTest_InvalidBlockSize_Float8<Float8E4M3FN, float>(3, 2, 1);
    DequantizeLinearOp21BlockedTest_InvalidBlockSize_Float8<Float8E4M3FN, MLFloat16>(3, 2, 3);
    DequantizeLinearOp21BlockedTest_InvalidBlockSize_Float8<Float8E5M2, float>(3, 2, 1);
    DequantizeLinearOp21BlockedTest_InvalidBlockSize_Float8<Float8E5M2, MLFloat16>(3, 2, 3);
  }
  if (enable_cpu) {
    DequantizeLinearOp21BlockedTest_InvalidBlockSize_Float8<Float8E4M3FNUZ, float>(3, 2, 1);
    DequantizeLinearOp21BlockedTest_InvalidBlockSize_Float8<Float8E4M3FNUZ, MLFloat16>(3, 2, 3);
    DequantizeLinearOp21BlockedTest_InvalidBlockSize_Float8<Float8E5M2FNUZ, float>(3, 2, 1);
    DequantizeLinearOp21BlockedTest_InvalidBlockSize_Float8<Float8E5M2FNUZ, MLFloat16>(3, 2, 3);
  }
}
#endif

// test DQ with blocked quantization succeed
template <typename Tin, typename Tout>
void DequantizeLinearOp21BlockedTest_Int4_Succeed(std::vector<int64_t>&& dims,
                                                  int64_t axis,
                                                  int64_t block_size,
                                                  std::vector<int>& x_,
                                                  std::vector<float>& x_scale_,
                                                  std::vector<int>& x_zero_point_,
                                                  std::vector<float>& y_,
                                                  std::unique_ptr<IExecutionProvider> ep = nullptr) {
  OpTester test("DequantizeLinear", 21);
  std::vector<int64_t> x_scale_shape;
  std::vector<Tout> x_scale, y;
  std::vector<Tin> x, x_zero_point;
  std::vector<std::unique_ptr<IExecutionProvider>> eps;
  eps.push_back(ep ? std::move(ep) : DefaultCpuExecutionProvider());
  int64_t non_neg_axis = axis < 0 ? axis + dims.size() : axis;
  bool use_zero_point = !x_zero_point_.empty();

  for (auto v : y_) y.push_back(Tout(v));
  for (auto v : x_scale_) x_scale.push_back(Tout(v));
  for (size_t i = 0, n = dims.size(); i < n; ++i) {
    x_scale_shape.push_back((int64_t)i == non_neg_axis ? (dims[i] + block_size - 1) / block_size : dims[i]);
  }

  size_t i = 0, n = x_.size();
  for (; i < n - 1; i += 2) x.push_back(Tin(x_[i], x_[i + 1]));
  if (i < n) x.push_back(Tin(x_[i], 0xF));

  if (use_zero_point) {
    i = 0, n = x_zero_point_.size();
    for (; i < n - 1; i += 2) x_zero_point.push_back(Tin(x_zero_point_[i], x_zero_point_[i + 1]));
    if (i < n) x_zero_point.push_back(Tin(x_zero_point_[i], 0xF));
  }

  test.AddInput<Tin>("x", dims, x);
  test.AddAttribute<int64_t>("axis", axis);
  test.AddAttribute<int64_t>("block_size", block_size);
  test.AddInput<Tout>("x_scale", x_scale_shape, x_scale);
  if (use_zero_point) test.AddInput<Tin>("x_zero_point", x_scale_shape, x_zero_point);
  test.AddOutput<Tout>("y", dims, y);
  test.Run(BaseTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &eps);
}

template <typename Tin, typename Tout>
void DequantizeLinearOp21BlockedTest_Int_Succeed(std::vector<int64_t>&& dims,
                                                 int64_t axis,
                                                 int64_t block_size,
                                                 std::vector<int>& x_,
                                                 std::vector<float>& x_scale_,
                                                 std::vector<int>& x_zero_point_,
                                                 std::vector<float>& y_) {
  OpTester test("DequantizeLinear", 21);
  std::vector<int64_t> x_scale_shape;
  std::vector<Tout> x_scale, y;
  std::vector<Tin> x, x_zero_point;
  std::vector<std::unique_ptr<IExecutionProvider>> eps;
  eps.push_back(DefaultCpuExecutionProvider());

  int64_t non_neg_axis = axis < 0 ? axis + dims.size() : axis;
  bool use_zero_point = !x_zero_point_.empty();

  for (auto v : y_) y.push_back(Tout(v));
  for (auto v : x_scale_) x_scale.push_back(Tout(v));
  for (size_t i = 0, n = dims.size(); i < n; ++i) {
    x_scale_shape.push_back((int64_t)i == non_neg_axis ? (dims[i] + block_size - 1) / block_size : dims[i]);
  }
  for (auto v : x_) x.push_back(v);
  if (use_zero_point)
    for (auto v : x_zero_point_) x_zero_point.push_back(v);

  test.AddInput<Tin>("x", dims, x);
  test.AddAttribute<int64_t>("axis", axis);
  test.AddAttribute<int64_t>("block_size", block_size);
  test.AddInput<Tout>("x_scale", x_scale_shape, x_scale);
  if (use_zero_point) test.AddInput<Tin>("x_zero_point", x_scale_shape, x_zero_point);
  test.AddOutput<Tout>("y", dims, y);
  test.Run(BaseTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &eps);
}

template <typename Tin, typename Tout>
void DequantizeLinearOp21BlockedTest_Float8_Succeed(std::vector<int64_t>&& dims,
                                                    int64_t axis,
                                                    int64_t block_size,
                                                    std::vector<int>& x_,
                                                    std::vector<float>& x_scale_,
                                                    std::vector<int>& x_zero_point_,
                                                    std::vector<float>& y_) {
  OpTester test("DequantizeLinear", 21);
  std::vector<int64_t> x_scale_shape;
  std::vector<Tout> x_scale, y;
  std::vector<Tin> x, x_zero_point;
  std::vector<std::unique_ptr<IExecutionProvider>> eps;
  eps.push_back(DefaultCpuExecutionProvider());

  int64_t non_neg_axis = axis < 0 ? axis + dims.size() : axis;
  bool use_zero_point = !x_zero_point_.empty();

  for (auto v : y_) y.push_back(Tout(v));
  for (auto v : x_scale_) x_scale.push_back(Tout(v));
  for (size_t i = 0, n = dims.size(); i < n; ++i) {
    x_scale_shape.push_back((int64_t)i == non_neg_axis ? (dims[i] + block_size - 1) / block_size : dims[i]);
  }

  for (auto v : x_) x.push_back(Tin(static_cast<float>(v)));
  if (use_zero_point) {
    for (auto v : x_zero_point_) x_zero_point.push_back(Tin(static_cast<float>(v)));
  }

  test.AddInput<Tin>("x", dims, x);
  test.AddAttribute<int64_t>("axis", axis);
  test.AddAttribute<int64_t>("block_size", block_size);
  test.AddInput<Tout>("x_scale", x_scale_shape, x_scale);
  if (use_zero_point) test.AddInput<Tin>("x_zero_point", x_scale_shape, x_zero_point);
  test.AddOutput<Tout>("y", dims, y);
  test.Run(BaseTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &eps);
}

TEST(DequantizeLinearOp21BlockedTest, SignedInt_NoZeroPoint_FirstAxis) {
  std::vector<float> x_scale{-2.0, -4.0, 3.5, 1.0, 2.0, 4.0, -3.5, -1.0};
  std::vector<int> zero_point;
  std::vector<int> x{-7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, -8};
  std::vector<float> y_2{14.0, 24.0, -17.5, -4.0, 6.0, 8.0, -3.5, 0.0, 2.0, 8.0, -10.5, -4.0, 10.0, 24.0, -24.5, 8.0};
  std::vector<float> y_3{14.0, 24.0, -17.5, -4.0, 6.0, 8.0, -3.5, 0.0, -2.0, -8.0, 10.5, 4.0, 10.0, 24.0, -24.5, 8.0};

  DequantizeLinearOp21BlockedTest_Int4_Succeed<Int4x2, float>({4, 2, 2}, 0, 2, x, x_scale, zero_point, y_2);
  DequantizeLinearOp21BlockedTest_Int4_Succeed<Int4x2, float>({4, 2, 2}, 0, 3, x, x_scale, zero_point, y_3);
  DequantizeLinearOp21BlockedTest_Int_Succeed<int8_t, float>({4, 2, 2}, 0, 2, x, x_scale, zero_point, y_2);
  DequantizeLinearOp21BlockedTest_Int_Succeed<int8_t, float>({4, 2, 2}, 0, 3, x, x_scale, zero_point, y_3);
  DequantizeLinearOp21BlockedTest_Int_Succeed<int16_t, float>({4, 2, 2}, 0, 2, x, x_scale, zero_point, y_2);
  DequantizeLinearOp21BlockedTest_Int_Succeed<int16_t, float>({4, 2, 2}, 0, 3, x, x_scale, zero_point, y_3);
  DequantizeLinearOp21BlockedTest_Int_Succeed<int32_t, float>({4, 2, 2}, 0, 2, x, x_scale, zero_point, y_2);
  DequantizeLinearOp21BlockedTest_Int_Succeed<int32_t, float>({4, 2, 2}, 0, 3, x, x_scale, zero_point, y_3);
  DequantizeLinearOp21BlockedTest_Int4_Succeed<Int4x2, MLFloat16>({4, 2, 2}, 0, 2, x, x_scale, zero_point, y_2);
  DequantizeLinearOp21BlockedTest_Int4_Succeed<Int4x2, MLFloat16>({4, 2, 2}, 0, 3, x, x_scale, zero_point, y_3);
  DequantizeLinearOp21BlockedTest_Int_Succeed<int8_t, MLFloat16>({4, 2, 2}, 0, 2, x, x_scale, zero_point, y_2);
  DequantizeLinearOp21BlockedTest_Int_Succeed<int8_t, MLFloat16>({4, 2, 2}, 0, 3, x, x_scale, zero_point, y_3);
  DequantizeLinearOp21BlockedTest_Int_Succeed<int16_t, MLFloat16>({4, 2, 2}, 0, 2, x, x_scale, zero_point, y_2);
  DequantizeLinearOp21BlockedTest_Int_Succeed<int16_t, MLFloat16>({4, 2, 2}, 0, 3, x, x_scale, zero_point, y_3);
  DequantizeLinearOp21BlockedTest_Int_Succeed<int32_t, MLFloat16>({4, 2, 2}, 0, 2, x, x_scale, zero_point, y_2);
  DequantizeLinearOp21BlockedTest_Int_Succeed<int32_t, MLFloat16>({4, 2, 2}, 0, 3, x, x_scale, zero_point, y_3);
}

TEST(DequantizeLinearOp21BlockedTest, SignedInt_NoZeroPoint_FirstAxis_Cuda) {
  std::vector<float> x_scale{-2.0, -4.0, 3.5, 1.0, 2.0, 4.0, -3.5, -1.0};
  std::vector<int> zero_point;
  std::vector<int> x{-7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, -8};
  std::vector<float> y_2{14.0, 24.0, -17.5, -4.0, 6.0, 8.0, -3.5, 0.0, 2.0, 8.0, -10.5, -4.0, 10.0, 24.0, -24.5, 8.0};
  std::vector<float> y_3{14.0, 24.0, -17.5, -4.0, 6.0, 8.0, -3.5, 0.0, -2.0, -8.0, 10.5, 4.0, 10.0, 24.0, -24.5, 8.0};

  DequantizeLinearOp21BlockedTest_Int4_Succeed<Int4x2, float>({4, 2, 2}, 0, 2, x, x_scale, zero_point, y_2,
                                                              DefaultCudaExecutionProvider());
  DequantizeLinearOp21BlockedTest_Int4_Succeed<Int4x2, float>({4, 2, 2}, 0, 3, x, x_scale, zero_point, y_3,
                                                              DefaultCudaExecutionProvider());
  DequantizeLinearOp21BlockedTest_Int4_Succeed<Int4x2, MLFloat16>({4, 2, 2}, 0, 2, x, x_scale, zero_point, y_2,
                                                                  DefaultCudaExecutionProvider());
  DequantizeLinearOp21BlockedTest_Int4_Succeed<Int4x2, MLFloat16>({4, 2, 2}, 0, 3, x, x_scale, zero_point, y_3,
                                                                  DefaultCudaExecutionProvider());
}

TEST(DequantizeLinearOp21BlockedTest, SignedInt_UseZeroPoint_FirstAxis) {
  std::vector<float> x_scale{-2.0, -4.0, 3.5, 1.0, 2.0, 4.0, -3.5, -1.0};
  std::vector<int> zero_point{-6, -4, -3, -1, 0, 2, 4, 7};
  std::vector<int> x{-7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, -8};
  std::vector<float> y_2{2.0, 8.0, -7.0, -3, -6.0, -8.0, 7.0, 1, 2.0, 0, 3.5, 3.0, 10.0, 16.0, -10.5, 15};
  std::vector<float> y_3{2.0, 8.0, -7.0, -3, -6.0, -8.0, 7.0, 1, -14.0, -24, 21, 5, 10.0, 16.0, -10.5, 15};

  DequantizeLinearOp21BlockedTest_Int4_Succeed<Int4x2, float>({4, 2, 2}, 0, 2, x, x_scale, zero_point, y_2);
  DequantizeLinearOp21BlockedTest_Int4_Succeed<Int4x2, float>({4, 2, 2}, 0, 3, x, x_scale, zero_point, y_3);
  DequantizeLinearOp21BlockedTest_Int_Succeed<int8_t, float>({4, 2, 2}, 0, 2, x, x_scale, zero_point, y_2);
  DequantizeLinearOp21BlockedTest_Int_Succeed<int8_t, float>({4, 2, 2}, 0, 3, x, x_scale, zero_point, y_3);
  DequantizeLinearOp21BlockedTest_Int_Succeed<int16_t, float>({4, 2, 2}, 0, 2, x, x_scale, zero_point, y_2);
  DequantizeLinearOp21BlockedTest_Int_Succeed<int16_t, float>({4, 2, 2}, 0, 3, x, x_scale, zero_point, y_3);
  DequantizeLinearOp21BlockedTest_Int4_Succeed<Int4x2, MLFloat16>({4, 2, 2}, 0, 2, x, x_scale, zero_point, y_2);
  DequantizeLinearOp21BlockedTest_Int4_Succeed<Int4x2, MLFloat16>({4, 2, 2}, 0, 3, x, x_scale, zero_point, y_3);
  DequantizeLinearOp21BlockedTest_Int_Succeed<int8_t, MLFloat16>({4, 2, 2}, 0, 2, x, x_scale, zero_point, y_2);
  DequantizeLinearOp21BlockedTest_Int_Succeed<int8_t, MLFloat16>({4, 2, 2}, 0, 3, x, x_scale, zero_point, y_3);
  DequantizeLinearOp21BlockedTest_Int_Succeed<int16_t, MLFloat16>({4, 2, 2}, 0, 2, x, x_scale, zero_point, y_2);
  DequantizeLinearOp21BlockedTest_Int_Succeed<int16_t, MLFloat16>({4, 2, 2}, 0, 3, x, x_scale, zero_point, y_3);
}

TEST(DequantizeLinearOp21BlockedTest, SignedInt_UseZeroPoint_FirstAxis_Cuda) {
  std::vector<float> x_scale{-2.0, -4.0, 3.5, 1.0, 2.0, 4.0, -3.5, -1.0};
  std::vector<int> zero_point{-6, -4, -3, -1, 0, 2, 4, 7};
  std::vector<int> x{-7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, -8};
  std::vector<float> y_2{2.0, 8.0, -7.0, -3, -6.0, -8.0, 7.0, 1, 2.0, 0, 3.5, 3.0, 10.0, 16.0, -10.5, 15};
  std::vector<float> y_3{2.0, 8.0, -7.0, -3, -6.0, -8.0, 7.0, 1, -14.0, -24, 21, 5, 10.0, 16.0, -10.5, 15};

  DequantizeLinearOp21BlockedTest_Int4_Succeed<Int4x2, float>({4, 2, 2}, 0, 2, x, x_scale, zero_point, y_2,
                                                              DefaultCudaExecutionProvider());
  DequantizeLinearOp21BlockedTest_Int4_Succeed<Int4x2, float>({4, 2, 2}, 0, 3, x, x_scale, zero_point, y_3,
                                                              DefaultCudaExecutionProvider());
  DequantizeLinearOp21BlockedTest_Int4_Succeed<Int4x2, MLFloat16>({4, 2, 2}, 0, 2, x, x_scale, zero_point, y_2,
                                                                  DefaultCudaExecutionProvider());
  DequantizeLinearOp21BlockedTest_Int4_Succeed<Int4x2, MLFloat16>({4, 2, 2}, 0, 3, x, x_scale, zero_point, y_3,
                                                                  DefaultCudaExecutionProvider());
}

TEST(DequantizeLinearOp21BlockedTest, SignedInt_NoZeroPoint_MiddleAxis) {
  std::vector<int> zero_point{};
  std::vector<float> x_scale{-2.0, -4.0, 3.5, 1.0, 2.0, 4.0, -3.5, -1.0};
  std::vector<int> x{-7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, -8};
  std::vector<float> y_2{14, 24, 10, 16, -10.5, -2, -3.5, 0, 2, 8, 6, 16, -17.5, -6, -24.5, 8};
  std::vector<float> y_3{14, 24, 10, 16, 6, 8, -3.5, 0, 2, 8, 6, 16, 10, 24, -24.5, 8};

  DequantizeLinearOp21BlockedTest_Int4_Succeed<Int4x2, float>({2, 4, 2}, 1, 2, x, x_scale, zero_point, y_2);
  DequantizeLinearOp21BlockedTest_Int4_Succeed<Int4x2, float>({2, 4, 2}, 1, 3, x, x_scale, zero_point, y_3);
  DequantizeLinearOp21BlockedTest_Int_Succeed<int8_t, float>({2, 4, 2}, 1, 2, x, x_scale, zero_point, y_2);
  DequantizeLinearOp21BlockedTest_Int_Succeed<int8_t, float>({2, 4, 2}, 1, 3, x, x_scale, zero_point, y_3);
  DequantizeLinearOp21BlockedTest_Int_Succeed<int16_t, float>({2, 4, 2}, 1, 2, x, x_scale, zero_point, y_2);
  DequantizeLinearOp21BlockedTest_Int_Succeed<int16_t, float>({2, 4, 2}, 1, 3, x, x_scale, zero_point, y_3);
  DequantizeLinearOp21BlockedTest_Int_Succeed<int32_t, float>({2, 4, 2}, 1, 2, x, x_scale, zero_point, y_2);
  DequantizeLinearOp21BlockedTest_Int_Succeed<int32_t, float>({2, 4, 2}, 1, 3, x, x_scale, zero_point, y_3);
  DequantizeLinearOp21BlockedTest_Int4_Succeed<Int4x2, MLFloat16>({2, 4, 2}, 1, 2, x, x_scale, zero_point, y_2);
  DequantizeLinearOp21BlockedTest_Int4_Succeed<Int4x2, MLFloat16>({2, 4, 2}, 1, 3, x, x_scale, zero_point, y_3);
  DequantizeLinearOp21BlockedTest_Int_Succeed<int8_t, MLFloat16>({2, 4, 2}, 1, 2, x, x_scale, zero_point, y_2);
  DequantizeLinearOp21BlockedTest_Int_Succeed<int8_t, MLFloat16>({2, 4, 2}, 1, 3, x, x_scale, zero_point, y_3);
  DequantizeLinearOp21BlockedTest_Int_Succeed<int16_t, MLFloat16>({2, 4, 2}, 1, 2, x, x_scale, zero_point, y_2);
  DequantizeLinearOp21BlockedTest_Int_Succeed<int16_t, MLFloat16>({2, 4, 2}, 1, 3, x, x_scale, zero_point, y_3);
  DequantizeLinearOp21BlockedTest_Int_Succeed<int32_t, MLFloat16>({2, 4, 2}, 1, 2, x, x_scale, zero_point, y_2);
  DequantizeLinearOp21BlockedTest_Int_Succeed<int32_t, MLFloat16>({2, 4, 2}, 1, 3, x, x_scale, zero_point, y_3);
}

TEST(DequantizeLinearOp21BlockedTest, SignedInt_NoZeroPoint_MiddleAxis_Cuda) {
  std::vector<int> zero_point{};
  std::vector<float> x_scale{-2.0, -4.0, 3.5, 1.0, 2.0, 4.0, -3.5, -1.0};
  std::vector<int> x{-7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, -8};
  std::vector<float> y_2{14, 24, 10, 16, -10.5, -2, -3.5, 0, 2, 8, 6, 16, -17.5, -6, -24.5, 8};
  std::vector<float> y_3{14, 24, 10, 16, 6, 8, -3.5, 0, 2, 8, 6, 16, 10, 24, -24.5, 8};

  DequantizeLinearOp21BlockedTest_Int4_Succeed<Int4x2, float>({2, 4, 2}, 1, 2, x, x_scale, zero_point, y_2,
                                                              DefaultCudaExecutionProvider());
  DequantizeLinearOp21BlockedTest_Int4_Succeed<Int4x2, float>({2, 4, 2}, 1, 3, x, x_scale, zero_point, y_3,
                                                              DefaultCudaExecutionProvider());
  DequantizeLinearOp21BlockedTest_Int4_Succeed<Int4x2, MLFloat16>({2, 4, 2}, 1, 2, x, x_scale, zero_point, y_2,
                                                                  DefaultCudaExecutionProvider());
  DequantizeLinearOp21BlockedTest_Int4_Succeed<Int4x2, MLFloat16>({2, 4, 2}, 1, 3, x, x_scale, zero_point, y_3,
                                                                  DefaultCudaExecutionProvider());
}

TEST(DequantizeLinearOp21BlockedTest, SignedInt_UseZeroPoint_MiddleAxis) {
  std::vector<float> x_scale{-2.0, -4.0, 3.5, 1.0, 2.0, 4.0, -3.5, -1.0};
  std::vector<int> zero_point{-6, -4, -3, -1, 0, 2, 4, 7};
  std::vector<int> x{-7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, -8};
  std::vector<float> y_2{2, 8, -2, 0, 0, -1, 7, 1, 2, 0, 6, 8, -3.5, 1, -10.5, 15};
  std::vector<float> y_3{2, 8, -2, 0, -6, -8, 7, 1, 2, 0, 6, 8, 10, 16, -10.5, 15};

  DequantizeLinearOp21BlockedTest_Int4_Succeed<Int4x2, float>({2, 4, 2}, 1, 2, x, x_scale, zero_point, y_2);
  DequantizeLinearOp21BlockedTest_Int4_Succeed<Int4x2, float>({2, 4, 2}, 1, 3, x, x_scale, zero_point, y_3);
  DequantizeLinearOp21BlockedTest_Int_Succeed<int8_t, float>({2, 4, 2}, 1, 2, x, x_scale, zero_point, y_2);
  DequantizeLinearOp21BlockedTest_Int_Succeed<int8_t, float>({2, 4, 2}, 1, 3, x, x_scale, zero_point, y_3);
  DequantizeLinearOp21BlockedTest_Int_Succeed<int16_t, float>({2, 4, 2}, 1, 2, x, x_scale, zero_point, y_2);
  DequantizeLinearOp21BlockedTest_Int_Succeed<int16_t, float>({2, 4, 2}, 1, 3, x, x_scale, zero_point, y_3);
  DequantizeLinearOp21BlockedTest_Int4_Succeed<Int4x2, MLFloat16>({2, 4, 2}, 1, 2, x, x_scale, zero_point, y_2);
  DequantizeLinearOp21BlockedTest_Int4_Succeed<Int4x2, MLFloat16>({2, 4, 2}, 1, 3, x, x_scale, zero_point, y_3);
  DequantizeLinearOp21BlockedTest_Int_Succeed<int8_t, MLFloat16>({2, 4, 2}, 1, 2, x, x_scale, zero_point, y_2);
  DequantizeLinearOp21BlockedTest_Int_Succeed<int8_t, MLFloat16>({2, 4, 2}, 1, 3, x, x_scale, zero_point, y_3);
  DequantizeLinearOp21BlockedTest_Int_Succeed<int16_t, MLFloat16>({2, 4, 2}, 1, 2, x, x_scale, zero_point, y_2);
  DequantizeLinearOp21BlockedTest_Int_Succeed<int16_t, MLFloat16>({2, 4, 2}, 1, 3, x, x_scale, zero_point, y_3);
}

TEST(DequantizeLinearOp21BlockedTest, SignedInt_UseZeroPoint_MiddleAxis_Cuda) {
  std::vector<float> x_scale{-2.0, -4.0, 3.5, 1.0, 2.0, 4.0, -3.5, -1.0};
  std::vector<int> zero_point{-6, -4, -3, -1, 0, 2, 4, 7};
  std::vector<int> x{-7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, -8};
  std::vector<float> y_2{2, 8, -2, 0, 0, -1, 7, 1, 2, 0, 6, 8, -3.5, 1, -10.5, 15};
  std::vector<float> y_3{2, 8, -2, 0, -6, -8, 7, 1, 2, 0, 6, 8, 10, 16, -10.5, 15};

  DequantizeLinearOp21BlockedTest_Int4_Succeed<Int4x2, float>({2, 4, 2}, 1, 2, x, x_scale, zero_point, y_2,
                                                              DefaultCudaExecutionProvider());
  DequantizeLinearOp21BlockedTest_Int4_Succeed<Int4x2, float>({2, 4, 2}, 1, 3, x, x_scale, zero_point, y_3,
                                                              DefaultCudaExecutionProvider());
  DequantizeLinearOp21BlockedTest_Int4_Succeed<Int4x2, MLFloat16>({2, 4, 2}, 1, 2, x, x_scale, zero_point, y_2,
                                                                  DefaultCudaExecutionProvider());
  DequantizeLinearOp21BlockedTest_Int4_Succeed<Int4x2, MLFloat16>({2, 4, 2}, 1, 3, x, x_scale, zero_point, y_3,
                                                                  DefaultCudaExecutionProvider());
}

TEST(DequantizeLinearOp21BlockedTest, SignedInt_NoZeroPoint_LastAxis) {
  std::vector<int> zero_point{};
  std::vector<float> x_scale{-2.0, -4.0, 3.5, 1.0, 2.0, 4.0, -3.5, -1.0};
  std::vector<int> x{-7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, -8};
  std::vector<float> y_2{14, 12, 20, 16, -10.5, -7, -1, 0, 2, 4, 12, 16, -17.5, -21, -7, 8};
  std::vector<float> y_3{14, 12, 10, 16, -10.5, -7, -3.5, 0, 2, 4, 6, 16, -17.5, -21, -24.5, 8};

  DequantizeLinearOp21BlockedTest_Int4_Succeed<Int4x2, float>({2, 2, 4}, 2, 2, x, x_scale, zero_point, y_2);
  DequantizeLinearOp21BlockedTest_Int4_Succeed<Int4x2, float>({2, 2, 4}, 2, 3, x, x_scale, zero_point, y_3);
  DequantizeLinearOp21BlockedTest_Int_Succeed<int8_t, float>({2, 2, 4}, 2, 2, x, x_scale, zero_point, y_2);
  DequantizeLinearOp21BlockedTest_Int_Succeed<int8_t, float>({2, 2, 4}, 2, 3, x, x_scale, zero_point, y_3);
  DequantizeLinearOp21BlockedTest_Int_Succeed<int16_t, float>({2, 2, 4}, 2, 2, x, x_scale, zero_point, y_2);
  DequantizeLinearOp21BlockedTest_Int_Succeed<int16_t, float>({2, 2, 4}, 2, 3, x, x_scale, zero_point, y_3);
  DequantizeLinearOp21BlockedTest_Int_Succeed<int32_t, float>({2, 2, 4}, 2, 2, x, x_scale, zero_point, y_2);
  DequantizeLinearOp21BlockedTest_Int_Succeed<int32_t, float>({2, 2, 4}, 2, 3, x, x_scale, zero_point, y_3);
  DequantizeLinearOp21BlockedTest_Int4_Succeed<Int4x2, MLFloat16>({2, 2, 4}, 2, 2, x, x_scale, zero_point, y_2);
  DequantizeLinearOp21BlockedTest_Int4_Succeed<Int4x2, MLFloat16>({2, 2, 4}, 2, 3, x, x_scale, zero_point, y_3);
  DequantizeLinearOp21BlockedTest_Int_Succeed<int8_t, MLFloat16>({2, 2, 4}, 2, 2, x, x_scale, zero_point, y_2);
  DequantizeLinearOp21BlockedTest_Int_Succeed<int8_t, MLFloat16>({2, 2, 4}, 2, 3, x, x_scale, zero_point, y_3);
  DequantizeLinearOp21BlockedTest_Int_Succeed<int16_t, MLFloat16>({2, 2, 4}, 2, 2, x, x_scale, zero_point, y_2);
  DequantizeLinearOp21BlockedTest_Int_Succeed<int16_t, MLFloat16>({2, 2, 4}, 2, 3, x, x_scale, zero_point, y_3);
  DequantizeLinearOp21BlockedTest_Int_Succeed<int32_t, MLFloat16>({2, 2, 4}, 2, 2, x, x_scale, zero_point, y_2);
  DequantizeLinearOp21BlockedTest_Int_Succeed<int32_t, MLFloat16>({2, 2, 4}, 2, 3, x, x_scale, zero_point, y_3);
}

TEST(DequantizeLinearOp21BlockedTest, SignedInt_NoZeroPoint_LastAxis_Cuda) {
  std::vector<int> zero_point{};
  std::vector<float> x_scale{-2.0, -4.0, 3.5, 1.0, 2.0, 4.0, -3.5, -1.0};
  std::vector<int> x{-7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, -8};
  std::vector<float> y_2{14, 12, 20, 16, -10.5, -7, -1, 0, 2, 4, 12, 16, -17.5, -21, -7, 8};
  std::vector<float> y_3{14, 12, 10, 16, -10.5, -7, -3.5, 0, 2, 4, 6, 16, -17.5, -21, -24.5, 8};

  DequantizeLinearOp21BlockedTest_Int4_Succeed<Int4x2, float>({2, 2, 4}, 2, 2, x, x_scale, zero_point, y_2,
                                                              DefaultCudaExecutionProvider());
  DequantizeLinearOp21BlockedTest_Int4_Succeed<Int4x2, float>({2, 2, 4}, 2, 3, x, x_scale, zero_point, y_3,
                                                              DefaultCudaExecutionProvider());
  DequantizeLinearOp21BlockedTest_Int4_Succeed<Int4x2, MLFloat16>({2, 2, 4}, 2, 2, x, x_scale, zero_point, y_2,
                                                                  DefaultCudaExecutionProvider());
  DequantizeLinearOp21BlockedTest_Int4_Succeed<Int4x2, MLFloat16>({2, 2, 4}, 2, 3, x, x_scale, zero_point, y_3,
                                                                  DefaultCudaExecutionProvider());
}

TEST(DequantizeLinearOp21BlockedTest, SignedInt_UseZeroPoint_LastAxis) {
  std::vector<float> x_scale{-2.0, -4.0, 3.5, 1.0, 2.0, 4.0, -3.5, -1.0};
  std::vector<int> zero_point{-6, -4, -3, -1, 0, 2, 4, 7};
  std::vector<int> x{-7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, -8};
  std::vector<float> y_2{2, 0, 4, 0, 0, 3.5, 0, 1, 2, 4, 4, 8, -3.5, -7, 0, 15};
  std::vector<float> y_3{2, 0, -2, 0, 0, 3.5, 7, 1, 2, 4, 6, 8, -3.5, -7, -10.5, 15};

  DequantizeLinearOp21BlockedTest_Int4_Succeed<Int4x2, float>({2, 2, 4}, 2, 2, x, x_scale, zero_point, y_2);
  DequantizeLinearOp21BlockedTest_Int4_Succeed<Int4x2, float>({2, 2, 4}, 2, 3, x, x_scale, zero_point, y_3);
  DequantizeLinearOp21BlockedTest_Int_Succeed<int8_t, float>({2, 2, 4}, 2, 2, x, x_scale, zero_point, y_2);
  DequantizeLinearOp21BlockedTest_Int_Succeed<int8_t, float>({2, 2, 4}, 2, 3, x, x_scale, zero_point, y_3);
  DequantizeLinearOp21BlockedTest_Int_Succeed<int16_t, float>({2, 2, 4}, 2, 2, x, x_scale, zero_point, y_2);
  DequantizeLinearOp21BlockedTest_Int_Succeed<int16_t, float>({2, 2, 4}, 2, 3, x, x_scale, zero_point, y_3);
  DequantizeLinearOp21BlockedTest_Int4_Succeed<Int4x2, MLFloat16>({2, 2, 4}, 2, 2, x, x_scale, zero_point, y_2);
  DequantizeLinearOp21BlockedTest_Int4_Succeed<Int4x2, MLFloat16>({2, 2, 4}, 2, 3, x, x_scale, zero_point, y_3);
  DequantizeLinearOp21BlockedTest_Int_Succeed<int8_t, MLFloat16>({2, 2, 4}, 2, 2, x, x_scale, zero_point, y_2);
  DequantizeLinearOp21BlockedTest_Int_Succeed<int8_t, MLFloat16>({2, 2, 4}, 2, 3, x, x_scale, zero_point, y_3);
  DequantizeLinearOp21BlockedTest_Int_Succeed<int16_t, MLFloat16>({2, 2, 4}, 2, 2, x, x_scale, zero_point, y_2);
  DequantizeLinearOp21BlockedTest_Int_Succeed<int16_t, MLFloat16>({2, 2, 4}, 2, 3, x, x_scale, zero_point, y_3);
}

TEST(DequantizeLinearOp21BlockedTest, SignedInt_UseZeroPoint_LastAxis_Cuda) {
  std::vector<float> x_scale{-2.0, -4.0, 3.5, 1.0, 2.0, 4.0, -3.5, -1.0};
  std::vector<int> zero_point{-6, -4, -3, -1, 0, 2, 4, 7};
  std::vector<int> x{-7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, -8};
  std::vector<float> y_2{2, 0, 4, 0, 0, 3.5, 0, 1, 2, 4, 4, 8, -3.5, -7, 0, 15};
  std::vector<float> y_3{2, 0, -2, 0, 0, 3.5, 7, 1, 2, 4, 6, 8, -3.5, -7, -10.5, 15};

  DequantizeLinearOp21BlockedTest_Int4_Succeed<Int4x2, float>({2, 2, 4}, 2, 2, x, x_scale, zero_point, y_2,
                                                              DefaultCudaExecutionProvider());
  DequantizeLinearOp21BlockedTest_Int4_Succeed<Int4x2, float>({2, 2, 4}, 2, 3, x, x_scale, zero_point, y_3,
                                                              DefaultCudaExecutionProvider());
  DequantizeLinearOp21BlockedTest_Int4_Succeed<Int4x2, MLFloat16>({2, 2, 4}, 2, 2, x, x_scale, zero_point, y_2,
                                                                  DefaultCudaExecutionProvider());
  DequantizeLinearOp21BlockedTest_Int4_Succeed<Int4x2, MLFloat16>({2, 2, 4}, 2, 3, x, x_scale, zero_point, y_3,
                                                                  DefaultCudaExecutionProvider());
}

TEST(DequantizeLinearOp21BlockedTest, UnsignedInt_NoZeroPoint_FirstAxis) {
  std::vector<float> x_scale{-2.0, -4.0, 3.5, 1.0, 2.0, 4.0, -3.5, -1.0};
  std::vector<int> zero_point{};
  std::vector<int> x{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
  std::vector<float> y_2{0, -4, 7, 3, -8, -20, 21, 7, 16, 36, -35, -11, 24, 52, -49, -15};
  std::vector<float> y_3{0, -4, 7, 3, -8, -20, 21, 7, -16, -36, 35, 11, 24, 52, -49, -15};

  DequantizeLinearOp21BlockedTest_Int4_Succeed<UInt4x2, float>({4, 2, 2}, 0, 2, x, x_scale, zero_point, y_2);
  DequantizeLinearOp21BlockedTest_Int4_Succeed<UInt4x2, float>({4, 2, 2}, 0, 3, x, x_scale, zero_point, y_3);
  DequantizeLinearOp21BlockedTest_Int_Succeed<uint8_t, float>({4, 2, 2}, 0, 2, x, x_scale, zero_point, y_2);
  DequantizeLinearOp21BlockedTest_Int_Succeed<uint8_t, float>({4, 2, 2}, 0, 3, x, x_scale, zero_point, y_3);
  DequantizeLinearOp21BlockedTest_Int_Succeed<uint16_t, float>({4, 2, 2}, 0, 2, x, x_scale, zero_point, y_2);
  DequantizeLinearOp21BlockedTest_Int_Succeed<uint16_t, float>({4, 2, 2}, 0, 3, x, x_scale, zero_point, y_3);
  DequantizeLinearOp21BlockedTest_Int4_Succeed<UInt4x2, MLFloat16>({4, 2, 2}, 0, 2, x, x_scale, zero_point, y_2);
  DequantizeLinearOp21BlockedTest_Int4_Succeed<UInt4x2, MLFloat16>({4, 2, 2}, 0, 3, x, x_scale, zero_point, y_3);
  DequantizeLinearOp21BlockedTest_Int_Succeed<uint8_t, MLFloat16>({4, 2, 2}, 0, 2, x, x_scale, zero_point, y_2);
  DequantizeLinearOp21BlockedTest_Int_Succeed<uint8_t, MLFloat16>({4, 2, 2}, 0, 3, x, x_scale, zero_point, y_3);
  DequantizeLinearOp21BlockedTest_Int_Succeed<uint16_t, MLFloat16>({4, 2, 2}, 0, 2, x, x_scale, zero_point, y_2);
  DequantizeLinearOp21BlockedTest_Int_Succeed<uint16_t, MLFloat16>({4, 2, 2}, 0, 3, x, x_scale, zero_point, y_3);
}

TEST(DequantizeLinearOp21BlockedTest, UnsignedInt_NoZeroPoint_FirstAxis_Cuda) {
  std::vector<float> x_scale{-2.0, -4.0, 3.5, 1.0, 2.0, 4.0, -3.5, -1.0};
  std::vector<int> zero_point{};
  std::vector<int> x{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
  std::vector<float> y_2{0, -4, 7, 3, -8, -20, 21, 7, 16, 36, -35, -11, 24, 52, -49, -15};
  std::vector<float> y_3{0, -4, 7, 3, -8, -20, 21, 7, -16, -36, 35, 11, 24, 52, -49, -15};

  DequantizeLinearOp21BlockedTest_Int4_Succeed<UInt4x2, float>({4, 2, 2}, 0, 2, x, x_scale, zero_point, y_2,
                                                               DefaultCudaExecutionProvider());
  DequantizeLinearOp21BlockedTest_Int4_Succeed<UInt4x2, float>({4, 2, 2}, 0, 3, x, x_scale, zero_point, y_3,
                                                               DefaultCudaExecutionProvider());
  DequantizeLinearOp21BlockedTest_Int4_Succeed<UInt4x2, MLFloat16>({4, 2, 2}, 0, 2, x, x_scale, zero_point, y_2,
                                                                   DefaultCudaExecutionProvider());
  DequantizeLinearOp21BlockedTest_Int4_Succeed<UInt4x2, MLFloat16>({4, 2, 2}, 0, 3, x, x_scale, zero_point, y_3,
                                                                   DefaultCudaExecutionProvider());
}

TEST(DequantizeLinearOp21BlockedTest, UnsignedInt_UseZeroPoint_FirstAxis) {
  std::vector<float> x_scale{-2.0, -4.0, 3.5, 1.0, 2.0, 4.0, -3.5, -1.0};
  std::vector<int> zero_point{2, 0, 1, 9, 13, 5, 11, 6};
  std::vector<int> x{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
  std::vector<float> y_2{4, -4, 3.5, -6, -4, -20, 17.5, -2, -10, 16, 3.5, -5, -2, 32, -10.5, -9};
  std::vector<float> y_3{4, -4, 3.5, -6, -4, -20, 17.5, -2, -12, -36, 31.5, 2, -2, 32, -10.5, -9};

  DequantizeLinearOp21BlockedTest_Int4_Succeed<UInt4x2, float>({4, 2, 2}, 0, 2, x, x_scale, zero_point, y_2);
  DequantizeLinearOp21BlockedTest_Int4_Succeed<UInt4x2, float>({4, 2, 2}, 0, 3, x, x_scale, zero_point, y_3);
  DequantizeLinearOp21BlockedTest_Int_Succeed<uint8_t, float>({4, 2, 2}, 0, 2, x, x_scale, zero_point, y_2);
  DequantizeLinearOp21BlockedTest_Int_Succeed<uint8_t, float>({4, 2, 2}, 0, 3, x, x_scale, zero_point, y_3);
  DequantizeLinearOp21BlockedTest_Int_Succeed<uint16_t, float>({4, 2, 2}, 0, 2, x, x_scale, zero_point, y_2);
  DequantizeLinearOp21BlockedTest_Int_Succeed<uint16_t, float>({4, 2, 2}, 0, 3, x, x_scale, zero_point, y_3);
  DequantizeLinearOp21BlockedTest_Int4_Succeed<UInt4x2, MLFloat16>({4, 2, 2}, 0, 2, x, x_scale, zero_point, y_2);
  DequantizeLinearOp21BlockedTest_Int4_Succeed<UInt4x2, MLFloat16>({4, 2, 2}, 0, 3, x, x_scale, zero_point, y_3);
  DequantizeLinearOp21BlockedTest_Int_Succeed<uint8_t, MLFloat16>({4, 2, 2}, 0, 2, x, x_scale, zero_point, y_2);
  DequantizeLinearOp21BlockedTest_Int_Succeed<uint8_t, MLFloat16>({4, 2, 2}, 0, 3, x, x_scale, zero_point, y_3);
  DequantizeLinearOp21BlockedTest_Int_Succeed<uint16_t, MLFloat16>({4, 2, 2}, 0, 2, x, x_scale, zero_point, y_2);
  DequantizeLinearOp21BlockedTest_Int_Succeed<uint16_t, MLFloat16>({4, 2, 2}, 0, 3, x, x_scale, zero_point, y_3);
}

TEST(DequantizeLinearOp21BlockedTest, UnsignedInt_UseZeroPoint_FirstAxis_Cuda) {
  std::vector<float> x_scale{-2.0, -4.0, 3.5, 1.0, 2.0, 4.0, -3.5, -1.0};
  std::vector<int> zero_point{2, 0, 1, 9, 13, 5, 11, 6};
  std::vector<int> x{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
  std::vector<float> y_2{4, -4, 3.5, -6, -4, -20, 17.5, -2, -10, 16, 3.5, -5, -2, 32, -10.5, -9};
  std::vector<float> y_3{4, -4, 3.5, -6, -4, -20, 17.5, -2, -12, -36, 31.5, 2, -2, 32, -10.5, -9};

  DequantizeLinearOp21BlockedTest_Int4_Succeed<UInt4x2, float>({4, 2, 2}, 0, 2, x, x_scale, zero_point, y_2,
                                                               DefaultCudaExecutionProvider());
  DequantizeLinearOp21BlockedTest_Int4_Succeed<UInt4x2, float>({4, 2, 2}, 0, 3, x, x_scale, zero_point, y_3,
                                                               DefaultCudaExecutionProvider());
  DequantizeLinearOp21BlockedTest_Int4_Succeed<UInt4x2, MLFloat16>({4, 2, 2}, 0, 2, x, x_scale, zero_point, y_2,
                                                                   DefaultCudaExecutionProvider());
  DequantizeLinearOp21BlockedTest_Int4_Succeed<UInt4x2, MLFloat16>({4, 2, 2}, 0, 3, x, x_scale, zero_point, y_3,
                                                                   DefaultCudaExecutionProvider());
}

TEST(DequantizeLinearOp21BlockedTest, UnsignedInt_NoZeroPoint_MiddleAxis) {
  std::vector<float> x_scale{-2.0, -4.0, 3.5, 1.0, 2.0, 4.0, -3.5, -1.0};
  std::vector<int> zero_point{};
  std::vector<int> x{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
  std::vector<float> y_2{0, -4, -4, -12, 14, 5, 21, 7, 16, 36, 20, 44, -42, -13, -49, -15};
  std::vector<float> y_3{0, -4, -4, -12, -8, -20, 21, 7, 16, 36, 20, 44, 24, 52, -49, -15};

  DequantizeLinearOp21BlockedTest_Int4_Succeed<UInt4x2, float>({2, 4, 2}, 1, 2, x, x_scale, zero_point, y_2);
  DequantizeLinearOp21BlockedTest_Int4_Succeed<UInt4x2, float>({2, 4, 2}, 1, 3, x, x_scale, zero_point, y_3);
  DequantizeLinearOp21BlockedTest_Int_Succeed<uint8_t, float>({2, 4, 2}, 1, 2, x, x_scale, zero_point, y_2);
  DequantizeLinearOp21BlockedTest_Int_Succeed<uint8_t, float>({2, 4, 2}, 1, 3, x, x_scale, zero_point, y_3);
  DequantizeLinearOp21BlockedTest_Int_Succeed<uint16_t, float>({2, 4, 2}, 1, 2, x, x_scale, zero_point, y_2);
  DequantizeLinearOp21BlockedTest_Int_Succeed<uint16_t, float>({2, 4, 2}, 1, 3, x, x_scale, zero_point, y_3);
  DequantizeLinearOp21BlockedTest_Int4_Succeed<UInt4x2, MLFloat16>({2, 4, 2}, 1, 2, x, x_scale, zero_point, y_2);
  DequantizeLinearOp21BlockedTest_Int4_Succeed<UInt4x2, MLFloat16>({2, 4, 2}, 1, 3, x, x_scale, zero_point, y_3);
  DequantizeLinearOp21BlockedTest_Int_Succeed<uint8_t, MLFloat16>({2, 4, 2}, 1, 2, x, x_scale, zero_point, y_2);
  DequantizeLinearOp21BlockedTest_Int_Succeed<uint8_t, MLFloat16>({2, 4, 2}, 1, 3, x, x_scale, zero_point, y_3);
  DequantizeLinearOp21BlockedTest_Int_Succeed<uint16_t, MLFloat16>({2, 4, 2}, 1, 2, x, x_scale, zero_point, y_2);
  DequantizeLinearOp21BlockedTest_Int_Succeed<uint16_t, MLFloat16>({2, 4, 2}, 1, 3, x, x_scale, zero_point, y_3);
}

TEST(DequantizeLinearOp21BlockedTest, UnsignedInt_NoZeroPoint_MiddleAxis_Cuda) {
  std::vector<float> x_scale{-2.0, -4.0, 3.5, 1.0, 2.0, 4.0, -3.5, -1.0};
  std::vector<int> zero_point{};
  std::vector<int> x{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
  std::vector<float> y_2{0, -4, -4, -12, 14, 5, 21, 7, 16, 36, 20, 44, -42, -13, -49, -15};
  std::vector<float> y_3{0, -4, -4, -12, -8, -20, 21, 7, 16, 36, 20, 44, 24, 52, -49, -15};

  DequantizeLinearOp21BlockedTest_Int4_Succeed<UInt4x2, float>({2, 4, 2}, 1, 2, x, x_scale, zero_point, y_2,
                                                               DefaultCudaExecutionProvider());
  DequantizeLinearOp21BlockedTest_Int4_Succeed<UInt4x2, float>({2, 4, 2}, 1, 3, x, x_scale, zero_point, y_3,
                                                               DefaultCudaExecutionProvider());
  DequantizeLinearOp21BlockedTest_Int4_Succeed<UInt4x2, MLFloat16>({2, 4, 2}, 1, 2, x, x_scale, zero_point, y_2,
                                                                   DefaultCudaExecutionProvider());
  DequantizeLinearOp21BlockedTest_Int4_Succeed<UInt4x2, MLFloat16>({2, 4, 2}, 1, 3, x, x_scale, zero_point, y_3,
                                                                   DefaultCudaExecutionProvider());
}

TEST(DequantizeLinearOp21BlockedTest, UnsignedInt_UseZeroPoint_MiddleAxis) {
  std::vector<float> x_scale{-2.0, -4.0, 3.5, 1.0, 2.0, 4.0, -3.5, -1.0};
  std::vector<int> zero_point{2, 0, 1, 9, 13, 5, 11, 6};
  std::vector<int> x{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
  std::vector<float> y_2{4, -4, 0, -12, 10.5, -4, 17.5, -2, -10, 16, -6, 24, -3.5, -7, -10.5, -9};
  std::vector<float> y_3{4, -4, 0, -12, -4, -20, 17.5, -2, -10, 16, -6, 24, -2, 32, -10.5, -9};

  DequantizeLinearOp21BlockedTest_Int4_Succeed<UInt4x2, float>({2, 4, 2}, 1, 2, x, x_scale, zero_point, y_2);
  DequantizeLinearOp21BlockedTest_Int4_Succeed<UInt4x2, float>({2, 4, 2}, 1, 3, x, x_scale, zero_point, y_3);
  DequantizeLinearOp21BlockedTest_Int_Succeed<uint8_t, float>({2, 4, 2}, 1, 2, x, x_scale, zero_point, y_2);
  DequantizeLinearOp21BlockedTest_Int_Succeed<uint8_t, float>({2, 4, 2}, 1, 3, x, x_scale, zero_point, y_3);
  DequantizeLinearOp21BlockedTest_Int_Succeed<uint16_t, float>({2, 4, 2}, 1, 2, x, x_scale, zero_point, y_2);
  DequantizeLinearOp21BlockedTest_Int_Succeed<uint16_t, float>({2, 4, 2}, 1, 3, x, x_scale, zero_point, y_3);
  DequantizeLinearOp21BlockedTest_Int4_Succeed<UInt4x2, MLFloat16>({2, 4, 2}, 1, 2, x, x_scale, zero_point, y_2);
  DequantizeLinearOp21BlockedTest_Int4_Succeed<UInt4x2, MLFloat16>({2, 4, 2}, 1, 3, x, x_scale, zero_point, y_3);
  DequantizeLinearOp21BlockedTest_Int_Succeed<uint8_t, MLFloat16>({2, 4, 2}, 1, 2, x, x_scale, zero_point, y_2);
  DequantizeLinearOp21BlockedTest_Int_Succeed<uint8_t, MLFloat16>({2, 4, 2}, 1, 3, x, x_scale, zero_point, y_3);
  DequantizeLinearOp21BlockedTest_Int_Succeed<uint16_t, MLFloat16>({2, 4, 2}, 1, 2, x, x_scale, zero_point, y_2);
  DequantizeLinearOp21BlockedTest_Int_Succeed<uint16_t, MLFloat16>({2, 4, 2}, 1, 3, x, x_scale, zero_point, y_3);
}

TEST(DequantizeLinearOp21BlockedTest, UnsignedInt_UseZeroPoint_MiddleAxis_Cuda) {
  std::vector<float> x_scale{-2.0, -4.0, 3.5, 1.0, 2.0, 4.0, -3.5, -1.0};
  std::vector<int> zero_point{2, 0, 1, 9, 13, 5, 11, 6};
  std::vector<int> x{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
  std::vector<float> y_2{4, -4, 0, -12, 10.5, -4, 17.5, -2, -10, 16, -6, 24, -3.5, -7, -10.5, -9};
  std::vector<float> y_3{4, -4, 0, -12, -4, -20, 17.5, -2, -10, 16, -6, 24, -2, 32, -10.5, -9};

  DequantizeLinearOp21BlockedTest_Int4_Succeed<UInt4x2, float>({2, 4, 2}, 1, 2, x, x_scale, zero_point, y_2,
                                                               DefaultCudaExecutionProvider());
  DequantizeLinearOp21BlockedTest_Int4_Succeed<UInt4x2, float>({2, 4, 2}, 1, 3, x, x_scale, zero_point, y_3,
                                                               DefaultCudaExecutionProvider());
  DequantizeLinearOp21BlockedTest_Int4_Succeed<UInt4x2, MLFloat16>({2, 4, 2}, 1, 2, x, x_scale, zero_point, y_2,
                                                                   DefaultCudaExecutionProvider());
  DequantizeLinearOp21BlockedTest_Int4_Succeed<UInt4x2, MLFloat16>({2, 4, 2}, 1, 3, x, x_scale, zero_point, y_3,
                                                                   DefaultCudaExecutionProvider());
}

TEST(DequantizeLinearOp21BlockedTest, UnsignedInt_NoZeroPoint_LastAxis) {
  std::vector<float> x_scale{-2.0, -4.0, 3.5, 1.0, 2.0, 4.0, -3.5, -1.0};
  std::vector<int> zero_point{};
  std::vector<int> x{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
  std::vector<float> y_2{0, -2, -8, -12, 14, 17.5, 6, 7, 16, 18, 40, 44, -42, -45.5, -14, -15};
  std::vector<float> y_3{0, -2, -4, -12, 14, 17.5, 21, 7, 16, 18, 20, 44, -42, -45.5, -49, -15};

  DequantizeLinearOp21BlockedTest_Int4_Succeed<UInt4x2, float>({2, 2, 4}, 2, 2, x, x_scale, zero_point, y_2);
  DequantizeLinearOp21BlockedTest_Int4_Succeed<UInt4x2, float>({2, 2, 4}, 2, 3, x, x_scale, zero_point, y_3);
  DequantizeLinearOp21BlockedTest_Int_Succeed<uint8_t, float>({2, 2, 4}, 2, 2, x, x_scale, zero_point, y_2);
  DequantizeLinearOp21BlockedTest_Int_Succeed<uint8_t, float>({2, 2, 4}, 2, 3, x, x_scale, zero_point, y_3);
  DequantizeLinearOp21BlockedTest_Int_Succeed<uint16_t, float>({2, 2, 4}, 2, 2, x, x_scale, zero_point, y_2);
  DequantizeLinearOp21BlockedTest_Int_Succeed<uint16_t, float>({2, 2, 4}, 2, 3, x, x_scale, zero_point, y_3);
  DequantizeLinearOp21BlockedTest_Int4_Succeed<UInt4x2, MLFloat16>({2, 2, 4}, 2, 2, x, x_scale, zero_point, y_2);
  DequantizeLinearOp21BlockedTest_Int4_Succeed<UInt4x2, MLFloat16>({2, 2, 4}, 2, 3, x, x_scale, zero_point, y_3);
  DequantizeLinearOp21BlockedTest_Int_Succeed<uint8_t, MLFloat16>({2, 2, 4}, 2, 2, x, x_scale, zero_point, y_2);
  DequantizeLinearOp21BlockedTest_Int_Succeed<uint8_t, MLFloat16>({2, 2, 4}, 2, 3, x, x_scale, zero_point, y_3);
  DequantizeLinearOp21BlockedTest_Int_Succeed<uint16_t, MLFloat16>({2, 2, 4}, 2, 2, x, x_scale, zero_point, y_2);
  DequantizeLinearOp21BlockedTest_Int_Succeed<uint16_t, MLFloat16>({2, 2, 4}, 2, 3, x, x_scale, zero_point, y_3);
}

TEST(DequantizeLinearOp21BlockedTest, UnsignedInt_NoZeroPoint_LastAxis_Cuda) {
  std::vector<float> x_scale{-2.0, -4.0, 3.5, 1.0, 2.0, 4.0, -3.5, -1.0};
  std::vector<int> zero_point{};
  std::vector<int> x{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
  std::vector<float> y_2{0, -2, -8, -12, 14, 17.5, 6, 7, 16, 18, 40, 44, -42, -45.5, -14, -15};
  std::vector<float> y_3{0, -2, -4, -12, 14, 17.5, 21, 7, 16, 18, 20, 44, -42, -45.5, -49, -15};

  DequantizeLinearOp21BlockedTest_Int4_Succeed<UInt4x2, float>({2, 2, 4}, 2, 2, x, x_scale, zero_point, y_2,
                                                               DefaultCudaExecutionProvider());
  DequantizeLinearOp21BlockedTest_Int4_Succeed<UInt4x2, float>({2, 2, 4}, 2, 3, x, x_scale, zero_point, y_3,
                                                               DefaultCudaExecutionProvider());
  DequantizeLinearOp21BlockedTest_Int4_Succeed<UInt4x2, MLFloat16>({2, 2, 4}, 2, 2, x, x_scale, zero_point, y_2,
                                                                   DefaultCudaExecutionProvider());
  DequantizeLinearOp21BlockedTest_Int4_Succeed<UInt4x2, MLFloat16>({2, 2, 4}, 2, 3, x, x_scale, zero_point, y_3,
                                                                   DefaultCudaExecutionProvider());
}

TEST(DequantizeLinearOp21BlockedTest, UnsignedInt_UseZeroPoint_LastAxis) {
  std::vector<float> x_scale{-2.0, -4.0, 3.5, 1.0, 2.0, 4.0, -3.5, -1.0};
  std::vector<int> zero_point{2, 0, 1, 9, 13, 5, 11, 6};
  std::vector<int> x{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
  std::vector<float> y_2{4, 2, -8, -12, 10.5, 14, -3, -2, -10, -8, 20, 24, -3.5, -7, -8, -9};
  std::vector<float> y_3{4, 2, 0, -12, 10.5, 14, 17.5, -2, -10, -8, -6, 24, -3.5, -7, -10.5, -9};

  DequantizeLinearOp21BlockedTest_Int4_Succeed<UInt4x2, float>({2, 2, 4}, 2, 2, x, x_scale, zero_point, y_2);
  DequantizeLinearOp21BlockedTest_Int4_Succeed<UInt4x2, float>({2, 2, 4}, 2, 3, x, x_scale, zero_point, y_3);
  DequantizeLinearOp21BlockedTest_Int_Succeed<uint8_t, float>({2, 2, 4}, 2, 2, x, x_scale, zero_point, y_2);
  DequantizeLinearOp21BlockedTest_Int_Succeed<uint8_t, float>({2, 2, 4}, 2, 3, x, x_scale, zero_point, y_3);
  DequantizeLinearOp21BlockedTest_Int_Succeed<uint16_t, float>({2, 2, 4}, 2, 2, x, x_scale, zero_point, y_2);
  DequantizeLinearOp21BlockedTest_Int_Succeed<uint16_t, float>({2, 2, 4}, 2, 3, x, x_scale, zero_point, y_3);
  DequantizeLinearOp21BlockedTest_Int4_Succeed<UInt4x2, MLFloat16>({2, 2, 4}, 2, 2, x, x_scale, zero_point, y_2);
  DequantizeLinearOp21BlockedTest_Int4_Succeed<UInt4x2, MLFloat16>({2, 2, 4}, 2, 3, x, x_scale, zero_point, y_3);
  DequantizeLinearOp21BlockedTest_Int_Succeed<uint8_t, MLFloat16>({2, 2, 4}, 2, 2, x, x_scale, zero_point, y_2);
  DequantizeLinearOp21BlockedTest_Int_Succeed<uint8_t, MLFloat16>({2, 2, 4}, 2, 3, x, x_scale, zero_point, y_3);
  DequantizeLinearOp21BlockedTest_Int_Succeed<uint16_t, MLFloat16>({2, 2, 4}, 2, 2, x, x_scale, zero_point, y_2);
  DequantizeLinearOp21BlockedTest_Int_Succeed<uint16_t, MLFloat16>({2, 2, 4}, 2, 3, x, x_scale, zero_point, y_3);
}

TEST(DequantizeLinearOp21BlockedTest, UnsignedInt_UseZeroPoint_LastAxis_Cuda) {
  std::vector<float> x_scale{-2.0, -4.0, 3.5, 1.0, 2.0, 4.0, -3.5, -1.0};
  std::vector<int> zero_point{2, 0, 1, 9, 13, 5, 11, 6};
  std::vector<int> x{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
  std::vector<float> y_2{4, 2, -8, -12, 10.5, 14, -3, -2, -10, -8, 20, 24, -3.5, -7, -8, -9};
  std::vector<float> y_3{4, 2, 0, -12, 10.5, 14, 17.5, -2, -10, -8, -6, 24, -3.5, -7, -10.5, -9};

  DequantizeLinearOp21BlockedTest_Int4_Succeed<UInt4x2, float>({2, 2, 4}, 2, 2, x, x_scale, zero_point, y_2,
                                                               DefaultCudaExecutionProvider());
  DequantizeLinearOp21BlockedTest_Int4_Succeed<UInt4x2, float>({2, 2, 4}, 2, 3, x, x_scale, zero_point, y_3,
                                                               DefaultCudaExecutionProvider());
  DequantizeLinearOp21BlockedTest_Int4_Succeed<UInt4x2, MLFloat16>({2, 2, 4}, 2, 2, x, x_scale, zero_point, y_2,
                                                                   DefaultCudaExecutionProvider());
  DequantizeLinearOp21BlockedTest_Int4_Succeed<UInt4x2, MLFloat16>({2, 2, 4}, 2, 3, x, x_scale, zero_point, y_3,
                                                                   DefaultCudaExecutionProvider());
}

#if !defined(DISABLE_FLOAT8_TYPES)
TEST(DequantizeLinearOp21BlockedTest, Float8_NoZeroPoint_FirstAxis) {
  constexpr int min_cuda_architecture = 11080;
  bool enable_cuda = (nullptr != DefaultCpuExecutionProvider().get()) && HasCudaEnvironment(min_cuda_architecture);
  bool enable_cpu = (nullptr != DefaultCpuExecutionProvider().get());

  std::vector<float> x_scale{-2.0, -4.0, 3.5, 1.0, 2.0, 4.0, -3.5, -1.0};
  std::vector<int> zero_point{};
  std::vector<int> x{-7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8};
  std::vector<float> y_2{14.0, 24.0, -17.5, -4.0, 6.0, 8.0, -3.5, 0.0, 2.0, 8.0, -10.5, -4.0, 10.0, 24.0, -24.5, -8.0};
  std::vector<float> y_3{14.0, 24.0, -17.5, -4.0, 6.0, 8.0, -3.5, 0.0, -2.0, -8.0, 10.5, 4.0, 10.0, 24.0, -24.5, -8.0};

  if (enable_cpu || enable_cuda) {
    DequantizeLinearOp21BlockedTest_Float8_Succeed<Float8E4M3FN, float>({4, 2, 2}, 0, 2, x, x_scale, zero_point, y_2);
    DequantizeLinearOp21BlockedTest_Float8_Succeed<Float8E4M3FN, MLFloat16>(
        {4, 2, 2}, 0, 2, x, x_scale, zero_point, y_2);
    DequantizeLinearOp21BlockedTest_Float8_Succeed<Float8E5M2, float>({4, 2, 2}, 0, 2, x, x_scale, zero_point, y_2);
    DequantizeLinearOp21BlockedTest_Float8_Succeed<Float8E5M2, MLFloat16>({4, 2, 2}, 0, 2, x, x_scale, zero_point, y_2);
    DequantizeLinearOp21BlockedTest_Float8_Succeed<Float8E4M3FN, float>({4, 2, 2}, 0, 3, x, x_scale, zero_point, y_3);
    DequantizeLinearOp21BlockedTest_Float8_Succeed<Float8E4M3FN, MLFloat16>(
        {4, 2, 2}, 0, 3, x, x_scale, zero_point, y_3);
    DequantizeLinearOp21BlockedTest_Float8_Succeed<Float8E5M2, float>({4, 2, 2}, 0, 3, x, x_scale, zero_point, y_3);
    DequantizeLinearOp21BlockedTest_Float8_Succeed<Float8E5M2, MLFloat16>({4, 2, 2}, 0, 3, x, x_scale, zero_point, y_3);
  }
  if (enable_cpu) {
    DequantizeLinearOp21BlockedTest_Float8_Succeed<Float8E4M3FNUZ, float>({4, 2, 2}, 0, 2, x, x_scale, zero_point, y_2);
    DequantizeLinearOp21BlockedTest_Float8_Succeed<Float8E4M3FNUZ, MLFloat16>(
        {4, 2, 2}, 0, 2, x, x_scale, zero_point, y_2);
    DequantizeLinearOp21BlockedTest_Float8_Succeed<Float8E5M2FNUZ, float>({4, 2, 2}, 0, 2, x, x_scale, zero_point, y_2);
    DequantizeLinearOp21BlockedTest_Float8_Succeed<Float8E5M2FNUZ, MLFloat16>(
        {4, 2, 2}, 0, 2, x, x_scale, zero_point, y_2);
    DequantizeLinearOp21BlockedTest_Float8_Succeed<Float8E4M3FNUZ, float>({4, 2, 2}, 0, 3, x, x_scale, zero_point, y_3);
    DequantizeLinearOp21BlockedTest_Float8_Succeed<Float8E4M3FNUZ, MLFloat16>(
        {4, 2, 2}, 0, 3, x, x_scale, zero_point, y_3);
    DequantizeLinearOp21BlockedTest_Float8_Succeed<Float8E5M2FNUZ, float>({4, 2, 2}, 0, 3, x, x_scale, zero_point, y_3);
    DequantizeLinearOp21BlockedTest_Float8_Succeed<Float8E5M2FNUZ, MLFloat16>(
        {4, 2, 2}, 0, 3, x, x_scale, zero_point, y_3);
  }
}

TEST(DequantizeLinearOp21BlockedTest, Float8_NoZeroPoint_MiddleAxis) {
  constexpr int min_cuda_architecture = 11080;
  bool enable_cuda = (nullptr != DefaultCpuExecutionProvider().get()) && HasCudaEnvironment(min_cuda_architecture);
  bool enable_cpu = (nullptr != DefaultCpuExecutionProvider().get());

  std::vector<int> zero_point{};
  std::vector<float> x_scale{-2.0, -4.0, 3.5, 1.0, 2.0, 4.0, -3.5, -1.0};
  std::vector<int> x{-7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8};
  std::vector<float> y_2{14, 24, 10, 16, -10.5, -2, -3.5, 0, 2, 8, 6, 16, -17.5, -6, -24.5, -8};
  std::vector<float> y_3{14, 24, 10, 16, 6, 8, -3.5, 0, 2, 8, 6, 16, 10, 24, -24.5, -8};

  if (enable_cpu || enable_cuda) {
    DequantizeLinearOp21BlockedTest_Float8_Succeed<Float8E4M3FN, float>({2, 4, 2}, 1, 2, x, x_scale, zero_point, y_2);
    DequantizeLinearOp21BlockedTest_Float8_Succeed<Float8E4M3FN, MLFloat16>(
        {2, 4, 2}, 1, 2, x, x_scale, zero_point, y_2);
    DequantizeLinearOp21BlockedTest_Float8_Succeed<Float8E5M2, float>({2, 4, 2}, 1, 2, x, x_scale, zero_point, y_2);
    DequantizeLinearOp21BlockedTest_Float8_Succeed<Float8E5M2, MLFloat16>({2, 4, 2}, 1, 2, x, x_scale, zero_point, y_2);
    DequantizeLinearOp21BlockedTest_Float8_Succeed<Float8E4M3FN, float>({2, 4, 2}, 1, 3, x, x_scale, zero_point, y_3);
    DequantizeLinearOp21BlockedTest_Float8_Succeed<Float8E4M3FN, MLFloat16>(
        {2, 4, 2}, 1, 3, x, x_scale, zero_point, y_3);
    DequantizeLinearOp21BlockedTest_Float8_Succeed<Float8E5M2, float>({2, 4, 2}, 1, 3, x, x_scale, zero_point, y_3);
    DequantizeLinearOp21BlockedTest_Float8_Succeed<Float8E5M2, MLFloat16>({2, 4, 2}, 1, 3, x, x_scale, zero_point, y_3);
  }
  if (enable_cpu) {
    DequantizeLinearOp21BlockedTest_Float8_Succeed<Float8E4M3FNUZ, float>({2, 4, 2}, 1, 2, x, x_scale, zero_point, y_2);
    DequantizeLinearOp21BlockedTest_Float8_Succeed<Float8E4M3FNUZ, MLFloat16>(
        {2, 4, 2}, 1, 2, x, x_scale, zero_point, y_2);
    DequantizeLinearOp21BlockedTest_Float8_Succeed<Float8E5M2FNUZ, float>({2, 4, 2}, 1, 2, x, x_scale, zero_point, y_2);
    DequantizeLinearOp21BlockedTest_Float8_Succeed<Float8E5M2FNUZ, MLFloat16>(
        {2, 4, 2}, 1, 2, x, x_scale, zero_point, y_2);
    DequantizeLinearOp21BlockedTest_Float8_Succeed<Float8E4M3FNUZ, float>({2, 4, 2}, 1, 3, x, x_scale, zero_point, y_3);
    DequantizeLinearOp21BlockedTest_Float8_Succeed<Float8E4M3FNUZ, MLFloat16>(
        {2, 4, 2}, 1, 3, x, x_scale, zero_point, y_3);
    DequantizeLinearOp21BlockedTest_Float8_Succeed<Float8E5M2FNUZ, float>({2, 4, 2}, 1, 3, x, x_scale, zero_point, y_3);
    DequantizeLinearOp21BlockedTest_Float8_Succeed<Float8E5M2FNUZ, MLFloat16>(
        {2, 4, 2}, 1, 3, x, x_scale, zero_point, y_3);
  }
}

TEST(DequantizeLinearOp21BlockedTest, Float8_NoZeroPoint_LastAxis) {
  constexpr int min_cuda_architecture = 11080;
  bool enable_cuda = (nullptr != DefaultCpuExecutionProvider().get()) && HasCudaEnvironment(min_cuda_architecture);
  bool enable_cpu = (nullptr != DefaultCpuExecutionProvider().get());

  std::vector<int> zero_point{};
  std::vector<float> x_scale{-2.0, -4.0, 3.5, 1.0, 2.0, 4.0, -3.5, -1.0};
  std::vector<int> x{-7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8};
  std::vector<float> y_2{14, 12, 20, 16, -10.5, -7, -1, 0, 2, 4, 12, 16, -17.5, -21, -7, -8};
  std::vector<float> y_3{14, 12, 10, 16, -10.5, -7, -3.5, 0, 2, 4, 6, 16, -17.5, -21, -24.5, -8};

  if (enable_cpu || enable_cuda) {
    DequantizeLinearOp21BlockedTest_Float8_Succeed<Float8E4M3FN, float>({2, 2, 4}, 2, 2, x, x_scale, zero_point, y_2);
    DequantizeLinearOp21BlockedTest_Float8_Succeed<Float8E4M3FN, MLFloat16>(
        {2, 2, 4}, 2, 2, x, x_scale, zero_point, y_2);
    DequantizeLinearOp21BlockedTest_Float8_Succeed<Float8E5M2, float>({2, 2, 4}, 2, 2, x, x_scale, zero_point, y_2);
    DequantizeLinearOp21BlockedTest_Float8_Succeed<Float8E5M2, MLFloat16>({2, 2, 4}, 2, 2, x, x_scale, zero_point, y_2);
    DequantizeLinearOp21BlockedTest_Float8_Succeed<Float8E4M3FN, float>({2, 2, 4}, 2, 3, x, x_scale, zero_point, y_3);
    DequantizeLinearOp21BlockedTest_Float8_Succeed<Float8E4M3FN, MLFloat16>(
        {2, 2, 4}, 2, 3, x, x_scale, zero_point, y_3);
    DequantizeLinearOp21BlockedTest_Float8_Succeed<Float8E5M2, float>({2, 2, 4}, 2, 3, x, x_scale, zero_point, y_3);
    DequantizeLinearOp21BlockedTest_Float8_Succeed<Float8E5M2, MLFloat16>({2, 2, 4}, 2, 3, x, x_scale, zero_point, y_3);
  }
  if (enable_cpu) {
    DequantizeLinearOp21BlockedTest_Float8_Succeed<Float8E4M3FNUZ, float>({2, 2, 4}, 2, 2, x, x_scale, zero_point, y_2);
    DequantizeLinearOp21BlockedTest_Float8_Succeed<Float8E4M3FNUZ, MLFloat16>(
        {2, 2, 4}, 2, 2, x, x_scale, zero_point, y_2);
    DequantizeLinearOp21BlockedTest_Float8_Succeed<Float8E5M2FNUZ, float>({2, 2, 4}, 2, 2, x, x_scale, zero_point, y_2);
    DequantizeLinearOp21BlockedTest_Float8_Succeed<Float8E5M2FNUZ, MLFloat16>(
        {2, 2, 4}, 2, 2, x, x_scale, zero_point, y_2);
    DequantizeLinearOp21BlockedTest_Float8_Succeed<Float8E4M3FNUZ, float>({2, 2, 4}, 2, 3, x, x_scale, zero_point, y_3);
    DequantizeLinearOp21BlockedTest_Float8_Succeed<Float8E4M3FNUZ, MLFloat16>(
        {2, 2, 4}, 2, 3, x, x_scale, zero_point, y_3);
    DequantizeLinearOp21BlockedTest_Float8_Succeed<Float8E5M2FNUZ, float>({2, 2, 4}, 2, 3, x, x_scale, zero_point, y_3);
    DequantizeLinearOp21BlockedTest_Float8_Succeed<Float8E5M2FNUZ, MLFloat16>(
        {2, 2, 4}, 2, 3, x, x_scale, zero_point, y_3);
  }
}
#endif
}  // namespace blocked_dequantization

namespace blocked_quantization {

template <typename Tout, typename Tin>
void QuantizeLinearOp21BlockedTest_InvalidBlockSize_Int(int64_t block_size,
                                                        int64_t scale_block_count,
                                                        int64_t zero_point_block_count) {
  OpTester test("QuantizeLinear", 21);
  std::vector<int64_t> dims{2, 4};
  std::vector<Tout> y, x_zero_point;
  std::vector<Tin> x, x_scale;
  SessionOptions so;
  std::vector<std::string> log_msgs;  // redirect error messages
  std::vector<std::unique_ptr<IExecutionProvider>> eps;
  eps.push_back(DefaultCpuExecutionProvider());
  so.user_logging_function = [](void* param, OrtLoggingLevel severity, const char* category,
                                const char* logid, const char* code_location, const char* message) {
    ORT_UNUSED_PARAMETER(severity);
    ORT_UNUSED_PARAMETER(category);
    ORT_UNUSED_PARAMETER(logid);
    ORT_UNUSED_PARAMETER(code_location);
    std::vector<std::string>* v_ptr = reinterpret_cast<std::vector<std::string>*>(param);
    std::vector<std::string>& msg_vector = *v_ptr;
    msg_vector.push_back(std::string(message));
  };
  so.user_logging_param = &log_msgs;
  so.session_logid = "QuantizeLinear";
  so.use_per_session_threads = false;
  so.session_log_verbosity_level = 1;
  so.graph_optimization_level = TransformerLevel::Default;

  for (int64_t i = 0, n = 2 * zero_point_block_count; i < n; ++i) x_zero_point.push_back(0);
  for (int64_t i = 0, n = 2 * scale_block_count; i < n; i++) x_scale.push_back(Tin(2.0f));
  for (int i = 0; i < 8; ++i) {
    x.push_back(Tin(static_cast<float>(i) * 2.0f));
    y.push_back(i);
  }

  test.AddInput<Tin>("x", dims, x);
  test.AddAttribute<int64_t>("axis", 1);
  test.AddAttribute<int64_t>("block_size", block_size);
  test.AddInput<Tin>("scale", {2, scale_block_count}, x_scale);
  test.AddInput<Tout>("zero_point", {2, zero_point_block_count}, x_zero_point);
  test.AddOutput<Tout>("y", dims, y);
  test.Run(so, OpTester::ExpectResult::kExpectFailure, "", {}, nullptr, &eps);
}

template <typename Tout, typename Tin>
void QuantizeLinearOp21BlockedTest_InvalidBlockSize_Int4(int64_t block_size,
                                                         int64_t scale_block_count,
                                                         int64_t zero_point_block_count,
                                                         std::unique_ptr<IExecutionProvider> ep = nullptr) {
  OpTester test("QuantizeLinear", 21);
  std::vector<int64_t> dims{2, 4};
  std::vector<Tout> x_zero_point, y;
  std::vector<Tin> x, x_scale;
  SessionOptions so;
  std::vector<std::string> log_msgs;  // redirect error messages
  std::vector<std::unique_ptr<IExecutionProvider>> eps;
  eps.push_back(ep ? std::move(ep) : DefaultCpuExecutionProvider());
  so.user_logging_function = [](void* param, OrtLoggingLevel severity, const char* category,
                                const char* logid, const char* code_location, const char* message) {
    ORT_UNUSED_PARAMETER(severity);
    ORT_UNUSED_PARAMETER(category);
    ORT_UNUSED_PARAMETER(logid);
    ORT_UNUSED_PARAMETER(code_location);
    std::vector<std::string>* v_ptr = reinterpret_cast<std::vector<std::string>*>(param);
    std::vector<std::string>& msg_vector = *v_ptr;
    msg_vector.push_back(std::string(message));
  };
  so.user_logging_param = &log_msgs;
  so.session_logid = "QuantizeLinear";
  so.use_per_session_threads = false;
  so.session_log_verbosity_level = 1;
  so.graph_optimization_level = TransformerLevel::Default;

  for (int64_t i = 0, n = zero_point_block_count; i < n; ++i) x_zero_point.push_back(Tout(0, 0));
  for (int64_t i = 0, n = 2 * scale_block_count; i < n; i++) x_scale.push_back(Tin(2.0f));
  for (int i = 0; i < 8; ++i) {
    if (i & 1) y.push_back(Tout(i - 1, i));
    x.push_back(Tin(static_cast<float>(i) * 2.0f));
  }

  test.AddInput<Tin>("x", dims, x);
  test.AddAttribute<int64_t>("axis", 1);
  test.AddAttribute<int64_t>("block_size", block_size);
  test.AddInput<Tin>("scale", {2, scale_block_count}, x_scale);
  test.AddInput<Tout>("zero_point", {2, zero_point_block_count}, x_zero_point);
  test.AddOutput<Tout>("y", dims, y);
  test.Run(so, OpTester::ExpectResult::kExpectFailure, "", {}, nullptr, &eps);
}

template <typename Tout, typename Tin>
void QuantizeLinearOp21BlockedTest_InvalidBlockSize_Float8(int64_t block_size,
                                                           int64_t scale_block_count,
                                                           int64_t zero_point_block_count) {
  OpTester test("QuantizeLinear", 21);
  std::vector<int64_t> dims{2, 4};
  std::vector<Tout> x_zero_point, y;
  std::vector<Tin> x, x_scale;
  SessionOptions so;
  std::vector<std::string> log_msgs;  // redirect error messages
  std::vector<std::unique_ptr<IExecutionProvider>> eps;
  eps.push_back(DefaultCpuExecutionProvider());
  so.user_logging_function = [](void* param, OrtLoggingLevel severity, const char* category,
                                const char* logid, const char* code_location, const char* message) {
    ORT_UNUSED_PARAMETER(severity);
    ORT_UNUSED_PARAMETER(category);
    ORT_UNUSED_PARAMETER(logid);
    ORT_UNUSED_PARAMETER(code_location);
    std::vector<std::string>* v_ptr = reinterpret_cast<std::vector<std::string>*>(param);
    std::vector<std::string>& msg_vector = *v_ptr;
    msg_vector.push_back(std::string(message));
  };
  so.user_logging_param = &log_msgs;
  so.session_logid = "QuantizeLinear";
  so.use_per_session_threads = false;
  so.session_log_verbosity_level = 1;
  so.graph_optimization_level = TransformerLevel::Default;

  for (int64_t i = 0, n = 2 * zero_point_block_count; i < n; i++) x_zero_point.push_back(Tout(0.0f));
  for (int64_t i = 0, n = 2 * scale_block_count; i < n; i++) x_scale.push_back(Tin(2.0f));
  for (int i = 0; i < 8; ++i) y.push_back(Tout(static_cast<float>(i)));
  for (int i = 0; i < 8; ++i) x.push_back(Tin(static_cast<float>(i) * 2.0f));

  test.AddInput<Tin>("x", dims, x);
  test.AddAttribute<int64_t>("axis", 1);
  test.AddAttribute<int64_t>("block_size", block_size);
  test.AddInput<Tin>("scale", {2, scale_block_count}, x_scale);
  test.AddInput<Tout>("zero_point", {2, zero_point_block_count}, x_zero_point);
  test.AddOutput<Tout>("y", dims, y);
  test.Run(so, OpTester::ExpectResult::kExpectFailure, "", {}, nullptr, &eps);
}

// test negative block size fail
TEST(QuantizeLinearOp21BlockedTest, NagativeBlockSize_Int) {
  QuantizeLinearOp21BlockedTest_InvalidBlockSize_Int4<Int4x2, float>(-1, 2, 2);
  QuantizeLinearOp21BlockedTest_InvalidBlockSize_Int4<Int4x2, MLFloat16>(-1, 2, 2);
  QuantizeLinearOp21BlockedTest_InvalidBlockSize_Int4<UInt4x2, float>(-2, 2, 2);
  QuantizeLinearOp21BlockedTest_InvalidBlockSize_Int4<UInt4x2, MLFloat16>(-2, 2, 2);
  QuantizeLinearOp21BlockedTest_InvalidBlockSize_Int<int8_t, float>(-3, 2, 2);
  QuantizeLinearOp21BlockedTest_InvalidBlockSize_Int<int8_t, MLFloat16>(-3, 2, 2);
  QuantizeLinearOp21BlockedTest_InvalidBlockSize_Int<uint8_t, float>(-4, 2, 2);
  QuantizeLinearOp21BlockedTest_InvalidBlockSize_Int<uint8_t, MLFloat16>(-4, 2, 2);
  QuantizeLinearOp21BlockedTest_InvalidBlockSize_Int<int16_t, float>(-5, 2, 2);
  QuantizeLinearOp21BlockedTest_InvalidBlockSize_Int<int16_t, MLFloat16>(-5, 2, 2);
  QuantizeLinearOp21BlockedTest_InvalidBlockSize_Int<uint16_t, float>(-6, 2, 2);
  QuantizeLinearOp21BlockedTest_InvalidBlockSize_Int<uint16_t, MLFloat16>(-1, 2, 2);
  QuantizeLinearOp21BlockedTest_InvalidBlockSize_Int<int32_t, float>(-1, 2, 2);
  QuantizeLinearOp21BlockedTest_InvalidBlockSize_Int<int32_t, MLFloat16>(-1, 2, 2);
}

TEST(QuantizeLinearOp21BlockedTest, NagativeBlockSize_Int_Cuda) {
  QuantizeLinearOp21BlockedTest_InvalidBlockSize_Int4<Int4x2, float>(-1, 2, 2, DefaultCudaExecutionProvider());
  QuantizeLinearOp21BlockedTest_InvalidBlockSize_Int4<Int4x2, MLFloat16>(-1, 2, 2, DefaultCudaExecutionProvider());
  QuantizeLinearOp21BlockedTest_InvalidBlockSize_Int4<UInt4x2, float>(-2, 2, 2, DefaultCudaExecutionProvider());
  QuantizeLinearOp21BlockedTest_InvalidBlockSize_Int4<UInt4x2, MLFloat16>(-2, 2, 2, DefaultCudaExecutionProvider());
}

#if !defined(DISABLE_FLOAT8_TYPES)
TEST(QuantizeLinearOp21BlockedTest, NagativeBlockSize_Float8) {
  constexpr int min_cuda_architecture = 11080;
  bool enable_cuda = (nullptr != DefaultCpuExecutionProvider().get()) && HasCudaEnvironment(min_cuda_architecture);
  bool enable_cpu = (nullptr != DefaultCpuExecutionProvider().get());

  if (enable_cpu || enable_cuda) {
    QuantizeLinearOp21BlockedTest_InvalidBlockSize_Float8<Float8E4M3FN, float>(-1, 2, 2);
    QuantizeLinearOp21BlockedTest_InvalidBlockSize_Float8<Float8E4M3FN, MLFloat16>(-2, 2, 2);
  }
  if (enable_cpu) {
    QuantizeLinearOp21BlockedTest_InvalidBlockSize_Float8<Float8E4M3FNUZ, float>(-3, 2, 2);
    QuantizeLinearOp21BlockedTest_InvalidBlockSize_Float8<Float8E4M3FNUZ, MLFloat16>(-4, 2, 2);
  }
  if (enable_cpu || enable_cuda) {
    QuantizeLinearOp21BlockedTest_InvalidBlockSize_Float8<Float8E5M2, float>(-5, 2, 2);
    QuantizeLinearOp21BlockedTest_InvalidBlockSize_Float8<Float8E5M2, MLFloat16>(-6, 2, 2);
  }
  if (enable_cpu) {
    QuantizeLinearOp21BlockedTest_InvalidBlockSize_Float8<Float8E5M2FNUZ, float>(-1, 2, 2);
    QuantizeLinearOp21BlockedTest_InvalidBlockSize_Float8<Float8E5M2FNUZ, MLFloat16>(-1, 2, 2);
  }
}
#endif

// test block size incompatible with x_scale shape fail
TEST(QuantizeLinearOp21BlockedTest, IncompatibleBlockSizeWithX_Int) {
  QuantizeLinearOp21BlockedTest_InvalidBlockSize_Int4<Int4x2, float>(3, 1, 1);
  QuantizeLinearOp21BlockedTest_InvalidBlockSize_Int4<Int4x2, MLFloat16>(3, 3, 3);
  QuantizeLinearOp21BlockedTest_InvalidBlockSize_Int4<UInt4x2, float>(3, 3, 3);
  QuantizeLinearOp21BlockedTest_InvalidBlockSize_Int4<UInt4x2, MLFloat16>(3, 1, 1);
  QuantizeLinearOp21BlockedTest_InvalidBlockSize_Int<int8_t, float>(3, 1, 1);
  QuantizeLinearOp21BlockedTest_InvalidBlockSize_Int<int8_t, MLFloat16>(3, 3, 3);
  QuantizeLinearOp21BlockedTest_InvalidBlockSize_Int<uint8_t, float>(3, 3, 3);
  QuantizeLinearOp21BlockedTest_InvalidBlockSize_Int<uint8_t, MLFloat16>(3, 1, 1);
  QuantizeLinearOp21BlockedTest_InvalidBlockSize_Int<int16_t, float>(3, 1, 1);
  QuantizeLinearOp21BlockedTest_InvalidBlockSize_Int<int16_t, MLFloat16>(3, 3, 3);
  QuantizeLinearOp21BlockedTest_InvalidBlockSize_Int<uint16_t, float>(3, 3, 3);
  QuantizeLinearOp21BlockedTest_InvalidBlockSize_Int<uint16_t, MLFloat16>(3, 1, 1);
  QuantizeLinearOp21BlockedTest_InvalidBlockSize_Int<int32_t, float>(3, 3, 3);
  QuantizeLinearOp21BlockedTest_InvalidBlockSize_Int<int32_t, MLFloat16>(3, 1, 1);
}

TEST(QuantizeLinearOp21BlockedTest, IncompatibleBlockSizeWithX_Int_Cuda) {
  QuantizeLinearOp21BlockedTest_InvalidBlockSize_Int4<Int4x2, float>(3, 1, 1, DefaultCudaExecutionProvider());
  QuantizeLinearOp21BlockedTest_InvalidBlockSize_Int4<Int4x2, MLFloat16>(3, 3, 3, DefaultCudaExecutionProvider());
  QuantizeLinearOp21BlockedTest_InvalidBlockSize_Int4<UInt4x2, float>(3, 3, 3, DefaultCudaExecutionProvider());
  QuantizeLinearOp21BlockedTest_InvalidBlockSize_Int4<UInt4x2, MLFloat16>(3, 1, 1, DefaultCudaExecutionProvider());
}

#if !defined(DISABLE_FLOAT8_TYPES)
TEST(QuantizeLinearOp21BlockedTest, IncompatibleBlockSizeWithX_Float8) {
  constexpr int min_cuda_architecture = 11080;
  bool enable_cuda = (nullptr != DefaultCpuExecutionProvider().get()) && HasCudaEnvironment(min_cuda_architecture);
  bool enable_cpu = (nullptr != DefaultCpuExecutionProvider().get());

  if (enable_cpu || enable_cuda) {
    QuantizeLinearOp21BlockedTest_InvalidBlockSize_Float8<Float8E4M3FN, float>(3, 1, 1);
    QuantizeLinearOp21BlockedTest_InvalidBlockSize_Float8<Float8E4M3FN, MLFloat16>(3, 3, 3);
    QuantizeLinearOp21BlockedTest_InvalidBlockSize_Float8<Float8E5M2, float>(3, 1, 1);
    QuantizeLinearOp21BlockedTest_InvalidBlockSize_Float8<Float8E5M2, MLFloat16>(3, 3, 3);
  }
  if (enable_cpu) {
    QuantizeLinearOp21BlockedTest_InvalidBlockSize_Float8<Float8E4M3FNUZ, float>(3, 1, 1);
    QuantizeLinearOp21BlockedTest_InvalidBlockSize_Float8<Float8E4M3FNUZ, MLFloat16>(3, 3, 3);
    QuantizeLinearOp21BlockedTest_InvalidBlockSize_Float8<Float8E5M2FNUZ, float>(3, 1, 1);
    QuantizeLinearOp21BlockedTest_InvalidBlockSize_Float8<Float8E5M2FNUZ, MLFloat16>(3, 3, 3);
  }
}
#endif

// test x_scale vs. x_zero_point shape incompatible fail
TEST(QuantizeLinearOp21BlockedTest, ScaleShapeUnmatchZeroPoint_Int) {
  QuantizeLinearOp21BlockedTest_InvalidBlockSize_Int4<Int4x2, float>(3, 2, 1);
  QuantizeLinearOp21BlockedTest_InvalidBlockSize_Int4<Int4x2, MLFloat16>(3, 2, 3);
  QuantizeLinearOp21BlockedTest_InvalidBlockSize_Int4<UInt4x2, float>(3, 2, 3);
  QuantizeLinearOp21BlockedTest_InvalidBlockSize_Int4<UInt4x2, MLFloat16>(3, 2, 1);
  QuantizeLinearOp21BlockedTest_InvalidBlockSize_Int<int8_t, float>(3, 2, 1);
  QuantizeLinearOp21BlockedTest_InvalidBlockSize_Int<int8_t, MLFloat16>(3, 2, 3);
  QuantizeLinearOp21BlockedTest_InvalidBlockSize_Int<uint8_t, float>(3, 2, 3);
  QuantizeLinearOp21BlockedTest_InvalidBlockSize_Int<uint8_t, MLFloat16>(3, 2, 1);
  QuantizeLinearOp21BlockedTest_InvalidBlockSize_Int<int16_t, float>(3, 2, 1);
  QuantizeLinearOp21BlockedTest_InvalidBlockSize_Int<int16_t, MLFloat16>(3, 2, 3);
  QuantizeLinearOp21BlockedTest_InvalidBlockSize_Int<uint16_t, float>(3, 2, 3);
  QuantizeLinearOp21BlockedTest_InvalidBlockSize_Int<uint16_t, MLFloat16>(3, 2, 1);
  QuantizeLinearOp21BlockedTest_InvalidBlockSize_Int<int32_t, float>(3, 2, 3);
  QuantizeLinearOp21BlockedTest_InvalidBlockSize_Int<int32_t, MLFloat16>(3, 2, 1);
}

TEST(QuantizeLinearOp21BlockedTest, ScaleShapeUnmatchZeroPoint_Int_Cuda) {
  QuantizeLinearOp21BlockedTest_InvalidBlockSize_Int4<Int4x2, float>(3, 2, 1, DefaultCudaExecutionProvider());
  QuantizeLinearOp21BlockedTest_InvalidBlockSize_Int4<Int4x2, MLFloat16>(3, 2, 3, DefaultCudaExecutionProvider());
  QuantizeLinearOp21BlockedTest_InvalidBlockSize_Int4<UInt4x2, float>(3, 2, 3, DefaultCudaExecutionProvider());
  QuantizeLinearOp21BlockedTest_InvalidBlockSize_Int4<UInt4x2, MLFloat16>(3, 2, 1, DefaultCudaExecutionProvider());
}

#if !defined(DISABLE_FLOAT8_TYPES)
TEST(QuantizeLinearOp21BlockedTest, ScaleShapeUnmatchZeroPoint_Float8) {
  constexpr int min_cuda_architecture = 11080;
  bool enable_cuda = (nullptr != DefaultCpuExecutionProvider().get()) && HasCudaEnvironment(min_cuda_architecture);
  bool enable_cpu = (nullptr != DefaultCpuExecutionProvider().get());

  if (enable_cpu || enable_cuda) {
    QuantizeLinearOp21BlockedTest_InvalidBlockSize_Float8<Float8E4M3FN, float>(3, 2, 1);
    QuantizeLinearOp21BlockedTest_InvalidBlockSize_Float8<Float8E4M3FN, MLFloat16>(3, 2, 3);
    QuantizeLinearOp21BlockedTest_InvalidBlockSize_Float8<Float8E5M2, float>(3, 2, 1);
    QuantizeLinearOp21BlockedTest_InvalidBlockSize_Float8<Float8E5M2, MLFloat16>(3, 2, 3);
  }
  if (enable_cpu) {
    QuantizeLinearOp21BlockedTest_InvalidBlockSize_Float8<Float8E4M3FNUZ, float>(3, 2, 1);
    QuantizeLinearOp21BlockedTest_InvalidBlockSize_Float8<Float8E4M3FNUZ, MLFloat16>(3, 2, 3);
    QuantizeLinearOp21BlockedTest_InvalidBlockSize_Float8<Float8E5M2FNUZ, float>(3, 2, 1);
    QuantizeLinearOp21BlockedTest_InvalidBlockSize_Float8<Float8E5M2FNUZ, MLFloat16>(3, 2, 3);
  }
}
#endif

// test DQ with blocked quantization succeed
template <typename Tout, typename Tin>
void QuantizeLinearOp21BlockedTest_Int4_Succeed(std::vector<int64_t>&& dims,
                                                int64_t axis,
                                                int64_t block_size,
                                                std::vector<float>& x_,
                                                std::vector<float>& scale_,
                                                std::vector<int>& zero_point_,
                                                std::vector<int>& y_,
                                                std::unique_ptr<IExecutionProvider> ep = nullptr) {
  OpTester test("QuantizeLinear", 21);
  std::vector<int64_t> scale_shape;
  std::vector<Tout> zero_point, y;
  std::vector<Tin> x, scale;
  std::vector<std::unique_ptr<IExecutionProvider>> eps;
  eps.push_back(ep ? std::move(ep) : DefaultCpuExecutionProvider());
  int64_t non_neg_axis = axis < 0 ? axis + dims.size() : axis;
  bool use_zero_point = !zero_point_.empty();

  for (auto v : x_) x.push_back(Tin(v));
  for (auto v : scale_) scale.push_back(Tin(v));
  for (size_t i = 0, n = dims.size(); i < n; ++i) {
    scale_shape.push_back((int64_t)i == non_neg_axis ? (dims[i] + block_size - 1) / block_size : dims[i]);
  }

  size_t i = 0, n = y_.size();
  for (; i < n - 1; i += 2) y.push_back(Tout(y_[i], y_[i + 1]));
  if (i < n) y.push_back(Tout(y_[i], 0xF));

  if (use_zero_point) {
    i = 0, n = zero_point_.size();
    for (; i < n - 1; i += 2) zero_point.push_back(Tout(zero_point_[i], zero_point_[i + 1]));
    if (i < n) zero_point.push_back(Tout(zero_point_[i], 0xF));
  }

  test.AddInput<Tin>("x", dims, x);
  test.AddAttribute<int64_t>("axis", axis);
  test.AddAttribute<int64_t>("block_size", block_size);
  test.AddInput<Tin>("y_scale", scale_shape, scale);
  if (use_zero_point) test.AddInput<Tout>("y_zero_point", scale_shape, zero_point);
  test.AddOutput<Tout>("y", dims, y);
  test.Run(BaseTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &eps);
}

template <typename Tout, typename Tin>
void QuantizeLinearOp21BlockedTest_Int_Succeed(std::vector<int64_t>&& dims,
                                               int64_t axis,
                                               int64_t block_size,
                                               std::vector<float>& x_,
                                               std::vector<float>& scale_,
                                               std::vector<int>& zero_point_,
                                               std::vector<int>& y_) {
  OpTester test("QuantizeLinear", 21);
  std::vector<int64_t> scale_shape;
  std::vector<Tout> zero_point, y;
  std::vector<Tin> x, scale;
  std::vector<std::unique_ptr<IExecutionProvider>> eps;
  eps.push_back(DefaultCpuExecutionProvider());

  int64_t non_neg_axis = axis < 0 ? axis + dims.size() : axis;
  bool use_zero_point = !zero_point_.empty();

  for (auto v : x_) x.push_back(Tin(v));
  for (auto v : scale_) scale.push_back(Tin(v));
  for (size_t i = 0, n = dims.size(); i < n; ++i) {
    scale_shape.push_back((int64_t)i == non_neg_axis ? (dims[i] + block_size - 1) / block_size : dims[i]);
  }
  for (auto v : y_) y.push_back(v);
  if (use_zero_point)
    for (auto v : zero_point_) zero_point.push_back(v);

  test.AddInput<Tin>("x", dims, x);
  test.AddAttribute<int64_t>("axis", axis);
  test.AddAttribute<int64_t>("block_size", block_size);
  test.AddInput<Tin>("scale", scale_shape, scale);
  if (use_zero_point) test.AddInput<Tout>("zero_point", scale_shape, zero_point);
  test.AddOutput<Tout>("y", dims, y);
  test.Run(BaseTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &eps);
}

template <typename Tout, typename Tin>
void QuantizeLinearOp21BlockedTest_Float8_Succeed(std::vector<int64_t>&& dims,
                                                  int64_t axis,
                                                  int64_t block_size,
                                                  std::vector<float>& x_,
                                                  std::vector<float>& scale_,
                                                  std::vector<float>& zero_point_,
                                                  std::vector<float>& y_) {
  OpTester test("QuantizeLinear", 21);
  std::vector<int64_t> scale_shape;
  std::vector<Tout> zero_point, y;
  std::vector<Tin> x, scale;
  std::vector<std::unique_ptr<IExecutionProvider>> eps;
  eps.push_back(DefaultCpuExecutionProvider());

  int64_t non_neg_axis = axis < 0 ? axis + dims.size() : axis;
  bool use_zero_point = !zero_point_.empty();

  for (auto v : x_) x.push_back(Tin(v));
  for (auto v : scale_) scale.push_back(Tin(v));
  for (size_t i = 0, n = dims.size(); i < n; ++i) {
    scale_shape.push_back((int64_t)i == non_neg_axis ? (dims[i] + block_size - 1) / block_size : dims[i]);
  }

  for (auto v : y_) y.push_back(Tout(static_cast<float>(v)));
  if (use_zero_point) {
    for (auto v : zero_point_) zero_point.push_back(Tout(static_cast<float>(v)));
  }

  test.AddInput<Tin>("x", dims, x);
  test.AddAttribute<int64_t>("axis", axis);
  test.AddAttribute<int64_t>("block_size", block_size);
  test.AddInput<Tin>("scale", scale_shape, scale);
  if (use_zero_point) test.AddInput<Tout>("zero_point", scale_shape, zero_point);
  test.AddOutput<Tout>("y", dims, y);
  test.Run(BaseTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &eps);
}

TEST(QuantizeLinearOp21BlockedTest, SignedInt4_NoZeroPoint_FirstAxis) {
  std::vector<float> y_scale{-2.0, -4.0, 3.5, 1.0, -2.0, -4.0, 3.5, 1.0, -2.0, -4.0, 3.5, 1.0, -2.0, -4.0, 3.5, 1.0,
                             2.0, 4.0, -3.5, -1.0, 2.0, 4.0, -3.5, -1.0, 2.0, 4.0, -3.5, -1.0, 2.0, 4.0, -3.5, -1.0};
  std::vector<int> zero_point{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                              0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  std::vector<float> x{14.0, 24.0, -17.5, -4, 14.0, 24.0, -17.5, -4, 14.0, 24.0, -17.5, -4, 14.0, 24.0, -17.5, -4,
                       6.0, 8.0, -3.5, 0.0, 6.0, 8.0, -3.5, 0.0, 6.0, 8.0, -3.5, 0.0, 6.0, 8.0, -3.5, 0.0,
                       2.0, 8.0, -10.5, -4.0, 2.0, 8.0, -10.5, -4.0, 2.0, 8.0, -10.5, -4.0, 2.0, 8.0, -10.5, -4.0,
                       10.0, 24.0, -24.5, 8.0, 10.0, 24.0, -24.5, 8.0, 10.0, 24.0, -24.5, 8.0, 10.0, 24.0, -24.5, 8.0};
  std::vector<int> y_2{-7, -6, -5, -4, -7, -6, -5, -4, -7, -6, -5, -4, -7, -6, -5, -4,
                       -3, -2, -1, 0, -3, -2, -1, 0, -3, -2, -1, 0, -3, -2, -1, 0,
                       1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4,
                       5, 6, 7, -8, 5, 6, 7, -8, 5, 6, 7, -8, 5, 6, 7, -8};
  std::vector<int> y_3{-7, -6, -5, -4, -7, -6, -5, -4, -7, -6, -5, -4, -7, -6, -5, -4,
                       -3, -2, -1, 0, -3, -2, -1, 0, -3, -2, -1, 0, -3, -2, -1, 0,
                       -1, -2, -3, -4, -1, -2, -3, -4, -1, -2, -3, -4, -1, -2, -3, -4,
                       5, 6, 7, -8, 5, 6, 7, -8, 5, 6, 7, -8, 5, 6, 7, -8};

  QuantizeLinearOp21BlockedTest_Int4_Succeed<Int4x2, float>({4, 8, 2}, 0, 2, x, y_scale, zero_point, y_2);
  QuantizeLinearOp21BlockedTest_Int4_Succeed<Int4x2, float>({4, 8, 2}, 0, 3, x, y_scale, zero_point, y_3);
  QuantizeLinearOp21BlockedTest_Int4_Succeed<Int4x2, MLFloat16>({4, 8, 2}, 0, 2, x, y_scale, zero_point, y_2);
  QuantizeLinearOp21BlockedTest_Int4_Succeed<Int4x2, MLFloat16>({4, 8, 2}, 0, 3, x, y_scale, zero_point, y_3);
}

TEST(QuantizeLinearOp21BlockedTest, SignedInt4_NoZeroPoint_FirstAxis_Cuda) {
  std::vector<float> y_scale{-2.0, -4.0, 3.5, 1.0, -2.0, -4.0, 3.5, 1.0, -2.0, -4.0, 3.5, 1.0, -2.0, -4.0, 3.5, 1.0,
                             2.0, 4.0, -3.5, -1.0, 2.0, 4.0, -3.5, -1.0, 2.0, 4.0, -3.5, -1.0, 2.0, 4.0, -3.5, -1.0};
  std::vector<int> zero_point{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                              0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  std::vector<float> x{14.0, 24.0, -17.5, -4, 14.0, 24.0, -17.5, -4, 14.0, 24.0, -17.5, -4, 14.0, 24.0, -17.5, -4,
                       6.0, 8.0, -3.5, 0.0, 6.0, 8.0, -3.5, 0.0, 6.0, 8.0, -3.5, 0.0, 6.0, 8.0, -3.5, 0.0,
                       2.0, 8.0, -10.5, -4.0, 2.0, 8.0, -10.5, -4.0, 2.0, 8.0, -10.5, -4.0, 2.0, 8.0, -10.5, -4.0,
                       10.0, 24.0, -24.5, 8.0, 10.0, 24.0, -24.5, 8.0, 10.0, 24.0, -24.5, 8.0, 10.0, 24.0, -24.5, 8.0};
  std::vector<int> y_2{-7, -6, -5, -4, -7, -6, -5, -4, -7, -6, -5, -4, -7, -6, -5, -4,
                       -3, -2, -1, 0, -3, -2, -1, 0, -3, -2, -1, 0, -3, -2, -1, 0,
                       1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4,
                       5, 6, 7, -8, 5, 6, 7, -8, 5, 6, 7, -8, 5, 6, 7, -8};
  std::vector<int> y_3{-7, -6, -5, -4, -7, -6, -5, -4, -7, -6, -5, -4, -7, -6, -5, -4,
                       -3, -2, -1, 0, -3, -2, -1, 0, -3, -2, -1, 0, -3, -2, -1, 0,
                       -1, -2, -3, -4, -1, -2, -3, -4, -1, -2, -3, -4, -1, -2, -3, -4,
                       5, 6, 7, -8, 5, 6, 7, -8, 5, 6, 7, -8, 5, 6, 7, -8};

  QuantizeLinearOp21BlockedTest_Int4_Succeed<Int4x2, float>({4, 8, 2}, 0, 2, x, y_scale, zero_point, y_2,
                                                            DefaultCudaExecutionProvider());
  QuantizeLinearOp21BlockedTest_Int4_Succeed<Int4x2, float>({4, 8, 2}, 0, 3, x, y_scale, zero_point, y_3,
                                                            DefaultCudaExecutionProvider());
  QuantizeLinearOp21BlockedTest_Int4_Succeed<Int4x2, MLFloat16>({4, 8, 2}, 0, 2, x, y_scale, zero_point, y_2,
                                                                DefaultCudaExecutionProvider());
  QuantizeLinearOp21BlockedTest_Int4_Succeed<Int4x2, MLFloat16>({4, 8, 2}, 0, 3, x, y_scale, zero_point, y_3,
                                                                DefaultCudaExecutionProvider());
}

TEST(QuantizeLinearOp21BlockedTest, SignedInt_NoZeroPoint_FirstAxis) {
  std::vector<float> y_scale{-2.0, -4.0, 3.5, 1.0, -2.0, -4.0, 3.5, 1.0, -2.0, -4.0, 3.5, 1.0, -2.0, -4.0, 3.5, 1.0,
                             2.0, 4.0, -3.5, -1.0, 2.0, 4.0, -3.5, -1.0, 2.0, 4.0, -3.5, -1.0, 2.0, 4.0, -3.5, -1.0};
  std::vector<int> zero_point{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                              0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  std::vector<float> x{14.0, 24.0, -17.5, -4, 14.0, 24.0, -17.5, -4, 14.0, 24.0, -17.5, -4, 14.0, 24.0, -17.5, -4,
                       6.0, 8.0, -3.5, 0.0, 6.0, 8.0, -3.5, 0.0, 6.0, 8.0, -3.5, 0.0, 6.0, 8.0, -3.5, 0.0,
                       2.0, 8.0, -10.5, -4.0, 2.0, 8.0, -10.5, -4.0, 2.0, 8.0, -10.5, -4.0, 2.0, 8.0, -10.5, -4.0,
                       10.0, 24.0, -24.5, 8.0, 10.0, 24.0, -24.5, 8.0, 10.0, 24.0, -24.5, 8.0, 10.0, 24.0, -24.5, 8.0};
  std::vector<int> y_2{-7, -6, -5, -4, -7, -6, -5, -4, -7, -6, -5, -4, -7, -6, -5, -4,
                       -3, -2, -1, 0, -3, -2, -1, 0, -3, -2, -1, 0, -3, -2, -1, 0,
                       1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4,
                       5, 6, 7, -8, 5, 6, 7, -8, 5, 6, 7, -8, 5, 6, 7, -8};
  std::vector<int> y_3{-7, -6, -5, -4, -7, -6, -5, -4, -7, -6, -5, -4, -7, -6, -5, -4,
                       -3, -2, -1, 0, -3, -2, -1, 0, -3, -2, -1, 0, -3, -2, -1, 0,
                       -1, -2, -3, -4, -1, -2, -3, -4, -1, -2, -3, -4, -1, -2, -3, -4,
                       5, 6, 7, -8, 5, 6, 7, -8, 5, 6, 7, -8, 5, 6, 7, -8};

  QuantizeLinearOp21BlockedTest_Int_Succeed<int8_t, float>({4, 8, 2}, 0, 2, x, y_scale, zero_point, y_2);
  QuantizeLinearOp21BlockedTest_Int_Succeed<int8_t, float>({4, 8, 2}, 0, 3, x, y_scale, zero_point, y_3);
  QuantizeLinearOp21BlockedTest_Int_Succeed<int16_t, float>({4, 8, 2}, 0, 2, x, y_scale, zero_point, y_2);
  QuantizeLinearOp21BlockedTest_Int_Succeed<int16_t, float>({4, 8, 2}, 0, 3, x, y_scale, zero_point, y_3);
  QuantizeLinearOp21BlockedTest_Int_Succeed<int8_t, MLFloat16>({4, 8, 2}, 0, 2, x, y_scale, zero_point, y_2);
  QuantizeLinearOp21BlockedTest_Int_Succeed<int8_t, MLFloat16>({4, 8, 2}, 0, 3, x, y_scale, zero_point, y_3);
  QuantizeLinearOp21BlockedTest_Int_Succeed<int16_t, MLFloat16>({4, 8, 2}, 0, 2, x, y_scale, zero_point, y_2);
  QuantizeLinearOp21BlockedTest_Int_Succeed<int16_t, MLFloat16>({4, 8, 2}, 0, 3, x, y_scale, zero_point, y_3);
}

TEST(QuantizeLinearOp21BlockedTest, SignedInt4_UseZeroPoint_FirstAxis) {
  std::vector<float> y_scale{-2.0, -4.0, 3.5, 1.0, -2.0, -4.0, 3.5, 1.0, -2.0, -4.0, 3.5, 1.0, -2.0, -4.0, 3.5, 1.0,
                             2.0, 4.0, -3.5, -1.0, 2.0, 4.0, -3.5, -1.0, 2.0, 4.0, -3.5, -1.0, 2.0, 4.0, -3.5, -1.0};
  std::vector<int> zero_point{-6, -4, -3, -1, -6, -4, -3, -1, -6, -4, -3, -1, -6, -4, -3, -1,
                              0, 2, 4, 7, 0, 2, 4, 7, 0, 2, 4, 7, 0, 2, 4, 7};
  std::vector<float> x{2.0, 8.0, -7.0, -3, 2.0, 8.0, -7.0, -3, 2.0, 8.0, -7.0, -3, 2.0, 8.0, -7.0, -3,
                       -6.0, -8.0, 7.0, 1, -6.0, -8.0, 7.0, 1, -6.0, -8.0, 7.0, 1, -6.0, -8.0, 7.0, 1,
                       2.0, 0, 3.5, 3.0, 2.0, 0, 3.5, 3.0, 2.0, 0, 3.5, 3.0, 2.0, 0, 3.5, 3.0,
                       10.0, 16.0, -10.5, 15, 10.0, 16.0, -10.5, 15, 10.0, 16.0, -10.5, 15, 10.0, 16.0, -10.5, 15};
  std::vector<int> y_2{-7, -6, -5, -4, -7, -6, -5, -4, -7, -6, -5, -4, -7, -6, -5, -4,
                       -3, -2, -1, 0, -3, -2, -1, 0, -3, -2, -1, 0, -3, -2, -1, 0,
                       1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4,
                       5, 6, 7, -8, 5, 6, 7, -8, 5, 6, 7, -8, 5, 6, 7, -8};
  std::vector<int> y_3{-7, -6, -5, -4, -7, -6, -5, -4, -7, -6, -5, -4, -7, -6, -5, -4,
                       -3, -2, -1, 0, -3, -2, -1, 0, -3, -2, -1, 0, -3, -2, -1, 0,
                       -7, -4, -2, 2, -7, -4, -2, 2, -7, -4, -2, 2, -7, -4, -2, 2,
                       5, 6, 7, -8, 5, 6, 7, -8, 5, 6, 7, -8, 5, 6, 7, -8};

  QuantizeLinearOp21BlockedTest_Int4_Succeed<Int4x2, float>({4, 8, 2}, 0, 2, x, y_scale, zero_point, y_2);
  QuantizeLinearOp21BlockedTest_Int4_Succeed<Int4x2, float>({4, 8, 2}, 0, 3, x, y_scale, zero_point, y_3);
  QuantizeLinearOp21BlockedTest_Int4_Succeed<Int4x2, MLFloat16>({4, 8, 2}, 0, 2, x, y_scale, zero_point, y_2);
  QuantizeLinearOp21BlockedTest_Int4_Succeed<Int4x2, MLFloat16>({4, 8, 2}, 0, 3, x, y_scale, zero_point, y_3);
}

TEST(QuantizeLinearOp21BlockedTest, SignedInt4_UseZeroPoint_FirstAxis_Cuda) {
  std::vector<float> y_scale{-2.0, -4.0, 3.5, 1.0, -2.0, -4.0, 3.5, 1.0, -2.0, -4.0, 3.5, 1.0, -2.0, -4.0, 3.5, 1.0,
                             2.0, 4.0, -3.5, -1.0, 2.0, 4.0, -3.5, -1.0, 2.0, 4.0, -3.5, -1.0, 2.0, 4.0, -3.5, -1.0};
  std::vector<int> zero_point{-6, -4, -3, -1, -6, -4, -3, -1, -6, -4, -3, -1, -6, -4, -3, -1,
                              0, 2, 4, 7, 0, 2, 4, 7, 0, 2, 4, 7, 0, 2, 4, 7};
  std::vector<float> x{2.0, 8.0, -7.0, -3, 2.0, 8.0, -7.0, -3, 2.0, 8.0, -7.0, -3, 2.0, 8.0, -7.0, -3,
                       -6.0, -8.0, 7.0, 1, -6.0, -8.0, 7.0, 1, -6.0, -8.0, 7.0, 1, -6.0, -8.0, 7.0, 1,
                       2.0, 0, 3.5, 3.0, 2.0, 0, 3.5, 3.0, 2.0, 0, 3.5, 3.0, 2.0, 0, 3.5, 3.0,
                       10.0, 16.0, -10.5, 15, 10.0, 16.0, -10.5, 15, 10.0, 16.0, -10.5, 15, 10.0, 16.0, -10.5, 15};
  std::vector<int> y_2{-7, -6, -5, -4, -7, -6, -5, -4, -7, -6, -5, -4, -7, -6, -5, -4,
                       -3, -2, -1, 0, -3, -2, -1, 0, -3, -2, -1, 0, -3, -2, -1, 0,
                       1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4,
                       5, 6, 7, -8, 5, 6, 7, -8, 5, 6, 7, -8, 5, 6, 7, -8};
  std::vector<int> y_3{-7, -6, -5, -4, -7, -6, -5, -4, -7, -6, -5, -4, -7, -6, -5, -4,
                       -3, -2, -1, 0, -3, -2, -1, 0, -3, -2, -1, 0, -3, -2, -1, 0,
                       -7, -4, -2, 2, -7, -4, -2, 2, -7, -4, -2, 2, -7, -4, -2, 2,
                       5, 6, 7, -8, 5, 6, 7, -8, 5, 6, 7, -8, 5, 6, 7, -8};

  QuantizeLinearOp21BlockedTest_Int4_Succeed<Int4x2, float>({4, 8, 2}, 0, 2, x, y_scale, zero_point, y_2,
                                                            DefaultCudaExecutionProvider());
  QuantizeLinearOp21BlockedTest_Int4_Succeed<Int4x2, float>({4, 8, 2}, 0, 3, x, y_scale, zero_point, y_3,
                                                            DefaultCudaExecutionProvider());
  QuantizeLinearOp21BlockedTest_Int4_Succeed<Int4x2, MLFloat16>({4, 8, 2}, 0, 2, x, y_scale, zero_point, y_2,
                                                                DefaultCudaExecutionProvider());
  QuantizeLinearOp21BlockedTest_Int4_Succeed<Int4x2, MLFloat16>({4, 8, 2}, 0, 3, x, y_scale, zero_point, y_3,
                                                                DefaultCudaExecutionProvider());
}

TEST(QuantizeLinearOp21BlockedTest, SignedInt_UseZeroPoint_FirstAxis) {
  std::vector<float> y_scale{-2.0, -4.0, 3.5, 1.0, -2.0, -4.0, 3.5, 1.0, -2.0, -4.0, 3.5, 1.0, -2.0, -4.0, 3.5, 1.0,
                             2.0, 4.0, -3.5, -1.0, 2.0, 4.0, -3.5, -1.0, 2.0, 4.0, -3.5, -1.0, 2.0, 4.0, -3.5, -1.0};
  std::vector<int> zero_point{-6, -4, -3, -1, -6, -4, -3, -1, -6, -4, -3, -1, -6, -4, -3, -1,
                              0, 2, 4, 7, 0, 2, 4, 7, 0, 2, 4, 7, 0, 2, 4, 7};
  std::vector<float> x{2.0, 8.0, -7.0, -3, 2.0, 8.0, -7.0, -3, 2.0, 8.0, -7.0, -3, 2.0, 8.0, -7.0, -3,
                       -6.0, -8.0, 7.0, 1, -6.0, -8.0, 7.0, 1, -6.0, -8.0, 7.0, 1, -6.0, -8.0, 7.0, 1,
                       2.0, 0, 3.5, 3.0, 2.0, 0, 3.5, 3.0, 2.0, 0, 3.5, 3.0, 2.0, 0, 3.5, 3.0,
                       10.0, 16.0, -10.5, 15, 10.0, 16.0, -10.5, 15, 10.0, 16.0, -10.5, 15, 10.0, 16.0, -10.5, 15};
  std::vector<int> y_2{-7, -6, -5, -4, -7, -6, -5, -4, -7, -6, -5, -4, -7, -6, -5, -4,
                       -3, -2, -1, 0, -3, -2, -1, 0, -3, -2, -1, 0, -3, -2, -1, 0,
                       1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4,
                       5, 6, 7, -8, 5, 6, 7, -8, 5, 6, 7, -8, 5, 6, 7, -8};
  std::vector<int> y_3{-7, -6, -5, -4, -7, -6, -5, -4, -7, -6, -5, -4, -7, -6, -5, -4,
                       -3, -2, -1, 0, -3, -2, -1, 0, -3, -2, -1, 0, -3, -2, -1, 0,
                       -7, -4, -2, 2, -7, -4, -2, 2, -7, -4, -2, 2, -7, -4, -2, 2,
                       5, 6, 7, -8, 5, 6, 7, -8, 5, 6, 7, -8, 5, 6, 7, -8};

  QuantizeLinearOp21BlockedTest_Int_Succeed<int8_t, float>({4, 8, 2}, 0, 2, x, y_scale, zero_point, y_2);
  QuantizeLinearOp21BlockedTest_Int_Succeed<int8_t, float>({4, 8, 2}, 0, 3, x, y_scale, zero_point, y_3);
  QuantizeLinearOp21BlockedTest_Int_Succeed<int16_t, float>({4, 8, 2}, 0, 2, x, y_scale, zero_point, y_2);
  QuantizeLinearOp21BlockedTest_Int_Succeed<int16_t, float>({4, 8, 2}, 0, 3, x, y_scale, zero_point, y_3);
  QuantizeLinearOp21BlockedTest_Int_Succeed<int8_t, MLFloat16>({4, 8, 2}, 0, 2, x, y_scale, zero_point, y_2);
  QuantizeLinearOp21BlockedTest_Int_Succeed<int8_t, MLFloat16>({4, 8, 2}, 0, 3, x, y_scale, zero_point, y_3);
  QuantizeLinearOp21BlockedTest_Int_Succeed<int16_t, MLFloat16>({4, 8, 2}, 0, 2, x, y_scale, zero_point, y_2);
  QuantizeLinearOp21BlockedTest_Int_Succeed<int16_t, MLFloat16>({4, 8, 2}, 0, 3, x, y_scale, zero_point, y_3);
}

TEST(QuantizeLinearOp21BlockedTest, SignedInt4_NoZeroPoint_MiddleAxis) {
  std::vector<int> zero_point{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                              0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  std::vector<float> y_scale{-2.0, -4.0, -2.0, -4.0, -2.0, -4.0, -2.0, -4.0, 3.5, 1.0, 3.5, 1.0, 3.5, 1.0, 3.5, 1.0,
                             2.0, 4.0, 2.0, 4.0, 2.0, 4.0, 2.0, 4.0, -3.5, -1.0, -3.5, -1.0, -3.5, -1.0, -3.5, -1.0};
  std::vector<float> x{14, 24, 14, 24, 14, 24, 14, 24, 10, 16, 10, 16, 10, 16, 10, 16,
                       -10.5, -2, -10.5, -2, -10.5, -2, -10.5, -2, -3.5, 0, -3.5, 0, -3.5, 0, -3.5, 0,
                       2, 8, 2, 8, 2, 8, 2, 8, 6, 16, 6, 16, 6, 16, 6, 16,
                       -17.5, -6, -17.5, -6, -17.5, -6, -17.5, -6, -24.5, 8, -24.5, 8, -24.5, 8, -24.5, 8};
  std::vector<int> y_2{-7, -6, -7, -6, -7, -6, -7, -6, -5, -4, -5, -4, -5, -4, -5, -4,
                       -3, -2, -3, -2, -3, -2, -3, -2, -1, 0, -1, 0, -1, 0, -1, 0,
                       1, 2, 1, 2, 1, 2, 1, 2, 3, 4, 3, 4, 3, 4, 3, 4,
                       5, 6, 5, 6, 5, 6, 5, 6, 7, -8, 7, -8, 7, -8, 7, -8};
  std::vector<int> y_3{-7, -6, -7, -6, -7, -6, -7, -6, -5, -4, -5, -4, -5, -4, -5, -4,
                       5, 0, 5, 0, 5, 0, 5, 0, -1, 0, -1, 0, -1, 0, -1, 0,
                       1, 2, 1, 2, 1, 2, 1, 2, 3, 4, 3, 4, 3, 4, 3, 4,
                       -8, -2, -8, -2, -8, -2, -8, -2, 7, -8, 7, -8, 7, -8, 7, -8};

  QuantizeLinearOp21BlockedTest_Int4_Succeed<Int4x2, float>({2, 4, 8}, 1, 2, x, y_scale, zero_point, y_2);
  QuantizeLinearOp21BlockedTest_Int4_Succeed<Int4x2, float>({2, 4, 8}, 1, 3, x, y_scale, zero_point, y_3);
  QuantizeLinearOp21BlockedTest_Int4_Succeed<Int4x2, MLFloat16>({2, 4, 8}, 1, 2, x, y_scale, zero_point, y_2);
  QuantizeLinearOp21BlockedTest_Int4_Succeed<Int4x2, MLFloat16>({2, 4, 8}, 1, 3, x, y_scale, zero_point, y_3);
}

TEST(QuantizeLinearOp21BlockedTest, SignedInt4_NoZeroPoint_MiddleAxis_Cuda) {
  std::vector<int> zero_point{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                              0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  std::vector<float> y_scale{-2.0, -4.0, -2.0, -4.0, -2.0, -4.0, -2.0, -4.0, 3.5, 1.0, 3.5, 1.0, 3.5, 1.0, 3.5, 1.0,
                             2.0, 4.0, 2.0, 4.0, 2.0, 4.0, 2.0, 4.0, -3.5, -1.0, -3.5, -1.0, -3.5, -1.0, -3.5, -1.0};
  std::vector<float> x{14, 24, 14, 24, 14, 24, 14, 24, 10, 16, 10, 16, 10, 16, 10, 16,
                       -10.5, -2, -10.5, -2, -10.5, -2, -10.5, -2, -3.5, 0, -3.5, 0, -3.5, 0, -3.5, 0,
                       2, 8, 2, 8, 2, 8, 2, 8, 6, 16, 6, 16, 6, 16, 6, 16,
                       -17.5, -6, -17.5, -6, -17.5, -6, -17.5, -6, -24.5, 8, -24.5, 8, -24.5, 8, -24.5, 8};
  std::vector<int> y_2{-7, -6, -7, -6, -7, -6, -7, -6, -5, -4, -5, -4, -5, -4, -5, -4,
                       -3, -2, -3, -2, -3, -2, -3, -2, -1, 0, -1, 0, -1, 0, -1, 0,
                       1, 2, 1, 2, 1, 2, 1, 2, 3, 4, 3, 4, 3, 4, 3, 4,
                       5, 6, 5, 6, 5, 6, 5, 6, 7, -8, 7, -8, 7, -8, 7, -8};
  std::vector<int> y_3{-7, -6, -7, -6, -7, -6, -7, -6, -5, -4, -5, -4, -5, -4, -5, -4,
                       5, 0, 5, 0, 5, 0, 5, 0, -1, 0, -1, 0, -1, 0, -1, 0,
                       1, 2, 1, 2, 1, 2, 1, 2, 3, 4, 3, 4, 3, 4, 3, 4,
                       -8, -2, -8, -2, -8, -2, -8, -2, 7, -8, 7, -8, 7, -8, 7, -8};

  QuantizeLinearOp21BlockedTest_Int4_Succeed<Int4x2, float>({2, 4, 8}, 1, 2, x, y_scale, zero_point, y_2,
                                                            DefaultCudaExecutionProvider());
  QuantizeLinearOp21BlockedTest_Int4_Succeed<Int4x2, float>({2, 4, 8}, 1, 3, x, y_scale, zero_point, y_3,
                                                            DefaultCudaExecutionProvider());
  QuantizeLinearOp21BlockedTest_Int4_Succeed<Int4x2, MLFloat16>({2, 4, 8}, 1, 2, x, y_scale, zero_point, y_2,
                                                                DefaultCudaExecutionProvider());
  QuantizeLinearOp21BlockedTest_Int4_Succeed<Int4x2, MLFloat16>({2, 4, 8}, 1, 3, x, y_scale, zero_point, y_3,
                                                                DefaultCudaExecutionProvider());
}

TEST(QuantizeLinearOp21BlockedTest, SignedInt_NoZeroPoint_MiddleAxis) {
  std::vector<int> zero_point{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                              0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  std::vector<float> y_scale{-2.0, -4.0, -2.0, -4.0, -2.0, -4.0, -2.0, -4.0, 3.5, 1.0, 3.5, 1.0, 3.5, 1.0, 3.5, 1.0,
                             2.0, 4.0, 2.0, 4.0, 2.0, 4.0, 2.0, 4.0, -3.5, -1.0, -3.5, -1.0, -3.5, -1.0, -3.5, -1.0};
  std::vector<float> x{14, 24, 14, 24, 14, 24, 14, 24, 10, 16, 10, 16, 10, 16, 10, 16,
                       -10.5, -2, -10.5, -2, -10.5, -2, -10.5, -2, -3.5, 0, -3.5, 0, -3.5, 0, -3.5, 0,
                       2, 8, 2, 8, 2, 8, 2, 8, 6, 16, 6, 16, 6, 16, 6, 16,
                       -17.5, -6, -17.5, -6, -17.5, -6, -17.5, -6, -24.5, 8, -24.5, 8, -24.5, 8, -24.5, 8};
  std::vector<int> y_2{-7, -6, -7, -6, -7, -6, -7, -6, -5, -4, -5, -4, -5, -4, -5, -4,
                       -3, -2, -3, -2, -3, -2, -3, -2, -1, 0, -1, 0, -1, 0, -1, 0,
                       1, 2, 1, 2, 1, 2, 1, 2, 3, 4, 3, 4, 3, 4, 3, 4,
                       5, 6, 5, 6, 5, 6, 5, 6, 7, -8, 7, -8, 7, -8, 7, -8};
  std::vector<int> y_3{-7, -6, -7, -6, -7, -6, -7, -6, -5, -4, -5, -4, -5, -4, -5, -4,
                       5, 0, 5, 0, 5, 0, 5, 0, -1, 0, -1, 0, -1, 0, -1, 0,
                       1, 2, 1, 2, 1, 2, 1, 2, 3, 4, 3, 4, 3, 4, 3, 4,
                       -9, -2, -9, -2, -9, -2, -9, -2, 7, -8, 7, -8, 7, -8, 7, -8};

  QuantizeLinearOp21BlockedTest_Int_Succeed<int8_t, float>({2, 4, 8}, 1, 2, x, y_scale, zero_point, y_2);
  QuantizeLinearOp21BlockedTest_Int_Succeed<int8_t, float>({2, 4, 8}, 1, 3, x, y_scale, zero_point, y_3);
  QuantizeLinearOp21BlockedTest_Int_Succeed<int16_t, float>({2, 4, 8}, 1, 2, x, y_scale, zero_point, y_2);
  QuantizeLinearOp21BlockedTest_Int_Succeed<int16_t, float>({2, 4, 8}, 1, 3, x, y_scale, zero_point, y_3);
  QuantizeLinearOp21BlockedTest_Int_Succeed<int8_t, MLFloat16>({2, 4, 8}, 1, 2, x, y_scale, zero_point, y_2);
  QuantizeLinearOp21BlockedTest_Int_Succeed<int8_t, MLFloat16>({2, 4, 8}, 1, 3, x, y_scale, zero_point, y_3);
  QuantizeLinearOp21BlockedTest_Int_Succeed<int16_t, MLFloat16>({2, 4, 8}, 1, 2, x, y_scale, zero_point, y_2);
  QuantizeLinearOp21BlockedTest_Int_Succeed<int16_t, MLFloat16>({2, 4, 8}, 1, 3, x, y_scale, zero_point, y_3);
}

TEST(QuantizeLinearOp21BlockedTest, SignedInt4_UseZeroPoint_MiddleAxis) {
  std::vector<int> zero_point{-6, -4, -6, -4, -6, -4, -6, -4, -3, -1, -3, -1, -3, -1, -3, -1,
                              0, 2, 0, 2, 0, 2, 0, 2, 4, 7, 4, 7, 4, 7, 4, 7};
  std::vector<float> y_scale{-2.0, -4.0, -2.0, -4.0, -2.0, -4.0, -2.0, -4.0, 3.5, 1.0, 3.5, 1.0, 3.5, 1.0, 3.5, 1.0,
                             2.0, 4.0, 2.0, 4.0, 2.0, 4.0, 2.0, 4.0, -3.5, -1.0, -3.5, -1.0, -3.5, -1.0, -3.5, -1.0};
  std::vector<float> x{2, 8, 2, 8, 2, 8, 2, 8, -2, 0, -2, 0, -2, 0, -2, 0,
                       0, -1, 0, -1, 0, -1, 0, -1, 7, 1, 7, 1, 7, 1, 7, 1,
                       2, 0, 2, 0, 2, 0, 2, 0, 6, 8, 6, 8, 6, 8, 6, 8,
                       -3.5, 1, -3.5, 1, -3.5, 1, -3.5, 1, -10.5, 15, -10.5, 15, -10.5, 15, -10.5, 15};
  std::vector<int> y_2{-7, -6, -7, -6, -7, -6, -7, -6, -5, -4, -5, -4, -5, -4, -5, -4,
                       -3, -2, -3, -2, -3, -2, -3, -2, -1, 0, -1, 0, -1, 0, -1, 0,
                       1, 2, 1, 2, 1, 2, 1, 2, 3, 4, 3, 4, 3, 4, 3, 4,
                       5, 6, 5, 6, 5, 6, 5, 6, 7, -8, 7, -8, 7, -8, 7, -8};
  std::vector<int> y_3{-7, -6, -7, -6, -7, -6, -7, -6, -5, -4, -5, -4, -5, -4, -5, -4,
                       -6, -4, -6, -4, -6, -4, -6, -4, -1, 0, -1, 0, -1, 0, -1, 0,
                       1, 2, 1, 2, 1, 2, 1, 2, 3, 4, 3, 4, 3, 4, 3, 4,
                       -2, 2, -2, 2, -2, 2, -2, 2, 7, -8, 7, -8, 7, -8, 7, -8};

  QuantizeLinearOp21BlockedTest_Int4_Succeed<Int4x2, float>({2, 4, 8}, 1, 2, x, y_scale, zero_point, y_2);
  QuantizeLinearOp21BlockedTest_Int4_Succeed<Int4x2, float>({2, 4, 8}, 1, 3, x, y_scale, zero_point, y_3);
  QuantizeLinearOp21BlockedTest_Int4_Succeed<Int4x2, MLFloat16>({2, 4, 8}, 1, 2, x, y_scale, zero_point, y_2);
  QuantizeLinearOp21BlockedTest_Int4_Succeed<Int4x2, MLFloat16>({2, 4, 8}, 1, 3, x, y_scale, zero_point, y_3);
}

TEST(QuantizeLinearOp21BlockedTest, SignedInt4_UseZeroPoint_MiddleAxis_Cuda) {
  std::vector<int> zero_point{-6, -4, -6, -4, -6, -4, -6, -4, -3, -1, -3, -1, -3, -1, -3, -1,
                              0, 2, 0, 2, 0, 2, 0, 2, 4, 7, 4, 7, 4, 7, 4, 7};
  std::vector<float> y_scale{-2.0, -4.0, -2.0, -4.0, -2.0, -4.0, -2.0, -4.0, 3.5, 1.0, 3.5, 1.0, 3.5, 1.0, 3.5, 1.0,
                             2.0, 4.0, 2.0, 4.0, 2.0, 4.0, 2.0, 4.0, -3.5, -1.0, -3.5, -1.0, -3.5, -1.0, -3.5, -1.0};
  std::vector<float> x{2, 8, 2, 8, 2, 8, 2, 8, -2, 0, -2, 0, -2, 0, -2, 0,
                       0, -1, 0, -1, 0, -1, 0, -1, 7, 1, 7, 1, 7, 1, 7, 1,
                       2, 0, 2, 0, 2, 0, 2, 0, 6, 8, 6, 8, 6, 8, 6, 8,
                       -3.5, 1, -3.5, 1, -3.5, 1, -3.5, 1, -10.5, 15, -10.5, 15, -10.5, 15, -10.5, 15};
  std::vector<int> y_2{-7, -6, -7, -6, -7, -6, -7, -6, -5, -4, -5, -4, -5, -4, -5, -4,
                       -3, -2, -3, -2, -3, -2, -3, -2, -1, 0, -1, 0, -1, 0, -1, 0,
                       1, 2, 1, 2, 1, 2, 1, 2, 3, 4, 3, 4, 3, 4, 3, 4,
                       5, 6, 5, 6, 5, 6, 5, 6, 7, -8, 7, -8, 7, -8, 7, -8};
  std::vector<int> y_3{-7, -6, -7, -6, -7, -6, -7, -6, -5, -4, -5, -4, -5, -4, -5, -4,
                       -6, -4, -6, -4, -6, -4, -6, -4, -1, 0, -1, 0, -1, 0, -1, 0,
                       1, 2, 1, 2, 1, 2, 1, 2, 3, 4, 3, 4, 3, 4, 3, 4,
                       -2, 2, -2, 2, -2, 2, -2, 2, 7, -8, 7, -8, 7, -8, 7, -8};

  QuantizeLinearOp21BlockedTest_Int4_Succeed<Int4x2, float>({2, 4, 8}, 1, 2, x, y_scale, zero_point, y_2,
                                                            DefaultCudaExecutionProvider());
  QuantizeLinearOp21BlockedTest_Int4_Succeed<Int4x2, float>({2, 4, 8}, 1, 3, x, y_scale, zero_point, y_3,
                                                            DefaultCudaExecutionProvider());
  QuantizeLinearOp21BlockedTest_Int4_Succeed<Int4x2, MLFloat16>({2, 4, 8}, 1, 2, x, y_scale, zero_point, y_2,
                                                                DefaultCudaExecutionProvider());
  QuantizeLinearOp21BlockedTest_Int4_Succeed<Int4x2, MLFloat16>({2, 4, 8}, 1, 3, x, y_scale, zero_point, y_3,
                                                                DefaultCudaExecutionProvider());
}

TEST(QuantizeLinearOp21BlockedTest, SignedInt_UseZeroPoint_MiddleAxis) {
  std::vector<int> zero_point{-6, -4, -6, -4, -6, -4, -6, -4, -3, -1, -3, -1, -3, -1, -3, -1,
                              0, 2, 0, 2, 0, 2, 0, 2, 4, 7, 4, 7, 4, 7, 4, 7};
  std::vector<float> y_scale{-2.0, -4.0, -2.0, -4.0, -2.0, -4.0, -2.0, -4.0, 3.5, 1.0, 3.5, 1.0, 3.5, 1.0, 3.5, 1.0,
                             2.0, 4.0, 2.0, 4.0, 2.0, 4.0, 2.0, 4.0, -3.5, -1.0, -3.5, -1.0, -3.5, -1.0, -3.5, -1.0};
  std::vector<float> x{2, 8, 2, 8, 2, 8, 2, 8, -2, 0, -2, 0, -2, 0, -2, 0,
                       0, -1, 0, -1, 0, -1, 0, -1, 7, 1, 7, 1, 7, 1, 7, 1,
                       2, 0, 2, 0, 2, 0, 2, 0, 6, 8, 6, 8, 6, 8, 6, 8,
                       -3.5, 1, -3.5, 1, -3.5, 1, -3.5, 1, -10.5, 15, -10.5, 15, -10.5, 15, -10.5, 15};
  std::vector<int> y_2{-7, -6, -7, -6, -7, -6, -7, -6, -5, -4, -5, -4, -5, -4, -5, -4,
                       -3, -2, -3, -2, -3, -2, -3, -2, -1, 0, -1, 0, -1, 0, -1, 0,
                       1, 2, 1, 2, 1, 2, 1, 2, 3, 4, 3, 4, 3, 4, 3, 4,
                       5, 6, 5, 6, 5, 6, 5, 6, 7, -8, 7, -8, 7, -8, 7, -8};
  std::vector<int> y_3{-7, -6, -7, -6, -7, -6, -7, -6, -5, -4, -5, -4, -5, -4, -5, -4,
                       -6, -4, -6, -4, -6, -4, -6, -4, -1, 0, -1, 0, -1, 0, -1, 0,
                       1, 2, 1, 2, 1, 2, 1, 2, 3, 4, 3, 4, 3, 4, 3, 4,
                       -2, 2, -2, 2, -2, 2, -2, 2, 7, -8, 7, -8, 7, -8, 7, -8};

  QuantizeLinearOp21BlockedTest_Int_Succeed<int8_t, float>({2, 4, 8}, 1, 2, x, y_scale, zero_point, y_2);
  QuantizeLinearOp21BlockedTest_Int_Succeed<int8_t, float>({2, 4, 8}, 1, 3, x, y_scale, zero_point, y_3);
  QuantizeLinearOp21BlockedTest_Int_Succeed<int16_t, float>({2, 4, 8}, 1, 2, x, y_scale, zero_point, y_2);
  QuantizeLinearOp21BlockedTest_Int_Succeed<int16_t, float>({2, 4, 8}, 1, 3, x, y_scale, zero_point, y_3);
  QuantizeLinearOp21BlockedTest_Int_Succeed<int8_t, MLFloat16>({2, 4, 8}, 1, 2, x, y_scale, zero_point, y_2);
  QuantizeLinearOp21BlockedTest_Int_Succeed<int8_t, MLFloat16>({2, 4, 8}, 1, 3, x, y_scale, zero_point, y_3);
  QuantizeLinearOp21BlockedTest_Int_Succeed<int16_t, MLFloat16>({2, 4, 8}, 1, 2, x, y_scale, zero_point, y_2);
  QuantizeLinearOp21BlockedTest_Int_Succeed<int16_t, MLFloat16>({2, 4, 8}, 1, 3, x, y_scale, zero_point, y_3);
}

TEST(QuantizeLinearOp21BlockedTest, SignedInt4_NoZeroPoint_LastAxis) {
  std::vector<int> zero_point{0, 0, 0, 0, 0, 0, 0, 0,
                              0, 0, 0, 0, 0, 0, 0, 0};
  std::vector<float> y_scale{-2.0, -4.0, -2.0, -4.0, 3.5, 1.0, 3.5, 1.0,
                             2.0, 4.0, 2.0, 4.0, -3.5, -1.0, -3.5, -1.0};
  std::vector<float> x{14, 12, 14, 12, 20, 16, 20, 16, 14, 12, 14, 12, 20, 16, 20, 16,
                       -10.5, -7, -10.5, -7, -1, 0, -1, 0, -10.5, -7, -10.5, -7, -1, 0, -1, 0,
                       2, 4, 2, 4, 12, 16, 12, 16, 2, 4, 2, 4, 12, 16, 12, 16,
                       -17.5, -21, -17.5, -21, -7, 8, -7, 8, -17.5, -21, -17.5, -21, -7, 8, -7, 8};
  std::vector<int> y_2{-7, -6, -7, -6, -5, -4, -5, -4, -7, -6, -7, -6, -5, -4, -5, -4,
                       -3, -2, -3, -2, -1, 0, -1, 0, -3, -2, -3, -2, -1, 0, -1, 0,
                       1, 2, 1, 2, 3, 4, 3, 4, 1, 2, 1, 2, 3, 4, 3, 4,
                       5, 6, 5, 6, 7, -8, 7, -8, 5, 6, 5, 6, 7, -8, 7, -8};
  std::vector<int> y_3{-7, -6, -7, -6, -8, -4, -5, -4, -7, -6, -7, -6, -8, -4, -5, -4,
                       -3, -2, -3, -2, 0, 0, -1, 0, -3, -2, -3, -2, 0, 0, -1, 0,
                       1, 2, 1, 2, 6, 4, 3, 4, 1, 2, 1, 2, 6, 4, 3, 4,
                       5, 6, 5, 6, 2, -8, 7, -8, 5, 6, 5, 6, 2, -8, 7, -8};

  QuantizeLinearOp21BlockedTest_Int4_Succeed<Int4x2, float>({2, 4, 8}, 2, 4, x, y_scale, zero_point, y_2);
  QuantizeLinearOp21BlockedTest_Int4_Succeed<Int4x2, float>({2, 4, 8}, 2, 5, x, y_scale, zero_point, y_3);
  QuantizeLinearOp21BlockedTest_Int4_Succeed<Int4x2, MLFloat16>({2, 4, 8}, 2, 4, x, y_scale, zero_point, y_2);
  QuantizeLinearOp21BlockedTest_Int4_Succeed<Int4x2, MLFloat16>({2, 4, 8}, 2, 5, x, y_scale, zero_point, y_3);
}

TEST(QuantizeLinearOp21BlockedTest, SignedInt4_NoZeroPoint_LastAxis_Cuda) {
  std::vector<int> zero_point{0, 0, 0, 0, 0, 0, 0, 0,
                              0, 0, 0, 0, 0, 0, 0, 0};
  std::vector<float> y_scale{-2.0, -4.0, -2.0, -4.0, 3.5, 1.0, 3.5, 1.0,
                             2.0, 4.0, 2.0, 4.0, -3.5, -1.0, -3.5, -1.0};
  std::vector<float> x{14, 12, 14, 12, 20, 16, 20, 16, 14, 12, 14, 12, 20, 16, 20, 16,
                       -10.5, -7, -10.5, -7, -1, 0, -1, 0, -10.5, -7, -10.5, -7, -1, 0, -1, 0,
                       2, 4, 2, 4, 12, 16, 12, 16, 2, 4, 2, 4, 12, 16, 12, 16,
                       -17.5, -21, -17.5, -21, -7, 8, -7, 8, -17.5, -21, -17.5, -21, -7, 8, -7, 8};
  std::vector<int> y_2{-7, -6, -7, -6, -5, -4, -5, -4, -7, -6, -7, -6, -5, -4, -5, -4,
                       -3, -2, -3, -2, -1, 0, -1, 0, -3, -2, -3, -2, -1, 0, -1, 0,
                       1, 2, 1, 2, 3, 4, 3, 4, 1, 2, 1, 2, 3, 4, 3, 4,
                       5, 6, 5, 6, 7, -8, 7, -8, 5, 6, 5, 6, 7, -8, 7, -8};
  std::vector<int> y_3{-7, -6, -7, -6, -8, -4, -5, -4, -7, -6, -7, -6, -8, -4, -5, -4,
                       -3, -2, -3, -2, 0, 0, -1, 0, -3, -2, -3, -2, 0, 0, -1, 0,
                       1, 2, 1, 2, 6, 4, 3, 4, 1, 2, 1, 2, 6, 4, 3, 4,
                       5, 6, 5, 6, 2, -8, 7, -8, 5, 6, 5, 6, 2, -8, 7, -8};

  QuantizeLinearOp21BlockedTest_Int4_Succeed<Int4x2, float>({2, 4, 8}, 2, 4, x, y_scale, zero_point, y_2,
                                                            DefaultCudaExecutionProvider());
  QuantizeLinearOp21BlockedTest_Int4_Succeed<Int4x2, float>({2, 4, 8}, 2, 5, x, y_scale, zero_point, y_3,
                                                            DefaultCudaExecutionProvider());
  QuantizeLinearOp21BlockedTest_Int4_Succeed<Int4x2, MLFloat16>({2, 4, 8}, 2, 4, x, y_scale, zero_point, y_2,
                                                                DefaultCudaExecutionProvider());
  QuantizeLinearOp21BlockedTest_Int4_Succeed<Int4x2, MLFloat16>({2, 4, 8}, 2, 5, x, y_scale, zero_point, y_3,
                                                                DefaultCudaExecutionProvider());
}

TEST(QuantizeLinearOp21BlockedTest, SignedInt_NoZeroPoint_LastAxis) {
  std::vector<int> zero_point{0, 0, 0, 0, 0, 0, 0, 0,
                              0, 0, 0, 0, 0, 0, 0, 0};
  std::vector<float> y_scale{-2.0, -4.0, -2.0, -4.0, 3.5, 1.0, 3.5, 1.0,
                             2.0, 4.0, 2.0, 4.0, -3.5, -1.0, -3.5, -1.0};
  std::vector<float> x{14, 12, 14, 12, 20, 16, 20, 16, 14, 12, 14, 12, 20, 16, 20, 16,
                       -10.5, -7, -10.5, -7, -1, 0, -1, 0, -10.5, -7, -10.5, -7, -1, 0, -1, 0,
                       2, 4, 2, 4, 12, 16, 12, 16, 2, 4, 2, 4, 12, 16, 12, 16,
                       -17.5, -21, -17.5, -21, -7, 8, -7, 8, -17.5, -21, -17.5, -21, -7, 8, -7, 8};
  std::vector<int> y_2{-7, -6, -7, -6, -5, -4, -5, -4, -7, -6, -7, -6, -5, -4, -5, -4,
                       -3, -2, -3, -2, -1, 0, -1, 0, -3, -2, -3, -2, -1, 0, -1, 0,
                       1, 2, 1, 2, 3, 4, 3, 4, 1, 2, 1, 2, 3, 4, 3, 4,
                       5, 6, 5, 6, 7, -8, 7, -8, 5, 6, 5, 6, 7, -8, 7, -8};
  std::vector<int> y_3{-7, -6, -7, -6, -10, -4, -5, -4, -7, -6, -7, -6, -10, -4, -5, -4,
                       -3, -2, -3, -2, 0, 0, -1, 0, -3, -2, -3, -2, 0, 0, -1, 0,
                       1, 2, 1, 2, 6, 4, 3, 4, 1, 2, 1, 2, 6, 4, 3, 4,
                       5, 6, 5, 6, 2, -8, 7, -8, 5, 6, 5, 6, 2, -8, 7, -8};

  QuantizeLinearOp21BlockedTest_Int_Succeed<int8_t, float>({2, 4, 8}, 2, 4, x, y_scale, zero_point, y_2);
  QuantizeLinearOp21BlockedTest_Int_Succeed<int8_t, float>({2, 4, 8}, 2, 5, x, y_scale, zero_point, y_3);
  QuantizeLinearOp21BlockedTest_Int_Succeed<int16_t, float>({2, 4, 8}, 2, 4, x, y_scale, zero_point, y_2);
  QuantizeLinearOp21BlockedTest_Int_Succeed<int16_t, float>({2, 4, 8}, 2, 5, x, y_scale, zero_point, y_3);
  QuantizeLinearOp21BlockedTest_Int_Succeed<int8_t, MLFloat16>({2, 4, 8}, 2, 4, x, y_scale, zero_point, y_2);
  QuantizeLinearOp21BlockedTest_Int_Succeed<int8_t, MLFloat16>({2, 4, 8}, 2, 5, x, y_scale, zero_point, y_3);
  QuantizeLinearOp21BlockedTest_Int_Succeed<int16_t, MLFloat16>({2, 4, 8}, 2, 4, x, y_scale, zero_point, y_2);
  QuantizeLinearOp21BlockedTest_Int_Succeed<int16_t, MLFloat16>({2, 4, 8}, 2, 5, x, y_scale, zero_point, y_3);
}

TEST(QuantizeLinearOp21BlockedTest, SignedInt4_UseZeroPoint_LastAxis) {
  std::vector<float> y_scale{-2.0, -4.0, -2.0, -4.0, 3.5, 1.0, 3.5, 1.0,
                             2.0, 4.0, 2.0, 4.0, -3.5, -1.0, -3.5, -1.0};
  std::vector<int> zero_point{-6, -4, -6, -4, -3, -1, -3, -1,
                              0, 2, 0, 2, 4, 7, 4, 7};
  std::vector<float> x{2, 0, 2, 0, 4, 0, 4, 0, 2, 0, 2, 0, 4, 0, 4, 0,
                       0, 3.5, 0, 3.5, 0, 1, 0, 1, 0, 3.5, 0, 3.5, 0, 1, 0, 1,
                       2, 4, 2, 4, 4, 8, 4, 8, 2, 4, 2, 4, 4, 8, 4, 8,
                       -3.5, -7, -3.5, -7, 0, 15, 0, 15, -3.5, -7, -3.5, -7, 0, 15, 0, 15};
  std::vector<int> y_2{-7, -6, -7, -6, -5, -4, -5, -4, -7, -6, -7, -6, -5, -4, -5, -4,
                       -3, -2, -3, -2, -1, 0, -1, 0, -3, -2, -3, -2, -1, 0, -1, 0,
                       1, 2, 1, 2, 3, 4, 3, 4, 1, 2, 1, 2, 3, 4, 3, 4,
                       5, 6, 5, 6, 7, -8, 7, -8, 5, 6, 5, 6, 7, -8, 7, -8};
  std::vector<int> y_3{-7, -6, -7, -6, -8, -4, -5, -4, -7, -6, -7, -6, -8, -4, -5, -4,
                       -3, -2, -3, -2, -3, 0, -1, 0, -3, -2, -3, -2, -3, 0, -1, 0,
                       1, 2, 1, 2, 2, 4, 3, 4, 1, 2, 1, 2, 2, 4, 3, 4,
                       5, 6, 5, 6, 4, -8, 7, -8, 5, 6, 5, 6, 4, -8, 7, -8};

  QuantizeLinearOp21BlockedTest_Int4_Succeed<Int4x2, float>({2, 4, 8}, 2, 4, x, y_scale, zero_point, y_2);
  QuantizeLinearOp21BlockedTest_Int4_Succeed<Int4x2, float>({2, 4, 8}, 2, 5, x, y_scale, zero_point, y_3);
  QuantizeLinearOp21BlockedTest_Int4_Succeed<Int4x2, MLFloat16>({2, 4, 8}, 2, 4, x, y_scale, zero_point, y_2);
  QuantizeLinearOp21BlockedTest_Int4_Succeed<Int4x2, MLFloat16>({2, 4, 8}, 2, 5, x, y_scale, zero_point, y_3);
}

TEST(QuantizeLinearOp21BlockedTest, SignedInt4_UseZeroPoint_LastAxis_Cuda) {
  std::vector<float> y_scale{-2.0, -4.0, -2.0, -4.0, 3.5, 1.0, 3.5, 1.0,
                             2.0, 4.0, 2.0, 4.0, -3.5, -1.0, -3.5, -1.0};
  std::vector<int> zero_point{-6, -4, -6, -4, -3, -1, -3, -1,
                              0, 2, 0, 2, 4, 7, 4, 7};
  std::vector<float> x{2, 0, 2, 0, 4, 0, 4, 0, 2, 0, 2, 0, 4, 0, 4, 0,
                       0, 3.5, 0, 3.5, 0, 1, 0, 1, 0, 3.5, 0, 3.5, 0, 1, 0, 1,
                       2, 4, 2, 4, 4, 8, 4, 8, 2, 4, 2, 4, 4, 8, 4, 8,
                       -3.5, -7, -3.5, -7, 0, 15, 0, 15, -3.5, -7, -3.5, -7, 0, 15, 0, 15};
  std::vector<int> y_2{-7, -6, -7, -6, -5, -4, -5, -4, -7, -6, -7, -6, -5, -4, -5, -4,
                       -3, -2, -3, -2, -1, 0, -1, 0, -3, -2, -3, -2, -1, 0, -1, 0,
                       1, 2, 1, 2, 3, 4, 3, 4, 1, 2, 1, 2, 3, 4, 3, 4,
                       5, 6, 5, 6, 7, -8, 7, -8, 5, 6, 5, 6, 7, -8, 7, -8};
  std::vector<int> y_3{-7, -6, -7, -6, -8, -4, -5, -4, -7, -6, -7, -6, -8, -4, -5, -4,
                       -3, -2, -3, -2, -3, 0, -1, 0, -3, -2, -3, -2, -3, 0, -1, 0,
                       1, 2, 1, 2, 2, 4, 3, 4, 1, 2, 1, 2, 2, 4, 3, 4,
                       5, 6, 5, 6, 4, -8, 7, -8, 5, 6, 5, 6, 4, -8, 7, -8};

  QuantizeLinearOp21BlockedTest_Int4_Succeed<Int4x2, float>({2, 4, 8}, 2, 4, x, y_scale, zero_point, y_2,
                                                            DefaultCudaExecutionProvider());
  QuantizeLinearOp21BlockedTest_Int4_Succeed<Int4x2, float>({2, 4, 8}, 2, 5, x, y_scale, zero_point, y_3,
                                                            DefaultCudaExecutionProvider());
  QuantizeLinearOp21BlockedTest_Int4_Succeed<Int4x2, MLFloat16>({2, 4, 8}, 2, 4, x, y_scale, zero_point, y_2,
                                                                DefaultCudaExecutionProvider());
  QuantizeLinearOp21BlockedTest_Int4_Succeed<Int4x2, MLFloat16>({2, 4, 8}, 2, 5, x, y_scale, zero_point, y_3,
                                                                DefaultCudaExecutionProvider());
}

TEST(QuantizeLinearOp21BlockedTest, SignedInt_UseZeroPoint_LastAxis) {
  std::vector<float> y_scale{-2.0, -4.0, -2.0, -4.0, 3.5, 1.0, 3.5, 1.0,
                             2.0, 4.0, 2.0, 4.0, -3.5, -1.0, -3.5, -1.0};
  std::vector<int> zero_point{-6, -4, -6, -4, -3, -1, -3, -1,
                              0, 2, 0, 2, 4, 7, 4, 7};
  std::vector<float> x{2, 0, 2, 0, 4, 0, 4, 0, 2, 0, 2, 0, 4, 0, 4, 0,
                       0, 3.5, 0, 3.5, 0, 1, 0, 1, 0, 3.5, 0, 3.5, 0, 1, 0, 1,
                       2, 4, 2, 4, 4, 8, 4, 8, 2, 4, 2, 4, 4, 8, 4, 8,
                       -3.5, -7, -3.5, -7, 0, 15, 0, 15, -3.5, -7, -3.5, -7, 0, 15, 0, 15};
  std::vector<int> y_2{-7, -6, -7, -6, -5, -4, -5, -4, -7, -6, -7, -6, -5, -4, -5, -4,
                       -3, -2, -3, -2, -1, 0, -1, 0, -3, -2, -3, -2, -1, 0, -1, 0,
                       1, 2, 1, 2, 3, 4, 3, 4, 1, 2, 1, 2, 3, 4, 3, 4,
                       5, 6, 5, 6, 7, -8, 7, -8, 5, 6, 5, 6, 7, -8, 7, -8};
  std::vector<int> y_3{-7, -6, -7, -6, -8, -4, -5, -4, -7, -6, -7, -6, -8, -4, -5, -4,
                       -3, -2, -3, -2, -3, 0, -1, 0, -3, -2, -3, -2, -3, 0, -1, 0,
                       1, 2, 1, 2, 2, 4, 3, 4, 1, 2, 1, 2, 2, 4, 3, 4,
                       5, 6, 5, 6, 4, -8, 7, -8, 5, 6, 5, 6, 4, -8, 7, -8};

  QuantizeLinearOp21BlockedTest_Int_Succeed<int8_t, float>({2, 4, 8}, 2, 4, x, y_scale, zero_point, y_2);
  QuantizeLinearOp21BlockedTest_Int_Succeed<int8_t, float>({2, 4, 8}, 2, 5, x, y_scale, zero_point, y_3);
  QuantizeLinearOp21BlockedTest_Int_Succeed<int16_t, float>({2, 4, 8}, 2, 4, x, y_scale, zero_point, y_2);
  QuantizeLinearOp21BlockedTest_Int_Succeed<int16_t, float>({2, 4, 8}, 2, 5, x, y_scale, zero_point, y_3);
  QuantizeLinearOp21BlockedTest_Int_Succeed<int8_t, MLFloat16>({2, 4, 8}, 2, 4, x, y_scale, zero_point, y_2);
  QuantizeLinearOp21BlockedTest_Int_Succeed<int8_t, MLFloat16>({2, 4, 8}, 2, 5, x, y_scale, zero_point, y_3);
  QuantizeLinearOp21BlockedTest_Int_Succeed<int16_t, MLFloat16>({2, 4, 8}, 2, 4, x, y_scale, zero_point, y_2);
  QuantizeLinearOp21BlockedTest_Int_Succeed<int16_t, MLFloat16>({2, 4, 8}, 2, 5, x, y_scale, zero_point, y_3);
}

TEST(QuantizeLinearOp21BlockedTest, UnsignedInt4_NoZeroPoint_FirstAxis) {
  std::vector<float> y_scale{-2.0, -4.0, 3.5, 1.0, -2.0, -4.0, 3.5, 1.0, -2.0, -4.0, 3.5, 1.0, -2.0, -4.0, 3.5, 1.0,
                             2.0, 4.0, -3.5, -1.0, 2.0, 4.0, -3.5, -1.0, 2.0, 4.0, -3.5, -1.0, 2.0, 4.0, -3.5, -1.0};
  std::vector<int> zero_point{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                              0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  std::vector<float> x{0, -4, 7, 3, 0, -4, 7, 3, 0, -4, 7, 3, 0, -4, 7, 3,
                       -8, -20, 21, 7, -8, -20, 21, 7, -8, -20, 21, 7, -8, -20, 21, 7,
                       16, 36, -35, -11, 16, 36, -35, -11, 16, 36, -35, -11, 16, 36, -35, -11,
                       24, 52, -49, -15, 24, 52, -49, -15, 24, 52, -49, -15, 24, 52, -49, -15};
  std::vector<int> y_2{0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3,
                       4, 5, 6, 7, 4, 5, 6, 7, 4, 5, 6, 7, 4, 5, 6, 7,
                       8, 9, 10, 11, 8, 9, 10, 11, 8, 9, 10, 11, 8, 9, 10, 11,
                       12, 13, 14, 15, 12, 13, 14, 15, 12, 13, 14, 15, 12, 13, 14, 15};
  std::vector<int> y_3{0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3,
                       4, 5, 6, 7, 4, 5, 6, 7, 4, 5, 6, 7, 4, 5, 6, 7,
                       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                       12, 13, 14, 15, 12, 13, 14, 15, 12, 13, 14, 15, 12, 13, 14, 15};

  QuantizeLinearOp21BlockedTest_Int4_Succeed<UInt4x2, float>({4, 8, 2}, 0, 2, x, y_scale, zero_point, y_2);
  QuantizeLinearOp21BlockedTest_Int4_Succeed<UInt4x2, float>({4, 8, 2}, 0, 3, x, y_scale, zero_point, y_3);
  QuantizeLinearOp21BlockedTest_Int4_Succeed<UInt4x2, MLFloat16>({4, 8, 2}, 0, 2, x, y_scale, zero_point, y_2);
  QuantizeLinearOp21BlockedTest_Int4_Succeed<UInt4x2, MLFloat16>({4, 8, 2}, 0, 3, x, y_scale, zero_point, y_3);
}

TEST(QuantizeLinearOp21BlockedTest, UnsignedInt4_NoZeroPoint_FirstAxis_Cuda) {
  std::vector<float> y_scale{-2.0, -4.0, 3.5, 1.0, -2.0, -4.0, 3.5, 1.0, -2.0, -4.0, 3.5, 1.0, -2.0, -4.0, 3.5, 1.0,
                             2.0, 4.0, -3.5, -1.0, 2.0, 4.0, -3.5, -1.0, 2.0, 4.0, -3.5, -1.0, 2.0, 4.0, -3.5, -1.0};
  std::vector<int> zero_point{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                              0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  std::vector<float> x{0, -4, 7, 3, 0, -4, 7, 3, 0, -4, 7, 3, 0, -4, 7, 3,
                       -8, -20, 21, 7, -8, -20, 21, 7, -8, -20, 21, 7, -8, -20, 21, 7,
                       16, 36, -35, -11, 16, 36, -35, -11, 16, 36, -35, -11, 16, 36, -35, -11,
                       24, 52, -49, -15, 24, 52, -49, -15, 24, 52, -49, -15, 24, 52, -49, -15};
  std::vector<int> y_2{0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3,
                       4, 5, 6, 7, 4, 5, 6, 7, 4, 5, 6, 7, 4, 5, 6, 7,
                       8, 9, 10, 11, 8, 9, 10, 11, 8, 9, 10, 11, 8, 9, 10, 11,
                       12, 13, 14, 15, 12, 13, 14, 15, 12, 13, 14, 15, 12, 13, 14, 15};
  std::vector<int> y_3{0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3,
                       4, 5, 6, 7, 4, 5, 6, 7, 4, 5, 6, 7, 4, 5, 6, 7,
                       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                       12, 13, 14, 15, 12, 13, 14, 15, 12, 13, 14, 15, 12, 13, 14, 15};

  QuantizeLinearOp21BlockedTest_Int4_Succeed<UInt4x2, float>({4, 8, 2}, 0, 2, x, y_scale, zero_point, y_2,
                                                             DefaultCudaExecutionProvider());
  QuantizeLinearOp21BlockedTest_Int4_Succeed<UInt4x2, float>({4, 8, 2}, 0, 3, x, y_scale, zero_point, y_3,
                                                             DefaultCudaExecutionProvider());
  QuantizeLinearOp21BlockedTest_Int4_Succeed<UInt4x2, MLFloat16>({4, 8, 2}, 0, 2, x, y_scale, zero_point, y_2,
                                                                 DefaultCudaExecutionProvider());
  QuantizeLinearOp21BlockedTest_Int4_Succeed<UInt4x2, MLFloat16>({4, 8, 2}, 0, 3, x, y_scale, zero_point, y_3,
                                                                 DefaultCudaExecutionProvider());
}

TEST(QuantizeLinearOp21BlockedTest, UnsignedInt_NoZeroPoint_FirstAxis) {
  std::vector<float> y_scale{-2.0, -4.0, 3.5, 1.0, -2.0, -4.0, 3.5, 1.0, -2.0, -4.0, 3.5, 1.0, -2.0, -4.0, 3.5, 1.0,
                             2.0, 4.0, -3.5, -1.0, 2.0, 4.0, -3.5, -1.0, 2.0, 4.0, -3.5, -1.0, 2.0, 4.0, -3.5, -1.0};
  std::vector<int> zero_point{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                              0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  std::vector<float> x{0, -4, 7, 3, 0, -4, 7, 3, 0, -4, 7, 3, 0, -4, 7, 3,
                       -8, -20, 21, 7, -8, -20, 21, 7, -8, -20, 21, 7, -8, -20, 21, 7,
                       16, 36, -35, -11, 16, 36, -35, -11, 16, 36, -35, -11, 16, 36, -35, -11,
                       24, 52, -49, -15, 24, 52, -49, -15, 24, 52, -49, -15, 24, 52, -49, -15};
  std::vector<int> y_2{0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3,
                       4, 5, 6, 7, 4, 5, 6, 7, 4, 5, 6, 7, 4, 5, 6, 7,
                       8, 9, 10, 11, 8, 9, 10, 11, 8, 9, 10, 11, 8, 9, 10, 11,
                       12, 13, 14, 15, 12, 13, 14, 15, 12, 13, 14, 15, 12, 13, 14, 15};
  std::vector<int> y_3{0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3,
                       4, 5, 6, 7, 4, 5, 6, 7, 4, 5, 6, 7, 4, 5, 6, 7,
                       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                       12, 13, 14, 15, 12, 13, 14, 15, 12, 13, 14, 15, 12, 13, 14, 15};

  QuantizeLinearOp21BlockedTest_Int_Succeed<uint8_t, float>({4, 8, 2}, 0, 2, x, y_scale, zero_point, y_2);
  QuantizeLinearOp21BlockedTest_Int_Succeed<uint8_t, float>({4, 8, 2}, 0, 3, x, y_scale, zero_point, y_3);
  QuantizeLinearOp21BlockedTest_Int_Succeed<uint16_t, float>({4, 8, 2}, 0, 2, x, y_scale, zero_point, y_2);
  QuantizeLinearOp21BlockedTest_Int_Succeed<uint16_t, float>({4, 8, 2}, 0, 3, x, y_scale, zero_point, y_3);
  QuantizeLinearOp21BlockedTest_Int_Succeed<uint8_t, MLFloat16>({4, 8, 2}, 0, 2, x, y_scale, zero_point, y_2);
  QuantizeLinearOp21BlockedTest_Int_Succeed<uint8_t, MLFloat16>({4, 8, 2}, 0, 3, x, y_scale, zero_point, y_3);
  QuantizeLinearOp21BlockedTest_Int_Succeed<uint16_t, MLFloat16>({4, 8, 2}, 0, 2, x, y_scale, zero_point, y_2);
  QuantizeLinearOp21BlockedTest_Int_Succeed<uint16_t, MLFloat16>({4, 8, 2}, 0, 3, x, y_scale, zero_point, y_3);
}

TEST(QuantizeLinearOp21BlockedTest, UnsignedInt4_UseZeroPoint_FirstAxis) {
  std::vector<float> y_scale{-2.0, -4.0, 3.5, 1.0, -2.0, -4.0, 3.5, 1.0, -2.0, -4.0, 3.5, 1.0, -2.0, -4.0, 3.5, 1.0,
                             2.0, 4.0, -3.5, -1.0, 2.0, 4.0, -3.5, -1.0, 2.0, 4.0, -3.5, -1.0, 2.0, 4.0, -3.5, -1.0};
  std::vector<int> zero_point{2, 0, 1, 9, 2, 0, 1, 9, 2, 0, 1, 9, 2, 0, 1, 9,
                              13, 5, 11, 6, 13, 5, 11, 6, 13, 5, 11, 6, 13, 5, 11, 6};
  std::vector<float> x{4, -4, 3.5, -6, 4, -4, 3.5, -6, 4, -4, 3.5, -6, 4, -4, 3.5, -6,
                       -4, -20, 17.5, -2, -4, -20, 17.5, -2, -4, -20, 17.5, -2, -4, -20, 17.5, -2,
                       -10, 16, 3.5, -5, -10, 16, 3.5, -5, -10, 16, 3.5, -5, -10, 16, 3.5, -5,
                       -2, 32, -10.5, -9, -2, 32, -10.5, -9, -2, 32, -10.5, -9, -2, 32, -10.5, -9};
  std::vector<int> y_2{0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3,
                       4, 5, 6, 7, 4, 5, 6, 7, 4, 5, 6, 7, 4, 5, 6, 7,
                       8, 9, 10, 11, 8, 9, 10, 11, 8, 9, 10, 11, 8, 9, 10, 11,
                       12, 13, 14, 15, 12, 13, 14, 15, 12, 13, 14, 15, 12, 13, 14, 15};
  std::vector<int> y_3{0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3,
                       4, 5, 6, 7, 4, 5, 6, 7, 4, 5, 6, 7, 4, 5, 6, 7,
                       7, 0, 2, 4, 7, 0, 2, 4, 7, 0, 2, 4, 7, 0, 2, 4,
                       12, 13, 14, 15, 12, 13, 14, 15, 12, 13, 14, 15, 12, 13, 14, 15};

  QuantizeLinearOp21BlockedTest_Int4_Succeed<UInt4x2, float>({4, 8, 2}, 0, 2, x, y_scale, zero_point, y_2);
  QuantizeLinearOp21BlockedTest_Int4_Succeed<UInt4x2, float>({4, 8, 2}, 0, 3, x, y_scale, zero_point, y_3);
  QuantizeLinearOp21BlockedTest_Int4_Succeed<UInt4x2, MLFloat16>({4, 8, 2}, 0, 2, x, y_scale, zero_point, y_2);
  QuantizeLinearOp21BlockedTest_Int4_Succeed<UInt4x2, MLFloat16>({4, 8, 2}, 0, 3, x, y_scale, zero_point, y_3);
}

TEST(QuantizeLinearOp21BlockedTest, UnsignedInt4_UseZeroPoint_FirstAxis_Cuda) {
  std::vector<float> y_scale{-2.0, -4.0, 3.5, 1.0, -2.0, -4.0, 3.5, 1.0, -2.0, -4.0, 3.5, 1.0, -2.0, -4.0, 3.5, 1.0,
                             2.0, 4.0, -3.5, -1.0, 2.0, 4.0, -3.5, -1.0, 2.0, 4.0, -3.5, -1.0, 2.0, 4.0, -3.5, -1.0};
  std::vector<int> zero_point{2, 0, 1, 9, 2, 0, 1, 9, 2, 0, 1, 9, 2, 0, 1, 9,
                              13, 5, 11, 6, 13, 5, 11, 6, 13, 5, 11, 6, 13, 5, 11, 6};
  std::vector<float> x{4, -4, 3.5, -6, 4, -4, 3.5, -6, 4, -4, 3.5, -6, 4, -4, 3.5, -6,
                       -4, -20, 17.5, -2, -4, -20, 17.5, -2, -4, -20, 17.5, -2, -4, -20, 17.5, -2,
                       -10, 16, 3.5, -5, -10, 16, 3.5, -5, -10, 16, 3.5, -5, -10, 16, 3.5, -5,
                       -2, 32, -10.5, -9, -2, 32, -10.5, -9, -2, 32, -10.5, -9, -2, 32, -10.5, -9};
  std::vector<int> y_2{0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3,
                       4, 5, 6, 7, 4, 5, 6, 7, 4, 5, 6, 7, 4, 5, 6, 7,
                       8, 9, 10, 11, 8, 9, 10, 11, 8, 9, 10, 11, 8, 9, 10, 11,
                       12, 13, 14, 15, 12, 13, 14, 15, 12, 13, 14, 15, 12, 13, 14, 15};
  std::vector<int> y_3{0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3,
                       4, 5, 6, 7, 4, 5, 6, 7, 4, 5, 6, 7, 4, 5, 6, 7,
                       7, 0, 2, 4, 7, 0, 2, 4, 7, 0, 2, 4, 7, 0, 2, 4,
                       12, 13, 14, 15, 12, 13, 14, 15, 12, 13, 14, 15, 12, 13, 14, 15};

  QuantizeLinearOp21BlockedTest_Int4_Succeed<UInt4x2, float>({4, 8, 2}, 0, 2, x, y_scale, zero_point, y_2,
                                                             DefaultCudaExecutionProvider());
  QuantizeLinearOp21BlockedTest_Int4_Succeed<UInt4x2, float>({4, 8, 2}, 0, 3, x, y_scale, zero_point, y_3,
                                                             DefaultCudaExecutionProvider());
  QuantizeLinearOp21BlockedTest_Int4_Succeed<UInt4x2, MLFloat16>({4, 8, 2}, 0, 2, x, y_scale, zero_point, y_2,
                                                                 DefaultCudaExecutionProvider());
  QuantizeLinearOp21BlockedTest_Int4_Succeed<UInt4x2, MLFloat16>({4, 8, 2}, 0, 3, x, y_scale, zero_point, y_3,
                                                                 DefaultCudaExecutionProvider());
}

TEST(QuantizeLinearOp21BlockedTest, UnsignedInt_UseZeroPoint_FirstAxis) {
  std::vector<float> y_scale{-2.0, -4.0, 3.5, 1.0, -2.0, -4.0, 3.5, 1.0, -2.0, -4.0, 3.5, 1.0, -2.0, -4.0, 3.5, 1.0,
                             2.0, 4.0, -3.5, -1.0, 2.0, 4.0, -3.5, -1.0, 2.0, 4.0, -3.5, -1.0, 2.0, 4.0, -3.5, -1.0};
  std::vector<int> zero_point{2, 0, 1, 9, 2, 0, 1, 9, 2, 0, 1, 9, 2, 0, 1, 9,
                              13, 5, 11, 6, 13, 5, 11, 6, 13, 5, 11, 6, 13, 5, 11, 6};
  std::vector<float> x{4, -4, 3.5, -6, 4, -4, 3.5, -6, 4, -4, 3.5, -6, 4, -4, 3.5, -6,
                       -4, -20, 17.5, -2, -4, -20, 17.5, -2, -4, -20, 17.5, -2, -4, -20, 17.5, -2,
                       -10, 16, 3.5, -5, -10, 16, 3.5, -5, -10, 16, 3.5, -5, -10, 16, 3.5, -5,
                       -2, 32, -10.5, -9, -2, 32, -10.5, -9, -2, 32, -10.5, -9, -2, 32, -10.5, -9};
  std::vector<int> y_2{0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3,
                       4, 5, 6, 7, 4, 5, 6, 7, 4, 5, 6, 7, 4, 5, 6, 7,
                       8, 9, 10, 11, 8, 9, 10, 11, 8, 9, 10, 11, 8, 9, 10, 11,
                       12, 13, 14, 15, 12, 13, 14, 15, 12, 13, 14, 15, 12, 13, 14, 15};
  std::vector<int> y_3{0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3,
                       4, 5, 6, 7, 4, 5, 6, 7, 4, 5, 6, 7, 4, 5, 6, 7,
                       7, 0, 2, 4, 7, 0, 2, 4, 7, 0, 2, 4, 7, 0, 2, 4,
                       12, 13, 14, 15, 12, 13, 14, 15, 12, 13, 14, 15, 12, 13, 14, 15};

  QuantizeLinearOp21BlockedTest_Int_Succeed<uint8_t, float>({4, 8, 2}, 0, 2, x, y_scale, zero_point, y_2);
  QuantizeLinearOp21BlockedTest_Int_Succeed<uint8_t, float>({4, 8, 2}, 0, 3, x, y_scale, zero_point, y_3);
  QuantizeLinearOp21BlockedTest_Int_Succeed<uint16_t, float>({4, 8, 2}, 0, 2, x, y_scale, zero_point, y_2);
  QuantizeLinearOp21BlockedTest_Int_Succeed<uint16_t, float>({4, 8, 2}, 0, 3, x, y_scale, zero_point, y_3);
  QuantizeLinearOp21BlockedTest_Int_Succeed<uint8_t, MLFloat16>({4, 8, 2}, 0, 2, x, y_scale, zero_point, y_2);
  QuantizeLinearOp21BlockedTest_Int_Succeed<uint8_t, MLFloat16>({4, 8, 2}, 0, 3, x, y_scale, zero_point, y_3);
  QuantizeLinearOp21BlockedTest_Int_Succeed<uint16_t, MLFloat16>({4, 8, 2}, 0, 2, x, y_scale, zero_point, y_2);
  QuantizeLinearOp21BlockedTest_Int_Succeed<uint16_t, MLFloat16>({4, 8, 2}, 0, 3, x, y_scale, zero_point, y_3);
}

TEST(QuantizeLinearOp21BlockedTest, UnsignedInt4_NoZeroPoint_MiddleAxis) {
  std::vector<float> y_scale{-2.0, -4.0, 3.5, 1.0, 2.0, 4.0, -3.5, -1.0, -2.0, -4.0, 3.5, 1.0, 2.0, 4.0, -3.5, -1.0,
                             -2.0, -4.0, 3.5, 1.0, 2.0, 4.0, -3.5, -1.0, -2.0, -4.0, 3.5, 1.0, 2.0, 4.0, -3.5, -1.0};
  std::vector<int> zero_point{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                              0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  std::vector<float> x{0, -4, -4, -12, 14, 5, 21, 7, 16, 36, 20, 44, -42, -13, -49, -15,
                       0, -4, -4, -12, 14, 5, 21, 7, 16, 36, 20, 44, -42, -13, -49, -15,
                       0, -4, -4, -12, 14, 5, 21, 7, 16, 36, 20, 44, -42, -13, -49, -15,
                       0, -4, -4, -12, 14, 5, 21, 7, 16, 36, 20, 44, -42, -13, -49, -15};
  std::vector<int> y_2{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
                       0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
                       0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
                       0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
  std::vector<int> y_3{0, 1, 2, 3, 0, 0, 6, 7, 8, 9, 10, 11, 0, 0, 14, 15,
                       0, 1, 2, 3, 0, 0, 6, 7, 8, 9, 10, 11, 0, 0, 14, 15,
                       0, 1, 2, 3, 0, 0, 6, 7, 8, 9, 10, 11, 0, 0, 14, 15,
                       0, 1, 2, 3, 0, 0, 6, 7, 8, 9, 10, 11, 0, 0, 14, 15};

  QuantizeLinearOp21BlockedTest_Int4_Succeed<UInt4x2, float>({8, 4, 2}, 1, 2, x, y_scale, zero_point, y_2);
  QuantizeLinearOp21BlockedTest_Int4_Succeed<UInt4x2, float>({8, 4, 2}, 1, 3, x, y_scale, zero_point, y_3);
  QuantizeLinearOp21BlockedTest_Int4_Succeed<UInt4x2, MLFloat16>({8, 4, 2}, 1, 2, x, y_scale, zero_point, y_2);
  QuantizeLinearOp21BlockedTest_Int4_Succeed<UInt4x2, MLFloat16>({8, 4, 2}, 1, 3, x, y_scale, zero_point, y_3);
}

TEST(QuantizeLinearOp21BlockedTest, UnsignedInt4_NoZeroPoint_MiddleAxis_Cuda) {
  std::vector<float> y_scale{-2.0, -4.0, 3.5, 1.0, 2.0, 4.0, -3.5, -1.0, -2.0, -4.0, 3.5, 1.0, 2.0, 4.0, -3.5, -1.0,
                             -2.0, -4.0, 3.5, 1.0, 2.0, 4.0, -3.5, -1.0, -2.0, -4.0, 3.5, 1.0, 2.0, 4.0, -3.5, -1.0};
  std::vector<int> zero_point{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                              0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  std::vector<float> x{0, -4, -4, -12, 14, 5, 21, 7, 16, 36, 20, 44, -42, -13, -49, -15,
                       0, -4, -4, -12, 14, 5, 21, 7, 16, 36, 20, 44, -42, -13, -49, -15,
                       0, -4, -4, -12, 14, 5, 21, 7, 16, 36, 20, 44, -42, -13, -49, -15,
                       0, -4, -4, -12, 14, 5, 21, 7, 16, 36, 20, 44, -42, -13, -49, -15};
  std::vector<int> y_2{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
                       0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
                       0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
                       0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
  std::vector<int> y_3{0, 1, 2, 3, 0, 0, 6, 7, 8, 9, 10, 11, 0, 0, 14, 15,
                       0, 1, 2, 3, 0, 0, 6, 7, 8, 9, 10, 11, 0, 0, 14, 15,
                       0, 1, 2, 3, 0, 0, 6, 7, 8, 9, 10, 11, 0, 0, 14, 15,
                       0, 1, 2, 3, 0, 0, 6, 7, 8, 9, 10, 11, 0, 0, 14, 15};

  QuantizeLinearOp21BlockedTest_Int4_Succeed<UInt4x2, float>({8, 4, 2}, 1, 2, x, y_scale, zero_point, y_2,
                                                             DefaultCudaExecutionProvider());
  QuantizeLinearOp21BlockedTest_Int4_Succeed<UInt4x2, float>({8, 4, 2}, 1, 3, x, y_scale, zero_point, y_3,
                                                             DefaultCudaExecutionProvider());
  QuantizeLinearOp21BlockedTest_Int4_Succeed<UInt4x2, MLFloat16>({8, 4, 2}, 1, 2, x, y_scale, zero_point, y_2,
                                                                 DefaultCudaExecutionProvider());
  QuantizeLinearOp21BlockedTest_Int4_Succeed<UInt4x2, MLFloat16>({8, 4, 2}, 1, 3, x, y_scale, zero_point, y_3,
                                                                 DefaultCudaExecutionProvider());
}

TEST(QuantizeLinearOp21BlockedTest, UnsignedInt_NoZeroPoint_MiddleAxis) {
  std::vector<float> y_scale{-2.0, -4.0, 3.5, 1.0, 2.0, 4.0, -3.5, -1.0, -2.0, -4.0, 3.5, 1.0, 2.0, 4.0, -3.5, -1.0,
                             -2.0, -4.0, 3.5, 1.0, 2.0, 4.0, -3.5, -1.0, -2.0, -4.0, 3.5, 1.0, 2.0, 4.0, -3.5, -1.0};
  std::vector<int> zero_point{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                              0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  std::vector<float> x{0, -4, -4, -12, 14, 5, 21, 7, 16, 36, 20, 44, -42, -13, -49, -15,
                       0, -4, -4, -12, 14, 5, 21, 7, 16, 36, 20, 44, -42, -13, -49, -15,
                       0, -4, -4, -12, 14, 5, 21, 7, 16, 36, 20, 44, -42, -13, -49, -15,
                       0, -4, -4, -12, 14, 5, 21, 7, 16, 36, 20, 44, -42, -13, -49, -15};
  std::vector<int> y_2{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
                       0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
                       0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
                       0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
  std::vector<int> y_3{0, 1, 2, 3, 0, 0, 6, 7, 8, 9, 10, 11, 0, 0, 14, 15,
                       0, 1, 2, 3, 0, 0, 6, 7, 8, 9, 10, 11, 0, 0, 14, 15,
                       0, 1, 2, 3, 0, 0, 6, 7, 8, 9, 10, 11, 0, 0, 14, 15,
                       0, 1, 2, 3, 0, 0, 6, 7, 8, 9, 10, 11, 0, 0, 14, 15};

  QuantizeLinearOp21BlockedTest_Int_Succeed<uint8_t, float>({8, 4, 2}, 1, 2, x, y_scale, zero_point, y_2);
  QuantizeLinearOp21BlockedTest_Int_Succeed<uint8_t, float>({8, 4, 2}, 1, 3, x, y_scale, zero_point, y_3);
  QuantizeLinearOp21BlockedTest_Int_Succeed<uint16_t, float>({8, 4, 2}, 1, 2, x, y_scale, zero_point, y_2);
  QuantizeLinearOp21BlockedTest_Int_Succeed<uint16_t, float>({8, 4, 2}, 1, 3, x, y_scale, zero_point, y_3);
  QuantizeLinearOp21BlockedTest_Int_Succeed<uint8_t, MLFloat16>({8, 4, 2}, 1, 2, x, y_scale, zero_point, y_2);
  QuantizeLinearOp21BlockedTest_Int_Succeed<uint8_t, MLFloat16>({8, 4, 2}, 1, 3, x, y_scale, zero_point, y_3);
  QuantizeLinearOp21BlockedTest_Int_Succeed<uint16_t, MLFloat16>({8, 4, 2}, 1, 2, x, y_scale, zero_point, y_2);
  QuantizeLinearOp21BlockedTest_Int_Succeed<uint16_t, MLFloat16>({8, 4, 2}, 1, 3, x, y_scale, zero_point, y_3);
}

TEST(QuantizeLinearOp21BlockedTest, UnsignedInt4_UseZeroPoint_MiddleAxis) {
  std::vector<float> y_scale{-2.0, -4.0, 3.5, 1.0, 2.0, 4.0, -3.5, -1.0, -2.0, -4.0, 3.5, 1.0, 2.0, 4.0, -3.5, -1.0,
                             -2.0, -4.0, 3.5, 1.0, 2.0, 4.0, -3.5, -1.0, -2.0, -4.0, 3.5, 1.0, 2.0, 4.0, -3.5, -1.0};
  std::vector<int> zero_point{2, 0, 1, 9, 13, 5, 11, 6, 2, 0, 1, 9, 13, 5, 11, 6,
                              2, 0, 1, 9, 13, 5, 11, 6, 2, 0, 1, 9, 13, 5, 11, 6};
  std::vector<float> x{4, -4, 0, -12, 10.5, -4, 17.5, -2, -10, 16, -6, 24, -3.5, -7, -10.5, -9,
                       4, -4, 0, -12, 10.5, -4, 17.5, -2, -10, 16, -6, 24, -3.5, -7, -10.5, -9,
                       4, -4, 0, -12, 10.5, -4, 17.5, -2, -10, 16, -6, 24, -3.5, -7, -10.5, -9,
                       4, -4, 0, -12, 10.5, -4, 17.5, -2, -10, 16, -6, 24, -3.5, -7, -10.5, -9};
  std::vector<int> y_2{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
                       0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
                       0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
                       0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
  std::vector<int> y_3{0, 1, 2, 3, 0, 1, 6, 7, 8, 9, 10, 11, 11, 3, 14, 15,
                       0, 1, 2, 3, 0, 1, 6, 7, 8, 9, 10, 11, 11, 3, 14, 15,
                       0, 1, 2, 3, 0, 1, 6, 7, 8, 9, 10, 11, 11, 3, 14, 15,
                       0, 1, 2, 3, 0, 1, 6, 7, 8, 9, 10, 11, 11, 3, 14, 15};

  QuantizeLinearOp21BlockedTest_Int4_Succeed<UInt4x2, float>({8, 4, 2}, 1, 2, x, y_scale, zero_point, y_2);
  QuantizeLinearOp21BlockedTest_Int4_Succeed<UInt4x2, float>({8, 4, 2}, 1, 3, x, y_scale, zero_point, y_3);
  QuantizeLinearOp21BlockedTest_Int4_Succeed<UInt4x2, MLFloat16>({8, 4, 2}, 1, 2, x, y_scale, zero_point, y_2);
  QuantizeLinearOp21BlockedTest_Int4_Succeed<UInt4x2, MLFloat16>({8, 4, 2}, 1, 3, x, y_scale, zero_point, y_3);
}

TEST(QuantizeLinearOp21BlockedTest, UnsignedInt4_UseZeroPoint_MiddleAxis_Cuda) {
  std::vector<float> y_scale{-2.0, -4.0, 3.5, 1.0, 2.0, 4.0, -3.5, -1.0, -2.0, -4.0, 3.5, 1.0, 2.0, 4.0, -3.5, -1.0,
                             -2.0, -4.0, 3.5, 1.0, 2.0, 4.0, -3.5, -1.0, -2.0, -4.0, 3.5, 1.0, 2.0, 4.0, -3.5, -1.0};
  std::vector<int> zero_point{2, 0, 1, 9, 13, 5, 11, 6, 2, 0, 1, 9, 13, 5, 11, 6,
                              2, 0, 1, 9, 13, 5, 11, 6, 2, 0, 1, 9, 13, 5, 11, 6};
  std::vector<float> x{4, -4, 0, -12, 10.5, -4, 17.5, -2, -10, 16, -6, 24, -3.5, -7, -10.5, -9,
                       4, -4, 0, -12, 10.5, -4, 17.5, -2, -10, 16, -6, 24, -3.5, -7, -10.5, -9,
                       4, -4, 0, -12, 10.5, -4, 17.5, -2, -10, 16, -6, 24, -3.5, -7, -10.5, -9,
                       4, -4, 0, -12, 10.5, -4, 17.5, -2, -10, 16, -6, 24, -3.5, -7, -10.5, -9};
  std::vector<int> y_2{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
                       0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
                       0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
                       0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
  std::vector<int> y_3{0, 1, 2, 3, 0, 1, 6, 7, 8, 9, 10, 11, 11, 3, 14, 15,
                       0, 1, 2, 3, 0, 1, 6, 7, 8, 9, 10, 11, 11, 3, 14, 15,
                       0, 1, 2, 3, 0, 1, 6, 7, 8, 9, 10, 11, 11, 3, 14, 15,
                       0, 1, 2, 3, 0, 1, 6, 7, 8, 9, 10, 11, 11, 3, 14, 15};

  QuantizeLinearOp21BlockedTest_Int4_Succeed<UInt4x2, float>({8, 4, 2}, 1, 2, x, y_scale, zero_point, y_2,
                                                             DefaultCudaExecutionProvider());
  QuantizeLinearOp21BlockedTest_Int4_Succeed<UInt4x2, float>({8, 4, 2}, 1, 3, x, y_scale, zero_point, y_3,
                                                             DefaultCudaExecutionProvider());
  QuantizeLinearOp21BlockedTest_Int4_Succeed<UInt4x2, MLFloat16>({8, 4, 2}, 1, 2, x, y_scale, zero_point, y_2,
                                                                 DefaultCudaExecutionProvider());
  QuantizeLinearOp21BlockedTest_Int4_Succeed<UInt4x2, MLFloat16>({8, 4, 2}, 1, 3, x, y_scale, zero_point, y_3,
                                                                 DefaultCudaExecutionProvider());
}

TEST(QuantizeLinearOp21BlockedTest, UnsignedInt_UseZeroPoint_MiddleAxis) {
  std::vector<float> y_scale{-2.0, -4.0, 3.5, 1.0, 2.0, 4.0, -3.5, -1.0, -2.0, -4.0, 3.5, 1.0, 2.0, 4.0, -3.5, -1.0,
                             -2.0, -4.0, 3.5, 1.0, 2.0, 4.0, -3.5, -1.0, -2.0, -4.0, 3.5, 1.0, 2.0, 4.0, -3.5, -1.0};
  std::vector<int> zero_point{2, 0, 1, 9, 13, 5, 11, 6, 2, 0, 1, 9, 13, 5, 11, 6,
                              2, 0, 1, 9, 13, 5, 11, 6, 2, 0, 1, 9, 13, 5, 11, 6};
  std::vector<float> x{4, -4, 0, -12, 10.5, -4, 17.5, -2, -10, 16, -6, 24, -3.5, -7, -10.5, -9,
                       4, -4, 0, -12, 10.5, -4, 17.5, -2, -10, 16, -6, 24, -3.5, -7, -10.5, -9,
                       4, -4, 0, -12, 10.5, -4, 17.5, -2, -10, 16, -6, 24, -3.5, -7, -10.5, -9,
                       4, -4, 0, -12, 10.5, -4, 17.5, -2, -10, 16, -6, 24, -3.5, -7, -10.5, -9};
  std::vector<int> y_2{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
                       0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
                       0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
                       0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
  std::vector<int> y_3{0, 1, 2, 3, 0, 1, 6, 7, 8, 9, 10, 11, 11, 3, 14, 15,
                       0, 1, 2, 3, 0, 1, 6, 7, 8, 9, 10, 11, 11, 3, 14, 15,
                       0, 1, 2, 3, 0, 1, 6, 7, 8, 9, 10, 11, 11, 3, 14, 15,
                       0, 1, 2, 3, 0, 1, 6, 7, 8, 9, 10, 11, 11, 3, 14, 15};

  QuantizeLinearOp21BlockedTest_Int_Succeed<uint8_t, float>({8, 4, 2}, 1, 2, x, y_scale, zero_point, y_2);
  QuantizeLinearOp21BlockedTest_Int_Succeed<uint8_t, float>({8, 4, 2}, 1, 3, x, y_scale, zero_point, y_3);
  QuantizeLinearOp21BlockedTest_Int_Succeed<uint16_t, float>({8, 4, 2}, 1, 2, x, y_scale, zero_point, y_2);
  QuantizeLinearOp21BlockedTest_Int_Succeed<uint16_t, float>({8, 4, 2}, 1, 3, x, y_scale, zero_point, y_3);
  QuantizeLinearOp21BlockedTest_Int_Succeed<uint8_t, MLFloat16>({8, 4, 2}, 1, 2, x, y_scale, zero_point, y_2);
  QuantizeLinearOp21BlockedTest_Int_Succeed<uint8_t, MLFloat16>({8, 4, 2}, 1, 3, x, y_scale, zero_point, y_3);
  QuantizeLinearOp21BlockedTest_Int_Succeed<uint16_t, MLFloat16>({8, 4, 2}, 1, 2, x, y_scale, zero_point, y_2);
  QuantizeLinearOp21BlockedTest_Int_Succeed<uint16_t, MLFloat16>({8, 4, 2}, 1, 3, x, y_scale, zero_point, y_3);
}

TEST(QuantizeLinearOp21BlockedTest, UnsignedInt4_NoZeroPoint_LastAxis) {
  std::vector<float> y_scale{-2.0, -4.0, -2.0, -4.0, 3.5, 1.0, 3.5, 1.0,
                             2.0, 4.0, 2.0, 4.0, -3.5, -1.0, -3.5, -1.0};
  std::vector<int> zero_point{0, 0, 0, 0, 0, 0, 0, 0,
                              0, 0, 0, 0, 0, 0, 0, 0};
  std::vector<float> x{0, -2, 0, -2, -8, -12, -8, -12, 0, -2, 0, -2, -8, -12, -8, -12,
                       14, 17.5, 14, 17.5, 6, 7, 6, 7, 14, 17.5, 14, 17.5, 6, 7, 6, 7,
                       16, 18, 16, 18, 40, 44, 40, 44, 16, 18, 16, 18, 40, 44, 40, 44,
                       -42, -45.5, -42, -45.5, -14, -15, -14, -15, -42, -45.5, -42, -45.5, -14, -15, -14, -15};
  std::vector<int> y_2{0, 1, 0, 1, 2, 3, 2, 3, 0, 1, 0, 1, 2, 3, 2, 3,
                       4, 5, 4, 5, 6, 7, 6, 7, 4, 5, 4, 5, 6, 7, 6, 7,
                       8, 9, 8, 9, 10, 11, 10, 11, 8, 9, 8, 9, 10, 11, 10, 11,
                       12, 13, 12, 13, 14, 15, 14, 15, 12, 13, 12, 13, 14, 15, 14, 15};
  std::vector<int> y_3{0, 1, 0, 1, 4, 3, 2, 3, 0, 1, 0, 1, 4, 3, 2, 3,
                       4, 5, 4, 5, 2, 7, 6, 7, 4, 5, 4, 5, 2, 7, 6, 7,
                       8, 9, 8, 9, 15, 11, 10, 11, 8, 9, 8, 9, 15, 11, 10, 11,
                       12, 13, 12, 13, 4, 15, 14, 15, 12, 13, 12, 13, 4, 15, 14, 15};

  QuantizeLinearOp21BlockedTest_Int4_Succeed<UInt4x2, float>({2, 4, 8}, 2, 4, x, y_scale, zero_point, y_2);
  QuantizeLinearOp21BlockedTest_Int4_Succeed<UInt4x2, float>({2, 4, 8}, 2, 5, x, y_scale, zero_point, y_3);
  QuantizeLinearOp21BlockedTest_Int4_Succeed<UInt4x2, MLFloat16>({2, 4, 8}, 2, 4, x, y_scale, zero_point, y_2);
  QuantizeLinearOp21BlockedTest_Int4_Succeed<UInt4x2, MLFloat16>({2, 4, 8}, 2, 5, x, y_scale, zero_point, y_3);
}

TEST(QuantizeLinearOp21BlockedTest, UnsignedInt4_NoZeroPoint_LastAxis_Cuda) {
  std::vector<float> y_scale{-2.0, -4.0, -2.0, -4.0, 3.5, 1.0, 3.5, 1.0,
                             2.0, 4.0, 2.0, 4.0, -3.5, -1.0, -3.5, -1.0};
  std::vector<int> zero_point{0, 0, 0, 0, 0, 0, 0, 0,
                              0, 0, 0, 0, 0, 0, 0, 0};
  std::vector<float> x{0, -2, 0, -2, -8, -12, -8, -12, 0, -2, 0, -2, -8, -12, -8, -12,
                       14, 17.5, 14, 17.5, 6, 7, 6, 7, 14, 17.5, 14, 17.5, 6, 7, 6, 7,
                       16, 18, 16, 18, 40, 44, 40, 44, 16, 18, 16, 18, 40, 44, 40, 44,
                       -42, -45.5, -42, -45.5, -14, -15, -14, -15, -42, -45.5, -42, -45.5, -14, -15, -14, -15};
  std::vector<int> y_2{0, 1, 0, 1, 2, 3, 2, 3, 0, 1, 0, 1, 2, 3, 2, 3,
                       4, 5, 4, 5, 6, 7, 6, 7, 4, 5, 4, 5, 6, 7, 6, 7,
                       8, 9, 8, 9, 10, 11, 10, 11, 8, 9, 8, 9, 10, 11, 10, 11,
                       12, 13, 12, 13, 14, 15, 14, 15, 12, 13, 12, 13, 14, 15, 14, 15};
  std::vector<int> y_3{0, 1, 0, 1, 4, 3, 2, 3, 0, 1, 0, 1, 4, 3, 2, 3,
                       4, 5, 4, 5, 2, 7, 6, 7, 4, 5, 4, 5, 2, 7, 6, 7,
                       8, 9, 8, 9, 15, 11, 10, 11, 8, 9, 8, 9, 15, 11, 10, 11,
                       12, 13, 12, 13, 4, 15, 14, 15, 12, 13, 12, 13, 4, 15, 14, 15};

  QuantizeLinearOp21BlockedTest_Int4_Succeed<UInt4x2, float>({2, 4, 8}, 2, 4, x, y_scale, zero_point, y_2,
                                                             DefaultCudaExecutionProvider());
  QuantizeLinearOp21BlockedTest_Int4_Succeed<UInt4x2, float>({2, 4, 8}, 2, 5, x, y_scale, zero_point, y_3,
                                                             DefaultCudaExecutionProvider());
  QuantizeLinearOp21BlockedTest_Int4_Succeed<UInt4x2, MLFloat16>({2, 4, 8}, 2, 4, x, y_scale, zero_point, y_2,
                                                                 DefaultCudaExecutionProvider());
  QuantizeLinearOp21BlockedTest_Int4_Succeed<UInt4x2, MLFloat16>({2, 4, 8}, 2, 5, x, y_scale, zero_point, y_3,
                                                                 DefaultCudaExecutionProvider());
}

TEST(QuantizeLinearOp21BlockedTest, UnsignedInt_NoZeroPoint_LastAxis) {
  std::vector<float> y_scale{-2.0, -4.0, -2.0, -4.0, 3.5, 1.0, 3.5, 1.0,
                             2.0, 4.0, 2.0, 4.0, -3.5, -1.0, -3.5, -1.0};
  std::vector<int> zero_point{0, 0, 0, 0, 0, 0, 0, 0,
                              0, 0, 0, 0, 0, 0, 0, 0};
  std::vector<float> x{0, -2, 0, -2, -8, -12, -8, -12, 0, -2, 0, -2, -8, -12, -8, -12,
                       14, 17.5, 14, 17.5, 6, 7, 6, 7, 14, 17.5, 14, 17.5, 6, 7, 6, 7,
                       16, 18, 16, 18, 40, 44, 40, 44, 16, 18, 16, 18, 40, 44, 40, 44,
                       -42, -45.5, -42, -45.5, -14, -15, -14, -15, -42, -45.5, -42, -45.5, -14, -15, -14, -15};
  std::vector<int> y_2{0, 1, 0, 1, 2, 3, 2, 3, 0, 1, 0, 1, 2, 3, 2, 3,
                       4, 5, 4, 5, 6, 7, 6, 7, 4, 5, 4, 5, 6, 7, 6, 7,
                       8, 9, 8, 9, 10, 11, 10, 11, 8, 9, 8, 9, 10, 11, 10, 11,
                       12, 13, 12, 13, 14, 15, 14, 15, 12, 13, 12, 13, 14, 15, 14, 15};
  std::vector<int> y_3{0, 1, 0, 1, 4, 3, 2, 3, 0, 1, 0, 1, 4, 3, 2, 3,
                       4, 5, 4, 5, 2, 7, 6, 7, 4, 5, 4, 5, 2, 7, 6, 7,
                       8, 9, 8, 9, 20, 11, 10, 11, 8, 9, 8, 9, 20, 11, 10, 11,
                       12, 13, 12, 13, 4, 15, 14, 15, 12, 13, 12, 13, 4, 15, 14, 15};

  QuantizeLinearOp21BlockedTest_Int_Succeed<uint8_t, float>({2, 4, 8}, 2, 4, x, y_scale, zero_point, y_2);
  QuantizeLinearOp21BlockedTest_Int_Succeed<uint8_t, float>({2, 4, 8}, 2, 5, x, y_scale, zero_point, y_3);
  QuantizeLinearOp21BlockedTest_Int_Succeed<uint16_t, float>({2, 4, 8}, 2, 4, x, y_scale, zero_point, y_2);
  QuantizeLinearOp21BlockedTest_Int_Succeed<uint16_t, float>({2, 4, 8}, 2, 5, x, y_scale, zero_point, y_3);
  QuantizeLinearOp21BlockedTest_Int_Succeed<uint8_t, MLFloat16>({2, 4, 8}, 2, 4, x, y_scale, zero_point, y_2);
  QuantizeLinearOp21BlockedTest_Int_Succeed<uint8_t, MLFloat16>({2, 4, 8}, 2, 5, x, y_scale, zero_point, y_3);
  QuantizeLinearOp21BlockedTest_Int_Succeed<uint16_t, MLFloat16>({2, 4, 8}, 2, 4, x, y_scale, zero_point, y_2);
  QuantizeLinearOp21BlockedTest_Int_Succeed<uint16_t, MLFloat16>({2, 4, 8}, 2, 5, x, y_scale, zero_point, y_3);
}

TEST(QuantizeLinearOp21BlockedTest, UnsignedInt4_UseZeroPoint_LastAxis) {
  std::vector<float> y_scale{-2.0, -4.0, -2.0, -4.0, 3.5, 1.0, 3.5, 1.0,
                             2.0, 4.0, 2.0, 4.0, -3.5, -1.0, -3.5, -1.0};
  std::vector<int> zero_point{2, 0, 2, 0, 1, 9, 1, 9,
                              13, 5, 13, 5, 11, 6, 11, 6};
  std::vector<float> x{4, 2, 4, 2, -8, -12, -8, -12, 4, 2, 4, 2, -8, -12, -8, -12,
                       10.5, 14, 10.5, 14, -3, -2, -3, -2, 10.5, 14, 10.5, 14, -3, -2, -3, -2,
                       -10, -8, -10, -8, 20, 24, 20, 24, -10, -8, -10, -8, 20, 24, 20, 24,
                       -3.5, -7, -3.5, -7, -8, -9, -8, -9, -3.5, -7, -3.5, -7, -8, -9, -8, -9};
  std::vector<int> y_2{0, 1, 0, 1, 2, 3, 2, 3, 0, 1, 0, 1, 2, 3, 2, 3,
                       4, 5, 4, 5, 6, 7, 6, 7, 4, 5, 4, 5, 6, 7, 6, 7,
                       8, 9, 8, 9, 10, 11, 10, 11, 8, 9, 8, 9, 10, 11, 10, 11,
                       12, 13, 12, 13, 14, 15, 14, 15, 12, 13, 12, 13, 14, 15, 14, 15};
  std::vector<int> y_3{0, 1, 0, 1, 6, 3, 2, 3, 0, 1, 0, 1, 6, 3, 2, 3,
                       4, 5, 4, 5, 0, 7, 6, 7, 4, 5, 4, 5, 0, 7, 6, 7,
                       8, 9, 8, 9, 15, 11, 10, 11, 8, 9, 8, 9, 15, 11, 10, 11,
                       12, 13, 12, 13, 13, 15, 14, 15, 12, 13, 12, 13, 13, 15, 14, 15};

  QuantizeLinearOp21BlockedTest_Int4_Succeed<UInt4x2, float>({2, 4, 8}, 2, 4, x, y_scale, zero_point, y_2);
  QuantizeLinearOp21BlockedTest_Int4_Succeed<UInt4x2, float>({2, 4, 8}, 2, 5, x, y_scale, zero_point, y_3);
  QuantizeLinearOp21BlockedTest_Int4_Succeed<UInt4x2, MLFloat16>({2, 4, 8}, 2, 4, x, y_scale, zero_point, y_2);
  QuantizeLinearOp21BlockedTest_Int4_Succeed<UInt4x2, MLFloat16>({2, 4, 8}, 2, 5, x, y_scale, zero_point, y_3);
}

TEST(QuantizeLinearOp21BlockedTest, UnsignedInt4_UseZeroPoint_LastAxis_Cuda) {
  std::vector<float> y_scale{-2.0, -4.0, -2.0, -4.0, 3.5, 1.0, 3.5, 1.0,
                             2.0, 4.0, 2.0, 4.0, -3.5, -1.0, -3.5, -1.0};
  std::vector<int> zero_point{2, 0, 2, 0, 1, 9, 1, 9,
                              13, 5, 13, 5, 11, 6, 11, 6};
  std::vector<float> x{4, 2, 4, 2, -8, -12, -8, -12, 4, 2, 4, 2, -8, -12, -8, -12,
                       10.5, 14, 10.5, 14, -3, -2, -3, -2, 10.5, 14, 10.5, 14, -3, -2, -3, -2,
                       -10, -8, -10, -8, 20, 24, 20, 24, -10, -8, -10, -8, 20, 24, 20, 24,
                       -3.5, -7, -3.5, -7, -8, -9, -8, -9, -3.5, -7, -3.5, -7, -8, -9, -8, -9};
  std::vector<int> y_2{0, 1, 0, 1, 2, 3, 2, 3, 0, 1, 0, 1, 2, 3, 2, 3,
                       4, 5, 4, 5, 6, 7, 6, 7, 4, 5, 4, 5, 6, 7, 6, 7,
                       8, 9, 8, 9, 10, 11, 10, 11, 8, 9, 8, 9, 10, 11, 10, 11,
                       12, 13, 12, 13, 14, 15, 14, 15, 12, 13, 12, 13, 14, 15, 14, 15};
  std::vector<int> y_3{0, 1, 0, 1, 6, 3, 2, 3, 0, 1, 0, 1, 6, 3, 2, 3,
                       4, 5, 4, 5, 0, 7, 6, 7, 4, 5, 4, 5, 0, 7, 6, 7,
                       8, 9, 8, 9, 15, 11, 10, 11, 8, 9, 8, 9, 15, 11, 10, 11,
                       12, 13, 12, 13, 13, 15, 14, 15, 12, 13, 12, 13, 13, 15, 14, 15};

  QuantizeLinearOp21BlockedTest_Int4_Succeed<UInt4x2, float>({2, 4, 8}, 2, 4, x, y_scale, zero_point, y_2,
                                                             DefaultCudaExecutionProvider());
  QuantizeLinearOp21BlockedTest_Int4_Succeed<UInt4x2, float>({2, 4, 8}, 2, 5, x, y_scale, zero_point, y_3,
                                                             DefaultCudaExecutionProvider());
  QuantizeLinearOp21BlockedTest_Int4_Succeed<UInt4x2, MLFloat16>({2, 4, 8}, 2, 4, x, y_scale, zero_point, y_2,
                                                                 DefaultCudaExecutionProvider());
  QuantizeLinearOp21BlockedTest_Int4_Succeed<UInt4x2, MLFloat16>({2, 4, 8}, 2, 5, x, y_scale, zero_point, y_3,
                                                                 DefaultCudaExecutionProvider());
}

TEST(QuantizeLinearOp21BlockedTest, UnsignedInt_UseZeroPoint_LastAxis) {
  std::vector<float> y_scale{-2.0, -4.0, -2.0, -4.0, 3.5, 1.0, 3.5, 1.0,
                             2.0, 4.0, 2.0, 4.0, -3.5, -1.0, -3.5, -1.0};
  std::vector<int> zero_point{2, 0, 2, 0, 1, 9, 1, 9,
                              13, 5, 13, 5, 11, 6, 11, 6};
  std::vector<float> x{4, 2, 4, 2, -8, -12, -8, -12, 4, 2, 4, 2, -8, -12, -8, -12,
                       10.5, 14, 10.5, 14, -3, -2, -3, -2, 10.5, 14, 10.5, 14, -3, -2, -3, -2,
                       -10, -8, -10, -8, 20, 24, 20, 24, -10, -8, -10, -8, 20, 24, 20, 24,
                       -3.5, -7, -3.5, -7, -8, -9, -8, -9, -3.5, -7, -3.5, -7, -8, -9, -8, -9};
  std::vector<int> y_2{0, 1, 0, 1, 2, 3, 2, 3, 0, 1, 0, 1, 2, 3, 2, 3,
                       4, 5, 4, 5, 6, 7, 6, 7, 4, 5, 4, 5, 6, 7, 6, 7,
                       8, 9, 8, 9, 10, 11, 10, 11, 8, 9, 8, 9, 10, 11, 10, 11,
                       12, 13, 12, 13, 14, 15, 14, 15, 12, 13, 12, 13, 14, 15, 14, 15};
  std::vector<int> y_3{0, 1, 0, 1, 6, 3, 2, 3, 0, 1, 0, 1, 6, 3, 2, 3,
                       4, 5, 4, 5, 0, 7, 6, 7, 4, 5, 4, 5, 0, 7, 6, 7,
                       8, 9, 8, 9, 23, 11, 10, 11, 8, 9, 8, 9, 23, 11, 10, 11,
                       12, 13, 12, 13, 13, 15, 14, 15, 12, 13, 12, 13, 13, 15, 14, 15};

  QuantizeLinearOp21BlockedTest_Int_Succeed<uint8_t, float>({2, 4, 8}, 2, 4, x, y_scale, zero_point, y_2);
  QuantizeLinearOp21BlockedTest_Int_Succeed<uint8_t, float>({2, 4, 8}, 2, 5, x, y_scale, zero_point, y_3);
  QuantizeLinearOp21BlockedTest_Int_Succeed<uint16_t, float>({2, 4, 8}, 2, 4, x, y_scale, zero_point, y_2);
  QuantizeLinearOp21BlockedTest_Int_Succeed<uint16_t, float>({2, 4, 8}, 2, 5, x, y_scale, zero_point, y_3);
  QuantizeLinearOp21BlockedTest_Int_Succeed<uint8_t, MLFloat16>({2, 4, 8}, 2, 4, x, y_scale, zero_point, y_2);
  QuantizeLinearOp21BlockedTest_Int_Succeed<uint8_t, MLFloat16>({2, 4, 8}, 2, 5, x, y_scale, zero_point, y_3);
  QuantizeLinearOp21BlockedTest_Int_Succeed<uint16_t, MLFloat16>({2, 4, 8}, 2, 4, x, y_scale, zero_point, y_2);
  QuantizeLinearOp21BlockedTest_Int_Succeed<uint16_t, MLFloat16>({2, 4, 8}, 2, 5, x, y_scale, zero_point, y_3);
}

#if !defined(DISABLE_FLOAT8_TYPES)
TEST(QuantizeLinearOp21BlockedTest, Float8_NoZeroPoint_FirstAxis) {
  constexpr int min_cuda_architecture = 11080;
  bool enable_cuda = (nullptr != DefaultCpuExecutionProvider().get()) && HasCudaEnvironment(min_cuda_architecture);
  bool enable_cpu = (nullptr != DefaultCpuExecutionProvider().get());

  std::vector<float> y_scale{-2.0, -4.0, 3.5, 1.0, -2.0, -4.0, 3.5, 1.0, -2.0, -4.0, 3.5, 1.0, -2.0, -4.0, 3.5, 1.0,
                             2.0, 4.0, -3.5, -1.0, 2.0, 4.0, -3.5, -1.0, 2.0, 4.0, -3.5, -1.0, 2.0, 4.0, -3.5, -1.0};
  std::vector<float> zero_point{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  std::vector<float> x{14.0, 24.0, -17.5, -4, 14.0, 24.0, -17.5, -4, 14.0, 24.0, -17.5, -4, 14.0, 24.0, -17.5, -4,
                       6.0, 8.0, -3.5, 0.0, 6.0, 8.0, -3.5, 0.0, 6.0, 8.0, -3.5, 0.0, 6.0, 8.0, -3.5, 0.0,
                       2.0, 8.0, -10.5, -4.0, 2.0, 8.0, -10.5, -4.0, 2.0, 8.0, -10.5, -4.0, 2.0, 8.0, -10.5, -4.0,
                       10.0, 24.0, -24.5, 8.0, 10.0, 24.0, -24.5, 8.0, 10.0, 24.0, -24.5, 8.0, 10.0, 24.0, -24.5, 8.0};
  std::vector<float> y_2{-7, -6, -5, -4, -7, -6, -5, -4, -7, -6, -5, -4, -7, -6, -5, -4,
                         -3, -2, -1, 0, -3, -2, -1, 0, -3, -2, -1, 0, -3, -2, -1, 0,
                         1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4,
                         5, 6, 7, -8, 5, 6, 7, -8, 5, 6, 7, -8, 5, 6, 7, -8};
  std::vector<float> y_3{-7, -6, -5, -4, -7, -6, -5, -4, -7, -6, -5, -4, -7, -6, -5, -4,
                         -3, -2, -1, 0, -3, -2, -1, 0, -3, -2, -1, 0, -3, -2, -1, 0,
                         -1, -2, -3, -4, -1, -2, -3, -4, -1, -2, -3, -4, -1, -2, -3, -4,
                         5, 6, 7, -8, 5, 6, 7, -8, 5, 6, 7, -8, 5, 6, 7, -8};

  if (enable_cpu || enable_cuda) {
    QuantizeLinearOp21BlockedTest_Float8_Succeed<Float8E4M3FN, float>({4, 8, 2}, 0, 2, x, y_scale, zero_point, y_2);
    QuantizeLinearOp21BlockedTest_Float8_Succeed<Float8E4M3FN, MLFloat16>(
        {4, 8, 2}, 0, 2, x, y_scale, zero_point, y_2);
    QuantizeLinearOp21BlockedTest_Float8_Succeed<Float8E5M2, float>({4, 8, 2}, 0, 2, x, y_scale, zero_point, y_2);
    QuantizeLinearOp21BlockedTest_Float8_Succeed<Float8E5M2, MLFloat16>({4, 8, 2}, 0, 2, x, y_scale, zero_point, y_2);
    QuantizeLinearOp21BlockedTest_Float8_Succeed<Float8E4M3FN, float>({4, 8, 2}, 0, 3, x, y_scale, zero_point, y_3);
    QuantizeLinearOp21BlockedTest_Float8_Succeed<Float8E4M3FN, MLFloat16>(
        {4, 8, 2}, 0, 3, x, y_scale, zero_point, y_3);
    QuantizeLinearOp21BlockedTest_Float8_Succeed<Float8E5M2, float>({4, 8, 2}, 0, 3, x, y_scale, zero_point, y_3);
    QuantizeLinearOp21BlockedTest_Float8_Succeed<Float8E5M2, MLFloat16>({4, 8, 2}, 0, 3, x, y_scale, zero_point, y_3);
  }
  if (enable_cpu) {
    QuantizeLinearOp21BlockedTest_Float8_Succeed<Float8E4M3FNUZ, float>({4, 8, 2}, 0, 2, x, y_scale, zero_point, y_2);
    QuantizeLinearOp21BlockedTest_Float8_Succeed<Float8E4M3FNUZ, MLFloat16>(
        {4, 8, 2}, 0, 2, x, y_scale, zero_point, y_2);
    QuantizeLinearOp21BlockedTest_Float8_Succeed<Float8E5M2FNUZ, float>({4, 8, 2}, 0, 2, x, y_scale, zero_point, y_2);
    QuantizeLinearOp21BlockedTest_Float8_Succeed<Float8E5M2FNUZ, MLFloat16>(
        {4, 8, 2}, 0, 2, x, y_scale, zero_point, y_2);
    QuantizeLinearOp21BlockedTest_Float8_Succeed<Float8E4M3FNUZ, float>({4, 8, 2}, 0, 3, x, y_scale, zero_point, y_3);
    QuantizeLinearOp21BlockedTest_Float8_Succeed<Float8E4M3FNUZ, MLFloat16>(
        {4, 8, 2}, 0, 3, x, y_scale, zero_point, y_3);
    QuantizeLinearOp21BlockedTest_Float8_Succeed<Float8E5M2FNUZ, float>({4, 8, 2}, 0, 3, x, y_scale, zero_point, y_3);
    QuantizeLinearOp21BlockedTest_Float8_Succeed<Float8E5M2FNUZ, MLFloat16>(
        {4, 8, 2}, 0, 3, x, y_scale, zero_point, y_3);
  }
}

TEST(QuantizeLinearOp21BlockedTest, Float8_NoZeroPoint_MiddleAxis) {
  constexpr int min_cuda_architecture = 11080;
  bool enable_cuda = (nullptr != DefaultCpuExecutionProvider().get()) && HasCudaEnvironment(min_cuda_architecture);
  bool enable_cpu = (nullptr != DefaultCpuExecutionProvider().get());

  std::vector<float> zero_point{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  std::vector<float> y_scale{-2.0, -4.0, 3.5, 1.0, 2.0, 4.0, -3.5, -1.0, -2.0, -4.0, 3.5, 1.0, 2.0, 4.0, -3.5, -1.0,
                             -2.0, -4.0, 3.5, 1.0, 2.0, 4.0, -3.5, -1.0, -2.0, -4.0, 3.5, 1.0, 2.0, 4.0, -3.5, -1.0};
  std::vector<float> x{14, 24, 10, 16, -10.5, -2, -3.5, 0, 2, 8, 6, 16, -17.5, -6, -24.5, 8,
                       14, 24, 10, 16, -10.5, -2, -3.5, 0, 2, 8, 6, 16, -17.5, -6, -24.5, 8,
                       14, 24, 10, 16, -10.5, -2, -3.5, 0, 2, 8, 6, 16, -17.5, -6, -24.5, 8,
                       14, 24, 10, 16, -10.5, -2, -3.5, 0, 2, 8, 6, 16, -17.5, -6, -24.5, 8};
  std::vector<float> y_2{-7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, -8,
                         -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, -8,
                         -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, -8,
                         -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, -8};
  std::vector<float> y_3{-7, -6, -5, -4, 5, 0.5, -1, 0, 1, 2, 3, 4, -9, -1.5, 7, -8,
                         -7, -6, -5, -4, 5, 0.5, -1, 0, 1, 2, 3, 4, -9, -1.5, 7, -8,
                         -7, -6, -5, -4, 5, 0.5, -1, 0, 1, 2, 3, 4, -9, -1.5, 7, -8,
                         -7, -6, -5, -4, 5, 0.5, -1, 0, 1, 2, 3, 4, -9, -1.5, 7, -8};

  if (enable_cpu || enable_cuda) {
    QuantizeLinearOp21BlockedTest_Float8_Succeed<Float8E4M3FN, float>({8, 4, 2}, 1, 2, x, y_scale, zero_point, y_2);
    QuantizeLinearOp21BlockedTest_Float8_Succeed<Float8E4M3FN, MLFloat16>(
        {8, 4, 2}, 1, 2, x, y_scale, zero_point, y_2);
    QuantizeLinearOp21BlockedTest_Float8_Succeed<Float8E5M2, float>({8, 4, 2}, 1, 2, x, y_scale, zero_point, y_2);
    QuantizeLinearOp21BlockedTest_Float8_Succeed<Float8E5M2, MLFloat16>({8, 4, 2}, 1, 2, x, y_scale, zero_point, y_2);
    QuantizeLinearOp21BlockedTest_Float8_Succeed<Float8E4M3FN, float>({8, 4, 2}, 1, 3, x, y_scale, zero_point, y_3);
    QuantizeLinearOp21BlockedTest_Float8_Succeed<Float8E4M3FN, MLFloat16>(
        {8, 4, 2}, 1, 3, x, y_scale, zero_point, y_3);
    QuantizeLinearOp21BlockedTest_Float8_Succeed<Float8E5M2, float>({8, 4, 2}, 1, 3, x, y_scale, zero_point, y_3);
    QuantizeLinearOp21BlockedTest_Float8_Succeed<Float8E5M2, MLFloat16>({8, 4, 2}, 1, 3, x, y_scale, zero_point, y_3);
  }
  if (enable_cpu) {
    QuantizeLinearOp21BlockedTest_Float8_Succeed<Float8E4M3FNUZ, float>({8, 4, 2}, 1, 2, x, y_scale, zero_point, y_2);
    QuantizeLinearOp21BlockedTest_Float8_Succeed<Float8E4M3FNUZ, MLFloat16>(
        {8, 4, 2}, 1, 2, x, y_scale, zero_point, y_2);
    QuantizeLinearOp21BlockedTest_Float8_Succeed<Float8E5M2FNUZ, float>({8, 4, 2}, 1, 2, x, y_scale, zero_point, y_2);
    QuantizeLinearOp21BlockedTest_Float8_Succeed<Float8E5M2FNUZ, MLFloat16>(
        {8, 4, 2}, 1, 2, x, y_scale, zero_point, y_2);
    QuantizeLinearOp21BlockedTest_Float8_Succeed<Float8E4M3FNUZ, float>({8, 4, 2}, 1, 3, x, y_scale, zero_point, y_3);
    QuantizeLinearOp21BlockedTest_Float8_Succeed<Float8E4M3FNUZ, MLFloat16>(
        {8, 4, 2}, 1, 3, x, y_scale, zero_point, y_3);
    QuantizeLinearOp21BlockedTest_Float8_Succeed<Float8E5M2FNUZ, float>({8, 4, 2}, 1, 3, x, y_scale, zero_point, y_3);
    QuantizeLinearOp21BlockedTest_Float8_Succeed<Float8E5M2FNUZ, MLFloat16>(
        {8, 4, 2}, 1, 3, x, y_scale, zero_point, y_3);
  }
}

TEST(QuantizeLinearOp21BlockedTest, Float8_NoZeroPoint_LastAxis) {
  constexpr int min_cuda_architecture = 11080;
  bool enable_cuda = (nullptr != DefaultCpuExecutionProvider().get()) && HasCudaEnvironment(min_cuda_architecture);
  bool enable_cpu = (nullptr != DefaultCpuExecutionProvider().get());

  std::vector<float> zero_point{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  std::vector<float> y_scale{-2.0, -4.0, 3.5, 1.0, 2.0, 4.0, -3.5, -1.0, -2.0, -4.0, 3.5, 1.0, 2.0, 4.0, -3.5, -1.0,
                             -2.0, -4.0, 3.5, 1.0, 2.0, 4.0, -3.5, -1.0, -2.0, -4.0, 3.5, 1.0, 2.0, 4.0, -3.5, -1.0};
  std::vector<float> x{14, 12, 20, 16, -10.5, -7, -3.5, 0, 2, 4, 12, 16, -17.5, -21, -7, 8,
                       14, 12, 20, 16, -10.5, -7, -3.5, 0, 2, 4, 12, 16, -17.5, -21, -7, 8,
                       14, 12, 20, 16, -10.5, -7, -3.5, 0, 2, 4, 12, 16, -17.5, -21, -7, 8,
                       14, 12, 20, 16, -10.5, -7, -3.5, 0, 2, 4, 12, 16, -17.5, -21, -7, 8};
  std::vector<float> y_2{-7, -6, -5, -4, -3, -2, -3.5, 0, 1, 2, 3, 4, 5, 6, 7, -8,
                         -7, -6, -5, -4, -3, -2, -3.5, 0, 1, 2, 3, 4, 5, 6, 7, -8,
                         -7, -6, -5, -4, -3, -2, -3.5, 0, 1, 2, 3, 4, 5, 6, 7, -8,
                         -7, -6, -5, -4, -3, -2, -3.5, 0, 1, 2, 3, 4, 5, 6, 7, -8};
  std::vector<float> y_3{-7, -6, -10, -4, -3, -2, -1, 0, 1, 2, 6, 4, 5, 6, 2, -8,
                         -7, -6, -10, -4, -3, -2, -1, 0, 1, 2, 6, 4, 5, 6, 2, -8,
                         -7, -6, -10, -4, -3, -2, -1, 0, 1, 2, 6, 4, 5, 6, 2, -8,
                         -7, -6, -10, -4, -3, -2, -1, 0, 1, 2, 6, 4, 5, 6, 2, -8};

  if (enable_cpu || enable_cuda) {
    QuantizeLinearOp21BlockedTest_Float8_Succeed<Float8E4M3FN, float>({8, 2, 4}, 2, 2, x, y_scale, zero_point, y_2);
    QuantizeLinearOp21BlockedTest_Float8_Succeed<Float8E4M3FN, MLFloat16>(
        {8, 2, 4}, 2, 2, x, y_scale, zero_point, y_2);
    QuantizeLinearOp21BlockedTest_Float8_Succeed<Float8E5M2, float>({8, 2, 4}, 2, 2, x, y_scale, zero_point, y_2);
    QuantizeLinearOp21BlockedTest_Float8_Succeed<Float8E5M2, MLFloat16>({8, 2, 4}, 2, 2, x, y_scale, zero_point, y_2);
    QuantizeLinearOp21BlockedTest_Float8_Succeed<Float8E4M3FN, float>({8, 2, 4}, 2, 3, x, y_scale, zero_point, y_3);
    QuantizeLinearOp21BlockedTest_Float8_Succeed<Float8E4M3FN, MLFloat16>(
        {8, 2, 4}, 2, 3, x, y_scale, zero_point, y_3);
    QuantizeLinearOp21BlockedTest_Float8_Succeed<Float8E5M2, float>({8, 2, 4}, 2, 3, x, y_scale, zero_point, y_3);
    QuantizeLinearOp21BlockedTest_Float8_Succeed<Float8E5M2, MLFloat16>({8, 2, 4}, 2, 3, x, y_scale, zero_point, y_3);
  }
  if (enable_cpu) {
    QuantizeLinearOp21BlockedTest_Float8_Succeed<Float8E4M3FNUZ, float>({8, 2, 4}, 2, 2, x, y_scale, zero_point, y_2);
    QuantizeLinearOp21BlockedTest_Float8_Succeed<Float8E4M3FNUZ, MLFloat16>(
        {8, 2, 4}, 2, 2, x, y_scale, zero_point, y_2);
    QuantizeLinearOp21BlockedTest_Float8_Succeed<Float8E5M2FNUZ, float>({8, 2, 4}, 2, 2, x, y_scale, zero_point, y_2);
    QuantizeLinearOp21BlockedTest_Float8_Succeed<Float8E5M2FNUZ, MLFloat16>(
        {8, 2, 4}, 2, 2, x, y_scale, zero_point, y_2);
    QuantizeLinearOp21BlockedTest_Float8_Succeed<Float8E4M3FNUZ, float>({8, 2, 4}, 2, 3, x, y_scale, zero_point, y_3);
    QuantizeLinearOp21BlockedTest_Float8_Succeed<Float8E4M3FNUZ, MLFloat16>(
        {8, 2, 4}, 2, 3, x, y_scale, zero_point, y_3);
    QuantizeLinearOp21BlockedTest_Float8_Succeed<Float8E5M2FNUZ, float>({8, 2, 4}, 2, 3, x, y_scale, zero_point, y_3);
    QuantizeLinearOp21BlockedTest_Float8_Succeed<Float8E5M2FNUZ, MLFloat16>(
        {8, 2, 4}, 2, 3, x, y_scale, zero_point, y_3);
  }
}
#endif
}  // namespace blocked_quantization
}  // namespace test
}  // namespace onnxruntime
