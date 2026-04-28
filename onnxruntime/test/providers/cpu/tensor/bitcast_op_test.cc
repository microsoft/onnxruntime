// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"
#include "test/providers/provider_test_utils.h"
#include "core/framework/to_tensor_proto_element_type.h"

#include <cstring>
#include <vector>

namespace onnxruntime {
namespace test {

template <typename SrcType, typename DstType>
void TestBitCastOp(const std::vector<int64_t>& shape,
                   const std::vector<SrcType>& input,
                   const std::vector<DstType>& expected_output) {
  OpTester test("BitCast", 26, onnxruntime::kOnnxDomain);
  test.AddAttribute<int64_t>("to", utils::ToTensorProtoElementType<DstType>());
  test.AddInput<SrcType>("input", shape, input);
  test.AddOutput<DstType>("output", shape, expected_output);
  // BitCast is CPU-only for now; exclude providers that don't support it.
  test.Run(OpTester::ExpectResult::kExpectSuccess, "",
           {kTensorrtExecutionProvider, kOpenVINOExecutionProvider});
}

// float32 and int32 are both 4 bytes.
// IEEE 754: 1.0f = 0x3F800000 = 1065353216 as int32
TEST(BitCastTest, Float32ToInt32) {
  std::vector<float> input = {0.0f, 1.0f, -1.0f, 0.5f};
  std::vector<int32_t> expected(input.size());
  std::memcpy(expected.data(), input.data(), input.size() * sizeof(float));

  TestBitCastOp<float, int32_t>({4}, input, expected);
}

TEST(BitCastTest, Int32ToFloat32) {
  // 0x3F800000 = 1065353216 -> 1.0f
  // 0x40000000 = 1073741824 -> 2.0f
  std::vector<int32_t> input = {0, 1065353216, 1073741824};
  std::vector<float> expected(input.size());
  std::memcpy(expected.data(), input.data(), input.size() * sizeof(int32_t));

  TestBitCastOp<int32_t, float>({3}, input, expected);
}

// double and int64 are both 8 bytes.
TEST(BitCastTest, DoubleToInt64) {
  std::vector<double> input = {0.0, 1.0, -1.0};
  std::vector<int64_t> expected(input.size());
  std::memcpy(expected.data(), input.data(), input.size() * sizeof(double));

  TestBitCastOp<double, int64_t>({3}, input, expected);
}

TEST(BitCastTest, Int64ToDouble) {
  std::vector<int64_t> input = {0, 4607182418800017408};  // 0 and 1.0 as int64
  std::vector<double> expected(input.size());
  std::memcpy(expected.data(), input.data(), input.size() * sizeof(int64_t));

  TestBitCastOp<int64_t, double>({2}, input, expected);
}

// float16 and uint16 are both 2 bytes.
TEST(BitCastTest, Float16ToUInt16) {
  std::vector<MLFloat16> input = {MLFloat16(0.0f), MLFloat16(1.0f), MLFloat16(0.5f)};
  std::vector<uint16_t> expected(input.size());
  std::memcpy(expected.data(), input.data(), input.size() * sizeof(MLFloat16));

  TestBitCastOp<MLFloat16, uint16_t>({3}, input, expected);
}

TEST(BitCastTest, UInt16ToFloat16) {
  std::vector<uint16_t> input = {0x0000, 0x3C00, 0x3800};  // 0.0, 1.0, 0.5 in float16
  std::vector<MLFloat16> expected;
  expected.reserve(input.size());
  for (auto v : input) {
    expected.push_back(MLFloat16::FromBits(v));
  }

  TestBitCastOp<uint16_t, MLFloat16>({3}, input, expected);
}

// BFloat16 and int16 are both 2 bytes.
TEST(BitCastTest, BFloat16ToInt16) {
  std::vector<BFloat16> input = {BFloat16(0.0f), BFloat16(1.0f)};
  std::vector<int16_t> expected(input.size());
  std::memcpy(expected.data(), input.data(), input.size() * sizeof(BFloat16));

  TestBitCastOp<BFloat16, int16_t>({2}, input, expected);
}

// int8 and uint8 are both 1 byte.
TEST(BitCastTest, Int8ToUInt8) {
  std::vector<int8_t> input = {0, 1, -1, 127, -128};
  std::vector<uint8_t> expected(input.size());
  std::memcpy(expected.data(), input.data(), input.size() * sizeof(int8_t));

  TestBitCastOp<int8_t, uint8_t>({5}, input, expected);
}

TEST(BitCastTest, UInt8ToInt8) {
  std::vector<uint8_t> input = {0, 1, 127, 128, 255};
  std::vector<int8_t> expected(input.size());
  std::memcpy(expected.data(), input.data(), input.size() * sizeof(uint8_t));

  TestBitCastOp<uint8_t, int8_t>({5}, input, expected);
}

// Same type (identity-like).
TEST(BitCastTest, Float32ToFloat32) {
  std::vector<float> input = {1.0f, 2.0f, 3.0f};
  TestBitCastOp<float, float>({3}, input, input);
}

// Multi-dimensional input.
TEST(BitCastTest, Float32ToInt32_2D) {
  std::vector<float> input = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  std::vector<int32_t> expected(input.size());
  std::memcpy(expected.data(), input.data(), input.size() * sizeof(float));

  TestBitCastOp<float, int32_t>({2, 3}, input, expected);
}

TEST(BitCastTest, Float32ToInt32_3D) {
  std::vector<float> input = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f,
                              7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f};
  std::vector<int32_t> expected(input.size());
  std::memcpy(expected.data(), input.data(), input.size() * sizeof(float));

  TestBitCastOp<float, int32_t>({2, 2, 3}, input, expected);
}

// Empty tensor.
TEST(BitCastTest, EmptyTensor) {
  std::vector<float> input = {};
  std::vector<int32_t> expected = {};
  TestBitCastOp<float, int32_t>({0}, input, expected);
}

// Scalar (0-dim) tensor.
TEST(BitCastTest, ScalarTensor) {
  std::vector<float> input = {42.0f};
  std::vector<int32_t> expected(1);
  std::memcpy(expected.data(), input.data(), sizeof(float));

  OpTester test("BitCast", 26, onnxruntime::kOnnxDomain);
  test.AddAttribute<int64_t>("to", utils::ToTensorProtoElementType<int32_t>());
  test.AddInput<float>("input", {}, input);
  test.AddOutput<int32_t>("output", {}, expected);
  test.Run(OpTester::ExpectResult::kExpectSuccess, "",
           {kTensorrtExecutionProvider, kOpenVINOExecutionProvider});
}

// uint32 and float32 (same size, 4 bytes).
TEST(BitCastTest, UInt32ToFloat32) {
  std::vector<uint32_t> input = {0, 0x3F800000, 0x40000000};  // 0.0f, 1.0f, 2.0f
  std::vector<float> expected(input.size());
  std::memcpy(expected.data(), input.data(), input.size() * sizeof(uint32_t));

  TestBitCastOp<uint32_t, float>({3}, input, expected);
}

// uint64 and double (same size, 8 bytes).
TEST(BitCastTest, UInt64ToDouble) {
  std::vector<uint64_t> input = {0, 0x3FF0000000000000ULL};  // 0.0 and 1.0 as uint64
  std::vector<double> expected(input.size());
  std::memcpy(expected.data(), input.data(), input.size() * sizeof(uint64_t));

  TestBitCastOp<uint64_t, double>({2}, input, expected);
}

// int16 and uint16 (same size, 2 bytes).
TEST(BitCastTest, Int16ToUInt16) {
  std::vector<int16_t> input = {0, 1, -1, 32767, -32768};
  std::vector<uint16_t> expected(input.size());
  std::memcpy(expected.data(), input.data(), input.size() * sizeof(int16_t));

  TestBitCastOp<int16_t, uint16_t>({5}, input, expected);
}

}  // namespace test
}  // namespace onnxruntime
