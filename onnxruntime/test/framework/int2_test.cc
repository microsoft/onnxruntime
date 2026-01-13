// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <array>
#include <vector>
#include <cstdint>
#include <filesystem>

#include "core/framework/int2.h"
#include "core/framework/data_types.h"
#include "core/framework/tensorprotoutils.h"
#include "core/platform/env.h"
#include "test/test_environment.h"
#include "gtest/gtest.h"

namespace onnxruntime {
namespace test {

// ==============================================
// Int2x4 Tests (signed 2-bit integer, 4 per byte)
// ==============================================

TEST(Int2_Tests, Int2x4_DefaultConstructor) {
  Int2x4 int2;
  EXPECT_EQ(static_cast<uint8_t>(int2.ToBits()), 0);
}

TEST(Int2_Tests, Int2x4_BitsConstructor) {
  // Pack 4 signed 2-bit values: val0=1, val1=-1 (0b11), val2=-2 (0b10), val3=0
  // Binary: 0b00'10'11'01 = 0x2D
  Int2x4 int2(std::byte{0x2D});
  EXPECT_EQ(int2.GetElem(0), 1);
  EXPECT_EQ(int2.GetElem(1), -1);  // 0b11 sign-extended is -1
  EXPECT_EQ(int2.GetElem(2), -2);  // 0b10 sign-extended is -2
  EXPECT_EQ(int2.GetElem(3), 0);
}

TEST(Int2_Tests, Int2x4_FourValueConstructor) {
  Int2x4 int2(1, -1, -2, 0);
  EXPECT_EQ(int2.GetElem(0), 1);
  EXPECT_EQ(int2.GetElem(1), -1);
  EXPECT_EQ(int2.GetElem(2), -2);
  EXPECT_EQ(int2.GetElem(3), 0);
}

TEST(Int2_Tests, Int2x4_GetSetElem) {
  Int2x4 int2;

  // Set and get each element
  int2.SetElem(0, 1);
  int2.SetElem(1, -1);
  int2.SetElem(2, -2);
  int2.SetElem(3, 0);

  EXPECT_EQ(int2.GetElem(0), 1);
  EXPECT_EQ(int2.GetElem(1), -1);
  EXPECT_EQ(int2.GetElem(2), -2);
  EXPECT_EQ(int2.GetElem(3), 0);
}

TEST(Int2_Tests, Int2x4_ValueRange) {
  // Verify min/max values
  EXPECT_EQ(Int2x4::min_val, -2);
  EXPECT_EQ(Int2x4::max_val, 1);

  // Test all valid signed 2-bit values: -2, -1, 0, 1
  Int2x4 int2(-2, -1, 0, 1);
  EXPECT_EQ(int2.GetElem(0), -2);
  EXPECT_EQ(int2.GetElem(1), -1);
  EXPECT_EQ(int2.GetElem(2), 0);
  EXPECT_EQ(int2.GetElem(3), 1);
}

TEST(Int2_Tests, Int2x4_CalcNumInt2Quads) {
  // 0 elements -> 0 bytes
  EXPECT_EQ(Int2x4::CalcNumInt2Quads(0), 0u);
  // 1 element -> 1 byte
  EXPECT_EQ(Int2x4::CalcNumInt2Quads(1), 1u);
  // 4 elements -> 1 byte
  EXPECT_EQ(Int2x4::CalcNumInt2Quads(4), 1u);
  // 5 elements -> 2 bytes
  EXPECT_EQ(Int2x4::CalcNumInt2Quads(5), 2u);
  // 8 elements -> 2 bytes
  EXPECT_EQ(Int2x4::CalcNumInt2Quads(8), 2u);
}

TEST(Int2_Tests, Int2x4_PackUnpack) {
  std::vector<int8_t> src_values = {1, -1, -2, 0, 1, -1, -2, 0};
  std::vector<Int2x4> packed(Int2x4::CalcNumInt2Quads(src_values.size()));

  // Pack
  bool pack_result = Int2x4::Pack(gsl::make_span(packed), gsl::make_span(src_values));
  EXPECT_TRUE(pack_result);

  // Unpack
  std::vector<int8_t> unpacked(src_values.size());
  bool unpack_result = Int2x4::Unpack(gsl::make_span(unpacked), gsl::make_span(packed));
  EXPECT_TRUE(unpack_result);

  // Verify
  for (size_t i = 0; i < src_values.size(); i++) {
    EXPECT_EQ(unpacked[i], src_values[i]) << "Mismatch at index " << i;
  }
}

TEST(Int2_Tests, Int2x4_PackUnpackOddElements) {
  // Test with non-multiple-of-4 element count
  std::vector<int8_t> src_values = {1, -1, -2};
  std::vector<Int2x4> packed(Int2x4::CalcNumInt2Quads(src_values.size()));

  // Pack
  bool pack_result = Int2x4::Pack(gsl::make_span(packed), gsl::make_span(src_values));
  EXPECT_TRUE(pack_result);

  // Unpack
  std::vector<int8_t> unpacked(src_values.size());
  bool unpack_result = Int2x4::Unpack(gsl::make_span(unpacked), gsl::make_span(packed));
  EXPECT_TRUE(unpack_result);

  // Verify
  for (size_t i = 0; i < src_values.size(); i++) {
    EXPECT_EQ(unpacked[i], src_values[i]) << "Mismatch at index " << i;
  }
}

// ==============================================
// UInt2x4 Tests (unsigned 2-bit integer, 4 per byte)
// ==============================================

TEST(Int2_Tests, UInt2x4_DefaultConstructor) {
  UInt2x4 uint2;
  EXPECT_EQ(static_cast<uint8_t>(uint2.ToBits()), 0);
}

TEST(Int2_Tests, UInt2x4_BitsConstructor) {
  // Pack 4 unsigned 2-bit values: val0=0, val1=1, val2=2, val3=3
  // Binary: 0b11'10'01'00 = 0xE4
  UInt2x4 uint2(std::byte{0xE4});
  EXPECT_EQ(uint2.GetElem(0), 0);
  EXPECT_EQ(uint2.GetElem(1), 1);
  EXPECT_EQ(uint2.GetElem(2), 2);
  EXPECT_EQ(uint2.GetElem(3), 3);
}

TEST(Int2_Tests, UInt2x4_FourValueConstructor) {
  UInt2x4 uint2(0, 1, 2, 3);
  EXPECT_EQ(uint2.GetElem(0), 0);
  EXPECT_EQ(uint2.GetElem(1), 1);
  EXPECT_EQ(uint2.GetElem(2), 2);
  EXPECT_EQ(uint2.GetElem(3), 3);
}

TEST(Int2_Tests, UInt2x4_GetSetElem) {
  UInt2x4 uint2;

  // Set and get each element
  uint2.SetElem(0, 0);
  uint2.SetElem(1, 1);
  uint2.SetElem(2, 2);
  uint2.SetElem(3, 3);

  EXPECT_EQ(uint2.GetElem(0), 0);
  EXPECT_EQ(uint2.GetElem(1), 1);
  EXPECT_EQ(uint2.GetElem(2), 2);
  EXPECT_EQ(uint2.GetElem(3), 3);
}

TEST(Int2_Tests, UInt2x4_ValueRange) {
  // Verify min/max values
  EXPECT_EQ(UInt2x4::min_val, 0);
  EXPECT_EQ(UInt2x4::max_val, 3);

  // Test all valid unsigned 2-bit values: 0, 1, 2, 3
  UInt2x4 uint2(0, 1, 2, 3);
  EXPECT_EQ(uint2.GetElem(0), 0);
  EXPECT_EQ(uint2.GetElem(1), 1);
  EXPECT_EQ(uint2.GetElem(2), 2);
  EXPECT_EQ(uint2.GetElem(3), 3);
}

TEST(Int2_Tests, UInt2x4_CalcNumInt2Quads) {
  // Same as Int2x4
  EXPECT_EQ(UInt2x4::CalcNumInt2Quads(0), 0u);
  EXPECT_EQ(UInt2x4::CalcNumInt2Quads(1), 1u);
  EXPECT_EQ(UInt2x4::CalcNumInt2Quads(4), 1u);
  EXPECT_EQ(UInt2x4::CalcNumInt2Quads(5), 2u);
}

TEST(Int2_Tests, UInt2x4_PackUnpack) {
  std::vector<uint8_t> src_values = {0, 1, 2, 3, 3, 2, 1, 0};
  std::vector<UInt2x4> packed(UInt2x4::CalcNumInt2Quads(src_values.size()));

  // Pack
  bool pack_result = UInt2x4::Pack(gsl::make_span(packed), gsl::make_span(src_values));
  EXPECT_TRUE(pack_result);

  // Unpack
  std::vector<uint8_t> unpacked(src_values.size());
  bool unpack_result = UInt2x4::Unpack(gsl::make_span(unpacked), gsl::make_span(packed));
  EXPECT_TRUE(unpack_result);

  // Verify
  for (size_t i = 0; i < src_values.size(); i++) {
    EXPECT_EQ(unpacked[i], src_values[i]) << "Mismatch at index " << i;
  }
}

TEST(Int2_Tests, UInt2x4_PackUnpackOddElements) {
  // Test with non-multiple-of-4 element count
  std::vector<uint8_t> src_values = {3, 2, 1};
  std::vector<UInt2x4> packed(UInt2x4::CalcNumInt2Quads(src_values.size()));

  // Pack
  bool pack_result = UInt2x4::Pack(gsl::make_span(packed), gsl::make_span(src_values));
  EXPECT_TRUE(pack_result);

  // Unpack
  std::vector<uint8_t> unpacked(src_values.size());
  bool unpack_result = UInt2x4::Unpack(gsl::make_span(unpacked), gsl::make_span(packed));
  EXPECT_TRUE(unpack_result);

  // Verify
  for (size_t i = 0; i < src_values.size(); i++) {
    EXPECT_EQ(unpacked[i], src_values[i]) << "Mismatch at index " << i;
  }
}

// ==============================================
// Additional edge case tests
// ==============================================

TEST(Int2_Tests, Int2x4_AllSameValue) {
  // All values are -2 (minimum signed value)
  Int2x4 int2_min(-2, -2, -2, -2);
  for (size_t i = 0; i < 4; i++) {
    EXPECT_EQ(int2_min.GetElem(i), -2);
  }

  // All values are 1 (maximum signed value)
  Int2x4 int2_max(1, 1, 1, 1);
  for (size_t i = 0; i < 4; i++) {
    EXPECT_EQ(int2_max.GetElem(i), 1);
  }
}

TEST(Int2_Tests, UInt2x4_AllSameValue) {
  // All values are 0 (minimum unsigned value)
  UInt2x4 uint2_min(0, 0, 0, 0);
  for (size_t i = 0; i < 4; i++) {
    EXPECT_EQ(uint2_min.GetElem(i), 0);
  }

  // All values are 3 (maximum unsigned value)
  UInt2x4 uint2_max(3, 3, 3, 3);
  for (size_t i = 0; i < 4; i++) {
    EXPECT_EQ(uint2_max.GetElem(i), 3);
  }
}

TEST(Int2_Tests, Int2x4_BitManipulation) {
  // Test that ToBits returns correct packed representation
  Int2x4 int2(0, 1, -1, -2);  // 0b00, 0b01, 0b11, 0b10
  // Expected: 0b10'11'01'00 = 0xB4
  EXPECT_EQ(static_cast<uint8_t>(int2.ToBits()), 0xB4);
}

// ==============================================
// TypeProto / TypeFromProto smoke checks
// ==============================================

TEST(Int2_Tests, TensorTypeFromONNXEnum_Int2UInt2) {
  auto* int2_type = DataTypeImpl::TensorTypeFromONNXEnum(ONNX_NAMESPACE::TensorProto_DataType_INT2);
  auto* uint2_type = DataTypeImpl::TensorTypeFromONNXEnum(ONNX_NAMESPACE::TensorProto_DataType_UINT2);

  ASSERT_NE(int2_type, nullptr);
  ASSERT_NE(uint2_type, nullptr);
  EXPECT_EQ(int2_type->GetElementType(), DataTypeImpl::GetType<Int2x4>());
  EXPECT_EQ(uint2_type->GetElementType(), DataTypeImpl::GetType<UInt2x4>());
}

TEST(Int2_Tests, TypeFromProto_TensorProto_Int2) {
  ONNX_NAMESPACE::TypeProto tp;
  tp.mutable_tensor_type()->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_INT2);
  auto mltype = DataTypeImpl::TypeFromProto(tp);
  ASSERT_NE(mltype, nullptr);
  const auto* tensor_type = mltype->AsTensorType();
  ASSERT_NE(tensor_type, nullptr);
  EXPECT_EQ(tensor_type->GetElementType(), DataTypeImpl::GetType<Int2x4>());
}

TEST(Int2_Tests, TensorProtoRoundTrip_Int2) {
  // Build a tiny TensorProto with raw_data containing 2 bytes (8 int2 elements packed)
  ONNX_NAMESPACE::TensorProto proto;
  proto.set_data_type(ONNX_NAMESPACE::TensorProto_DataType_INT2);
  proto.add_dims(8);
  // pack values [1, -1, -2, 0, 1, -1, -2, 0]
  std::array<int8_t, 8> values = {1, -1, -2, 0, 1, -1, -2, 0};
  std::vector<Int2x4> packed(Int2x4::CalcNumInt2Quads(values.size()));
  ASSERT_TRUE(Int2x4::Pack(gsl::make_span(packed), gsl::make_span(values)));
  proto.set_raw_data(packed.data(), packed.size() * sizeof(Int2x4));

  Tensor result;
  // Use CreateTensorFromTensorProto which pre-allocates the tensor with proper shape
  ORT_THROW_IF_ERROR(utils::CreateTensorFromTensorProto(onnxruntime::Env::Default(), std::filesystem::path{}, proto, result));
  ASSERT_TRUE(result.IsDataType<Int2x4>());
  const auto* data = result.Data<Int2x4>();
  std::vector<int8_t> unpacked(values.size());
  ASSERT_TRUE(Int2x4::Unpack(gsl::make_span(unpacked), gsl::make_span(data, packed.size())));
  for (size_t i = 0; i < values.size(); ++i) {
    EXPECT_EQ(unpacked[i], values[i]) << "Mismatch at index " << i;
  }
}

TEST(Int2_Tests, UInt2x4_BitManipulation) {
  // Test that ToBits returns correct packed representation
  UInt2x4 uint2(3, 2, 1, 0);  // 0b11, 0b10, 0b01, 0b00
  // Expected: 0b00'01'10'11 = 0x1B
  EXPECT_EQ(static_cast<uint8_t>(uint2.ToBits()), 0x1B);
}

}  // namespace test
}  // namespace onnxruntime
