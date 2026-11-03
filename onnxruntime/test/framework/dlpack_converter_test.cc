// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#if defined(ENABLE_DLPACK)

#include <array>
#include <cstring>

#include "core/dlpack/dlpack_converter.h"
#include "core/framework/allocator.h"
#include "gtest/gtest.h"

namespace onnxruntime::test {
namespace {

struct DataTypeTestCase {
  ONNX_NAMESPACE::TensorProto_DataType ort_type;
  DLDataTypeCode code;
  uint8_t bits;
};

constexpr DataTypeTestCase data_types[] = {
    {ONNX_NAMESPACE::TensorProto_DataType_FLOAT, kDLFloat, 32},
    {ONNX_NAMESPACE::TensorProto_DataType_DOUBLE, kDLFloat, 64},
    {ONNX_NAMESPACE::TensorProto_DataType_FLOAT16, kDLFloat, 16},
    {ONNX_NAMESPACE::TensorProto_DataType_BFLOAT16, kDLBfloat, 16},
    {ONNX_NAMESPACE::TensorProto_DataType_BOOL, kDLBool, 8},
    {ONNX_NAMESPACE::TensorProto_DataType_INT8, kDLInt, 8},
    {ONNX_NAMESPACE::TensorProto_DataType_INT16, kDLInt, 16},
    {ONNX_NAMESPACE::TensorProto_DataType_INT32, kDLInt, 32},
    {ONNX_NAMESPACE::TensorProto_DataType_INT64, kDLInt, 64},
    {ONNX_NAMESPACE::TensorProto_DataType_UINT8, kDLUInt, 8},
    {ONNX_NAMESPACE::TensorProto_DataType_UINT16, kDLUInt, 16},
    {ONNX_NAMESPACE::TensorProto_DataType_UINT32, kDLUInt, 32},
    {ONNX_NAMESPACE::TensorProto_DataType_UINT64, kDLUInt, 64},
#if !defined(DISABLE_FLOAT8_TYPES)
    {ONNX_NAMESPACE::TensorProto_DataType_FLOAT8E4M3FN, kDLFloat8_e4m3fn, 8},
    {ONNX_NAMESPACE::TensorProto_DataType_FLOAT8E4M3FNUZ, kDLFloat8_e4m3fnuz, 8},
    {ONNX_NAMESPACE::TensorProto_DataType_FLOAT8E5M2, kDLFloat8_e5m2, 8},
    {ONNX_NAMESPACE::TensorProto_DataType_FLOAT8E5M2FNUZ, kDLFloat8_e5m2fnuz, 8},
    {ONNX_NAMESPACE::TensorProto_DataType_FLOAT8E8M0, kDLFloat8_e8m0fnu, 8},
#endif
#if !defined(DISABLE_FLOAT4_TYPES)
    {ONNX_NAMESPACE::TensorProto_DataType_FLOAT4E2M1, kDLFloat4_e2m1fn, 4},
#endif
    {ONNX_NAMESPACE::TensorProto_DataType_INT4, kDLInt, 4},
    {ONNX_NAMESPACE::TensorProto_DataType_UINT4, kDLUInt, 4},
    {ONNX_NAMESPACE::TensorProto_DataType_INT2, kDLInt, 2},
    {ONNX_NAMESPACE::TensorProto_DataType_UINT2, kDLUInt, 2},
};

TEST(DlpackConverterTest, AllDataTypesRoundTrip) {
  auto allocator = std::make_shared<CPUAllocator>();
  for (const auto& dtype : data_types) {
    SCOPED_TRACE(dtype.ort_type);
    for (const TensorShape& shape : {TensorShape{2, 5}, TensorShape{}, TensorShape{0, 5}}) {
      SCOPED_TRACE(shape.ToString());
      OrtValue original;
      Tensor::InitOrtValue(DataTypeImpl::TensorTypeFromONNXEnum(dtype.ort_type)->GetElementType(),
                           shape, allocator, original);
      auto* tensor = original.GetMutable<Tensor>();
      const size_t bytes = tensor->SizeInBytes();
      if (bytes > 0) {
        std::memset(tensor->MutableDataRaw(), 0x35, bytes);
      }
      auto* packed = dlpack::OrtValueToDlpack(original);
      EXPECT_EQ(packed->dl_tensor.dtype.code, dtype.code);
      EXPECT_EQ(packed->dl_tensor.dtype.bits, dtype.bits);
      EXPECT_EQ(packed->dl_tensor.dtype.lanes, 1);
      if (shape.NumDimensions() != 0) {
        ASSERT_NE(packed->dl_tensor.strides, nullptr);
        EXPECT_EQ(packed->dl_tensor.strides[shape.NumDimensions() - 1], shape.Size() == 0 ? 0 : 1);
      }

      OrtValue imported = dlpack::DlpackToOrtValue(packed);
      EXPECT_EQ(imported.Get<Tensor>().GetElementType(), dtype.ort_type);
      EXPECT_EQ(imported.Get<Tensor>().Shape(), shape);
      EXPECT_EQ(imported.Get<Tensor>().SizeInBytes(), bytes);
      EXPECT_EQ(imported.Get<Tensor>().DataRaw(), tensor->DataRaw());
      original = OrtValue{};
      if (bytes > 0) {
        const auto* data = static_cast<const uint8_t*>(imported.Get<Tensor>().DataRaw());
        EXPECT_EQ(data[0], 0x35);
        EXPECT_EQ(data[bytes - 1], 0x35);
      }
    }
  }
}

TEST(DlpackConverterTest, ImportDataTypesWithOffset) {
  std::array<uint8_t, 128> data{};
  for (const auto& dtype : data_types) {
    SCOPED_TRACE(dtype.ort_type);
    int64_t shape[] = {5};
    DLManagedTensor packed{};
    packed.dl_tensor = {data.data(), {kDLCPU, 0}, 1, {static_cast<uint8_t>(dtype.code), dtype.bits, 1}, shape, nullptr, 8};
    int deletion_count = 0;
    packed.manager_ctx = &deletion_count;
    packed.deleter = [](DLManagedTensor* self) { ++*static_cast<int*>(self->manager_ctx); };
    {
      OrtValue imported = dlpack::DlpackToOrtValue(&packed);
      EXPECT_EQ(imported.Get<Tensor>().GetElementType(), dtype.ort_type);
      EXPECT_EQ(imported.Get<Tensor>().DataRaw(), data.data() + 8);
      EXPECT_EQ(imported.Get<Tensor>().SizeInBytes(), (5 * dtype.bits + 7) / 8);
      EXPECT_EQ(deletion_count, 0);
    }
    EXPECT_EQ(deletion_count, 1);
  }
}

TEST(DlpackConverterTest, RejectUnsupportedDataTypes) {
  const DLDataType unsupported[] = {
      {kDLBool, 1, 1},
      {kDLBool, 16, 1},
      {kDLInt, 6, 1},
      {kDLUInt, 1, 1},
      {kDLFloat, 8, 1},
      {kDLBfloat, 8, 1},
      {kDLOpaqueHandle, 64, 1},
      {kDLComplex, 64, 1},
      {kDLFloat8_e3m4, 8, 1},
      {kDLFloat8_e4m3, 8, 1},
      {kDLFloat8_e4m3b11fnuz, 8, 1},
      {kDLFloat8_e4m3fn, 16, 1},
      {kDLFloat4_e2m1fn, 8, 1},
      {kDLFloat6_e2m3fn, 6, 1},
      {kDLFloat6_e3m2fn, 6, 1},
      {kDLFloat, 32, 2},
      {kDLInt, 8, 0},
  };
  for (const auto& dtype : unsupported) {
    SCOPED_TRACE(static_cast<int>(dtype.code));
    SCOPED_TRACE(static_cast<int>(dtype.bits));
    DLManagedTensor packed{};
    packed.dl_tensor.device = {kDLCPU, 0};
    packed.dl_tensor.dtype = dtype;
    EXPECT_THROW(dlpack::DlpackToOrtValue(&packed), OnnxRuntimeException);
  }

  OrtValue strings;
  Tensor::InitOrtValue(DataTypeImpl::GetType<std::string>(), TensorShape{1},
                       std::make_shared<CPUAllocator>(), strings);
  EXPECT_THROW(dlpack::OrtValueToDlpack(strings), OnnxRuntimeException);
}

TEST(DlpackConverterTest, ImportWithoutDeleter) {
  bool data[] = {true, false};
  int64_t shape[] = {2};
  DLManagedTensor packed{};
  packed.dl_tensor = {data, {kDLCPU, 0}, 1, {kDLBool, 8, 1}, shape, nullptr, 0};
  {
    OrtValue imported = dlpack::DlpackToOrtValue(&packed);
    EXPECT_EQ(imported.Get<Tensor>().Data<bool>(), data);
    EXPECT_TRUE(imported.Get<Tensor>().Data<bool>()[0]);
    EXPECT_FALSE(imported.Get<Tensor>().Data<bool>()[1]);
  }
  EXPECT_TRUE(data[0]);
  EXPECT_FALSE(data[1]);
}

TEST(DlpackConverterTest, EmptyTensorWithOverflowingDimensionProduct) {
  const TensorShape shape{0, 4294967296LL, 4294967296LL};
  OrtValue original;
  Tensor::InitOrtValue(DataTypeImpl::GetType<float>(), shape,
                       std::make_shared<CPUAllocator>(), original);
  ASSERT_EQ(original.Get<Tensor>().SizeInBytes(), 0);
  auto* packed = dlpack::OrtValueToDlpack(original);
  ASSERT_NE(packed->dl_tensor.strides, nullptr);
  for (size_t i = 0; i < shape.NumDimensions(); ++i) {
    EXPECT_EQ(packed->dl_tensor.strides[i], 0);
  }
  OrtValue imported = dlpack::DlpackToOrtValue(packed);
  EXPECT_EQ(imported.Get<Tensor>().Shape(), shape);
  EXPECT_EQ(imported.Get<Tensor>().SizeInBytes(), 0);
}

#ifdef ENABLE_STRIDED_TENSORS
TEST(DlpackConverterTest, RejectNonContiguousExport) {
  OrtValue value;
  Tensor::InitOrtValue(DataTypeImpl::GetType<float>(), TensorShape{2, 3},
                       std::make_shared<CPUAllocator>(), value);
  auto* tensor = value.GetMutable<Tensor>();
  const int64_t strides[] = {0, 1};
  tensor->SetShapeAndStrides(TensorShape{2, 3}, strides);
  ASSERT_FALSE(tensor->IsContiguous());
  EXPECT_THROW(dlpack::OrtValueToDlpack(value), OnnxRuntimeException);
}
#endif

}  // namespace
}  // namespace onnxruntime::test

#endif  // ENABLE_DLPACK
