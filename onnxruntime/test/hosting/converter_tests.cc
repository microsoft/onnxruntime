// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"
#include "gmock/gmock.h"

#include "core/framework/tensor.h"
#include "core/framework/allocatormgr.h"
#include "test/framework/test_utils.h"
#include "test/test_environment.h"
#include "hosting/converter.h"

namespace onnxruntime {
namespace hosting {
namespace test {

IExecutionProvider* TestCPUExecutionProvider() {
  static CPUExecutionProviderInfo info;
  static CPUExecutionProvider cpu_provider(info);
  return &cpu_provider;
}

TEST(PositiveTests, MLDataTypeToTensorProtoDataTypeTests) {
  auto logger = ::onnxruntime::test::DefaultLoggingManager().DefaultLogger();

  MLDataType ml_data_type = DataTypeImpl::GetType<float>();
  onnx::TensorProto_DataType result = onnxruntime::hosting::MLDataTypeToTensorProtoDataType(ml_data_type, logger);
  EXPECT_EQ(result, onnx::TensorProto_DataType_FLOAT);

  ml_data_type = DataTypeImpl::GetType<onnxruntime::MLFloat16>();
  result = onnxruntime::hosting::MLDataTypeToTensorProtoDataType(ml_data_type, logger);
  EXPECT_EQ(result, onnx::TensorProto_DataType_FLOAT16);

  ml_data_type = DataTypeImpl::GetType<onnxruntime::BFloat16>();
  result = onnxruntime::hosting::MLDataTypeToTensorProtoDataType(ml_data_type, logger);
  EXPECT_EQ(result, onnx::TensorProto_DataType_BFLOAT16);

  ml_data_type = DataTypeImpl::GetType<double>();
  result = onnxruntime::hosting::MLDataTypeToTensorProtoDataType(ml_data_type, logger);
  EXPECT_EQ(result, onnx::TensorProto_DataType_DOUBLE);

  ml_data_type = DataTypeImpl::GetType<uint8_t>();
  result = onnxruntime::hosting::MLDataTypeToTensorProtoDataType(ml_data_type, logger);
  EXPECT_EQ(result, onnx::TensorProto_DataType_UINT8);

  ml_data_type = DataTypeImpl::GetType<int8_t>();
  result = onnxruntime::hosting::MLDataTypeToTensorProtoDataType(ml_data_type, logger);
  EXPECT_EQ(result, onnx::TensorProto_DataType_INT8);

  ml_data_type = DataTypeImpl::GetType<uint16_t>();
  result = onnxruntime::hosting::MLDataTypeToTensorProtoDataType(ml_data_type, logger);
  EXPECT_EQ(result, onnx::TensorProto_DataType_UINT16);

  ml_data_type = DataTypeImpl::GetType<int16_t>();
  result = onnxruntime::hosting::MLDataTypeToTensorProtoDataType(ml_data_type, logger);
  EXPECT_EQ(result, onnx::TensorProto_DataType_INT16);

  ml_data_type = DataTypeImpl::GetType<uint32_t>();
  result = onnxruntime::hosting::MLDataTypeToTensorProtoDataType(ml_data_type, logger);
  EXPECT_EQ(result, onnx::TensorProto_DataType_UINT32);

  ml_data_type = DataTypeImpl::GetType<int32_t>();
  result = onnxruntime::hosting::MLDataTypeToTensorProtoDataType(ml_data_type, logger);
  EXPECT_EQ(result, onnx::TensorProto_DataType_INT32);

  ml_data_type = DataTypeImpl::GetType<uint64_t>();
  result = onnxruntime::hosting::MLDataTypeToTensorProtoDataType(ml_data_type, logger);
  EXPECT_EQ(result, onnx::TensorProto_DataType_UINT64);

  ml_data_type = DataTypeImpl::GetType<int64_t>();
  result = onnxruntime::hosting::MLDataTypeToTensorProtoDataType(ml_data_type, logger);
  EXPECT_EQ(result, onnx::TensorProto_DataType_INT64);

  ml_data_type = DataTypeImpl::GetType<std::string>();
  result = onnxruntime::hosting::MLDataTypeToTensorProtoDataType(ml_data_type, logger);
  EXPECT_EQ(result, onnx::TensorProto_DataType_STRING);

  ml_data_type = DataTypeImpl::GetType<bool>();
  result = onnxruntime::hosting::MLDataTypeToTensorProtoDataType(ml_data_type, logger);
  EXPECT_EQ(result, onnx::TensorProto_DataType_BOOL);

  ml_data_type = DataTypeImpl::GetTensorType<bool>();
  result = onnxruntime::hosting::MLDataTypeToTensorProtoDataType(ml_data_type, logger);
  EXPECT_EQ(result, onnx::TensorProto_DataType_UNDEFINED);
}

TEST(PositiveTests, MLValue2TensorProtoTestsFloat2Raw) {
  auto logger = ::onnxruntime::test::DefaultLoggingManager().DefaultLogger();

  std::vector<int64_t> dims_mul_x = {3, 2};
  std::vector<float> values_mul_x = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  MLValue ml_value;
  onnxruntime::test::CreateMLValue<float>(TestCPUExecutionProvider()->GetAllocator(0, OrtMemTypeDefault), dims_mul_x, values_mul_x, &ml_value);

  onnx::TensorProto tp;
  common::Status status = onnxruntime::hosting::MLValue2TensorProto(ml_value, /* using_raw_data */ true, logger, tp);
  EXPECT_TRUE(status.IsOK());

  // Verify dimensions
  const auto& dims = tp.dims();
  std::vector<int64_t> tensor_shape_vec(static_cast<size_t>(dims.size()));
  for (int i = 0; i < dims.size(); ++i) {
    EXPECT_EQ(dims[i], dims_mul_x[i]);
  }

  // Verify data
  auto raw = tp.raw_data().data();
  auto raw_len = tp.raw_data().size();
  float* tensor_data = (float*)raw;
  for (size_t j = 0; j < raw_len / sizeof(float); ++j) {
    EXPECT_FLOAT_EQ(tensor_data[j], values_mul_x[j]);
  }
}

TEST(PositiveTests, MLValue2TensorProtoTestsInt322Raw) {
  auto logger = ::onnxruntime::test::DefaultLoggingManager().DefaultLogger();

  std::vector<int64_t> dims_mul_x = {3, 2};
  std::vector<int32_t> values_mul_x = {1, 2, 3, 4, 5, 6};
  MLValue ml_value;
  onnxruntime::test::CreateMLValue<int32_t>(TestCPUExecutionProvider()->GetAllocator(0, OrtMemTypeDefault), dims_mul_x, values_mul_x, &ml_value);

  onnx::TensorProto tp;
  common::Status status = onnxruntime::hosting::MLValue2TensorProto(ml_value, /* using_raw_data */ true, logger, tp);
  EXPECT_TRUE(status.IsOK());

  // Verify dimensions
  const auto& dims = tp.dims();
  std::vector<int64_t> tensor_shape_vec(static_cast<size_t>(dims.size()));
  for (int i = 0; i < dims.size(); ++i) {
    EXPECT_EQ(dims[i], dims_mul_x[i]);
  }

  // Verify data
  auto raw = tp.raw_data().data();
  auto raw_len = tp.raw_data().size();
  int32_t* tensor_data = (int32_t*)raw;
  for (size_t j = 0; j < raw_len / sizeof(int32_t); ++j) {
    EXPECT_EQ(tensor_data[j], values_mul_x[j]);
  }
}

TEST(PositiveTests, MLValue2TensorProtoTestsUInt322Raw) {
  auto logger = ::onnxruntime::test::DefaultLoggingManager().DefaultLogger();

  std::vector<int64_t> dims_mul_x = {3, 2};
  std::vector<uint32_t> values_mul_x = {1, 2, 3, 4, 5, 6};
  MLValue ml_value;
  onnxruntime::test::CreateMLValue<uint32_t>(TestCPUExecutionProvider()->GetAllocator(0, OrtMemTypeDefault), dims_mul_x, values_mul_x, &ml_value);

  onnx::TensorProto tp;
  common::Status status = onnxruntime::hosting::MLValue2TensorProto(ml_value, /* using_raw_data */ true, logger, tp);
  EXPECT_TRUE(status.IsOK());

  // Verify dimensions
  const auto& dims = tp.dims();
  std::vector<int64_t> tensor_shape_vec(static_cast<size_t>(dims.size()));
  for (int i = 0; i < dims.size(); ++i) {
    EXPECT_EQ(dims[i], dims_mul_x[i]);
  }

  // Verify data
  auto raw = tp.raw_data().data();
  auto raw_len = tp.raw_data().size();
  uint64_t* tensor_data = (uint64_t*)raw;
  for (size_t j = 0; j < raw_len / sizeof(uint64_t); ++j) {
    EXPECT_EQ(tensor_data[j] >> 32, values_mul_x[j * 2 + 1]);
    EXPECT_EQ(tensor_data[j] & 0xFFFFFFFF, values_mul_x[j * 2]);
  }
}

}  // namespace test
}  // namespace hosting
}  // namespace onnxruntime