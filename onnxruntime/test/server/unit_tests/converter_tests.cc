// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"
#include "gmock/gmock.h"

#include "core/framework/tensor.h"
#include "core/framework/allocatormgr.h"
#include "test/framework/test_utils.h"
#include "test/test_environment.h"
#include "server/converter.h"

namespace onnxruntime {
namespace server {
namespace test {

void CreateMLValueBool(AllocatorPtr alloc, const std::vector<int64_t>& dims, const bool* value, MLValue* p_mlvalue);

IExecutionProvider* TestCPUExecutionProvider() {
  static CPUExecutionProviderInfo info;
  static CPUExecutionProvider cpu_provider(info);
  return &cpu_provider;
}

TEST(MLDataTypeToTensorProtoDataTypeTests, MLDataTypeToTensorProtoDataTypeTests) {
  auto logger = std::make_unique<onnxruntime::logging::Logger>(::onnxruntime::test::DefaultLoggingManager().DefaultLogger());

  MLDataType ml_data_type = DataTypeImpl::GetType<float>();
  onnx::TensorProto_DataType result = onnxruntime::server::MLDataTypeToTensorProtoDataType(ml_data_type);
  EXPECT_EQ(result, onnx::TensorProto_DataType_FLOAT);

  ml_data_type = DataTypeImpl::GetType<onnxruntime::MLFloat16>();
  result = onnxruntime::server::MLDataTypeToTensorProtoDataType(ml_data_type);
  EXPECT_EQ(result, onnx::TensorProto_DataType_FLOAT16);

  ml_data_type = DataTypeImpl::GetType<onnxruntime::BFloat16>();
  result = onnxruntime::server::MLDataTypeToTensorProtoDataType(ml_data_type);
  EXPECT_EQ(result, onnx::TensorProto_DataType_BFLOAT16);

  ml_data_type = DataTypeImpl::GetType<double>();
  result = onnxruntime::server::MLDataTypeToTensorProtoDataType(ml_data_type);
  EXPECT_EQ(result, onnx::TensorProto_DataType_DOUBLE);

  ml_data_type = DataTypeImpl::GetType<uint8_t>();
  result = onnxruntime::server::MLDataTypeToTensorProtoDataType(ml_data_type);
  EXPECT_EQ(result, onnx::TensorProto_DataType_UINT8);

  ml_data_type = DataTypeImpl::GetType<int8_t>();
  result = onnxruntime::server::MLDataTypeToTensorProtoDataType(ml_data_type);
  EXPECT_EQ(result, onnx::TensorProto_DataType_INT8);

  ml_data_type = DataTypeImpl::GetType<uint16_t>();
  result = onnxruntime::server::MLDataTypeToTensorProtoDataType(ml_data_type);
  EXPECT_EQ(result, onnx::TensorProto_DataType_UINT16);

  ml_data_type = DataTypeImpl::GetType<int16_t>();
  result = onnxruntime::server::MLDataTypeToTensorProtoDataType(ml_data_type);
  EXPECT_EQ(result, onnx::TensorProto_DataType_INT16);

  ml_data_type = DataTypeImpl::GetType<uint32_t>();
  result = onnxruntime::server::MLDataTypeToTensorProtoDataType(ml_data_type);
  EXPECT_EQ(result, onnx::TensorProto_DataType_UINT32);

  ml_data_type = DataTypeImpl::GetType<int32_t>();
  result = onnxruntime::server::MLDataTypeToTensorProtoDataType(ml_data_type);
  EXPECT_EQ(result, onnx::TensorProto_DataType_INT32);

  ml_data_type = DataTypeImpl::GetType<uint64_t>();
  result = onnxruntime::server::MLDataTypeToTensorProtoDataType(ml_data_type);
  EXPECT_EQ(result, onnx::TensorProto_DataType_UINT64);

  ml_data_type = DataTypeImpl::GetType<int64_t>();
  result = onnxruntime::server::MLDataTypeToTensorProtoDataType(ml_data_type);
  EXPECT_EQ(result, onnx::TensorProto_DataType_INT64);

  ml_data_type = DataTypeImpl::GetType<std::string>();
  result = onnxruntime::server::MLDataTypeToTensorProtoDataType(ml_data_type);
  EXPECT_EQ(result, onnx::TensorProto_DataType_STRING);

  ml_data_type = DataTypeImpl::GetType<bool>();
  result = onnxruntime::server::MLDataTypeToTensorProtoDataType(ml_data_type);
  EXPECT_EQ(result, onnx::TensorProto_DataType_BOOL);

  ml_data_type = DataTypeImpl::GetTensorType<bool>();
  result = onnxruntime::server::MLDataTypeToTensorProtoDataType(ml_data_type);
  EXPECT_EQ(result, onnx::TensorProto_DataType_UNDEFINED);
}

TEST(MLValueToTensorProtoTests, FloatToRaw) {
  auto logger = std::make_unique<onnxruntime::logging::Logger>(::onnxruntime::test::DefaultLoggingManager().DefaultLogger());

  std::vector<int64_t> dims_mul_x = {3, 2};
  std::vector<float> values_mul_x = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  MLValue ml_value;
  onnxruntime::test::CreateMLValue<float>(TestCPUExecutionProvider()->GetAllocator(0, OrtMemTypeDefault), dims_mul_x, values_mul_x, &ml_value);

  onnx::TensorProto tp;
  common::Status status = onnxruntime::server::MLValueToTensorProto(ml_value, /* using_raw_data */ true, std::move(logger), tp);
  EXPECT_TRUE(status.IsOK());

  // Verify data type
  EXPECT_TRUE(tp.has_data_type());
  EXPECT_EQ(tp.data_type(), onnx::TensorProto_DataType_FLOAT);

  // Verify data location
  EXPECT_TRUE(tp.has_data_location());
  EXPECT_EQ(tp.data_location(), onnx::TensorProto_DataLocation_DEFAULT);

  // Verify dimensions
  const auto& dims = tp.dims();
  std::vector<int64_t> tensor_shape_vec(static_cast<size_t>(dims.size()));
  for (int i = 0; i < dims.size(); ++i) {
    EXPECT_EQ(dims[i], dims_mul_x[i]);
  }

  // Verify data
  EXPECT_TRUE(tp.has_raw_data());
  auto count = tp.raw_data().size() / sizeof(float);
  EXPECT_EQ(count, 6);

  auto raw = tp.raw_data().data();
  const auto* tensor_data = reinterpret_cast<const float*>(raw);
  for (size_t j = 0; j < count; ++j) {
    EXPECT_FLOAT_EQ(tensor_data[j], values_mul_x[j]);
  }
}

TEST(MLValueToTensorProtoTests, FloatToFloatData) {
  auto logger = std::make_unique<onnxruntime::logging::Logger>(::onnxruntime::test::DefaultLoggingManager().DefaultLogger());

  std::vector<int64_t> dims_mul_x = {3, 2};
  std::vector<float> values_mul_x = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  MLValue ml_value;
  onnxruntime::test::CreateMLValue<float>(TestCPUExecutionProvider()->GetAllocator(0, OrtMemTypeDefault), dims_mul_x, values_mul_x, &ml_value);

  onnx::TensorProto tp;
  common::Status status = onnxruntime::server::MLValueToTensorProto(ml_value, /* using_raw_data */ false, std::move(logger), tp);
  EXPECT_TRUE(status.IsOK());

  // Verify data type
  EXPECT_TRUE(tp.has_data_type());
  EXPECT_EQ(tp.data_type(), onnx::TensorProto_DataType_FLOAT);

  // Verify data location
  EXPECT_FALSE(tp.has_data_location());

  // Verify dimensions
  const auto& dims = tp.dims();
  std::vector<int64_t> tensor_shape_vec(static_cast<size_t>(dims.size()));
  for (int i = 0; i < dims.size(); ++i) {
    EXPECT_EQ(dims[i], dims_mul_x[i]);
  }

  // Verify data
  EXPECT_FALSE(tp.has_raw_data());
  EXPECT_EQ(tp.float_data().size(), 6);
  auto data = tp.float_data().data();
  for (int j = 0; j < tp.float_data().size(); ++j) {
    EXPECT_FLOAT_EQ(data[j], values_mul_x[j]);
  }
}

TEST(MLValueToTensorProtoTests, Int32ToRaw) {
  auto logger = std::make_unique<onnxruntime::logging::Logger>(::onnxruntime::test::DefaultLoggingManager().DefaultLogger());

  std::vector<int64_t> dims_mul_x = {3, 2};
  std::vector<int32_t> values_mul_x = {1, 2, 3, 4, 5, 6};
  MLValue ml_value;
  onnxruntime::test::CreateMLValue<int32_t>(TestCPUExecutionProvider()->GetAllocator(0, OrtMemTypeDefault), dims_mul_x, values_mul_x, &ml_value);

  onnx::TensorProto tp;
  common::Status status = onnxruntime::server::MLValueToTensorProto(ml_value, /* using_raw_data */ true, std::move(logger), tp);
  EXPECT_TRUE(status.IsOK());

  // Verify data type
  EXPECT_TRUE(tp.has_data_type());
  EXPECT_EQ(tp.data_type(), onnx::TensorProto_DataType_INT32);

  // Verify data location
  EXPECT_TRUE(tp.has_data_location());
  EXPECT_EQ(tp.data_location(), onnx::TensorProto_DataLocation_DEFAULT);

  // Verify dimensions
  const auto& dims = tp.dims();
  std::vector<int64_t> tensor_shape_vec(static_cast<size_t>(dims.size()));
  for (int i = 0; i < dims.size(); ++i) {
    EXPECT_EQ(dims[i], dims_mul_x[i]);
  }

  // Verify data
  EXPECT_TRUE(tp.has_raw_data());
  auto count = tp.raw_data().size() / sizeof(int32_t);
  EXPECT_EQ(count, 6);

  auto raw = tp.raw_data().data();
  const auto* tensor_data = reinterpret_cast<const int32_t*>(raw);
  for (size_t j = 0; j < count; ++j) {
    EXPECT_EQ(tensor_data[j], values_mul_x[j]);
  }
}

TEST(MLValueToTensorProtoTests, Int32ToInt32Data) {
  auto logger = std::make_unique<onnxruntime::logging::Logger>(::onnxruntime::test::DefaultLoggingManager().DefaultLogger());

  std::vector<int64_t> dims_mul_x = {3, 2};
  std::vector<int32_t> values_mul_x = {1, 2, 3, 4, 5, 6};
  MLValue ml_value;
  onnxruntime::test::CreateMLValue<int32_t>(TestCPUExecutionProvider()->GetAllocator(0, OrtMemTypeDefault), dims_mul_x, values_mul_x, &ml_value);

  onnx::TensorProto tp;
  common::Status status = onnxruntime::server::MLValueToTensorProto(ml_value, /* using_raw_data */ false, std::move(logger), tp);
  EXPECT_TRUE(status.IsOK());

  // Verify data type
  EXPECT_TRUE(tp.has_data_type());
  EXPECT_EQ(tp.data_type(), onnx::TensorProto_DataType_INT32);

  // Verify data location
  EXPECT_FALSE(tp.has_data_location());

  // Verify dimensions
  const auto& dims = tp.dims();
  std::vector<int64_t> tensor_shape_vec(static_cast<size_t>(dims.size()));
  for (int i = 0; i < dims.size(); ++i) {
    EXPECT_EQ(dims[i], dims_mul_x[i]);
  }

  // Verify data
  EXPECT_FALSE(tp.has_raw_data());
  EXPECT_EQ(tp.int32_data().size(), 6);
  auto data = tp.int32_data().data();
  for (int j = 0; j < tp.int32_data().size(); ++j) {
    EXPECT_EQ(data[j], values_mul_x[j]);
  }
}

TEST(MLValueToTensorProtoTests, UInt8ToRaw) {
  auto logger = std::make_unique<onnxruntime::logging::Logger>(::onnxruntime::test::DefaultLoggingManager().DefaultLogger());

  std::vector<int64_t> dims_mul_x = {3, 2};
  std::vector<uint8_t> values_mul_x = {1, 2, 3, 4, 5, 6};
  MLValue ml_value;
  onnxruntime::test::CreateMLValue<uint8_t>(TestCPUExecutionProvider()->GetAllocator(0, OrtMemTypeDefault), dims_mul_x, values_mul_x, &ml_value);

  onnx::TensorProto tp;
  common::Status status = onnxruntime::server::MLValueToTensorProto(ml_value, /* using_raw_data */ true, std::move(logger), tp);
  EXPECT_TRUE(status.IsOK());

  // Verify data type
  EXPECT_TRUE(tp.has_data_type());
  EXPECT_EQ(tp.data_type(), onnx::TensorProto_DataType_UINT8);

  // Verify data location
  EXPECT_TRUE(tp.has_data_location());
  EXPECT_EQ(tp.data_location(), onnx::TensorProto_DataLocation_DEFAULT);

  // Verify dimensions
  const auto& dims = tp.dims();
  std::vector<int64_t> tensor_shape_vec(static_cast<size_t>(dims.size()));
  for (int i = 0; i < dims.size(); ++i) {
    EXPECT_EQ(dims[i], dims_mul_x[i]);
  }

  // Verify data
  EXPECT_TRUE(tp.has_raw_data());
  auto count = tp.raw_data().size() / sizeof(uint8_t);
  EXPECT_EQ(count, 6);

  auto raw = tp.raw_data().data();
  const auto* tensor_data = reinterpret_cast<const uint8_t*>(raw);
  for (size_t j = 0; j < count; ++j) {
    EXPECT_EQ(tensor_data[j], values_mul_x[j]);
  }
}

TEST(MLValueToTensorProtoTests, UInt8ToInt32Data) {
  auto logger = std::make_unique<onnxruntime::logging::Logger>(::onnxruntime::test::DefaultLoggingManager().DefaultLogger());

  std::vector<int64_t> dims_mul_x = {3, 2};
  std::vector<uint8_t> values_mul_x = {1, 2, 3, 4, 5, 6};
  MLValue ml_value;
  onnxruntime::test::CreateMLValue<uint8_t>(TestCPUExecutionProvider()->GetAllocator(0, OrtMemTypeDefault), dims_mul_x, values_mul_x, &ml_value);

  onnx::TensorProto tp;
  common::Status status = onnxruntime::server::MLValueToTensorProto(ml_value, /* using_raw_data */ false, std::move(logger), tp);
  EXPECT_TRUE(status.IsOK());

  // Verify data type
  EXPECT_TRUE(tp.has_data_type());
  EXPECT_EQ(tp.data_type(), onnx::TensorProto_DataType_UINT8);

  // Verify data location
  EXPECT_FALSE(tp.has_data_location());

  // Verify dimensions
  const auto& dims = tp.dims();
  std::vector<int64_t> tensor_shape_vec(static_cast<size_t>(dims.size()));
  for (int i = 0; i < dims.size(); ++i) {
    EXPECT_EQ(dims[i], dims_mul_x[i]);
  }

  // Verify data
  EXPECT_FALSE(tp.has_raw_data());
  auto count = tp.int32_data().size() * (sizeof(int32_t) / sizeof(uint8_t));
  EXPECT_EQ(count, 8);
  auto data = tp.int32_data().data();
  const auto* data8 = reinterpret_cast<const uint8_t*>(data);
  for (int x = 0; x < 6; ++x) {
    EXPECT_EQ(data8[x], values_mul_x[x]);
  }
}

TEST(MLValueToTensorProtoTests, Int8ToRaw) {
  auto logger = std::make_unique<onnxruntime::logging::Logger>(::onnxruntime::test::DefaultLoggingManager().DefaultLogger());

  std::vector<int64_t> dims_mul_x = {3, 2};
  std::vector<int8_t> values_mul_x = {1, 2, 3, 4, 5, 6};
  MLValue ml_value;
  onnxruntime::test::CreateMLValue<int8_t>(TestCPUExecutionProvider()->GetAllocator(0, OrtMemTypeDefault), dims_mul_x, values_mul_x, &ml_value);

  onnx::TensorProto tp;
  common::Status status = onnxruntime::server::MLValueToTensorProto(ml_value, /* using_raw_data */ true, std::move(logger), tp);
  EXPECT_TRUE(status.IsOK());

  // Verify data type
  EXPECT_TRUE(tp.has_data_type());
  EXPECT_EQ(tp.data_type(), onnx::TensorProto_DataType_INT8);

  // Verify data location
  EXPECT_TRUE(tp.has_data_location());
  EXPECT_EQ(tp.data_location(), onnx::TensorProto_DataLocation_DEFAULT);

  // Verify dimensions
  const auto& dims = tp.dims();
  std::vector<int64_t> tensor_shape_vec(static_cast<size_t>(dims.size()));
  for (int i = 0; i < dims.size(); ++i) {
    EXPECT_EQ(dims[i], dims_mul_x[i]);
  }

  // Verify data
  EXPECT_TRUE(tp.has_raw_data());
  auto count = tp.raw_data().size() / sizeof(uint8_t);
  EXPECT_EQ(count, 6);

  auto raw = tp.raw_data().data();
  const auto* tensor_data = reinterpret_cast<const int8_t*>(raw);
  for (size_t j = 0; j < count; ++j) {
    EXPECT_EQ(tensor_data[j], values_mul_x[j]);
  }
}

TEST(MLValueToTensorProtoTests, Int8ToInt32Data) {
  auto logger = std::make_unique<onnxruntime::logging::Logger>(::onnxruntime::test::DefaultLoggingManager().DefaultLogger());

  std::vector<int64_t> dims_mul_x = {3, 2};
  std::vector<int8_t> values_mul_x = {1, 2, 3, 4, 5, 6};
  MLValue ml_value;
  onnxruntime::test::CreateMLValue<int8_t>(TestCPUExecutionProvider()->GetAllocator(0, OrtMemTypeDefault), dims_mul_x, values_mul_x, &ml_value);

  onnx::TensorProto tp;
  common::Status status = onnxruntime::server::MLValueToTensorProto(ml_value, /* using_raw_data */ false, std::move(logger), tp);
  EXPECT_TRUE(status.IsOK());

  // Verify data type
  EXPECT_TRUE(tp.has_data_type());
  EXPECT_EQ(tp.data_type(), onnx::TensorProto_DataType_INT8);

  // Verify data location
  EXPECT_FALSE(tp.has_data_location());

  // Verify dimensions
  const auto& dims = tp.dims();
  std::vector<int64_t> tensor_shape_vec(static_cast<size_t>(dims.size()));
  for (int i = 0; i < dims.size(); ++i) {
    EXPECT_EQ(dims[i], dims_mul_x[i]);
  }

  // Verify data
  EXPECT_FALSE(tp.has_raw_data());
  auto count = tp.int32_data().size();
  EXPECT_EQ(count, 2);
  auto data = tp.int32_data().data();
  const auto* data8 = reinterpret_cast<const int8_t*>(data);
  for (int x = 0; x < 6; ++x) {
    EXPECT_EQ(data8[x], values_mul_x[x]);
  }
}

TEST(MLValueToTensorProtoTests, UInt16ToRaw) {
  auto logger = std::make_unique<onnxruntime::logging::Logger>(::onnxruntime::test::DefaultLoggingManager().DefaultLogger());

  std::vector<int64_t> dims_mul_x = {3, 3};
  std::vector<uint16_t> values_mul_x = {1, 2, 3, 4, 5, 6, 7, 8, 9};
  MLValue ml_value;
  onnxruntime::test::CreateMLValue<uint16_t>(TestCPUExecutionProvider()->GetAllocator(0, OrtMemTypeDefault), dims_mul_x, values_mul_x, &ml_value);

  onnx::TensorProto tp;
  common::Status status = onnxruntime::server::MLValueToTensorProto(ml_value, /* using_raw_data */ true, std::move(logger), tp);
  EXPECT_TRUE(status.IsOK());

  // Verify data type
  EXPECT_TRUE(tp.has_data_type());
  EXPECT_EQ(tp.data_type(), onnx::TensorProto_DataType_UINT16);

  // Verify data location
  EXPECT_TRUE(tp.has_data_location());
  EXPECT_EQ(tp.data_location(), onnx::TensorProto_DataLocation_DEFAULT);

  // Verify dimensions
  const auto& dims = tp.dims();
  std::vector<int64_t> tensor_shape_vec(static_cast<size_t>(dims.size()));
  for (int i = 0; i < dims.size(); ++i) {
    EXPECT_EQ(dims[i], dims_mul_x[i]);
  }

  // Verify data
  EXPECT_TRUE(tp.has_raw_data());
  auto count = tp.raw_data().size() / sizeof(uint16_t);
  EXPECT_EQ(count, 9);

  auto raw = tp.raw_data().data();
  const auto* tensor_data = reinterpret_cast<const uint16_t*>(raw);
  for (size_t j = 0; j < count; ++j) {
    EXPECT_EQ(tensor_data[j], values_mul_x[j]);
  }
}

TEST(MLValueToTensorProtoTests, UInt16ToInt32Data) {
  auto logger = std::make_unique<onnxruntime::logging::Logger>(::onnxruntime::test::DefaultLoggingManager().DefaultLogger());

  std::vector<int64_t> dims_mul_x = {3, 3};
  std::vector<uint16_t> values_mul_x = {1, 2, 3, 4, 5, 6, 7, 8, 9};
  MLValue ml_value;
  onnxruntime::test::CreateMLValue<uint16_t>(TestCPUExecutionProvider()->GetAllocator(0, OrtMemTypeDefault), dims_mul_x, values_mul_x, &ml_value);

  onnx::TensorProto tp;
  common::Status status = onnxruntime::server::MLValueToTensorProto(ml_value, /* using_raw_data */ false, std::move(logger), tp);
  EXPECT_TRUE(status.IsOK());

  // Verify data type
  EXPECT_TRUE(tp.has_data_type());
  EXPECT_EQ(tp.data_type(), onnx::TensorProto_DataType_UINT16);

  // Verify data location
  EXPECT_FALSE(tp.has_data_location());

  // Verify dimensions
  const auto& dims = tp.dims();
  std::vector<int64_t> tensor_shape_vec(static_cast<size_t>(dims.size()));
  for (int i = 0; i < dims.size(); ++i) {
    EXPECT_EQ(dims[i], dims_mul_x[i]);
  }

  // Verify data
  EXPECT_FALSE(tp.has_raw_data());
  auto count = tp.int32_data().size();
  EXPECT_EQ(count, 5);
  auto data = tp.int32_data().data();
  const auto* data16 = reinterpret_cast<const uint16_t*>(data);
  for (int x = 0; x < 9; ++x) {
    EXPECT_EQ(data16[x], values_mul_x[x]);
  }
}

TEST(MLValueToTensorProtoTests, Int16ToRaw) {
  auto logger = std::make_unique<onnxruntime::logging::Logger>(::onnxruntime::test::DefaultLoggingManager().DefaultLogger());

  std::vector<int64_t> dims_mul_x = {3, 2};
  std::vector<int16_t> values_mul_x = {1, 2, 3, 4, 5, 6};
  MLValue ml_value;
  onnxruntime::test::CreateMLValue<int16_t>(TestCPUExecutionProvider()->GetAllocator(0, OrtMemTypeDefault), dims_mul_x, values_mul_x, &ml_value);

  onnx::TensorProto tp;
  common::Status status = onnxruntime::server::MLValueToTensorProto(ml_value, /* using_raw_data */ true, std::move(logger), tp);
  EXPECT_TRUE(status.IsOK());

  // Verify data type
  EXPECT_TRUE(tp.has_data_type());
  EXPECT_EQ(tp.data_type(), onnx::TensorProto_DataType_INT16);

  // Verify data location
  EXPECT_TRUE(tp.has_data_location());
  EXPECT_EQ(tp.data_location(), onnx::TensorProto_DataLocation_DEFAULT);

  // Verify dimensions
  const auto& dims = tp.dims();
  std::vector<int64_t> tensor_shape_vec(static_cast<size_t>(dims.size()));
  for (int i = 0; i < dims.size(); ++i) {
    EXPECT_EQ(dims[i], dims_mul_x[i]);
  }

  // Verify data
  EXPECT_TRUE(tp.has_raw_data());
  auto count = tp.raw_data().size() / sizeof(uint16_t);
  EXPECT_EQ(count, 6);

  auto raw = tp.raw_data().data();
  const auto* tensor_data = reinterpret_cast<const int16_t*>(raw);
  for (size_t j = 0; j < count; ++j) {
    EXPECT_EQ(tensor_data[j], values_mul_x[j]);
  }
}

TEST(MLValueToTensorProtoTests, Int16ToInt32Data) {
  auto logger = std::make_unique<onnxruntime::logging::Logger>(::onnxruntime::test::DefaultLoggingManager().DefaultLogger());

  std::vector<int64_t> dims_mul_x = {3, 2};
  std::vector<int16_t> values_mul_x = {1, 2, 3, 4, 5, 6};
  MLValue ml_value;
  onnxruntime::test::CreateMLValue<int16_t>(TestCPUExecutionProvider()->GetAllocator(0, OrtMemTypeDefault), dims_mul_x, values_mul_x, &ml_value);

  onnx::TensorProto tp;
  common::Status status = onnxruntime::server::MLValueToTensorProto(ml_value, /* using_raw_data */ false, std::move(logger), tp);
  EXPECT_TRUE(status.IsOK());

  // Verify data type
  EXPECT_TRUE(tp.has_data_type());
  EXPECT_EQ(tp.data_type(), onnx::TensorProto_DataType_INT16);

  // Verify data location
  EXPECT_FALSE(tp.has_data_location());

  // Verify dimensions
  const auto& dims = tp.dims();
  std::vector<int64_t> tensor_shape_vec(static_cast<size_t>(dims.size()));
  for (int i = 0; i < dims.size(); ++i) {
    EXPECT_EQ(dims[i], dims_mul_x[i]);
  }

  // Verify data
  EXPECT_FALSE(tp.has_raw_data());
  auto count = tp.int32_data().size() * (sizeof(int32_t) / sizeof(int16_t));
  EXPECT_EQ(count, 6);
  auto data = tp.int32_data().data();
  const auto* data16 = reinterpret_cast<const int16_t*>(data);
  for (int x = 0; x < 6; ++x) {
    EXPECT_EQ(data16[x], values_mul_x[x]);
  }
}

TEST(MLValueToTensorProtoTests, BoolToRaw) {
  auto logger = std::make_unique<onnxruntime::logging::Logger>(::onnxruntime::test::DefaultLoggingManager().DefaultLogger());

  std::vector<int64_t> dims_mul_x = {3, 2};
  bool values_mul_x[] = {true, false, false, true, true, false};
  MLValue ml_value;
  CreateMLValueBool(TestCPUExecutionProvider()->GetAllocator(0, OrtMemTypeDefault), dims_mul_x, values_mul_x, &ml_value);

  onnx::TensorProto tp;
  common::Status status = onnxruntime::server::MLValueToTensorProto(ml_value, /* using_raw_data */ true, std::move(logger), tp);
  EXPECT_TRUE(status.IsOK());

  // Verify data type
  EXPECT_TRUE(tp.has_data_type());
  EXPECT_EQ(tp.data_type(), onnx::TensorProto_DataType_BOOL);

  // Verify data location
  EXPECT_TRUE(tp.has_data_location());
  EXPECT_EQ(tp.data_location(), onnx::TensorProto_DataLocation_DEFAULT);

  // Verify dimensions
  const auto& dims = tp.dims();
  std::vector<int64_t> tensor_shape_vec(static_cast<size_t>(dims.size()));
  for (int i = 0; i < dims.size(); ++i) {
    EXPECT_EQ(dims[i], dims_mul_x[i]);
  }

  // Verify data
  EXPECT_TRUE(tp.has_raw_data());
  auto count = tp.raw_data().size() / sizeof(bool);
  EXPECT_EQ(count, 6);

  auto raw = tp.raw_data().data();
  const auto* tensor_data = reinterpret_cast<const bool*>(raw);
  for (size_t j = 0; j < count; ++j) {
    EXPECT_EQ(tensor_data[j], values_mul_x[j]);
  }
}

TEST(MLValueToTensorProtoTests, BoolToInt32Data) {
  auto logger = std::make_unique<onnxruntime::logging::Logger>(::onnxruntime::test::DefaultLoggingManager().DefaultLogger());

  std::vector<int64_t> dims_mul_x = {3, 2};
  bool values_mul_x[] = {true, false, false, true, true, false};
  MLValue ml_value;
  CreateMLValueBool(TestCPUExecutionProvider()->GetAllocator(0, OrtMemTypeDefault), dims_mul_x, values_mul_x, &ml_value);

  onnx::TensorProto tp;
  common::Status status = onnxruntime::server::MLValueToTensorProto(ml_value, /* using_raw_data */ false, std::move(logger), tp);
  EXPECT_TRUE(status.IsOK());

  // Verify data type
  EXPECT_TRUE(tp.has_data_type());
  EXPECT_EQ(tp.data_type(), onnx::TensorProto_DataType_BOOL);

  // Verify data location
  EXPECT_FALSE(tp.has_data_location());

  // Verify dimensions
  const auto& dims = tp.dims();
  std::vector<int64_t> tensor_shape_vec(static_cast<size_t>(dims.size()));
  for (int i = 0; i < dims.size(); ++i) {
    EXPECT_EQ(dims[i], dims_mul_x[i]);
  }

  // Verify data
  EXPECT_FALSE(tp.has_raw_data());
  auto count = tp.int32_data().size();
  EXPECT_EQ(count, 2);
  auto data = tp.int32_data().data();
  const auto* data16 = reinterpret_cast<const bool*>(data);
  for (int x = 0; x < 6; ++x) {
    EXPECT_EQ(data16[x], values_mul_x[x]);
  }
}

TEST(MLValueToTensorProtoTests, Float16ToRaw) {
  auto logger = std::make_unique<onnxruntime::logging::Logger>(::onnxruntime::test::DefaultLoggingManager().DefaultLogger());

  std::vector<int64_t> dims_mul_x = {3, 2};
  std::vector<onnxruntime::MLFloat16> values_mul_x{
      onnxruntime::MLFloat16(1),
      onnxruntime::MLFloat16(2),
      onnxruntime::MLFloat16(3),
      onnxruntime::MLFloat16(4),
      onnxruntime::MLFloat16(5),
      onnxruntime::MLFloat16(6)};
  MLValue ml_value;
  onnxruntime::test::CreateMLValue<onnxruntime::MLFloat16>(TestCPUExecutionProvider()->GetAllocator(0, OrtMemTypeDefault), dims_mul_x, values_mul_x, &ml_value);

  onnx::TensorProto tp;
  common::Status status = onnxruntime::server::MLValueToTensorProto(ml_value, /* using_raw_data */ true, std::move(logger), tp);
  EXPECT_TRUE(status.IsOK());

  // Verify data type
  EXPECT_TRUE(tp.has_data_type());
  EXPECT_EQ(tp.data_type(), onnx::TensorProto_DataType_FLOAT16);

  // Verify data location
  EXPECT_TRUE(tp.has_data_location());
  EXPECT_EQ(tp.data_location(), onnx::TensorProto_DataLocation_DEFAULT);

  // Verify dimensions
  const auto& dims = tp.dims();
  std::vector<int64_t> tensor_shape_vec(static_cast<size_t>(dims.size()));
  for (int i = 0; i < dims.size(); ++i) {
    EXPECT_EQ(dims[i], dims_mul_x[i]);
  }

  // Verify data
  EXPECT_TRUE(tp.has_raw_data());
  auto count = tp.raw_data().size() / sizeof(onnxruntime::MLFloat16);
  EXPECT_EQ(count, 6);

  auto raw = tp.raw_data().data();
  const auto* tensor_data = reinterpret_cast<const onnxruntime::MLFloat16*>(raw);
  for (size_t j = 0; j < count; ++j) {
    EXPECT_EQ(tensor_data[j], values_mul_x[j]);
  }
}

TEST(MLValueToTensorProtoTests, FloatToInt32Data) {
  auto logger = std::make_unique<onnxruntime::logging::Logger>(::onnxruntime::test::DefaultLoggingManager().DefaultLogger());

  std::vector<int64_t> dims_mul_x = {3, 2};
  std::vector<onnxruntime::MLFloat16> values_mul_x{
      onnxruntime::MLFloat16(1),
      onnxruntime::MLFloat16(2),
      onnxruntime::MLFloat16(3),
      onnxruntime::MLFloat16(4),
      onnxruntime::MLFloat16(5),
      onnxruntime::MLFloat16(6)};
  MLValue ml_value;
  onnxruntime::test::CreateMLValue<onnxruntime::MLFloat16>(TestCPUExecutionProvider()->GetAllocator(0, OrtMemTypeDefault), dims_mul_x, values_mul_x, &ml_value);

  onnx::TensorProto tp;
  common::Status status = onnxruntime::server::MLValueToTensorProto(ml_value, /* using_raw_data */ false, std::move(logger), tp);
  EXPECT_TRUE(status.IsOK());

  // Verify data type
  EXPECT_TRUE(tp.has_data_type());
  EXPECT_EQ(tp.data_type(), onnx::TensorProto_DataType_FLOAT16);

  // Verify data location
  EXPECT_FALSE(tp.has_data_location());

  // Verify dimensions
  const auto& dims = tp.dims();
  std::vector<int64_t> tensor_shape_vec(static_cast<size_t>(dims.size()));
  for (int i = 0; i < dims.size(); ++i) {
    EXPECT_EQ(dims[i], dims_mul_x[i]);
  }

  // Verify data
  EXPECT_FALSE(tp.has_raw_data());
  auto count = tp.int32_data().size();
  EXPECT_EQ(count, 3);
  auto data = tp.int32_data().data();
  const auto* data16 = reinterpret_cast<const onnxruntime::MLFloat16*>(data);
  for (int x = 0; x < 6; ++x) {
    EXPECT_EQ(data16[x], values_mul_x[x]);
  }
}

TEST(MLValueToTensorProtoTests, BFloat16ToRaw) {
  auto logger = std::make_unique<onnxruntime::logging::Logger>(::onnxruntime::test::DefaultLoggingManager().DefaultLogger());

  std::vector<int64_t> dims_mul_x = {3, 2};
  std::vector<onnxruntime::BFloat16> values_mul_x{
      onnxruntime::BFloat16(1.0f),
      onnxruntime::BFloat16(2.0f),
      onnxruntime::BFloat16(3.0f),
      onnxruntime::BFloat16(4.0f),
      onnxruntime::BFloat16(5.0f),
      onnxruntime::BFloat16(6.0f)};
  MLValue ml_value;
  onnxruntime::test::CreateMLValue<onnxruntime::BFloat16>(TestCPUExecutionProvider()->GetAllocator(0, OrtMemTypeDefault), dims_mul_x, values_mul_x, &ml_value);

  onnx::TensorProto tp;
  common::Status status = onnxruntime::server::MLValueToTensorProto(ml_value, /* using_raw_data */ true, std::move(logger), tp);
  EXPECT_TRUE(status.IsOK());

  // Verify data type
  EXPECT_TRUE(tp.has_data_type());
  EXPECT_EQ(tp.data_type(), onnx::TensorProto_DataType_BFLOAT16);

  // Verify data location
  EXPECT_TRUE(tp.has_data_location());
  EXPECT_EQ(tp.data_location(), onnx::TensorProto_DataLocation_DEFAULT);

  // Verify dimensions
  const auto& dims = tp.dims();
  std::vector<int64_t> tensor_shape_vec(static_cast<size_t>(dims.size()));
  for (int i = 0; i < dims.size(); ++i) {
    EXPECT_EQ(dims[i], dims_mul_x[i]);
  }

  // Verify data
  EXPECT_TRUE(tp.has_raw_data());
  auto count = tp.raw_data().size() / sizeof(uint16_t);
  EXPECT_EQ(count, 6);

  auto raw = tp.raw_data().data();
  const auto* tensor_data = reinterpret_cast<const uint16_t*>(raw);
  for (size_t j = 0; j < count; ++j) {
    EXPECT_EQ(tensor_data[j], values_mul_x[j].val);
  }
}

TEST(MLValueToTensorProtoTests, BFloatToInt32Data) {
  auto logger = std::make_unique<onnxruntime::logging::Logger>(::onnxruntime::test::DefaultLoggingManager().DefaultLogger());

  std::vector<int64_t> dims_mul_x = {3, 2};
  std::vector<onnxruntime::BFloat16> values_mul_x{
      onnxruntime::BFloat16(1.0f),
      onnxruntime::BFloat16(2.0f),
      onnxruntime::BFloat16(3.0f),
      onnxruntime::BFloat16(4.0f),
      onnxruntime::BFloat16(5.0f),
      onnxruntime::BFloat16(6.0f)};
  MLValue ml_value;
  onnxruntime::test::CreateMLValue<onnxruntime::BFloat16>(TestCPUExecutionProvider()->GetAllocator(0, OrtMemTypeDefault), dims_mul_x, values_mul_x, &ml_value);

  onnx::TensorProto tp;
  common::Status status = onnxruntime::server::MLValueToTensorProto(ml_value, /* using_raw_data */ false, std::move(logger), tp);
  EXPECT_TRUE(status.IsOK());

  // Verify data type
  EXPECT_TRUE(tp.has_data_type());
  EXPECT_EQ(tp.data_type(), onnx::TensorProto_DataType_BFLOAT16);

  // Verify data location
  EXPECT_FALSE(tp.has_data_location());

  // Verify dimensions
  const auto& dims = tp.dims();
  std::vector<int64_t> tensor_shape_vec(static_cast<size_t>(dims.size()));
  for (int i = 0; i < dims.size(); ++i) {
    EXPECT_EQ(dims[i], dims_mul_x[i]);
  }

  // Verify data
  EXPECT_FALSE(tp.has_raw_data());
  auto count = tp.int32_data().size();
  EXPECT_EQ(count, 3);
  auto data = tp.int32_data().data();
  const auto* data16 = reinterpret_cast<const uint16_t*>(data);
  for (int x = 0; x < 6; ++x) {
    EXPECT_EQ(data16[x], values_mul_x[x].val);
  }
}

TEST(MLValueToTensorProtoTests, StringToStringData) {
  auto logger = std::make_unique<onnxruntime::logging::Logger>(::onnxruntime::test::DefaultLoggingManager().DefaultLogger());

  std::vector<int64_t> dims_mul_x = {3, 2};
  std::vector<std::string> values_mul_x{"A", "BC", "DEF", "123", "45", "6"};
  MLValue ml_value;
  onnxruntime::test::AllocateMLValue<std::string>(TestCPUExecutionProvider()->GetAllocator(0, OrtMemTypeDefault), dims_mul_x, &ml_value);

  Tensor* mutable_tensor = ml_value.GetMutable<Tensor>();
  std::string* mutable_data = mutable_tensor->MutableData<std::string>();
  for (size_t i = 0; i < values_mul_x.size(); ++i) {
    mutable_data[i] = values_mul_x[i];
  }

  onnx::TensorProto tp;
  common::Status status = onnxruntime::server::MLValueToTensorProto(ml_value, /* using_raw_data */ false, std::move(logger), tp);
  EXPECT_TRUE(status.IsOK());

  // Verify data type
  EXPECT_TRUE(tp.has_data_type());
  EXPECT_EQ(tp.data_type(), onnx::TensorProto_DataType_STRING);

  // Verify data location
  EXPECT_FALSE(tp.has_data_location());

  // Verify dimensions
  const auto& dims = tp.dims();
  std::vector<int64_t> tensor_shape_vec(static_cast<size_t>(dims.size()));
  for (int i = 0; i < dims.size(); ++i) {
    EXPECT_EQ(dims[i], dims_mul_x[i]);
  }

  // Verify data
  EXPECT_FALSE(tp.has_raw_data());
  auto count = tp.string_data().size();
  EXPECT_EQ(count, 6);
  const auto* data = tp.string_data().data();
  for (int x = 0; x < 6; ++x) {
    EXPECT_STREQ(data[x]->c_str(), values_mul_x[x].c_str());
  }
}

TEST(MLValueToTensorProtoTests, Int64ToRaw) {
  auto logger = std::make_unique<onnxruntime::logging::Logger>(::onnxruntime::test::DefaultLoggingManager().DefaultLogger());

  std::vector<int64_t> dims_mul_x = {3, 2};
  std::vector<int64_t> values_mul_x = {1, 2, 3, 4, 5, 6};
  MLValue ml_value;
  onnxruntime::test::CreateMLValue<int64_t>(TestCPUExecutionProvider()->GetAllocator(0, OrtMemTypeDefault), dims_mul_x, values_mul_x, &ml_value);

  onnx::TensorProto tp;
  common::Status status = onnxruntime::server::MLValueToTensorProto(ml_value, /* using_raw_data */ true, std::move(logger), tp);
  EXPECT_TRUE(status.IsOK());

  // Verify data type
  EXPECT_TRUE(tp.has_data_type());
  EXPECT_EQ(tp.data_type(), onnx::TensorProto_DataType_INT64);

  // Verify data location
  EXPECT_TRUE(tp.has_data_location());
  EXPECT_EQ(tp.data_location(), onnx::TensorProto_DataLocation_DEFAULT);

  // Verify dimensions
  const auto& dims = tp.dims();
  std::vector<int64_t> tensor_shape_vec(static_cast<size_t>(dims.size()));
  for (int i = 0; i < dims.size(); ++i) {
    EXPECT_EQ(dims[i], dims_mul_x[i]);
  }

  // Verify data
  EXPECT_TRUE(tp.has_raw_data());
  auto count = tp.raw_data().size() / sizeof(int64_t);
  EXPECT_EQ(count, 6);

  auto raw = tp.raw_data().data();
  const auto* tensor_data = reinterpret_cast<const int64_t*>(raw);
  for (size_t j = 0; j < count; ++j) {
    EXPECT_EQ(tensor_data[j], values_mul_x[j]);
  }
}

TEST(MLValueToTensorProtoTests, Int64ToInt64Data) {
  auto logger = std::make_unique<onnxruntime::logging::Logger>(::onnxruntime::test::DefaultLoggingManager().DefaultLogger());

  std::vector<int64_t> dims_mul_x = {3, 2};
  std::vector<int64_t> values_mul_x = {1, 2, 3, 4, 5, 6};
  MLValue ml_value;
  onnxruntime::test::CreateMLValue<int64_t>(TestCPUExecutionProvider()->GetAllocator(0, OrtMemTypeDefault), dims_mul_x, values_mul_x, &ml_value);

  onnx::TensorProto tp;
  common::Status status = onnxruntime::server::MLValueToTensorProto(ml_value, /* using_raw_data */ false, std::move(logger), tp);
  EXPECT_TRUE(status.IsOK());

  // Verify data type
  EXPECT_TRUE(tp.has_data_type());
  EXPECT_EQ(tp.data_type(), onnx::TensorProto_DataType_INT64);

  // Verify data location
  EXPECT_FALSE(tp.has_data_location());

  // Verify dimensions
  const auto& dims = tp.dims();
  std::vector<int64_t> tensor_shape_vec(static_cast<size_t>(dims.size()));
  for (int i = 0; i < dims.size(); ++i) {
    EXPECT_EQ(dims[i], dims_mul_x[i]);
  }

  // Verify data
  EXPECT_FALSE(tp.has_raw_data());
  EXPECT_EQ(tp.int64_data().size(), 6);
  auto data = tp.int64_data().data();
  for (int j = 0; j < tp.int64_data().size(); ++j) {
    EXPECT_EQ(data[j], values_mul_x[j]);
  }
}

TEST(MLValueToTensorProtoTests, UInt32ToRaw) {
  auto logger = std::make_unique<onnxruntime::logging::Logger>(::onnxruntime::test::DefaultLoggingManager().DefaultLogger());

  std::vector<int64_t> dims_mul_x = {3, 2};
  std::vector<uint32_t> values_mul_x = {1, 2, 3, 4, 5, 6};
  MLValue ml_value;
  onnxruntime::test::CreateMLValue<uint32_t>(TestCPUExecutionProvider()->GetAllocator(0, OrtMemTypeDefault), dims_mul_x, values_mul_x, &ml_value);

  onnx::TensorProto tp;
  common::Status status = onnxruntime::server::MLValueToTensorProto(ml_value, /* using_raw_data */ true, std::move(logger), tp);
  EXPECT_TRUE(status.IsOK());

  // Verify data type
  EXPECT_TRUE(tp.has_data_type());
  EXPECT_EQ(tp.data_type(), onnx::TensorProto_DataType_UINT32);

  // Verify data location
  EXPECT_TRUE(tp.has_data_location());
  EXPECT_EQ(tp.data_location(), onnx::TensorProto_DataLocation_DEFAULT);

  // Verify dimensions
  const auto& dims = tp.dims();
  std::vector<int64_t> tensor_shape_vec(static_cast<size_t>(dims.size()));
  for (int i = 0; i < dims.size(); ++i) {
    EXPECT_EQ(dims[i], dims_mul_x[i]);
  }

  // Verify data
  EXPECT_TRUE(tp.has_raw_data());
  auto count = tp.raw_data().size() / sizeof(uint32_t);
  EXPECT_EQ(count, 6);

  auto raw = tp.raw_data().data();
  auto* tensor_data = (uint32_t*)raw;
  for (size_t j = 0; j < count; ++j) {
    EXPECT_EQ(tensor_data[j], values_mul_x[j]);
  }
}

TEST(MLValueToTensorProtoTests, UInt32ToUint64Data) {
  auto logger = std::make_unique<onnxruntime::logging::Logger>(::onnxruntime::test::DefaultLoggingManager().DefaultLogger());

  std::vector<int64_t> dims_mul_x = {3, 2};
  std::vector<uint32_t> values_mul_x = {1, 2, 3, 4, 5, 6};
  MLValue ml_value;
  onnxruntime::test::CreateMLValue<uint32_t>(TestCPUExecutionProvider()->GetAllocator(0, OrtMemTypeDefault), dims_mul_x, values_mul_x, &ml_value);

  onnx::TensorProto tp;
  common::Status status = onnxruntime::server::MLValueToTensorProto(ml_value, /* using_raw_data */ false, std::move(logger), tp);
  EXPECT_TRUE(status.IsOK());

  // Verify data type
  EXPECT_TRUE(tp.has_data_type());
  EXPECT_EQ(tp.data_type(), onnx::TensorProto_DataType_UINT32);

  // Verify data location
  EXPECT_FALSE(tp.has_data_location());

  // Verify dimensions
  const auto& dims = tp.dims();
  std::vector<int64_t> tensor_shape_vec(static_cast<size_t>(dims.size()));
  for (int i = 0; i < dims.size(); ++i) {
    EXPECT_EQ(dims[i], dims_mul_x[i]);
  }

  // Verify data
  EXPECT_FALSE(tp.has_raw_data());
  auto count = tp.uint64_data().size() * (sizeof(uint64_t) / sizeof(uint32_t));
  EXPECT_EQ(count, 6);

  auto data = tp.uint64_data().data();
  const auto* data32 = reinterpret_cast<const uint32_t*>(data);
  for (size_t x = 0; x < count; ++x) {
    EXPECT_EQ(data32[x], values_mul_x[x]);
  }
}

TEST(MLValueToTensorProtoTests, UInt64ToRaw) {
  auto logger = std::make_unique<onnxruntime::logging::Logger>(::onnxruntime::test::DefaultLoggingManager().DefaultLogger());

  std::vector<int64_t> dims_mul_x = {3, 2};
  std::vector<uint64_t> values_mul_x = {1, 2, 3, 4, 5, 6};
  MLValue ml_value;
  onnxruntime::test::CreateMLValue<uint64_t>(TestCPUExecutionProvider()->GetAllocator(0, OrtMemTypeDefault), dims_mul_x, values_mul_x, &ml_value);

  onnx::TensorProto tp;
  common::Status status = onnxruntime::server::MLValueToTensorProto(ml_value, /* using_raw_data */ true, std::move(logger), tp);
  EXPECT_TRUE(status.IsOK());

  // Verify data type
  EXPECT_TRUE(tp.has_data_type());
  EXPECT_EQ(tp.data_type(), onnx::TensorProto_DataType_UINT64);

  // Verify data location
  EXPECT_TRUE(tp.has_data_location());
  EXPECT_EQ(tp.data_location(), onnx::TensorProto_DataLocation_DEFAULT);

  // Verify dimensions
  const auto& dims = tp.dims();
  std::vector<int64_t> tensor_shape_vec(static_cast<size_t>(dims.size()));
  for (int i = 0; i < dims.size(); ++i) {
    EXPECT_EQ(dims[i], dims_mul_x[i]);
  }

  // Verify data
  EXPECT_TRUE(tp.has_raw_data());
  auto count = tp.raw_data().size() / sizeof(uint64_t);
  EXPECT_EQ(count, 6);

  auto raw = tp.raw_data().data();
  const auto* tensor_data = reinterpret_cast<const uint64_t*>(raw);
  for (size_t j = 0; j < count; ++j) {
    EXPECT_EQ(tensor_data[j], values_mul_x[j]);
  }
}

TEST(MLValueToTensorProtoTests, UInt64ToInt64Data) {
  auto logger = std::make_unique<onnxruntime::logging::Logger>(::onnxruntime::test::DefaultLoggingManager().DefaultLogger());

  std::vector<int64_t> dims_mul_x = {3, 2};
  std::vector<uint64_t> values_mul_x = {1, 2, 3, 4, 5, 6};
  MLValue ml_value;
  onnxruntime::test::CreateMLValue<uint64_t>(TestCPUExecutionProvider()->GetAllocator(0, OrtMemTypeDefault), dims_mul_x, values_mul_x, &ml_value);

  onnx::TensorProto tp;
  common::Status status = onnxruntime::server::MLValueToTensorProto(ml_value, /* using_raw_data */ false, std::move(logger), tp);
  EXPECT_TRUE(status.IsOK());

  // Verify data type
  EXPECT_TRUE(tp.has_data_type());
  EXPECT_EQ(tp.data_type(), onnx::TensorProto_DataType_UINT64);

  // Verify data location
  EXPECT_FALSE(tp.has_data_location());

  // Verify dimensions
  const auto& dims = tp.dims();
  std::vector<int64_t> tensor_shape_vec(static_cast<size_t>(dims.size()));
  for (int i = 0; i < dims.size(); ++i) {
    EXPECT_EQ(dims[i], dims_mul_x[i]);
  }

  // Verify data
  EXPECT_FALSE(tp.has_raw_data());
  EXPECT_EQ(tp.uint64_data().size(), 6);
  auto data = tp.uint64_data().data();
  for (int j = 0; j < tp.uint64_data().size(); ++j) {
    EXPECT_EQ(data[j], values_mul_x[j]);
  }
}

TEST(MLValueToTensorProtoTests, DoubleToRaw) {
  auto logger = std::make_unique<onnxruntime::logging::Logger>(::onnxruntime::test::DefaultLoggingManager().DefaultLogger());

  std::vector<int64_t> dims_mul_x = {3, 2};
  std::vector<double> values_mul_x = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
  MLValue ml_value;
  onnxruntime::test::CreateMLValue<double>(TestCPUExecutionProvider()->GetAllocator(0, OrtMemTypeDefault), dims_mul_x, values_mul_x, &ml_value);

  onnx::TensorProto tp;
  common::Status status = onnxruntime::server::MLValueToTensorProto(ml_value, /* using_raw_data */ true, std::move(logger), tp);
  EXPECT_TRUE(status.IsOK());

  // Verify data type
  EXPECT_TRUE(tp.has_data_type());
  EXPECT_EQ(tp.data_type(), onnx::TensorProto_DataType_DOUBLE);

  // Verify data location
  EXPECT_TRUE(tp.has_data_location());
  EXPECT_EQ(tp.data_location(), onnx::TensorProto_DataLocation_DEFAULT);

  // Verify dimensions
  const auto& dims = tp.dims();
  std::vector<int64_t> tensor_shape_vec(static_cast<size_t>(dims.size()));
  for (int i = 0; i < dims.size(); ++i) {
    EXPECT_EQ(dims[i], dims_mul_x[i]);
  }

  // Verify data
  EXPECT_TRUE(tp.has_raw_data());
  auto count = tp.raw_data().size() / sizeof(uint64_t);
  EXPECT_EQ(count, 6);

  auto raw = tp.raw_data().data();
  const auto* tensor_data = reinterpret_cast<const double*>(raw);
  for (size_t j = 0; j < count; ++j) {
    EXPECT_DOUBLE_EQ(tensor_data[j], values_mul_x[j]);
  }
}

TEST(MLValueToTensorProtoTests, DoubleToInt64Data) {
  auto logger = std::make_unique<onnxruntime::logging::Logger>(::onnxruntime::test::DefaultLoggingManager().DefaultLogger());

  std::vector<int64_t> dims_mul_x = {3, 2};
  std::vector<double> values_mul_x = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
  MLValue ml_value;
  onnxruntime::test::CreateMLValue<double>(TestCPUExecutionProvider()->GetAllocator(0, OrtMemTypeDefault), dims_mul_x, values_mul_x, &ml_value);

  onnx::TensorProto tp;
  common::Status status = onnxruntime::server::MLValueToTensorProto(ml_value, /* using_raw_data */ false, std::move(logger), tp);
  EXPECT_TRUE(status.IsOK());

  // Verify data type
  EXPECT_TRUE(tp.has_data_type());
  EXPECT_EQ(tp.data_type(), onnx::TensorProto_DataType_DOUBLE);

  // Verify data location
  EXPECT_FALSE(tp.has_data_location());

  // Verify dimensions
  const auto& dims = tp.dims();
  std::vector<int64_t> tensor_shape_vec(static_cast<size_t>(dims.size()));
  for (int i = 0; i < dims.size(); ++i) {
    EXPECT_EQ(dims[i], dims_mul_x[i]);
  }

  // Verify data
  EXPECT_FALSE(tp.has_raw_data());
  EXPECT_EQ(tp.double_data().size(), 6);
  auto data = tp.double_data().data();
  for (int j = 0; j < tp.double_data().size(); ++j) {
    EXPECT_DOUBLE_EQ(data[j], values_mul_x[j]);
  }
}

void CreateMLValueBool(AllocatorPtr alloc,
                       const std::vector<int64_t>& dims,
                       const bool* value,
                       MLValue* p_mlvalue) {
  TensorShape shape(dims);
  auto element_type = DataTypeImpl::GetType<bool>();
  std::unique_ptr<Tensor> p_tensor = std::make_unique<Tensor>(element_type,
                                                              shape,
                                                              alloc);
  memcpy(p_tensor->MutableData<bool>(), &value[0], element_type->Size() * shape.Size());
  p_mlvalue->Init(p_tensor.release(),
                  DataTypeImpl::GetType<Tensor>(),
                  DataTypeImpl::GetType<Tensor>()->GetDeleteFunc());
}

}  // namespace test
}  // namespace server
}  // namespace onnxruntime