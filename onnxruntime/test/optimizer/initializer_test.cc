// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/optimizer/initializer.h"

#include <cstdio>
#include <fstream>
#include <numeric>
#include <type_traits>

#include "gsl/gsl"

#include "gtest/gtest.h"

#include "core/common/common.h"
#include "core/framework/endian_utils.h"
#include "test/util/include/asserts.h"
#include "test/util/include/file_util.h"

namespace onnxruntime {
namespace test {
namespace {
template <typename T>
Status WriteExternalDataFile(gsl::span<const T> data, const PathString& path, ScopedFileDeleter& file_deleter) {
  std::vector<unsigned char> data_bytes(data.size_bytes());
  ORT_RETURN_IF_ERROR(onnxruntime::utils::WriteLittleEndian(data, gsl::make_span(data_bytes)));
  std::ofstream out{path, std::ios::binary | std::ios::trunc};
  ORT_RETURN_IF_NOT(out && out.write(reinterpret_cast<const char*>(data_bytes.data()), data_bytes.size()),
                    "out && out.write(data_bytes.data(), data_bytes.size()) was false");
  file_deleter = ScopedFileDeleter{path};
  return Status::OK();
}

void SetTensorProtoExternalData(const std::string& key, const std::string& value,
                                ONNX_NAMESPACE::TensorProto& tensor_proto) {
  auto* external_data = tensor_proto.mutable_external_data();
  auto kvp_it = std::find_if(
      external_data->begin(), external_data->end(),
      [&key](const ONNX_NAMESPACE::StringStringEntryProto& kvp) { return kvp.key() == key; });
  auto* kvp = kvp_it != external_data->end() ? &(*kvp_it) : external_data->Add();
  kvp->set_key(key);
  kvp->set_value(value);
}
}  // namespace

TEST(OptimizerInitializerTest, LoadExternalData) {
  const std::vector<int32_t> tensor_data = []() {
    std::vector<int32_t> tensor_data(100);
    std::iota(tensor_data.begin(), tensor_data.end(), 0);
    return tensor_data;
  }();
  const gsl::span<const int> tensor_data_span = gsl::make_span(tensor_data);
  const auto tensor_data_dir_path = Path::Parse(ORT_TSTR("."));
  const auto tensor_data_dir_relative_path = Path::Parse(ORT_TSTR("OptimizerInitializerTest_LoadExternalData.bin"));
  ScopedFileDeleter file_deleter{};

  ASSERT_STATUS_OK(WriteExternalDataFile(
      tensor_data_span, (tensor_data_dir_path / tensor_data_dir_relative_path).ToPathString(), file_deleter));

  const auto tensor_proto_base =
      [&]() {
        ONNX_NAMESPACE::TensorProto tensor_proto{};
        tensor_proto.set_name("test");
        tensor_proto.add_dims(tensor_data.size());
        tensor_proto.set_data_type(ONNX_NAMESPACE::TensorProto_DataType_INT32);
        tensor_proto.set_data_location(ONNX_NAMESPACE::TensorProto_DataLocation_EXTERNAL);
        SetTensorProtoExternalData("location", ToMBString(tensor_data_dir_relative_path.ToPathString()), tensor_proto);
        SetTensorProtoExternalData("offset", "0", tensor_proto);
        SetTensorProtoExternalData("length", std::to_string(tensor_data.size() * sizeof(int32_t)), tensor_proto);
        return tensor_proto;
      }();

  auto check_initializer_load =
      [&](size_t offset, size_t length) {
        ONNX_NAMESPACE::TensorProto tensor_proto{tensor_proto_base};
        tensor_proto.clear_dims();
        tensor_proto.add_dims(length);
        SetTensorProtoExternalData("offset", std::to_string(offset * sizeof(int32_t)), tensor_proto);
        SetTensorProtoExternalData("length", std::to_string(length * sizeof(int32_t)), tensor_proto);

        if (offset + length <= tensor_data_span.size()) {
          Initializer i(tensor_proto, tensor_data_dir_path);
          EXPECT_EQ(gsl::make_span(i.data<int32_t>(), i.size()), tensor_data_span.subspan(offset, length));
        } else {
          EXPECT_THROW(Initializer i(tensor_proto, tensor_data_dir_path), OnnxRuntimeException);
        }
      };

  check_initializer_load(0, tensor_data.size());
  check_initializer_load(tensor_data.size() / 2, tensor_data.size() / 3);

  // bad offset and length
  check_initializer_load(tensor_data.size() - 1, 2);
  check_initializer_load(0, tensor_data.size() + 1);

  // bad model paths
  EXPECT_THROW(Initializer i(tensor_proto_base, Path{}), OnnxRuntimeException);
  EXPECT_THROW(Initializer i(tensor_proto_base, Path::Parse(ORT_TSTR("invalid/directory"))), OnnxRuntimeException);

  // bad length
  {
    ONNX_NAMESPACE::TensorProto tensor_proto{tensor_proto_base};
    tensor_proto.clear_dims();
    SetTensorProtoExternalData("length", std::to_string(tensor_data.size() * sizeof(int32_t) + 1), tensor_proto);

    EXPECT_THROW(Initializer i(tensor_proto, tensor_data_dir_path), OnnxRuntimeException);
  }
}

template <typename T>
ONNX_NAMESPACE::TensorProto_DataType GetTensorProtoDataType();

#define CppTypeToTensorProto_DataType(CppType, TP_DataType)                \
  template <>                                                              \
  ONNX_NAMESPACE::TensorProto_DataType GetTensorProtoDataType<CppType>() { \
    return ONNX_NAMESPACE::TP_DataType;                                    \
  }

CppTypeToTensorProto_DataType(int8_t, TensorProto_DataType_INT8)
CppTypeToTensorProto_DataType(uint8_t, TensorProto_DataType_UINT8)
CppTypeToTensorProto_DataType(int32_t, TensorProto_DataType_INT32)
CppTypeToTensorProto_DataType(int64_t, TensorProto_DataType_INT64)
CppTypeToTensorProto_DataType(uint16_t, TensorProto_DataType_FLOAT16)
CppTypeToTensorProto_DataType(float, TensorProto_DataType_FLOAT)
CppTypeToTensorProto_DataType(double, TensorProto_DataType_DOUBLE)

template <typename T>
void TestInitializerRawData() {
  std::vector<T> data{
      0, 1, 2, 3,
      4, 5, 6, 7,
      8, 9, 10, 11};

  ONNX_NAMESPACE::TensorProto tensor_proto;
  tensor_proto.set_data_type(GetTensorProtoDataType<T>());
  tensor_proto.set_name("OptimizerInitializerTest_RawData");
  tensor_proto.add_dims(3);
  tensor_proto.add_dims(4);
  tensor_proto.set_raw_data(data.data(), data.size() * sizeof(T));

  Initializer init(tensor_proto, Path());

  for (size_t idx = 0; idx < data.size(); idx++) {
    EXPECT_EQ(data[idx], init.data<T>()[idx]);
  }
}

TEST(OptimizerInitializerTest, RawData) {
  TestInitializerRawData<int8_t>();
  TestInitializerRawData<uint8_t>();
  TestInitializerRawData<int32_t>();
  TestInitializerRawData<int64_t>();
  TestInitializerRawData<uint16_t>();
  TestInitializerRawData<float>();
  TestInitializerRawData<double>();
}

template <typename T>
void TestInitializerDataField() {
  std::vector<T> data{
      0, 1, 2, 3,
      4, 5, 6, 7,
      8, 9, 10, 11};

  auto dt = GetTensorProtoDataType<T>();
  ONNX_NAMESPACE::TensorProto tensor_proto;
  tensor_proto.set_data_type(GetTensorProtoDataType<T>());
  tensor_proto.set_name("OptimizerInitializerTest_DataField");
  tensor_proto.add_dims(3);
  tensor_proto.add_dims(4);
  for (size_t idx = 0; idx < data.size(); idx++) {
    if (dt == ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_INT8 ||
        dt == ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_UINT8 ||
        dt == ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_INT32 ||
        dt == ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_FLOAT16) {
      tensor_proto.add_int32_data(data[idx]);
    } else {
      ORT_NOT_IMPLEMENTED("tensor type ", GetTensorProtoDataType<T>(), " is not supported");
    }
  }

  Initializer init(tensor_proto, Path());

  for (size_t idx = 0; idx < data.size(); idx++) {
    EXPECT_EQ(data[idx], init.data<T>()[idx]);
  }
}

#define TestInitializerDataFieldSpecialized(type)                \
  template <>                                                    \
  void TestInitializerDataField<type>() {                        \
    std::vector<type> data{                                      \
        0, 1, 2, 3,                                              \
        4, 5, 6, 7,                                              \
        8, 9, 10, 11};                                           \
                                                                 \
    ONNX_NAMESPACE::TensorProto tensor_proto;                    \
    tensor_proto.set_data_type(GetTensorProtoDataType<type>());  \
    tensor_proto.set_name("OptimizerInitializerTest_DataField"); \
    tensor_proto.add_dims(3);                                    \
    tensor_proto.add_dims(4);                                    \
    for (size_t idx = 0; idx < data.size(); idx++) {             \
      tensor_proto.add_##type##_data(data[idx]);                 \
    }                                                            \
                                                                 \
    Initializer init(tensor_proto, Path());                      \
                                                                 \
    for (size_t idx = 0; idx < data.size(); idx++) {             \
      EXPECT_EQ(data[idx], init.data<type>()[idx]);              \
    }                                                            \
  }

typedef int64_t int64;
TestInitializerDataFieldSpecialized(float)
TestInitializerDataFieldSpecialized(double)
TestInitializerDataFieldSpecialized(int64)

TEST(OptimizerInitializerTest, DataField) {
  TestInitializerDataField<int8_t>();
  TestInitializerDataField<uint8_t>();
  TestInitializerDataField<int32_t>();
  TestInitializerDataField<int64_t>();
  TestInitializerDataField<uint16_t>();
  TestInitializerDataField<float>();
  TestInitializerDataField<double>();
}

}  // namespace test
}  // namespace onnxruntime
