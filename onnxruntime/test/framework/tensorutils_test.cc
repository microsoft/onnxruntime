// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/framework/tensorprotoutils.h"
#include "core/graph/onnx_protobuf.h"
#include "gtest/gtest.h"

using namespace ::onnxruntime::utils;
using namespace ONNX_NAMESPACE;

namespace onnxruntime {
namespace test {
namespace {
template <typename T>
Status UnpackTensorWrapper(const ONNX_NAMESPACE::TensorProto& tensor, /*out*/ T* p_data, int64_t expected_size) {
  if (tensor.has_raw_data())
    return UnpackTensor(tensor, tensor.raw_data().data(), tensor.raw_data().size(), p_data, expected_size);
  return UnpackTensor(tensor, nullptr, 0, p_data, expected_size);
}
}  // namespace

//T must be float for double, and it must match with the 'type' argument
template <typename T>
void test_unpack_float_tensor(TensorProto_DataType type) {
  TensorProto float_tensor_proto;
  float_tensor_proto.set_data_type(type);
  T f[4] = {1.1f, 2.2f, 3.3f, 4.4f};
  const size_t len = sizeof(T) * 4;
  char rawdata[len];
  for (int i = 0; i < 4; ++i) {
    memcpy(rawdata + i * sizeof(T), &(f[i]), sizeof(T));
  }
  float_tensor_proto.set_raw_data(rawdata, len);
  T float_data2[4];
  auto status = UnpackTensorWrapper(float_tensor_proto, float_data2, 4);
  EXPECT_TRUE(status.IsOK()) << status.ErrorMessage();
  EXPECT_EQ(1.1f, float_data2[0]);
  EXPECT_EQ(2.2f, float_data2[1]);
  EXPECT_EQ(3.3f, float_data2[2]);
  EXPECT_EQ(4.4f, float_data2[3]);
}

TEST(TensorParseTest, TensorUtilsTest) {
  TensorProto bool_tensor_proto;
  bool_tensor_proto.set_data_type(TensorProto_DataType_BOOL);
  bool_tensor_proto.add_int32_data(1);

  bool bool_data[1];
  auto status = UnpackTensorWrapper(bool_tensor_proto, bool_data, 1);
  EXPECT_TRUE(status.IsOK()) << status.ErrorMessage();
  EXPECT_TRUE(bool_data[0]);

  float float_data[1];
  status = UnpackTensorWrapper(bool_tensor_proto, float_data, 1);
  EXPECT_FALSE(status.IsOK());

  test_unpack_float_tensor<float>(TensorProto_DataType_FLOAT);
  test_unpack_float_tensor<double>(TensorProto_DataType_DOUBLE);

  TensorProto string_tensor_proto;
  string_tensor_proto.set_data_type(TensorProto_DataType_STRING);
  string_tensor_proto.add_string_data("a");
  string_tensor_proto.add_string_data("b");

  std::string string_data[2];
  status = UnpackTensorWrapper(string_tensor_proto, string_data, 2);
  EXPECT_TRUE(status.IsOK()) << status.ErrorMessage();
  EXPECT_EQ("a", string_data[0]);
  EXPECT_EQ("b", string_data[1]);

  status = UnpackTensorWrapper(bool_tensor_proto, string_data, 2);
  EXPECT_FALSE(status.IsOK());
}
}  // namespace test
}  // namespace onnxruntime
