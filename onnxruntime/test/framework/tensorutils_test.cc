// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/framework/tensorprotoutils.h"
#include "core/graph/onnx_protobuf.h"
#include "test/util/include/asserts.h"
#include "file_util.h"

#include "gtest/gtest.h"
#include "gmock/gmock.h"

#ifdef _WIN32
#include <Windows.h>
#endif

using namespace ::onnxruntime::utils;
using namespace ONNX_NAMESPACE;

namespace onnxruntime {
namespace test {

//T must be float for double, and it must match with the 'type' argument
template <typename T>
void TestUnpackFloatTensor(TensorProto_DataType type, const Path& model_path) {
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
  auto status = UnpackTensor(float_tensor_proto, model_path, float_data2, 4);
  EXPECT_TRUE(status.IsOK()) << status.ErrorMessage();
  EXPECT_EQ(1.1f, float_data2[0]);
  EXPECT_EQ(2.2f, float_data2[1]);
  EXPECT_EQ(3.3f, float_data2[2]);
  EXPECT_EQ(4.4f, float_data2[3]);
}

TEST(TensorProtoUtilsTest, UnpackTensor) {
  TensorProto bool_tensor_proto;
  // Path is required for loading external data.
  // Using empty path here since this test does not test
  // external data utils
  Path model_path;
  bool_tensor_proto.set_data_type(TensorProto_DataType_BOOL);
  bool_tensor_proto.add_int32_data(1);

  bool bool_data[1];
  auto status = UnpackTensor(bool_tensor_proto, model_path, bool_data, 1);
  EXPECT_TRUE(status.IsOK()) << status.ErrorMessage();
  EXPECT_TRUE(bool_data[0]);

  float float_data[1];
  status = UnpackTensor(bool_tensor_proto, model_path, float_data, 1);
  EXPECT_FALSE(status.IsOK());

  TestUnpackFloatTensor<float>(TensorProto_DataType_FLOAT, model_path);
  TestUnpackFloatTensor<double>(TensorProto_DataType_DOUBLE, model_path);

  TensorProto string_tensor_proto;
  string_tensor_proto.set_data_type(TensorProto_DataType_STRING);
  string_tensor_proto.add_string_data("a");
  string_tensor_proto.add_string_data("b");

  std::string string_data[2];
  status = UnpackTensor(string_tensor_proto, model_path, string_data, 2);
  EXPECT_TRUE(status.IsOK()) << status.ErrorMessage();
  EXPECT_EQ("a", string_data[0]);
  EXPECT_EQ("b", string_data[1]);

  status = UnpackTensor(bool_tensor_proto, model_path, string_data, 2);
  EXPECT_FALSE(status.IsOK());
}

namespace {
template <typename T>
std::vector<T> CreateValues() {
  return {1, 2, 3, 4};
}

template <>
std::vector<std::string> CreateValues<std::string>() {
  return {"one", "two", "three", "four"};
}

template <>
std::vector<bool> CreateValues() {
  return {true, false, false, true};
}

template <>
std::vector<MLFloat16> CreateValues<MLFloat16>() {
  return {MLFloat16(0.f), MLFloat16(1.f), MLFloat16(2.f), MLFloat16(3.f)};
}

template <>
std::vector<BFloat16> CreateValues<BFloat16>() {
  return {BFloat16(0.f), BFloat16(1.f), BFloat16(2.f), BFloat16(3.f)};
}

template <typename T>
void WriteDataToFile(FILE* fp, const std::vector<T>& test_data) {
  size_t size_in_bytes = test_data.size() * sizeof(T);
  ASSERT_EQ(size_in_bytes, fwrite(test_data.data(), 1, size_in_bytes, fp));
}

std::unique_ptr<bool[]> BoolDataFromVector(const std::vector<bool>& test_data) {
  auto arr = onnxruntime::make_unique<bool[]>(test_data.size());
  std::copy(std::begin(test_data), std::end(test_data), arr.get());
  return arr;
}

// work around std::vector<bool> storing data in bits
template <>
void WriteDataToFile<bool>(FILE* fp, const std::vector<bool>& test_data) {
  auto arr = BoolDataFromVector(test_data);
  size_t size_in_bytes = test_data.size() * sizeof(bool);
  ASSERT_EQ(size_in_bytes, fwrite(arr.get(), 1, size_in_bytes, fp));
}

template <typename T>
void CreateTensorWithExternalData(TensorProto_DataType type, const std::vector<T>& test_data,
                                  std::basic_string<ORTCHAR_T>& filename,
                                  TensorProto& tensor_proto) {
  // Create external data
  FILE* fp;
  CreateTestFile(fp, filename);
  WriteDataToFile(fp, test_data);
  ASSERT_EQ(0, fclose(fp));

  // set the tensor_proto to reference this external data
  onnx::StringStringEntryProto* location = tensor_proto.mutable_external_data()->Add();
  location->set_key("location");
  location->set_value(ToMBString(filename));
  tensor_proto.mutable_dims()->Add(test_data.size());
  tensor_proto.set_data_location(onnx::TensorProto_DataLocation_EXTERNAL);
  tensor_proto.set_data_type(type);
}

template <typename T>
void UnpackAndValidate(const TensorProto& tensor_proto, const Path& model_path, const std::vector<T>& test_data) {
  // Unpack tensor with external data
  std::vector<T> val(test_data.size());
  auto st = utils::UnpackTensor(tensor_proto, model_path, val.data(), test_data.size());
  ASSERT_TRUE(st.IsOK()) << st.ErrorMessage();

  // Validate data
  for (size_t i = 0; i < test_data.size(); i++) {
    ASSERT_TRUE(val[i] == test_data[i]);  // need to use ASSERT_TRUE with '==' to handle MFLoat16 and BFloat16
  }
}

template <>
void UnpackAndValidate<bool>(const TensorProto& tensor_proto, const Path& model_path,
                             const std::vector<bool>& test_data) {
  // Unpack tensor with external data
  auto arr = onnxruntime::make_unique<bool[]>(test_data.size());
  auto st = utils::UnpackTensor(tensor_proto, model_path, arr.get(), test_data.size());
  ASSERT_TRUE(st.IsOK()) << st.ErrorMessage();

  // Validate data
  for (size_t i = 0; i < test_data.size(); i++) {
    ASSERT_TRUE(arr[i] == test_data[i]);
  }
}

template <typename T>
void TestUnpackExternalTensor(TensorProto_DataType type, const Path& model_path) {
  // Create external data
  std::basic_string<ORTCHAR_T> filename(ORT_TSTR("tensor_XXXXXX"));
  TensorProto tensor_proto;
  auto test_data = CreateValues<T>();
  CreateTensorWithExternalData<T>(type, test_data, filename, tensor_proto);
  std::unique_ptr<ORTCHAR_T, decltype(&DeleteFileFromDisk)> file_deleter(const_cast<ORTCHAR_T*>(filename.c_str()),
                                                                         DeleteFileFromDisk);
  UnpackAndValidate(tensor_proto, model_path, test_data);
}
}  // namespace
TEST(TensorProtoUtilsTest, UnpackTensorWithExternalData) {
  Path model_path;
  TestUnpackExternalTensor<float>(TensorProto_DataType_FLOAT, model_path);
  TestUnpackExternalTensor<double>(TensorProto_DataType_DOUBLE, model_path);
  TestUnpackExternalTensor<int32_t>(TensorProto_DataType_INT32, model_path);
  TestUnpackExternalTensor<int8_t>(TensorProto_DataType_INT8, model_path);
  TestUnpackExternalTensor<MLFloat16>(TensorProto_DataType_FLOAT16, model_path);
  TestUnpackExternalTensor<BFloat16>(TensorProto_DataType_BFLOAT16, model_path);
  TestUnpackExternalTensor<bool>(TensorProto_DataType_BOOL, model_path);
}

template <typename T>
static NodeProto CreateConstantNode(const std::string& attrib_name, AttributeProto_AttributeType type,
                                    std::function<void(AttributeProto&)> add_data) {
  NodeProto constant_node;
  constant_node.set_op_type("Constant");
  constant_node.add_output("Constant_output");

  AttributeProto& attrib = *constant_node.mutable_attribute()->Add();

  attrib.set_name(attrib_name);
  attrib.set_type(type);
  add_data(attrib);

  return constant_node;
}

template <typename T>
static void TestConstantNodeConversion(const std::string& attrib_name,
                                       AttributeProto_AttributeType type,
                                       std::function<void(AttributeProto&, const std::vector<T>& data)> add_data,
                                       std::function<std::vector<T>(const TensorProto&)> get_data,
                                       int64_t num_elements) {
  auto input = CreateValues<T>();
  if (num_elements == -1) {
    num_elements = static_cast<int64_t>(input.size());
  } else {
    input.resize(num_elements);
  }

  auto c = CreateConstantNode<T>(
      attrib_name, type,
      [&input, &add_data](AttributeProto& attrib) { add_data(attrib, input); });

  TensorProto tp;
  Path model_path;
  EXPECT_STATUS_OK(utils::ConstantNodeProtoToTensorProto(c, model_path, tp));

  EXPECT_THAT(get_data(tp), ::testing::ContainerEq(input));
}

TEST(TensorProtoUtilsTest, ConstantTensorProto) {
  TestConstantNodeConversion<float>(
      "value_float", AttributeProto_AttributeType_FLOAT,
      [](AttributeProto& attrib, const std::vector<float>& data) { attrib.set_f(data[0]); },
      [](const TensorProto& tp) {
        return std::vector<float>(tp.float_data().cbegin(), tp.float_data().cend());
      },
      1);

  TestConstantNodeConversion<float>(
      "value_floats", AttributeProto_AttributeType_FLOATS,
      [](AttributeProto& attrib, const std::vector<float>& data) {
        *attrib.mutable_floats() = {data.cbegin(), data.cend()};
      },
      [](const TensorProto& tp) {
        return std::vector<float>(tp.float_data().cbegin(), tp.float_data().cend());
      },
      -1);

  TestConstantNodeConversion<int64_t>(
      "value_int", AttributeProto_AttributeType_INT,
      [](AttributeProto& attrib, const std::vector<int64_t>& data) { attrib.set_i(data[0]); },
      [](const TensorProto& tp) {
        return std::vector<int64_t>(tp.int64_data().cbegin(), tp.int64_data().cend());
      },
      1);

  TestConstantNodeConversion<int64_t>(
      "value_ints", AttributeProto_AttributeType_INTS,
      [](AttributeProto& attrib, const std::vector<int64_t>& data) {
        *attrib.mutable_ints() = {data.cbegin(), data.cend()};
      },
      [](const TensorProto& tp) {
        return std::vector<int64_t>(tp.int64_data().cbegin(), tp.int64_data().cend());
      },
      -1);

  TestConstantNodeConversion<std::string>(
      "value_string", AttributeProto_AttributeType_STRING,
      [](AttributeProto& attrib, const std::vector<std::string>& data) { attrib.set_s(data[0]); },
      [](const TensorProto& tp) {
        return std::vector<std::string>(tp.string_data().cbegin(), tp.string_data().cend());
      },
      1);

  TestConstantNodeConversion<std::string>(
      "value_strings", AttributeProto_AttributeType_STRINGS,
      [](AttributeProto& attrib, const std::vector<std::string>& data) {
        // for (const auto& s : data)
        *attrib.mutable_strings() = {data.cbegin(), data.cend()};
      },
      [](const TensorProto& tp) {
        return std::vector<std::string>(tp.string_data().cbegin(), tp.string_data().cend());
      },
      -1);

  // sparse_tensor is covered by SparseTensorConversionTests.TestConstantNodeConversion
}

template <typename T>
static NodeProto CreateConstantNodeWithExternalData(TensorProto_DataType type, PathString& tensor_filename,
                                                    const std::vector<T>& test_data) {
  NodeProto constant_node;
  constant_node.set_op_type("Constant");
  constant_node.add_output("Constant_output");

  AttributeProto& attrib = *constant_node.mutable_attribute()->Add();
  attrib.set_name("attrib");
  attrib.set_type(AttributeProto_AttributeType_TENSOR);

  TensorProto& tp = *attrib.mutable_t();
  CreateTensorWithExternalData<T>(type, test_data, tensor_filename, tp);

  return constant_node;
}

template <typename T>
static void TestConstantNodeConversionWithExternalData(TensorProto_DataType type) {
  // Create a constant node with external data
  auto test_data = CreateValues<T>();
  Path model_path;
  PathString tensor_filename(ORT_TSTR("tensor_XXXXXX"));
  auto c = CreateConstantNodeWithExternalData<T>(type, tensor_filename, test_data);
  std::unique_ptr<ORTCHAR_T, decltype(&DeleteFileFromDisk)> file_deleter(const_cast<ORTCHAR_T*>(tensor_filename.c_str()),
                                                                         DeleteFileFromDisk);

  // Convert NodeProto to tensorproto (with external data)
  TensorProto tp;
  EXPECT_STATUS_OK(utils::ConstantNodeProtoToTensorProto(c, model_path, tp));

  // Unpack tensor and validate the data
  std::vector<T> val(test_data.size());
  auto st = utils::UnpackTensor(tp, model_path, val.data(), test_data.size());
  ASSERT_TRUE(st.IsOK()) << st.ErrorMessage();
  for (size_t i = 0; i < test_data.size(); i++) {
    ASSERT_EQ(val[i], test_data[i]);
  }
}

TEST(TensorProtoUtilsTest, ConstantTensorProtoWithExternalData) {
  TestConstantNodeConversionWithExternalData<float>(TensorProto_DataType_FLOAT);
  TestConstantNodeConversionWithExternalData<double>(TensorProto_DataType_DOUBLE);
}
}  // namespace test
}  // namespace onnxruntime
