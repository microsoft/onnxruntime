// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/common/inlined_containers.h"
#include "core/common/parse_string.h"
#include "core/framework/prepacked_weights.h"
#include "core/framework/prepacked_weights_container.h"
#include "core/framework/tensorprotoutils.h"
#include "core/graph/onnx_protobuf.h"
#include "test/util/include/asserts.h"
#include "file_util.h"

#include <cstdint>
#include <limits>

#include "gtest/gtest.h"
#include "gmock/gmock.h"

#ifdef _WIN32
#include <Windows.h>
#endif

using namespace ::onnxruntime::utils;
using namespace ONNX_NAMESPACE;

namespace onnxruntime {
namespace test {

// if `expected_error_message_substring` is nullptr, parsing is expected to be successful
static void TestExternalDataInfoParsingOffsetAndLengthWithStrings(
    std::string_view offset_str,
    std::string_view length_str,
    const char* expected_error_message_substring = nullptr) {
  SCOPED_TRACE(MakeString("offset: \"", offset_str, "\", length: \"", length_str, "\""));

  ONNX_NAMESPACE::TensorProto tensor_proto;
  const std::filesystem::path kExternalDataPath("test.bin");

  tensor_proto.set_data_location(ONNX_NAMESPACE::TensorProto_DataLocation::TensorProto_DataLocation_EXTERNAL);

  auto* location_entry = tensor_proto.add_external_data();
  location_entry->set_key("location");
  location_entry->set_value(ToUTF8String(kExternalDataPath.native()));

  auto* offset_entry = tensor_proto.add_external_data();
  offset_entry->set_key("offset");
  offset_entry->set_value(offset_str.data(), offset_str.size());

  auto* length_entry = tensor_proto.add_external_data();
  length_entry->set_key("length");
  length_entry->set_value(length_str.data(), length_str.size());

  std::unique_ptr<ExternalDataInfo> external_data_info{};
  const auto create_status = ExternalDataInfo::Create(tensor_proto.external_data(), external_data_info);
  if (expected_error_message_substring) {
    ASSERT_STATUS_NOT_OK_AND_HAS_SUBSTR(create_status, expected_error_message_substring);
    return;
  }
  ASSERT_STATUS_OK(create_status);

  // if we got this far, assume that offset_str and length_str are able to be parsed.
  const auto expected_offset = ParseStringWithClassicLocale<ExternalDataInfo::OFFSET_TYPE>(offset_str);
  const auto expected_length = ParseStringWithClassicLocale<size_t>(length_str);

  ASSERT_EQ(external_data_info->GetOffset(), expected_offset);
  ASSERT_EQ(external_data_info->GetLength(), expected_length);
}

// if `expected_error_message_substring` is nullptr, parsing is expected to be successful
static void TestExternalDataInfoParsingOffsetAndLength(intmax_t offset,
                                                       uintmax_t length,
                                                       const char* expected_error_message_substring = nullptr) {
  TestExternalDataInfoParsingOffsetAndLengthWithStrings(std::to_string(offset), std::to_string(length),
                                                        expected_error_message_substring);
}

TEST(TensorProtoUtilsTest, ParseExternalDataInfoOffsetAndLength) {
  TestExternalDataInfoParsingOffsetAndLength(0, 0);

  TestExternalDataInfoParsingOffsetAndLength(0, 1024);
  TestExternalDataInfoParsingOffsetAndLength(0, std::numeric_limits<size_t>::max());

  TestExternalDataInfoParsingOffsetAndLength(1024, 1024);
  TestExternalDataInfoParsingOffsetAndLength(std::numeric_limits<ExternalDataInfo::OFFSET_TYPE>::max(), 1024);

  {
    // assuming that this value is too large to fit in either size_t or ExternalDataInfo::OFFSET_TYPE
    const std::string_view two_to_the_65th_power = "36893488147419103232";
    const std::string_view zero = "0";
    TestExternalDataInfoParsingOffsetAndLengthWithStrings(two_to_the_65th_power, zero, "Failed to parse value");
    TestExternalDataInfoParsingOffsetAndLengthWithStrings(zero, two_to_the_65th_power, "Failed to parse value");
  }

  // TODO should ExternalDataInfo::Create() also reject negative offset values?
}

// Test ExternalData functionality
TEST(TensorProtoUtilsTest, SetExternalDataInformation) {
  ONNX_NAMESPACE::TensorProto tensor_proto;
  const std::filesystem::path kExternalDataPath("test.bin");
  constexpr const int64_t init_offset = 100;
  constexpr const size_t init_length = 200;

  ExternalDataInfo::SetExternalLocationToProto(kExternalDataPath, init_offset, init_length, tensor_proto);

  ASSERT_EQ(tensor_proto.data_location(), ONNX_NAMESPACE::TensorProto_DataLocation::TensorProto_DataLocation_EXTERNAL);
  ASSERT_EQ(tensor_proto.external_data_size(), 3);
  ASSERT_EQ(tensor_proto.external_data(0).key(), "location");
  ASSERT_EQ(tensor_proto.external_data(0).value(), ToUTF8String(kExternalDataPath.native()));
  ASSERT_EQ(tensor_proto.external_data(1).key(), "offset");
  ASSERT_EQ(tensor_proto.external_data(1).value(), std::to_string(init_offset));
  ASSERT_EQ(tensor_proto.external_data(2).key(), "length");
  ASSERT_EQ(tensor_proto.external_data(2).value(), std::to_string(init_length));

  PrepackedKeyToBlobMap key_to_blob;
  constexpr bool save_mode_on = true;
  PrepackedWeightsForGraph prepacked_for_graph(key_to_blob, save_mode_on);
  PrePackedWeights prepacked_weights;
  const std::string init_name = "test_initializer";
  const std::string blob_key = "test_key";

  std::array<float, 2> kData = {1.2345f, 2.4690f};
  const size_t buffer_size = kData.size() * sizeof(float);

  prepacked_weights.buffers_.push_back(BufferUniquePtr(kData.data(), BufferDeleter(nullptr)));
  prepacked_weights.buffer_sizes_.push_back(buffer_size);
  // Write a second entry like this
  prepacked_weights.buffers_.push_back(BufferUniquePtr(kData.data(), BufferDeleter(nullptr)));
  prepacked_weights.buffer_sizes_.push_back(buffer_size);

  prepacked_for_graph.WritePackedMaybeForSave(init_name, blob_key, std::move(prepacked_weights));

  constexpr const int64_t starting_offset = 300;
  int64_t external_offset = starting_offset;
  std::stringstream ss;
  const auto* blobs_for_weight = prepacked_for_graph.GetKeysForWeightForSaving(init_name);
  ASSERT_TRUE(blobs_for_weight != nullptr);
  InlinedHashSet<std::string> blob_keys{blobs_for_weight->begin(), blobs_for_weight->end()};
  ASSERT_TRUE(ExternalDataInfo::WritePrepackedToFileAndAddToProto(prepacked_for_graph,
                                                                  blob_keys,
                                                                  true, 1024 * 1024, 0,
                                                                  ss, external_offset,
                                                                  tensor_proto));

  auto external_data_info = std::make_unique<ExternalDataInfo>();
  ASSERT_STATUS_OK(ExternalDataInfo::Create(tensor_proto.external_data(), external_data_info));

  // This should have prepacked_data entry with two blobs for a single key.
  ASSERT_TRUE(external_data_info->HasPrepackedInfo());
  auto prepacked_infos = external_data_info->TakePrepackedInfos();
  ASSERT_EQ(prepacked_infos.size(), 1U);
  ASSERT_TRUE(prepacked_infos.count(blob_key) > 0);

  int64_t final_offset = starting_offset;
  for (const auto& blob_info : prepacked_infos[blob_key]) {
    int64_t offset = std::get<0>(blob_info);
    ASSERT_EQ(offset, final_offset);
    size_t length = std::get<1>(blob_info);
    std::string checksum = std::get<2>(blob_info);  // currently "0"
    final_offset = offset + length;
    ASSERT_EQ(length, buffer_size);
    ASSERT_EQ(checksum, "0");
  }
  ASSERT_EQ(final_offset, external_offset);
}

// T must be float for double, and it must match with the 'type' argument
template <typename T>
void TestUnpackFloatTensor(TensorProto_DataType type, const std::filesystem::path& model_path) {
  TensorProto float_tensor_proto;
  float_tensor_proto.set_data_type(type);
  T f[4] = {1.1f, 2.2f, 3.3f, 4.4f};
  constexpr size_t len = sizeof(T) * 4;
  char rawdata[len];
  for (int i = 0; i < 4; ++i) {
    memcpy(rawdata + i * sizeof(T), &(f[i]), sizeof(T));
  }
  utils::SetRawDataInTensorProto(float_tensor_proto, rawdata, len);
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
  std::filesystem::path model_path;
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
void ConvertEndianessForVector(const std::vector<T>& test_data) {
  const size_t element_size = sizeof(T);
  const size_t num_elements = test_data.size();
  char* bytes = reinterpret_cast<char*>(const_cast<T*>(test_data.data()));
  for (size_t i = 0; i < num_elements; ++i) {
    char* start_byte = bytes + i * element_size;
    char* end_byte = start_byte + element_size - 1;
    for (size_t count = 0; count < element_size / 2; ++count) {
      std::swap(*start_byte++, *end_byte--);
    }
  }
}

template <typename T>
void WriteDataToFile(FILE* fp, const std::vector<T>& test_data) {
  if constexpr (endian::native != endian::little) {
    ConvertEndianessForVector(test_data);
  }
  size_t size_in_bytes = test_data.size() * sizeof(T);
  ASSERT_EQ(size_in_bytes, fwrite(test_data.data(), 1, size_in_bytes, fp));
}

std::unique_ptr<bool[]> BoolDataFromVector(const std::vector<bool>& test_data) {
  auto arr = std::make_unique<bool[]>(test_data.size());
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
  location->set_value(ToUTF8String(filename));
  tensor_proto.mutable_dims()->Add(test_data.size());
  tensor_proto.set_data_location(onnx::TensorProto_DataLocation_EXTERNAL);
  tensor_proto.set_data_type(type);
}

template <typename T>
void UnpackAndValidate(const TensorProto& tensor_proto, const std::filesystem::path& model_path, const std::vector<T>& test_data) {
  // Unpack tensor with external data
  std::vector<T> val(test_data.size());
  auto st = utils::UnpackTensor(tensor_proto, model_path, val.data(), test_data.size());
  ASSERT_TRUE(st.IsOK()) << st.ErrorMessage();
  if constexpr (endian::native != endian::little) {
    ConvertEndianessForVector(val);
  }

  // Validate data
  for (size_t i = 0; i < test_data.size(); i++) {
    ASSERT_TRUE(val[i] == test_data[i]);  // need to use ASSERT_TRUE with '==' to handle MFLoat16 and BFloat16
  }
}

template <>
void UnpackAndValidate<bool>(const TensorProto& tensor_proto, const std::filesystem::path& model_path,
                             const std::vector<bool>& test_data) {
  // Unpack tensor with external data
  auto arr = std::make_unique<bool[]>(test_data.size());
  auto st = utils::UnpackTensor(tensor_proto, model_path, arr.get(), test_data.size());
  ASSERT_TRUE(st.IsOK()) << st.ErrorMessage();

  // Validate data
  for (size_t i = 0; i < test_data.size(); i++) {
    ASSERT_TRUE(arr[i] == test_data[i]);
  }
}

template <typename T>
void TestUnpackExternalTensor(TensorProto_DataType type, const std::filesystem::path& model_path) {
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
  std::filesystem::path model_path;
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
  std::filesystem::path model_path;
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
  std::filesystem::path model_path;
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
  if constexpr (endian::native != endian::little) {
    ConvertEndianessForVector(val);
  }
  for (size_t i = 0; i < test_data.size(); i++) {
    ASSERT_EQ(val[i], test_data[i]);
  }
}

TEST(TensorProtoUtilsTest, ConstantTensorProtoWithExternalData) {
  TestConstantNodeConversionWithExternalData<float>(TensorProto_DataType_FLOAT);
  TestConstantNodeConversionWithExternalData<double>(TensorProto_DataType_DOUBLE);
}

// =============================================================================
// External data slice spec tests
// =============================================================================
namespace {

// Adds (key, value) to tensor_proto.external_data().
void AddExtEntry(ONNX_NAMESPACE::TensorProto& tp, std::string_view key, std::string_view value) {
  auto* e = tp.add_external_data();
  e->set_key(std::string(key));
  e->set_value(std::string(value));
}

// Builds an FP32 source tensor of `source_shape` (row-major), writes it to a
// temporary file, and returns the file path plus the source data as a vector.
std::pair<std::filesystem::path, std::vector<float>> WriteFp32SourceFile(
    const std::vector<int64_t>& source_shape,
    ScopedFileDeleter& deleter) {
  size_t total = 1;
  for (auto d : source_shape) total *= static_cast<size_t>(d);
  std::vector<float> data;
  data.reserve(total);
  for (size_t i = 0; i < total; ++i) {
    data.push_back(static_cast<float>(i));  // 0, 1, 2, ...
  }

  std::basic_string<ORTCHAR_T> filename_template = ORT_TSTR("ort_slice_test_XXXXXX");
  FILE* fp = nullptr;
  CreateTestFile(fp, filename_template);
  ORT_ENFORCE(fp != nullptr);
  std::fwrite(data.data(), sizeof(float), data.size(), fp);
  std::fclose(fp);
  deleter = ScopedFileDeleter(filename_template);
  return {std::filesystem::path(filename_template), std::move(data)};
}

ONNX_NAMESPACE::TensorProto MakeSlicedFp32Proto(
    const std::filesystem::path& file_path,
    const std::vector<int64_t>& source_shape,
    const std::vector<int64_t>& slice_starts,
    const std::vector<int64_t>& slice_sizes,
    int64_t source_byte_offset = 0) {
  ONNX_NAMESPACE::TensorProto tp;
  tp.set_name("sliced_w");
  tp.set_data_type(ONNX_NAMESPACE::TensorProto::FLOAT);
  for (auto d : slice_sizes) tp.add_dims(d);
  tp.set_data_location(ONNX_NAMESPACE::TensorProto_DataLocation::TensorProto_DataLocation_EXTERNAL);
  AddExtEntry(tp, "location", ToUTF8String(file_path.native()));
  AddExtEntry(tp, "offset", std::to_string(source_byte_offset));

  auto join = [](const std::vector<int64_t>& v) {
    std::string out;
    for (size_t i = 0; i < v.size(); ++i) {
      if (i) out.push_back(',');
      out.append(std::to_string(v[i]));
    }
    return out;
  };
  AddExtEntry(tp, ExternalDataInfo::kSourceShapeKey, join(source_shape));
  AddExtEntry(tp, ExternalDataInfo::kSliceStartsKey, join(slice_starts));
  AddExtEntry(tp, ExternalDataInfo::kSliceSizesKey, join(slice_sizes));
  return tp;
}

// Returns the expected sliced FP32 values (row-major gather from `source`).
std::vector<float> ExpectedSlice(const std::vector<float>& source,
                                 const std::vector<int64_t>& source_shape,
                                 const std::vector<int64_t>& slice_starts,
                                 const std::vector<int64_t>& slice_sizes) {
  // Compute row-major strides for source.
  const size_t rank = source_shape.size();
  std::vector<size_t> strides(rank, 1);
  for (size_t d = rank; d-- > 1;) {
    strides[d - 1] = strides[d] * static_cast<size_t>(source_shape[d]);
  }
  std::vector<float> out;
  size_t slice_total = 1;
  for (auto d : slice_sizes) slice_total *= static_cast<size_t>(d);
  out.reserve(slice_total);

  std::vector<int64_t> idx(rank, 0);
  for (size_t k = 0; k < slice_total; ++k) {
    size_t off = 0;
    for (size_t d = 0; d < rank; ++d) {
      off += static_cast<size_t>(slice_starts[d] + idx[d]) * strides[d];
    }
    out.push_back(source[off]);
    // Increment row-major index in slice space.
    for (size_t d = rank; d-- > 0;) {
      ++idx[d];
      if (idx[d] < slice_sizes[d]) break;
      idx[d] = 0;
    }
  }
  return out;
}

}  // namespace

TEST(ExternalDataSliceTest, ParseValidSliceSpec) {
  ONNX_NAMESPACE::TensorProto tp;
  tp.set_data_location(ONNX_NAMESPACE::TensorProto_DataLocation::TensorProto_DataLocation_EXTERNAL);
  AddExtEntry(tp, "location", "weights.bin");
  AddExtEntry(tp, "offset", "1024");
  AddExtEntry(tp, ExternalDataInfo::kSourceShapeKey, "2048,8192");
  AddExtEntry(tp, ExternalDataInfo::kSliceStartsKey, "0,2048");
  AddExtEntry(tp, ExternalDataInfo::kSliceSizesKey, "1024,4096");

  std::unique_ptr<ExternalDataInfo> info;
  ASSERT_STATUS_OK(ExternalDataInfo::Create(tp.external_data(), info));
  ASSERT_TRUE(info->HasSliceSpec());
  ASSERT_EQ(info->GetSourceShape(), (std::vector<int64_t>{2048, 8192}));
  ASSERT_EQ(info->GetSliceStarts(), (std::vector<int64_t>{0, 2048}));
  ASSERT_EQ(info->GetSliceSizes(), (std::vector<int64_t>{1024, 4096}));
  ASSERT_EQ(info->GetOffset(), 1024);
}

TEST(ExternalDataSliceTest, ParseDefaultStartsToZero) {
  ONNX_NAMESPACE::TensorProto tp;
  tp.set_data_location(ONNX_NAMESPACE::TensorProto_DataLocation::TensorProto_DataLocation_EXTERNAL);
  AddExtEntry(tp, "location", "weights.bin");
  AddExtEntry(tp, ExternalDataInfo::kSourceShapeKey, "8,16");
  AddExtEntry(tp, ExternalDataInfo::kSliceSizesKey, "4,16");
  // No slice_starts: should default to all zeros.

  std::unique_ptr<ExternalDataInfo> info;
  ASSERT_STATUS_OK(ExternalDataInfo::Create(tp.external_data(), info));
  ASSERT_TRUE(info->HasSliceSpec());
  ASSERT_EQ(info->GetSliceStarts(), (std::vector<int64_t>{0, 0}));
}

TEST(ExternalDataSliceTest, ParseDefaultSizesToRemainder) {
  ONNX_NAMESPACE::TensorProto tp;
  tp.set_data_location(ONNX_NAMESPACE::TensorProto_DataLocation::TensorProto_DataLocation_EXTERNAL);
  AddExtEntry(tp, "location", "weights.bin");
  AddExtEntry(tp, ExternalDataInfo::kSourceShapeKey, "2048,8192");
  AddExtEntry(tp, ExternalDataInfo::kSliceStartsKey, "1024,2048");
  // No slice_sizes: should default to source_shape - slice_starts.

  std::unique_ptr<ExternalDataInfo> info;
  ASSERT_STATUS_OK(ExternalDataInfo::Create(tp.external_data(), info));
  ASSERT_TRUE(info->HasSliceSpec());
  ASSERT_EQ(info->GetSliceSizes(), (std::vector<int64_t>{1024, 6144}));
}

TEST(ExternalDataSliceTest, ParseDefaultBothToWholeTensor) {
  // Only source_shape supplied => slice equals the entire source. This is a no-op slice
  // (semantically identical to omitting the slice spec) and must be accepted.
  ONNX_NAMESPACE::TensorProto tp;
  tp.set_data_location(ONNX_NAMESPACE::TensorProto_DataLocation::TensorProto_DataLocation_EXTERNAL);
  AddExtEntry(tp, "location", "weights.bin");
  AddExtEntry(tp, ExternalDataInfo::kSourceShapeKey, "3,5");
  // No slice_starts, no slice_sizes.

  std::unique_ptr<ExternalDataInfo> info;
  ASSERT_STATUS_OK(ExternalDataInfo::Create(tp.external_data(), info));
  ASSERT_TRUE(info->HasSliceSpec());
  ASSERT_EQ(info->GetSliceStarts(), (std::vector<int64_t>{0, 0}));
  ASSERT_EQ(info->GetSliceSizes(), (std::vector<int64_t>{3, 5}));
}

TEST(ExternalDataSliceTest, ParseRejectsRankMismatch) {
  ONNX_NAMESPACE::TensorProto tp;
  tp.set_data_location(ONNX_NAMESPACE::TensorProto_DataLocation::TensorProto_DataLocation_EXTERNAL);
  AddExtEntry(tp, "location", "weights.bin");
  AddExtEntry(tp, ExternalDataInfo::kSourceShapeKey, "8,16");
  AddExtEntry(tp, ExternalDataInfo::kSliceSizesKey, "4");  // wrong rank
  std::unique_ptr<ExternalDataInfo> info;
  ASSERT_STATUS_NOT_OK_AND_HAS_SUBSTR(ExternalDataInfo::Create(tp.external_data(), info),
                                       "does not match");
}

TEST(ExternalDataSliceTest, ParseRejectsSliceStartsWithoutSourceShape) {
  ONNX_NAMESPACE::TensorProto tp;
  tp.set_data_location(ONNX_NAMESPACE::TensorProto_DataLocation::TensorProto_DataLocation_EXTERNAL);
  AddExtEntry(tp, "location", "weights.bin");
  AddExtEntry(tp, ExternalDataInfo::kSliceStartsKey, "0,0");
  std::unique_ptr<ExternalDataInfo> info;
  ASSERT_STATUS_NOT_OK_AND_HAS_SUBSTR(ExternalDataInfo::Create(tp.external_data(), info),
                                       "without");
}

TEST(ExternalDataSliceTest, ParseRejectsOutOfBoundsSlice) {
  ONNX_NAMESPACE::TensorProto tp;
  tp.set_data_location(ONNX_NAMESPACE::TensorProto_DataLocation::TensorProto_DataLocation_EXTERNAL);
  AddExtEntry(tp, "location", "weights.bin");
  AddExtEntry(tp, ExternalDataInfo::kSourceShapeKey, "8,16");
  AddExtEntry(tp, ExternalDataInfo::kSliceStartsKey, "5,0");
  AddExtEntry(tp, ExternalDataInfo::kSliceSizesKey, "4,16");  // 5 + 4 > 8
  std::unique_ptr<ExternalDataInfo> info;
  ASSERT_STATUS_NOT_OK_AND_HAS_SUBSTR(ExternalDataInfo::Create(tp.external_data(), info),
                                       "out of bounds");
}

TEST(ExternalDataSliceTest, ReadFp32_OuterAxisOnly) {
  // Source (4, 6), slice rows [1, 4) => shape (3, 6) — fully contiguous block.
  ScopedFileDeleter deleter;
  auto [file_path, source] = WriteFp32SourceFile({4, 6}, deleter);
  auto tp = MakeSlicedFp32Proto(file_path, /*source_shape*/ {4, 6},
                                /*slice_starts*/ {1, 0}, /*slice_sizes*/ {3, 6});

  std::vector<uint8_t> bytes;
  ASSERT_STATUS_OK(UnpackInitializerData(tp, file_path, bytes));
  ASSERT_EQ(bytes.size(), 3u * 6u * sizeof(float));
  const float* read = reinterpret_cast<const float*>(bytes.data());
  auto expected = ExpectedSlice(source, {4, 6}, {1, 0}, {3, 6});
  for (size_t i = 0; i < expected.size(); ++i) {
    ASSERT_EQ(read[i], expected[i]) << "mismatch at " << i;
  }
}

TEST(ExternalDataSliceTest, ReadFp32_2D_InnerAndOuter) {
  // Source (4, 6), slice (rows [1,3), cols [2,5)) => shape (2, 3). Non-contiguous.
  ScopedFileDeleter deleter;
  auto [file_path, source] = WriteFp32SourceFile({4, 6}, deleter);
  auto tp = MakeSlicedFp32Proto(file_path, {4, 6}, {1, 2}, {2, 3});

  std::vector<uint8_t> bytes;
  ASSERT_STATUS_OK(UnpackInitializerData(tp, file_path, bytes));
  ASSERT_EQ(bytes.size(), 2u * 3u * sizeof(float));
  const float* read = reinterpret_cast<const float*>(bytes.data());
  auto expected = ExpectedSlice(source, {4, 6}, {1, 2}, {2, 3});
  for (size_t i = 0; i < expected.size(); ++i) {
    ASSERT_EQ(read[i], expected[i]) << "mismatch at " << i;
  }
}

TEST(ExternalDataSliceTest, ReadFp32_3D_MixedSlice) {
  // Source (2, 3, 4), slice (depth [0,2), rows [1,3), cols [1,3)) => shape (2, 2, 2).
  ScopedFileDeleter deleter;
  auto [file_path, source] = WriteFp32SourceFile({2, 3, 4}, deleter);
  auto tp = MakeSlicedFp32Proto(file_path, {2, 3, 4}, {0, 1, 1}, {2, 2, 2});

  std::vector<uint8_t> bytes;
  ASSERT_STATUS_OK(UnpackInitializerData(tp, file_path, bytes));
  ASSERT_EQ(bytes.size(), 2u * 2u * 2u * sizeof(float));
  const float* read = reinterpret_cast<const float*>(bytes.data());
  auto expected = ExpectedSlice(source, {2, 3, 4}, {0, 1, 1}, {2, 2, 2});
  for (size_t i = 0; i < expected.size(); ++i) {
    ASSERT_EQ(read[i], expected[i]) << "mismatch at " << i;
  }
}

TEST(ExternalDataSliceTest, ReadFp32_WholeTensorSliceIsIdentity) {
  // slice covers the entire source => one read.
  ScopedFileDeleter deleter;
  auto [file_path, source] = WriteFp32SourceFile({3, 5}, deleter);
  auto tp = MakeSlicedFp32Proto(file_path, {3, 5}, {0, 0}, {3, 5});

  std::vector<uint8_t> bytes;
  ASSERT_STATUS_OK(UnpackInitializerData(tp, file_path, bytes));
  ASSERT_EQ(bytes.size(), source.size() * sizeof(float));
  const float* read = reinterpret_cast<const float*>(bytes.data());
  for (size_t i = 0; i < source.size(); ++i) {
    ASSERT_EQ(read[i], source[i]);
  }
}

TEST(ExternalDataSliceTest, ReadFp32_WithSourceByteOffset) {
  // Place the source tensor at byte offset 64 in a file that has 16 floats of padding first.
  std::vector<float> padding(16, -1.0f);
  std::vector<float> source;
  for (size_t i = 0; i < 12; ++i) source.push_back(static_cast<float>(100 + i));

  std::basic_string<ORTCHAR_T> filename_template = ORT_TSTR("ort_slice_off_test_XXXXXX");
  FILE* fp = nullptr;
  CreateTestFile(fp, filename_template);
  std::fwrite(padding.data(), sizeof(float), padding.size(), fp);
  std::fwrite(source.data(), sizeof(float), source.size(), fp);
  std::fclose(fp);
  ScopedFileDeleter deleter{filename_template};
  std::filesystem::path file_path(filename_template);

  // Source is (3, 4), slice (rows [1,3), cols [1,4)) => (2, 3).
  auto tp = MakeSlicedFp32Proto(file_path, {3, 4}, {1, 1}, {2, 3},
                                /*source_byte_offset*/ static_cast<int64_t>(padding.size() * sizeof(float)));

  std::vector<uint8_t> bytes;
  ASSERT_STATUS_OK(UnpackInitializerData(tp, file_path, bytes));
  ASSERT_EQ(bytes.size(), 2u * 3u * sizeof(float));
  const float* read = reinterpret_cast<const float*>(bytes.data());
  auto expected = ExpectedSlice(source, {3, 4}, {1, 1}, {2, 3});
  for (size_t i = 0; i < expected.size(); ++i) {
    ASSERT_EQ(read[i], expected[i]);
  }
}

TEST(ExternalDataSliceTest, ReadInt4_OuterAxisSliceIsOk) {
  // Source (4, 8) UINT4, packed last-dim => 8/2 = 4 bytes per row, 16 bytes total.
  // Slice rows [1,3), keep full innermost dim => shape (2, 8) => 8 bytes.
  std::vector<uint8_t> packed_source;  // row-major, 2 elements per byte
  packed_source.reserve(16);
  for (int i = 0; i < 16; ++i) packed_source.push_back(static_cast<uint8_t>(i + 1));

  std::basic_string<ORTCHAR_T> filename_template = ORT_TSTR("ort_slice_int4_XXXXXX");
  FILE* fp = nullptr;
  CreateTestFile(fp, filename_template);
  std::fwrite(packed_source.data(), 1, packed_source.size(), fp);
  std::fclose(fp);
  ScopedFileDeleter deleter{filename_template};
  std::filesystem::path file_path(filename_template);

  ONNX_NAMESPACE::TensorProto tp;
  tp.set_name("w_int4");
  tp.set_data_type(ONNX_NAMESPACE::TensorProto::UINT4);
  tp.add_dims(2);
  tp.add_dims(8);
  tp.set_data_location(ONNX_NAMESPACE::TensorProto_DataLocation::TensorProto_DataLocation_EXTERNAL);
  AddExtEntry(tp, "location", ToUTF8String(file_path.native()));
  AddExtEntry(tp, "offset", "0");
  AddExtEntry(tp, ExternalDataInfo::kSourceShapeKey, "4,8");
  AddExtEntry(tp, ExternalDataInfo::kSliceStartsKey, "1,0");
  AddExtEntry(tp, ExternalDataInfo::kSliceSizesKey, "2,8");

  std::vector<uint8_t> bytes;
  ASSERT_STATUS_OK(UnpackInitializerData(tp, file_path, bytes));
  // Expect 2 rows of 8 elements = 2 * 4 packed bytes = 8 bytes total.
  ASSERT_EQ(bytes.size(), 8u);
  // Should equal bytes 4..11 of the source (rows 1 and 2, 4 bytes each).
  for (size_t i = 0; i < 8; ++i) {
    ASSERT_EQ(bytes[i], packed_source[4 + i]);
  }
}

TEST(ExternalDataSliceTest, ReadInt4_InnerAxisEvenAlignedSliceIsOk) {
  // Source (4, 8) UINT4, packed last-dim => 8/2 = 4 bytes per row, 16 bytes total.
  // Slice all rows but inner dim [2, 6) => shape (4, 4) = 2 bytes per row => 8 bytes.
  // slice_start=2 and slice_size=4 are both EVEN, so the byte boundary is exact:
  // each row's bytes [1, 3) of the source row map to dst row.
  std::vector<uint8_t> packed_source;
  packed_source.reserve(16);
  for (int i = 0; i < 16; ++i) packed_source.push_back(static_cast<uint8_t>(i + 1));

  std::basic_string<ORTCHAR_T> filename_template = ORT_TSTR("ort_slice_int4_inner_XXXXXX");
  FILE* fp = nullptr;
  CreateTestFile(fp, filename_template);
  std::fwrite(packed_source.data(), 1, packed_source.size(), fp);
  std::fclose(fp);
  ScopedFileDeleter deleter{filename_template};
  std::filesystem::path file_path(filename_template);

  ONNX_NAMESPACE::TensorProto tp;
  tp.set_name("w_int4_inner");
  tp.set_data_type(ONNX_NAMESPACE::TensorProto::UINT4);
  tp.add_dims(4);
  tp.add_dims(4);
  tp.set_data_location(ONNX_NAMESPACE::TensorProto_DataLocation::TensorProto_DataLocation_EXTERNAL);
  AddExtEntry(tp, "location", ToUTF8String(file_path.native()));
  AddExtEntry(tp, "offset", "0");
  AddExtEntry(tp, ExternalDataInfo::kSourceShapeKey, "4,8");
  AddExtEntry(tp, ExternalDataInfo::kSliceStartsKey, "0,2");
  AddExtEntry(tp, ExternalDataInfo::kSliceSizesKey, "4,4");

  std::vector<uint8_t> bytes;
  ASSERT_STATUS_OK(UnpackInitializerData(tp, file_path, bytes));
  // Expect 4 rows of 4 elements = 4 * 2 packed bytes = 8 bytes total.
  ASSERT_EQ(bytes.size(), 8u);
  // For each source row r, copy bytes [1, 3) i.e. source_byte_idx = r*4 + 1 and +2.
  for (size_t r = 0; r < 4; ++r) {
    ASSERT_EQ(bytes[r * 2 + 0], packed_source[r * 4 + 1]) << "row " << r << " byte 0";
    ASSERT_EQ(bytes[r * 2 + 1], packed_source[r * 4 + 2]) << "row " << r << " byte 1";
  }
}

TEST(ExternalDataSliceTest, ReadInt4_InnerAxisOddAlignedSliceIsRejected) {
  std::vector<uint8_t> packed_source(16, 0xAB);
  std::basic_string<ORTCHAR_T> filename_template = ORT_TSTR("ort_slice_int4_rej_XXXXXX");
  FILE* fp = nullptr;
  CreateTestFile(fp, filename_template);
  std::fwrite(packed_source.data(), 1, packed_source.size(), fp);
  std::fclose(fp);
  ScopedFileDeleter deleter{filename_template};
  std::filesystem::path file_path(filename_template);

  ONNX_NAMESPACE::TensorProto tp;
  tp.set_name("w_int4_bad");
  tp.set_data_type(ONNX_NAMESPACE::TensorProto::UINT4);
  tp.add_dims(4);
  tp.add_dims(3);  // odd inner slice_size from source 8
  tp.set_data_location(ONNX_NAMESPACE::TensorProto_DataLocation::TensorProto_DataLocation_EXTERNAL);
  AddExtEntry(tp, "location", ToUTF8String(file_path.native()));
  AddExtEntry(tp, "offset", "0");
  AddExtEntry(tp, ExternalDataInfo::kSourceShapeKey, "4,8");
  AddExtEntry(tp, ExternalDataInfo::kSliceStartsKey, "0,0");
  AddExtEntry(tp, ExternalDataInfo::kSliceSizesKey, "4,3");

  std::vector<uint8_t> bytes;
  ASSERT_STATUS_NOT_OK_AND_HAS_SUBSTR(UnpackInitializerData(tp, file_path, bytes),
                                       "byte-aligned");
}

}  // namespace test
}  // namespace onnxruntime
