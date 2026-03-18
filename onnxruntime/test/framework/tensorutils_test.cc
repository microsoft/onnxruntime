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
#include <fstream>
#include <utility>

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

// Test fixture for creating temporary directories and files for path validation tests.
class PathValidationTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Create a temporary directory for the tests.
    base_dir_ = std::filesystem::temp_directory_path() / "PathValidationTest";
    outside_dir_ = std::filesystem::temp_directory_path() / "outside";
    std::filesystem::create_directories(base_dir_);
    std::filesystem::create_directories(outside_dir_);
  }

  void TearDown() override {
    // Clean up the temporary directory.
    std::filesystem::remove_all(base_dir_);
    std::filesystem::remove_all(outside_dir_);

    for (const auto& other_dir : other_dirs_) {
      std::filesystem::remove_all(other_dir);
    }

    for (const auto& other_file : other_files_) {
      std::filesystem::remove(other_file);
    }
  }

  // Create directory that will be removed during test teardown.
  void CreateDirectories(std::filesystem::path dir) {
    std::filesystem::create_directories(dir);
    other_dirs_.push_back(std::move(dir));
  }

  // Create empty file that will be removed during test teardown.
  void CreateEmptyFile(std::filesystem::path file_path) {
    std::ofstream{file_path};
    other_files_.push_back(std::move(file_path));
  }

  std::filesystem::path base_dir_;
  std::filesystem::path outside_dir_;
  std::vector<std::filesystem::path> other_dirs_;
  std::vector<std::filesystem::path> other_files_;
};

// Test cases for ValidateExternalDataPath.
TEST_F(PathValidationTest, ValidateExternalDataPath) {
  std::filesystem::path model_path = base_dir_ / "model.onnx";
  std::filesystem::path cwd = std::filesystem::current_path();
  const bool is_cwd_root = cwd == cwd.root_path();

  // Create empty external data files that we'll need for testing.
  CreateEmptyFile(base_dir_ / "data.bin");
  CreateDirectories(base_dir_ / "sub");
  CreateEmptyFile(base_dir_ / "sub" / "data.bin");
  CreateEmptyFile(cwd / "data.bin");
  CreateDirectories(cwd / "abc");
  CreateEmptyFile(cwd / "abc" / "data.bin");
  CreateEmptyFile(cwd / "data..bin");

  // Valid relative path.
  ASSERT_STATUS_OK(utils::ValidateExternalDataPath(model_path, "data.bin"));

  // Empty location not allowed.
  {
    Status status = utils::ValidateExternalDataPath(model_path, "");
    ASSERT_THAT(status.ErrorMessage(), ::testing::HasSubstr("Empty external data path"));
  }

  // Path with ".." that escapes the base directory.
  ASSERT_FALSE(utils::ValidateExternalDataPath(model_path, "../data.bin").IsOK());

  // Absolute path.
  {
    Status status;
#ifdef _WIN32
    status = utils::ValidateExternalDataPath(model_path, "C:\\data.bin");
    ASSERT_THAT(status.ErrorMessage(), ::testing::HasSubstr("Absolute path not allowed"));

    status = utils::ValidateExternalDataPath("", "C:\\data.bin");
    ASSERT_THAT(status.ErrorMessage(), ::testing::HasSubstr("Absolute path not allowed"));
#endif  // Absolute path.

    // Paths starting from / should be rejected even on Windows.
    status = utils::ValidateExternalDataPath(model_path, "/data.bin");
    ASSERT_THAT(status.ErrorMessage(), ::testing::HasSubstr("Absolute path not allowed"));

    status = utils::ValidateExternalDataPath("", "/data.bin");
    ASSERT_THAT(status.ErrorMessage(), ::testing::HasSubstr("Absolute path not allowed"));
  }

  // Windows vs Unix path separators.
#ifdef _WIN32
  ASSERT_STATUS_OK(utils::ValidateExternalDataPath(model_path, "sub\\data.bin"));
#endif
  ASSERT_STATUS_OK(utils::ValidateExternalDataPath(model_path, "sub/data.bin"));

  // Model in a directory that does not exist.
  ASSERT_FALSE(utils::ValidateExternalDataPath("non_existent_dir/model.onnx", "data.bin").IsOK());

  // Model path is a bare filename (no directory component).
  ASSERT_STATUS_OK(utils::ValidateExternalDataPath("model.onnx", "data.bin"));
  ASSERT_EQ(utils::ValidateExternalDataPath("model.onnx", "../data.bin").IsOK(), is_cwd_root);

  // Model relative path checks.
  ASSERT_STATUS_OK(utils::ValidateExternalDataPath("./model.onnx", "data.bin"));
  ASSERT_EQ(utils::ValidateExternalDataPath("./model.onnx", "../data.bin").IsOK(), is_cwd_root);
#ifdef _WIN32
  ASSERT_STATUS_OK(utils::ValidateExternalDataPath(".\\model.onnx", "data.bin"));
  ASSERT_EQ(utils::ValidateExternalDataPath(".\\model.onnx", "../data.bin").IsOK(), is_cwd_root);
#endif

  ASSERT_STATUS_OK(utils::ValidateExternalDataPath("./abc/model.onnx", "data.bin"));
#ifdef _WIN32
  ASSERT_STATUS_OK(utils::ValidateExternalDataPath(".\\abc\\model.onnx", "data.bin"));
#endif

  //
  // Tests for an empty model path (model loaded from bytes).
  // The model path would be empty when 1) the session loads a model from bytes and 2) the application does not
  // set an external file folder path via the session config option
  // kOrtSessionOptionsModelExternalInitializersFileFolderPath.
  //

  // A simple filename is ok (would not escape current working directory).
  ASSERT_STATUS_OK(utils::ValidateExternalDataPath("", "data.bin"));
  ASSERT_STATUS_OK(utils::ValidateExternalDataPath("", "./data.bin"));

  // A ".." that is not a path component (part of the filename) is ok
  ASSERT_STATUS_OK(utils::ValidateExternalDataPath("", "data..bin"));

  // A path that would escape the current working directory is invalid.
  ASSERT_EQ(utils::ValidateExternalDataPath("", "../data.bin").IsOK(), is_cwd_root);

  // A path that uses ".." but would not escape the current working directory should be fine.
  ASSERT_STATUS_OK(utils::ValidateExternalDataPath("", "a/../data.bin"));

  // A path with multiple internal ".." that would escape current working direction should fail.
  ASSERT_EQ(utils::ValidateExternalDataPath("", "a/../../data.bin").IsOK(), is_cwd_root);
}

TEST_F(PathValidationTest, ValidateExternalDataPathWithSymlinkInside) {
  // Symbolic link that points inside the base directory.
  auto model_path = base_dir_ / "model.onnx";
  try {
    auto target = base_dir_ / "target.bin";
    std::ofstream{target};
    auto link = base_dir_ / "link.bin";
    std::filesystem::create_symlink(target, link);
  } catch (const std::exception& e) {
    GTEST_SKIP() << "Skipping symlink tests since symlink creation is not supported in this environment. Exception: "
                 << e.what();
  }
  ASSERT_STATUS_OK(utils::ValidateExternalDataPath(model_path, "link.bin"));
}

TEST_F(PathValidationTest, ValidateExternalDataPathWithSymlinkOutside) {
  // Symbolic link that points outside the base directory.
  auto model_path = base_dir_ / "model.onnx";
  auto outside_target = outside_dir_ / "outside.bin";
  try {
    {
      std::ofstream{outside_target};
      auto outside_link = base_dir_ / "outside_link.bin";
      std::filesystem::create_symlink(outside_target, outside_link);
    }
  } catch (const std::exception& e) {
    GTEST_SKIP() << "Skipping symlink tests since symlink creation is not supported in this environment. Exception: " << e.what();
  }
  ASSERT_FALSE(utils::ValidateExternalDataPath(model_path, "outside_link.bin").IsOK());
}

TEST_F(PathValidationTest, ValidateExternalDataPathEmptyModelPathWithSymlinkInside) {
  // Test external data path validation when the model path is empty.
  // Specifically tests that the following scenario is valid:
  //   - A symbolic link within the current working directory pointing to a file still within CWD.
  try {
    std::filesystem::path cwd = std::filesystem::current_path();
    std::filesystem::path sub_dir = cwd / "symlink_test_subdir";
    CreateDirectories(sub_dir);

    std::filesystem::path target = sub_dir / "target_inside.bin";
    std::filesystem::path symlink = sub_dir / "link_inside.bin";
    std::ofstream{target};
    std::filesystem::create_symlink(target, symlink);
  } catch (const std::exception& e) {
    GTEST_SKIP() << "Skipping test due to failure setting up directory and symlink files: "
                 << e.what();
  }

  EXPECT_STATUS_OK(utils::ValidateExternalDataPath("", "./symlink_test_subdir/link_inside.bin"));
}

TEST_F(PathValidationTest, ValidateExternalDataPathEmptyModelPathWithSymlinkOutside) {
  // Test external data path validation when the model path is empty.
  // Specifically tests that the following scenario is NOT valid:
  //  - A symbolic link within the current working directory pointing to a file outside CWD.
  try {
    std::filesystem::path cwd = std::filesystem::current_path();
    std::filesystem::path sub_dir = cwd / "symlink_test_subdir2";
    CreateDirectories(sub_dir);

    // Check if we can actually make a file outside of the current working directory (i.e., in a temp dir).
    // This is only possible if the current working directory is NOT the same as the temp directory.
    // Otherwise, we need to skip this test. This happens in Android CI.
    auto [cwd_end, outside_end] = std::mismatch(cwd.begin(), cwd.end(), outside_dir_.begin(), outside_dir_.end());
    if (cwd_end == cwd.end()) {
      GTEST_SKIP() << "Skipping test that needs to create a symlink outside of the cwd because the cwd is the same as "
                   << "the temp dir. cwd: " << cwd << " outside_dir_: " << outside_dir_;
    }

    std::filesystem::path outside_target = outside_dir_ / "outside_for_empty_basedir.bin";
    std::filesystem::path symlink = sub_dir / "outside_link.bin";
    std::ofstream{outside_target};
    std::filesystem::create_symlink(outside_target, symlink);
  } catch (const std::exception& e) {
    GTEST_SKIP() << "Skipping test due to failure setting up directory and symlink files: "
                 << e.what();
  }

  Status status = utils::ValidateExternalDataPath("", "./symlink_test_subdir2/outside_link.bin");
  ASSERT_FALSE(status.IsOK());
  EXPECT_THAT(status.ErrorMessage(), testing::HasSubstr("escapes working directory"));
}

// Tests for ValidateEmbeddedTensorProtoDataSizeAndShape and embedded initializer size limits

TEST(TensorProtoDataSizeShapeValidationTest, ValidTensorProtoWithRawData) {
  // A valid float tensor with 4 elements and matching raw_data
  TensorProto tensor_proto;
  tensor_proto.set_name("valid_raw");
  tensor_proto.set_data_type(TensorProto_DataType_FLOAT);
  tensor_proto.add_dims(2);
  tensor_proto.add_dims(2);
  // 4 floats = 16 bytes
  std::string raw(16, '\0');
  tensor_proto.set_raw_data(raw);

  ASSERT_STATUS_OK(utils::ValidateEmbeddedTensorProtoDataSizeAndShape(tensor_proto));
}

TEST(TensorProtoDataSizeShapeValidationTest, ValidTensorProtoWithTypedData) {
  // A valid float tensor with typed float_data
  TensorProto tensor_proto;
  tensor_proto.set_name("valid_typed");
  tensor_proto.set_data_type(TensorProto_DataType_FLOAT);
  tensor_proto.add_dims(3);
  tensor_proto.add_float_data(1.0f);
  tensor_proto.add_float_data(2.0f);
  tensor_proto.add_float_data(3.0f);

  ASSERT_STATUS_OK(utils::ValidateEmbeddedTensorProtoDataSizeAndShape(tensor_proto));
}

TEST(TensorProtoDataSizeShapeValidationTest, ValidZeroElementTensor) {
  // A valid zero-element tensor (one dim is 0)
  TensorProto tensor_proto;
  tensor_proto.set_name("zero_elem");
  tensor_proto.set_data_type(TensorProto_DataType_FLOAT);
  tensor_proto.add_dims(0);
  tensor_proto.add_dims(5);

  ASSERT_STATUS_OK(utils::ValidateEmbeddedTensorProtoDataSizeAndShape(tensor_proto));
}

TEST(TensorProtoDataSizeShapeValidationTest, LargeDimsNoDataRejected) {
  // Malicious: large dims but no data at all
  TensorProto tensor_proto;
  tensor_proto.set_name("malicious_no_data");
  tensor_proto.set_data_type(TensorProto_DataType_FLOAT);
  tensor_proto.add_dims(10000);
  tensor_proto.add_dims(10000);
  // No raw_data or float_data set

  auto status = utils::ValidateEmbeddedTensorProtoDataSizeAndShape(tensor_proto);
  ASSERT_FALSE(status.IsOK());
  EXPECT_THAT(status.ErrorMessage(), ::testing::HasSubstr("does not match expected count from shape"));
}

TEST(TensorProtoDataSizeShapeValidationTest, LargeDimsSmallRawDataRejected) {
  // Malicious: large dims with tiny raw_data
  TensorProto tensor_proto;
  tensor_proto.set_name("malicious_small_raw");
  tensor_proto.set_data_type(TensorProto_DataType_FLOAT);
  tensor_proto.add_dims(10000);
  tensor_proto.add_dims(10000);
  // Only 4 bytes of raw data (1 float), but shape says 100M elements
  std::string raw(4, '\0');
  tensor_proto.set_raw_data(raw);

  auto status = utils::ValidateEmbeddedTensorProtoDataSizeAndShape(tensor_proto);
  ASSERT_FALSE(status.IsOK());
  EXPECT_THAT(status.ErrorMessage(), ::testing::HasSubstr("does not match expected size from shape and data type"));
}

TEST(TensorProtoDataSizeShapeValidationTest, LargeDimsSmallTypedDataRejected) {
  // Malicious: large dims with just a few typed data elements
  TensorProto tensor_proto;
  tensor_proto.set_name("malicious_small_typed");
  tensor_proto.set_data_type(TensorProto_DataType_FLOAT);
  tensor_proto.add_dims(10000);
  tensor_proto.add_dims(10000);
  tensor_proto.add_float_data(1.0f);

  auto status = utils::ValidateEmbeddedTensorProtoDataSizeAndShape(tensor_proto);
  ASSERT_FALSE(status.IsOK());
  EXPECT_THAT(status.ErrorMessage(), ::testing::HasSubstr("does not match expected count from shape"));
}

TEST(TensorProtoDataSizeShapeValidationTest, EmbeddedInitializerExceeding2GiBRejected) {
  // A tensor whose declared shape exceeds 2 GiB should be rejected by TensorProtoToOrtValue and
  // CreateTensorFromTensorProto.
  TensorProto tensor_proto;
  tensor_proto.set_name("too_large");
  tensor_proto.set_data_type(TensorProto_DataType_FLOAT);
  // 536870913 floats * 4 bytes = 2147483652 bytes > 2 GiB
  tensor_proto.add_dims(536870913);
  // No data — the 2 GiB check should trigger before the consistency check

  // Test call to TensorProtoToOrtValue
  {
    OrtValue ort_value;
    auto status = utils::TensorProtoToOrtValue(Env::Default(), std::filesystem::path{},
                                               tensor_proto, CPUAllocator::DefaultInstance(), ort_value);
    ASSERT_FALSE(status.IsOK());
    EXPECT_THAT(status.ErrorMessage(), ::testing::HasSubstr("exceeds the 2147483648 byte limit"));
  }

  // Test call to CreateTensorFromTensorProto
  {
    Tensor tensor;
    auto status = utils::CreateTensorFromTensorProto(Env::Default(), std::filesystem::path{},
                                                     tensor_proto, tensor);
    ASSERT_FALSE(status.IsOK());
    EXPECT_THAT(status.ErrorMessage(), ::testing::HasSubstr("exceeds the 2147483648 byte limit"));
  }
}

TEST(TensorProtoDataSizeShapeValidationTest, ValidStringTensorProto) {
  // A valid string tensor with matching string_data
  TensorProto tensor_proto;
  tensor_proto.set_name("valid_string");
  tensor_proto.set_data_type(TensorProto_DataType_STRING);
  tensor_proto.add_dims(2);
  tensor_proto.add_string_data("hello");
  tensor_proto.add_string_data("world");

  ASSERT_STATUS_OK(utils::ValidateEmbeddedTensorProtoDataSizeAndShape(tensor_proto));
}

TEST(TensorProtoDataSizeShapeValidationTest, StringTensorWithMismatchedCountRejected) {
  TensorProto tensor_proto;
  tensor_proto.set_name("bad_string");
  tensor_proto.set_data_type(TensorProto_DataType_STRING);
  tensor_proto.add_dims(100);
  tensor_proto.add_string_data("only_one");

  auto status = utils::ValidateEmbeddedTensorProtoDataSizeAndShape(tensor_proto);
  ASSERT_FALSE(status.IsOK());
  EXPECT_THAT(status.ErrorMessage(), ::testing::HasSubstr("does not match expected count from shape"));
}

TEST(TensorProtoDataSizeShapeValidationTest, NegativeDimsRejected) {
  TensorProto tensor_proto;
  tensor_proto.set_name("negative_dims");
  tensor_proto.set_data_type(TensorProto_DataType_FLOAT);
  tensor_proto.add_dims(-1);
  tensor_proto.add_dims(10);

  auto status = utils::ValidateEmbeddedTensorProtoDataSizeAndShape(tensor_proto);
  ASSERT_FALSE(status.IsOK());
  EXPECT_THAT(status.ErrorMessage(), ::testing::HasSubstr("negative dimensions"));
}

#if !defined(__wasm__)
// Tests for external data file size validation in ReadExternalDataForTensor.
// These verify that the file size is checked before allocating memory for the tensor.

TEST(TensorProtoDataSizeShapeValidationTest, ExternalDataFileTooSmallForDeclaredShape) {
  // Create a small external data file with 4 floats (16 bytes)
  std::basic_string<ORTCHAR_T> filename(ORT_TSTR("ext_small_XXXXXX"));
  FILE* fp;
  CreateTestFile(fp, filename);
  const float small_data[] = {1.0f, 2.0f, 3.0f, 4.0f};
  ASSERT_EQ(sizeof(small_data), fwrite(small_data, 1, sizeof(small_data), fp));
  ASSERT_EQ(0, fclose(fp));
  std::unique_ptr<ORTCHAR_T, decltype(&DeleteFileFromDisk)> file_deleter(
      const_cast<ORTCHAR_T*>(filename.c_str()), DeleteFileFromDisk);

  // Declare a tensor with 1000 floats (4000 bytes) but the file only has 16 bytes
  TensorProto tensor_proto;
  tensor_proto.set_name("malicious_external");
  tensor_proto.set_data_type(TensorProto_DataType_FLOAT);
  tensor_proto.add_dims(1000);
  tensor_proto.set_data_location(TensorProto_DataLocation_EXTERNAL);
  auto* location = tensor_proto.add_external_data();
  location->set_key("location");
  location->set_value(ToUTF8String(filename));

  std::vector<uint8_t> unpacked_tensor;
  auto status = utils::UnpackInitializerData(tensor_proto, std::filesystem::path{}, unpacked_tensor);
  ASSERT_FALSE(status.IsOK());
  EXPECT_THAT(status.ErrorMessage(), ::testing::HasSubstr("out of bounds"));
}

TEST(TensorProtoDataSizeShapeValidationTest, ExternalDataOffsetPushesReadPastEndOfFile) {
  // Create an external data file with 4 floats (16 bytes)
  std::basic_string<ORTCHAR_T> filename(ORT_TSTR("ext_offset_XXXXXX"));
  FILE* fp;
  CreateTestFile(fp, filename);
  const float data[] = {1.0f, 2.0f, 3.0f, 4.0f};
  ASSERT_EQ(sizeof(data), fwrite(data, 1, sizeof(data), fp));
  ASSERT_EQ(0, fclose(fp));
  std::unique_ptr<ORTCHAR_T, decltype(&DeleteFileFromDisk)> file_deleter(
      const_cast<ORTCHAR_T*>(filename.c_str()), DeleteFileFromDisk);

  // Declare a tensor with 4 floats (16 bytes) but at offset 8, so read needs bytes [8..24) but file is only 16 bytes
  TensorProto tensor_proto;
  tensor_proto.set_name("offset_external");
  tensor_proto.set_data_type(TensorProto_DataType_FLOAT);
  tensor_proto.add_dims(4);
  tensor_proto.set_data_location(TensorProto_DataLocation_EXTERNAL);
  auto* location = tensor_proto.add_external_data();
  location->set_key("location");
  location->set_value(ToUTF8String(filename));
  auto* offset = tensor_proto.add_external_data();
  offset->set_key("offset");
  offset->set_value("8");

  std::vector<uint8_t> unpacked_tensor;
  auto status = utils::UnpackInitializerData(tensor_proto, std::filesystem::path{}, unpacked_tensor);
  ASSERT_FALSE(status.IsOK());
  EXPECT_THAT(status.ErrorMessage(), ::testing::HasSubstr("out of bounds"));
}

TEST(TensorProtoDataSizeShapeValidationTest, ExternalDataValidFileSizeSucceeds) {
  // Create an external data file with exactly 4 floats (16 bytes)
  std::basic_string<ORTCHAR_T> filename(ORT_TSTR("ext_valid_XXXXXX"));
  FILE* fp;
  CreateTestFile(fp, filename);
  const float data[] = {1.0f, 2.0f, 3.0f, 4.0f};
  ASSERT_EQ(sizeof(data), fwrite(data, 1, sizeof(data), fp));
  ASSERT_EQ(0, fclose(fp));
  std::unique_ptr<ORTCHAR_T, decltype(&DeleteFileFromDisk)> file_deleter(
      const_cast<ORTCHAR_T*>(filename.c_str()), DeleteFileFromDisk);

  // Declare a tensor with matching shape (4 floats = 16 bytes)
  TensorProto tensor_proto;
  tensor_proto.set_name("valid_external");
  tensor_proto.set_data_type(TensorProto_DataType_FLOAT);
  tensor_proto.add_dims(4);
  tensor_proto.set_data_location(TensorProto_DataLocation_EXTERNAL);
  auto* location = tensor_proto.add_external_data();
  location->set_key("location");
  location->set_value(ToUTF8String(filename));

  std::vector<uint8_t> unpacked_tensor;
  ASSERT_STATUS_OK(utils::UnpackInitializerData(tensor_proto, std::filesystem::path{}, unpacked_tensor));
  ASSERT_EQ(unpacked_tensor.size(), sizeof(data));
}
#endif  // !defined(__wasm__)

}  // namespace test
}  // namespace onnxruntime
