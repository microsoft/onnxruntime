// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// needs to be included first to get around onnxruntime\cmake\external\onnx\onnx/common/constants.h(14): error C2513: 'bool': no variable declared before '='

#include "TestCase.h"

#include <cctype>
#include <filesystem>
#include <fstream>
#include <memory>
#include <sstream>
#include <map>
#include <regex>
#include <set>
#include <string>

#include "callback.h"
#include "heap_buffer.h"
#include "mem_buffer.h"
#include "onnx_model_info.h"
#include "pb_helper.h"
#include "tensorprotoutils.h"

#include "core/common/logging/logging.h"
#include "core/common/common.h"
#include "core/platform/env.h"
#include "core/platform/ort_mutex.h"
#include "core/platform/path_lib.h"
#include "core/session/onnxruntime_cxx_api.h"
#include "core/framework/allocator.h"
#include "core/framework/TensorSeq.h"
#include "re2/re2.h"

using namespace onnxruntime;
using namespace onnxruntime::common;
using google::protobuf::RepeatedPtrField;

static constexpr int protobuf_block_size_in_bytes = 4 * 1024 * 1024;

const std::string TestModelInfo::unknown_version = "unknown version";

namespace {

template <typename T>
inline Ort::Value CreateTensorWithDataAsOrtValue(const Ort::MemoryInfo& info,
                                                 OrtAllocator*,
                                                 const std::vector<int64_t>& dims,
                                                 std::vector<T>& input) {
  return Ort::Value::CreateTensor<T>(static_cast<const OrtMemoryInfo*>(info), input.data(), input.size() * sizeof(T),
                                     dims.data(), dims.size());
}

template <>
inline Ort::Value CreateTensorWithDataAsOrtValue(const Ort::MemoryInfo&,
                                                 OrtAllocator* allocator,
                                                 const std::vector<int64_t>& dims,
                                                 std::vector<std::string>& input) {
  auto tensor_value = Ort::Value::CreateTensor(allocator, dims.data(), dims.size(),
                                               ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING);

  std::vector<const char*> p_str;
  for (const auto& s : input) {
    p_str.push_back(s.c_str());
  }

  tensor_value.FillStringTensor(p_str.data(), p_str.size());
  return tensor_value;
}

template <typename key_type, typename value_type>
Ort::Value PbMapToOrtValue(const google::protobuf::Map<key_type, value_type>& map) {
  auto info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);
  Ort::AllocatorWithDefaultOptions allocator;
  const size_t ele_count = map.size();
  std::vector<int64_t> dims(1, static_cast<int64_t>(ele_count));
  std::vector<key_type> keys(ele_count);
  std::vector<value_type> values(ele_count);
  size_t i = 0;
  for (auto& kvp : map) {
    keys[i] = kvp.first;
    values[i] = kvp.second;
    ++i;
  }

  //// See helper above
  auto ort_keys = CreateTensorWithDataAsOrtValue(info, allocator, dims, keys);
  auto ort_values = CreateTensorWithDataAsOrtValue(info, allocator, dims, values);
  return Ort::Value::CreateMap(ort_keys, ort_values);
}

template <typename T>
Ort::Value VectorProtoToOrtValue(const RepeatedPtrField<T>& input) {
  auto info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);
  Ort::AllocatorWithDefaultOptions allocator;
  std::vector<Ort::Value> seq;
  seq.reserve(input.size());
  for (const T& v : input) {
    // create key tensor
    const auto& map = v.v();
    size_t ele_count = map.size();
    using key_type = typename std::remove_reference<decltype(v.v())>::type::key_type;
    using value_type = typename std::remove_reference<decltype(v.v())>::type::mapped_type;
    std::vector<int64_t> dims(1, static_cast<int64_t>(ele_count));
    std::vector<key_type> keys(ele_count);
    std::vector<value_type> values(ele_count);
    size_t i = 0;
    for (auto& kvp : map) {
      keys[i] = kvp.first;
      values[i] = kvp.second;
      ++i;
    }

    auto ort_keys = CreateTensorWithDataAsOrtValue(info, allocator, dims, keys);
    auto ort_values = CreateTensorWithDataAsOrtValue(info, allocator, dims, values);
    auto ort_map = Ort::Value::CreateMap(ort_keys, ort_values);
    seq.push_back(std::move(ort_map));
  }
  return Ort::Value::CreateSequence(seq);
}

template <typename CHAR_T>
static int ExtractFileNo(const std::basic_string<CHAR_T>& name) {
  size_t p1 = name.rfind('.');
  size_t p2 = name.rfind('_', p1);
  ++p2;
  std::basic_string<CHAR_T> number_str = name.substr(p2, p1 - p2);
  const CHAR_T* start = number_str.c_str();
  const CHAR_T* end = number_str.c_str();
  long ret = OrtStrtol(start, const_cast<CHAR_T**>(&end));
  if (end == start) {
    ORT_THROW("parse file name failed");
  }
  return static_cast<int>(ret);
}
using PATH_STRING_TYPE = std::basic_string<PATH_CHAR_TYPE>;

static void SortFileNames(std::vector<std::basic_string<PATH_CHAR_TYPE>>& input_pb_files) {
  if (input_pb_files.size() <= 1) return;
  std::sort(input_pb_files.begin(), input_pb_files.end(),
            [](const std::basic_string<PATH_CHAR_TYPE>& left, const std::basic_string<PATH_CHAR_TYPE>& right) -> bool {
              std::basic_string<PATH_CHAR_TYPE> leftname = GetLastComponent(left);
              std::basic_string<PATH_CHAR_TYPE> rightname = GetLastComponent(right);
              int left1 = ExtractFileNo(leftname);
              int right1 = ExtractFileNo(rightname);
              return left1 < right1;
            });

  for (size_t i = 0; i != input_pb_files.size(); ++i) {
    int fileno = ExtractFileNo(GetLastComponent(input_pb_files[i]));
    if (static_cast<size_t>(fileno) != i) {
      std::basic_ostringstream<PATH_CHAR_TYPE> oss;
      oss << input_pb_files[0];
      for (size_t j = 1; j != input_pb_files.size(); ++j)
        oss << ORT_TSTR(" ") << input_pb_files[j];
      ORT_THROW("illegal input file name:", ToUTF8String(oss.str()));
    }
  }
}

Ort::Value TensorToOrtValue(const ONNX_NAMESPACE::TensorProto& t, onnxruntime::test::HeapBuffer& b) {
  size_t len = 0;
  auto status = onnxruntime::test::GetSizeInBytesFromTensorProto<0>(t, &len);
  if (!status.IsOK()) {
    ORT_THROW(status.ToString());
  }
  void* p = len == 0 ? nullptr : b.AllocMemory(len);
  Ort::Value temp_value{nullptr};
  onnxruntime::test::OrtCallback d;
  auto cpu_memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);
  status = onnxruntime::test::TensorProtoToMLValue(t, onnxruntime::test::MemBuffer(p, len, *static_cast<OrtMemoryInfo*>(cpu_memory_info)),
                                                   temp_value, d);
  if (!status.IsOK()) {
    ORT_THROW(status.ToString());
  }
  if (d.f) {
    b.AddDeleter(d);
  }
  return temp_value;
}

void LoopDataFile(int test_data_pb_fd, bool is_input, const TestModelInfo& modelinfo,
                  std::unordered_map<std::string, Ort::Value>& name_data_map, onnxruntime::test::HeapBuffer& b,
                  std::ostringstream& oss) {
  google::protobuf::io::FileInputStream f(test_data_pb_fd, protobuf_block_size_in_bytes);
  f.SetCloseOnDelete(true);
  google::protobuf::io::CodedInputStream coded_input(&f);
  bool clean_eof = false;
  [[maybe_unused]] int item_id = 1;
  for (proto::TraditionalMLData data;
       ParseDelimitedFromCodedStream(&data, &coded_input, &clean_eof);
       ++item_id, data.Clear()) {
    ORT_TRY {
      Ort::Value gvalue{nullptr};
      switch (data.values_case()) {
        case proto::TraditionalMLData::kVectorMapStringToFloat:
          gvalue = VectorProtoToOrtValue(data.vector_map_string_to_float().v());
          break;
        case proto::TraditionalMLData::kVectorMapInt64ToFloat:
          gvalue = VectorProtoToOrtValue(data.vector_map_int64_to_float().v());
          break;
        case proto::TraditionalMLData::kMapStringToString:
          gvalue = PbMapToOrtValue(data.map_string_to_string().v());
          break;
        case proto::TraditionalMLData::kMapStringToInt64:
          gvalue = PbMapToOrtValue(data.map_string_to_int64().v());
          break;
        case proto::TraditionalMLData::kMapStringToFloat:
          gvalue = PbMapToOrtValue(data.map_string_to_float().v());
          break;
        case proto::TraditionalMLData::kMapStringToDouble:
          gvalue = PbMapToOrtValue(data.map_string_to_double().v());
          break;
        case proto::TraditionalMLData::kMapInt64ToString:
          gvalue = PbMapToOrtValue(data.map_int64_to_string().v());
          break;
        case proto::TraditionalMLData::kMapInt64ToInt64:
          gvalue = PbMapToOrtValue(data.map_int64_to_int64().v());
          break;
        case proto::TraditionalMLData::kMapInt64ToFloat:
          gvalue = PbMapToOrtValue(data.map_int64_to_float().v());
          break;
        case proto::TraditionalMLData::kMapInt64ToDouble:
          gvalue = PbMapToOrtValue(data.map_int64_to_double().v());
          break;
        case proto::TraditionalMLData::kTensor: {
          gvalue = TensorToOrtValue(data.tensor(), b);
        } break;
        default:
          ORT_NOT_IMPLEMENTED("unknown data type inside TraditionalMLData");
      }
      if (!data.debug_info().empty()) {
        oss << ":" << data.debug_info();
      }
      std::string value_name = data.name();
      if (value_name.empty()) {
        const size_t c = name_data_map.size();
        value_name = is_input ? modelinfo.GetInputName(c) : modelinfo.GetOutputName(c);
      }

      auto p = name_data_map.emplace(value_name, std::move(gvalue));
      if (!p.second) {
        ORT_THROW("duplicated test data name");
        break;
      }
    }
    ORT_CATCH(onnxruntime::NotImplementedException & ex) {
      ORT_HANDLE_EXCEPTION([&]() {
        std::ostringstream oss2;
        oss2 << "load the " << item_id << "-th item failed," << ex.what();
        ORT_NOT_IMPLEMENTED(oss2.str());
      });
    }
    ORT_CATCH(const std::exception& ex) {
      ORT_HANDLE_EXCEPTION([&]() {
        std::ostringstream oss2;
        oss2 << "load the " << item_id << "-th item failed," << ex.what();
        ORT_THROW(oss2.str());
      });
    }
  }
  if (!clean_eof) {
    ORT_THROW("parse input file failed, has extra unparsed data");
  }
}

}  // namespace

#if !defined(ORT_MINIMAL_BUILD)
std::unique_ptr<TestModelInfo> TestModelInfo::LoadOnnxModel(_In_ const PATH_CHAR_TYPE* model_url) {
  return std::make_unique<OnnxModelInfo>(model_url);
}
#endif

std::unique_ptr<TestModelInfo> TestModelInfo::LoadOrtModel(_In_ const PATH_CHAR_TYPE* model_url) {
  return std::make_unique<OnnxModelInfo>(model_url, true);
}

/**
 * test_case_dir must have contents of:
 * model.onnx
 * ???/input_??.pb
 * ???/output_??.pb
 * ???/input_??.pb
 * ???/output_??.pb
 */
class OnnxTestCase : public ITestCase {
 private:
  std::string test_case_name_;
  mutable std::vector<std::string> debuginfo_strings_;
  mutable onnxruntime::OrtMutex m_;

  std::vector<std::basic_string<PATH_CHAR_TYPE>> test_data_dirs_;

  std::string GetDatasetDebugInfoString(size_t dataset_id) const override {
    std::lock_guard<OrtMutex> l(m_);
    if (dataset_id < debuginfo_strings_.size()) {
      return debuginfo_strings_[dataset_id];
    }
    // return empty string
    return std::string();
  }

  void ConvertTestData(const ONNX_NAMESPACE::TensorProto& test_data_pb,
                       onnxruntime::test::HeapBuffer& b,
                       bool is_input, size_t i,
                       std::unordered_map<std::string, Ort::Value>& out) const;

  void ConvertTestData(const ONNX_NAMESPACE::SequenceProto& test_data_pb,
                       onnxruntime::test::HeapBuffer& b,
                       bool is_input, size_t i,
                       std::unordered_map<std::string, Ort::Value>& out) const;

#if !defined(DISABLE_OPTIONAL_TYPE)
  void ConvertTestData(const ONNX_NAMESPACE::OptionalProto& test_data_pb,
                       onnxruntime::test::HeapBuffer& b,
                       bool is_input, size_t i,
                       std::unordered_map<std::string, Ort::Value>& out) const;
#endif

  std::once_flag model_parsed_;
  std::once_flag config_parsed_;
  double per_sample_tolerance_;
  double relative_per_sample_tolerance_;
  bool post_processing_;
  std::unique_ptr<TestModelInfo> model_info_;
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(OnnxTestCase);

 public:
  OnnxTestCase(const std::string& test_case_name, _In_ std::unique_ptr<TestModelInfo> model,
               double default_per_sample_tolerance, double default_relative_per_sample_tolerance);
  void GetPerSampleTolerance(double* value) const override;
  void GetRelativePerSampleTolerance(double* value) const override;
  void GetPostProcessing(bool* value) const override;

  const ONNX_NAMESPACE::ValueInfoProto* GetInputInfoFromModel(size_t i) const override {
    return model_info_->GetInputInfoFromModel(i);
  }

  const ONNX_NAMESPACE::ValueInfoProto* GetOutputInfoFromModel(size_t i) const override {
    return model_info_->GetOutputInfoFromModel(i);
  }

  size_t GetDataCount() const override { return test_data_dirs_.size(); }
  const std::string& GetNodeName() const override { return model_info_->GetNodeName(); }
  const PATH_CHAR_TYPE* GetModelUrl() const override { return model_info_->GetModelUrl(); }
  const std::string& GetTestCaseName() const override { return test_case_name_; }
  std::string GetTestCaseVersion() const override { return model_info_->GetNominalOpsetVersion(); }

  void LoadTestData(size_t id, onnxruntime::test::HeapBuffer& b, std::unordered_map<std::string, Ort::Value>&,
                    bool is_input) const override;
};

std::unique_ptr<ITestCase> CreateOnnxTestCase(const std::string& test_case_name,
                                              std::unique_ptr<TestModelInfo> model,
                                              double default_per_sample_tolerance,
                                              double default_relative_per_sample_tolerance) {
  return std::make_unique<OnnxTestCase>(test_case_name, std::move(model),
                                        default_per_sample_tolerance,
                                        default_relative_per_sample_tolerance);
}

void OnnxTestCase::GetPerSampleTolerance(double* value) const {
  *value = per_sample_tolerance_;
}

void OnnxTestCase::GetRelativePerSampleTolerance(double* value) const {
  *value = relative_per_sample_tolerance_;
}

void OnnxTestCase::GetPostProcessing(bool* value) const {
  *value = post_processing_;
}

// CentOS lacks find_if
template <class Iter, class Pred>
inline Iter find_with_pred(Iter first, Iter last, Pred p) {
  while (first != last) {
    if (p(*first)) {
      break;
    }
    ++first;
  }
  return first;
}

static std::string trim_str(const std::string& in) {
  std::string s = in;
  s.erase(s.begin(), find_with_pred(s.begin(), s.end(), [](int ch) {
            return !std::isspace(ch);
          }));
  s.erase(find_with_pred(s.rbegin(), s.rend(), [](int ch) {
            return !std::isspace(ch);
          }).base(),
          s.end());
  return s;
}

static bool read_config_file(const std::basic_string<PATH_CHAR_TYPE>& path, std::map<std::string, std::string>& fc) {
  std::ifstream infile(path);
  if (!infile.good()) {
    return false;
  }

  for (std::string line; std::getline(infile, line);) {
    std::istringstream ss(line);
    if (line.empty()) {
      continue;
    }
    std::vector<std::string> tokens;
    for (std::string token; std::getline(ss, token, ':');) {
      std::string trimmed_token = trim_str(token);
      if (trimmed_token.empty()) {
        continue;
      }
      tokens.push_back(trimmed_token);
    }
    fc[tokens[0]] = tokens[1];
  }
  return true;
}

// load tensors from disk
template <typename PATH_STRING_TYPE>
static void LoadTensor(const PATH_STRING_TYPE& pb_file, ONNX_NAMESPACE::TensorProto& input_pb) {
  int tensor_fd;
  auto st = Env::Default().FileOpenRd(pb_file, tensor_fd);
  if (!st.IsOK()) {
    ORT_THROW("open file '", ToUTF8String(pb_file), "' failed:", st.ErrorMessage());
  }
  google::protobuf::io::FileInputStream f(tensor_fd, protobuf_block_size_in_bytes);
  f.SetCloseOnDelete(true);
  if (!input_pb.ParseFromZeroCopyStream(&f)) {
    ORT_THROW("parse file '", ToUTF8String(pb_file), "' failed");
  }
}

// load sequence tensors from disk
template <typename PATH_STRING_TYPE>
static void LoadSequenceTensor(const PATH_STRING_TYPE& pb_file, ONNX_NAMESPACE::SequenceProto& input_pb) {
  int tensor_fd;
  auto st = Env::Default().FileOpenRd(pb_file, tensor_fd);
  if (!st.IsOK()) {
    ORT_THROW("open file '", ToUTF8String(pb_file), "' failed:", st.ErrorMessage());
  }
  google::protobuf::io::FileInputStream f(tensor_fd, protobuf_block_size_in_bytes);
  f.SetCloseOnDelete(true);
  if (!input_pb.ParseFromZeroCopyStream(&f)) {
    ORT_THROW("parse file '", ToUTF8String(pb_file), "' failed");
  }
}

#if !defined(DISABLE_OPTIONAL_TYPE)
template <typename PATH_STRING_TYPE>
static void LoadOptional(const PATH_STRING_TYPE& pb_file,
                         ONNX_NAMESPACE::OptionalProto& input_pb) {
  int tensor_fd;
  auto st = Env::Default().FileOpenRd(pb_file, tensor_fd);
  if (!st.IsOK()) {
    ORT_THROW("open file '", ToUTF8String(pb_file), "' failed:", st.ErrorMessage());
  }
  google::protobuf::io::FileInputStream f(tensor_fd, protobuf_block_size_in_bytes);
  f.SetCloseOnDelete(true);
  if (!input_pb.ParseFromZeroCopyStream(&f)) {
    ORT_THROW("parse file '", ToUTF8String(pb_file), "' failed");
  }
}
#endif

void OnnxTestCase::LoadTestData(size_t id, onnxruntime::test::HeapBuffer& b,
                                std::unordered_map<std::string, Ort::Value>& name_data_map,
                                bool is_input) const {
  if (id >= test_data_dirs_.size()) {
    ORT_THROW("index out of bound");
  }

  PATH_STRING_TYPE test_data_pb = ConcatPathComponent(
      test_data_dirs_[id], (is_input ? ORT_TSTR("inputs.pb") : ORT_TSTR("outputs.pb")));
  int test_data_pb_fd;
  auto st = Env::Default().FileOpenRd(test_data_pb, test_data_pb_fd);
  if (st.IsOK()) {  // has an all-in-one input file
    std::ostringstream oss;
    {
      std::lock_guard<OrtMutex> l(m_);
      oss << debuginfo_strings_[id];
    }
    ORT_TRY {
      LoopDataFile(test_data_pb_fd, is_input, *model_info_, name_data_map, b, oss);
    }
    ORT_CATCH(const std::exception& ex) {
      ORT_HANDLE_EXCEPTION([&]() {
        std::ostringstream oss2;
        oss2 << "parse data file \"" << ToUTF8String(test_data_pb) << "\" failed:" << ex.what();
        ORT_THROW(oss.str());
      });
    }

    {
      std::lock_guard<OrtMutex> l(m_);
      debuginfo_strings_[id] = oss.str();
    }
    return;
  }

  std::vector<PATH_STRING_TYPE> test_data_pb_files;

  const PATH_STRING_TYPE& dir_path = test_data_dirs_[id];
  LoopDir(dir_path,
          [&test_data_pb_files, &dir_path, is_input](const PATH_CHAR_TYPE* filename, OrtFileType f_type) -> bool {
            if (filename[0] == '.') return true;
            if (f_type != OrtFileType::TYPE_REG) return true;
            std::basic_string<PATH_CHAR_TYPE> filename_str = filename;
            if (!HasExtensionOf(filename_str, ORT_TSTR("pb"))) return true;
            const std::basic_string<PATH_CHAR_TYPE> file_prefix =
                is_input ? ORT_TSTR("input_") : ORT_TSTR("output_");
            if (!filename_str.compare(0, file_prefix.length(), file_prefix)) {
              std::basic_string<PATH_CHAR_TYPE> p = ConcatPathComponent(dir_path, filename_str);
              test_data_pb_files.push_back(p);
            }
            return true;
          });

  SortFileNames(test_data_pb_files);

  for (size_t i = 0; i < test_data_pb_files.size(); ++i) {
    const ONNX_NAMESPACE::ValueInfoProto* value_info_proto = is_input ? model_info_->GetInputInfoFromModel(i) : model_info_->GetOutputInfoFromModel(i);
    if (!value_info_proto->has_type()) {
      ORT_THROW("Model ", is_input ? "input " : "output ", i, " is missing type info");
    }

    if (value_info_proto->type().has_tensor_type()) {
      ONNX_NAMESPACE::TensorProto test_pb;
      LoadTensor(test_data_pb_files[i], test_pb);
      ConvertTestData(test_pb, b, is_input, i, name_data_map);
    } else if (value_info_proto->type().has_sequence_type()) {
      ONNX_NAMESPACE::SequenceProto test_pb;
      LoadSequenceTensor(test_data_pb_files[i], test_pb);
      ConvertTestData(test_pb, b, is_input, i, name_data_map);
    }
#if !defined(DISABLE_OPTIONAL_TYPE)
    else if (value_info_proto->type().has_optional_type()) {
      ONNX_NAMESPACE::OptionalProto test_pb;
      LoadOptional(test_data_pb_files[i], test_pb);
      ConvertTestData(test_pb, b, is_input, i, name_data_map);
    }
#endif
    else {
      ORT_THROW("Unsupported type for the ", is_input ? "input " : "output ", i, " in the test runner");
    }
  }
}

void OnnxTestCase::ConvertTestData(const ONNX_NAMESPACE::TensorProto& test_data_pb,
                                   onnxruntime::test::HeapBuffer& b,
                                   bool is_input, size_t i,
                                   std::unordered_map<std::string, Ort::Value>& out) const {
  const std::string& name = test_data_pb.name();
  const std::string& name_finalized = !name.empty()
                                          ? name
                                          : (is_input ? model_info_->GetInputName(i) : model_info_->GetOutputName(i));

  size_t len = 0;

  auto status = onnxruntime::test::GetSizeInBytesFromTensorProto<0>(test_data_pb, &len);
  if (!status.IsOK()) {
    ORT_THROW(status.ToString());
  }
  void* p = len == 0 ? nullptr : b.AllocMemory(len);
  Ort::Value v1{nullptr};
  onnxruntime::test::OrtCallback d;
  OrtMemoryInfo cpu_memory_info(onnxruntime::CPU, OrtDeviceAllocator, OrtDevice(), 0, OrtMemTypeDefault);
  status = onnxruntime::test::TensorProtoToMLValue(test_data_pb, onnxruntime::test::MemBuffer(p, len, cpu_memory_info),
                                                   v1, d);
  if (!status.IsOK()) {
    ORT_THROW(status.ToString());
  }
  if (d.f) {
    b.AddDeleter(d);
  }
  out.emplace(name_finalized, std::move(v1));
}

void OnnxTestCase::ConvertTestData(const ONNX_NAMESPACE::SequenceProto& test_data_pb,
                                   onnxruntime::test::HeapBuffer& b,
                                   bool is_input, size_t i,
                                   std::unordered_map<std::string, Ort::Value>& out) const {
  const std::string& name = test_data_pb.name();
  const std::string& name_finalized = !name.empty()
                                          ? name
                                          : (is_input ? model_info_->GetInputName(i) : model_info_->GetOutputName(i));

  size_t len = 0;

  std::vector<Ort::Value> seq;
  if (test_data_pb.elem_type() != ONNX_NAMESPACE::SequenceProto_DataType_TENSOR) {
    ORT_THROW("Only parsing a sequence of tensors is currently supported");
  }
  const auto& tensors = test_data_pb.tensor_values();
  const size_t val = tensors.size();
  seq.reserve(val);

  for (auto it = tensors.cbegin(); it != tensors.cend(); ++it) {
    auto status = onnxruntime::test::GetSizeInBytesFromTensorProto<0>(*it, &len);
    if (!status.IsOK()) {
      ORT_THROW(status.ToString());
    }
    void* p = len == 0 ? nullptr : b.AllocMemory(len);
    Ort::Value v1{nullptr};
    onnxruntime::test::OrtCallback d;
    OrtMemoryInfo cpu_memory_info(onnxruntime::CPU, OrtDeviceAllocator, OrtDevice(), 0, OrtMemTypeDefault);
    status = onnxruntime::test::TensorProtoToMLValue(*it, onnxruntime::test::MemBuffer(p, len, cpu_memory_info),
                                                     v1, d);
    if (!status.IsOK()) {
      ORT_THROW(status.ToString());
    }
    if (d.f) {
      b.AddDeleter(d);
    }

    seq.push_back(std::move(v1));
  }

  if (seq.size() == 0) {
    // TODO: implement support for creating empty sequences. Not urgent yet since we don't have real world models.
    // For now, only the single node ONNX test - `test_loop13_seq` requires it (will keep it disabled for now).
    ORT_THROW("Creation of empty sequences is currently not supported in the test runner");
  } else {
    out.emplace(name_finalized, Ort::Value::CreateSequence(seq));
  }
}

#if !defined(DISABLE_OPTIONAL_TYPE)
void OnnxTestCase::ConvertTestData(const ONNX_NAMESPACE::OptionalProto& test_data_pb,
                                   onnxruntime::test::HeapBuffer& b,
                                   bool is_input, size_t i,
                                   std::unordered_map<std::string, Ort::Value>& out) const {
  // Optional Tensor
  if (test_data_pb.elem_type() ==
      ONNX_NAMESPACE::OptionalProto_DataType::OptionalProto_DataType_TENSOR) {
    // The optional tensor is not "None", deal with it as a regular tensor
    if (test_data_pb.has_tensor_value()) {
      ConvertTestData(test_data_pb.tensor_value(), b, is_input, i, out);
    } else {
      // Process None
      // If is_input is true, don't include the None in the feeds
      // If is_input is false, include it in the fetches, so that we can validate
      // whether we received a None output from ORT.

      if (!is_input) {
        const std::string& name = test_data_pb.name();
        const std::string& name_finalized = !name.empty()
                                                ? name
                                                : (is_input ? model_info_->GetInputName(i) : model_info_->GetOutputName(i));

        // Our API doesn't support creating None OrtValue,
        // so we place an nullptr into the expected values.
        Ort::Value value{nullptr};
        out.emplace(name_finalized, std::move(value));
      }
    }
  }  // Optional Sequence Tensor
  else if (test_data_pb.elem_type() ==
           ONNX_NAMESPACE::OptionalProto_DataType::OptionalProto_DataType_SEQUENCE) {
    // The optional sequence tensor is not "None", deal with it as a regular tensor
    if (test_data_pb.has_sequence_value()) {
      // ConvertTestData() ensures that sequence contains only tensors - we do no need
      // a redundant check here
      ConvertTestData(test_data_pb.sequence_value(), b, is_input, i, out);
    } else {
      // Process None
      // If is_input is true, don't include the None in the feeds
      // If is_input is false, include it in the fetches, so that we can validate
      // whether we received a None output from ORT.

      if (!is_input) {
        const std::string& name = test_data_pb.name();
        const std::string& name_finalized = !name.empty()
                                                ? name
                                                : (is_input ? model_info_->GetInputName(i) : model_info_->GetOutputName(i));

        // Our API doesn't support creating None OrtValue,
        // so we place an nullptr into the expected values.
        Ort::Value value{nullptr};
        out.emplace(name_finalized, std::move(value));
      }
    }
  }
}
#endif

OnnxTestCase::OnnxTestCase(const std::string& test_case_name, _In_ std::unique_ptr<TestModelInfo> model,
                           double default_per_sample_tolerance, double default_relative_per_sample_tolerance)
    : test_case_name_(test_case_name), model_info_(std::move(model)) {
  std::basic_string<PATH_CHAR_TYPE> test_case_dir = model_info_->GetDir();

  // parse config
  std::basic_string<PATH_CHAR_TYPE> config_path =
      ConcatPathComponent(test_case_dir, ORT_TSTR("config.txt"));
  /* Note: protobuf-lite doesn't support reading protobuf files as text-format. Config.txt is exactly that.
     That's the reason I've to parse the file in a different way to read the configs. Currently
     this affects 2 tests - fp16_tiny_yolov2 and fp16_inception_v1. It's not clear why we've to use protobuf
     to represent simple config files that have only key-value pairs.
   */
  std::map<std::string, std::string> fc;
  per_sample_tolerance_ = default_per_sample_tolerance;
  relative_per_sample_tolerance_ = default_relative_per_sample_tolerance;
  post_processing_ = false;
  if (read_config_file(config_path, fc)) {
    if (fc.count("per_sample_tolerance") > 0) {
      per_sample_tolerance_ = stod(fc["per_sample_tolerance"]);
    }
    if (fc.count("relative_per_sample_tolerance") > 0) {
      relative_per_sample_tolerance_ = stod(fc["relative_per_sample_tolerance"]);
    }
    if (fc.count("post_processing") > 0) {
      post_processing_ = fc["post_processing"] == "true";
    }
  }

  LoopDir(test_case_dir, [&test_case_dir, this](const PATH_CHAR_TYPE* filename, OrtFileType f_type) -> bool {
    if (filename[0] == '.') return true;
    if (f_type == OrtFileType::TYPE_DIR) {
      std::basic_string<PATH_CHAR_TYPE> p = ConcatPathComponent(test_case_dir, filename);
      test_data_dirs_.push_back(p);
      debuginfo_strings_.push_back(ToUTF8String(p));
    }
    return true;
  });
}

void LoadTests(const std::vector<std::basic_string<PATH_CHAR_TYPE>>& input_paths,
               const std::vector<std::basic_string<PATH_CHAR_TYPE>>& whitelisted_test_cases,
               const TestTolerances& tolerances,
               const std::unordered_set<std::basic_string<ORTCHAR_T>>& disabled_tests,
               std::unique_ptr<std::set<BrokenTest>> broken_tests,
               std::unique_ptr<std::set<std::string>> broken_tests_keyword_set,
               const std::function<void(std::unique_ptr<ITestCase>)>& process_function) {
  std::vector<std::basic_string<PATH_CHAR_TYPE>> paths(input_paths);
  while (!paths.empty()) {
    std::basic_string<PATH_CHAR_TYPE> node_data_root_path = paths.back();
    paths.pop_back();
    std::basic_string<PATH_CHAR_TYPE> my_dir_name = GetLastComponent(node_data_root_path);
    LoopDir(node_data_root_path, [&](const PATH_CHAR_TYPE* filename, OrtFileType f_type) -> bool {
      if (filename[0] == '.') return true;
      if (f_type == OrtFileType::TYPE_DIR) {
        std::basic_string<PATH_CHAR_TYPE> p = ConcatPathComponent(node_data_root_path, filename);
        paths.push_back(p);
        return true;
      }

      std::basic_string<PATH_CHAR_TYPE> filename_str = filename;
      bool is_onnx_format = HasExtensionOf(filename_str, ORT_TSTR("onnx"));
      bool is_ort_format = HasExtensionOf(filename_str, ORT_TSTR("ort"));
      bool is_valid_model = false;

#if !defined(ORT_MINIMAL_BUILD)
      is_valid_model = is_onnx_format;
#endif

      is_valid_model = is_valid_model || is_ort_format;
      if (!is_valid_model)
        return true;

      std::basic_string<PATH_CHAR_TYPE> test_case_name = my_dir_name;
      if (test_case_name.compare(0, 5, ORT_TSTR("test_")) == 0) test_case_name = test_case_name.substr(5);

      if (!whitelisted_test_cases.empty() && std::find(whitelisted_test_cases.begin(), whitelisted_test_cases.end(),
                                                       test_case_name) == whitelisted_test_cases.end()) {
        return true;
      }
      if (disabled_tests.find(test_case_name) != disabled_tests.end()) return true;

      std::basic_string<PATH_CHAR_TYPE> p = ConcatPathComponent(node_data_root_path, filename_str);

      std::unique_ptr<TestModelInfo> model_info;

      if (is_onnx_format) {
#if !defined(ORT_MINIMAL_BUILD)
        model_info = TestModelInfo::LoadOnnxModel(p.c_str());
#else
        ORT_THROW("onnx model is not supported in this build");
#endif
      } else if (is_ort_format) {
        model_info = TestModelInfo::LoadOrtModel(p.c_str());
      } else {
        ORT_NOT_IMPLEMENTED(ToUTF8String(filename_str), " is not supported");
      }

      auto test_case_dir = model_info->GetDir();
      auto test_case_name_in_log = test_case_name + ORT_TSTR(" in ") + test_case_dir;

#if !defined(ORT_MINIMAL_BUILD) && !defined(USE_QNN)
      // to skip some models like *-int8 or *-qdq
      if ((reinterpret_cast<OnnxModelInfo*>(model_info.get()))->HasDomain(ONNX_NAMESPACE::AI_ONNX_TRAINING_DOMAIN) ||
          (reinterpret_cast<OnnxModelInfo*>(model_info.get()))->HasDomain(ONNX_NAMESPACE::AI_ONNX_PREVIEW_TRAINING_DOMAIN)) {
        fprintf(stderr, "Skip test case:: %s %s\n", ToUTF8String(test_case_name_in_log).c_str(), " as it has training domain");
        return true;
      }
#endif

      bool has_test_data = false;
      LoopDir(test_case_dir, [&](const PATH_CHAR_TYPE* filename, OrtFileType f_type) -> bool {
        if (filename[0] == '.') return true;
        if (f_type == OrtFileType::TYPE_DIR) {
          has_test_data = true;
          return false;
        }
        return true;
      });
      if (!has_test_data) {
        fprintf(stderr, "Skip test case:: %s %s\n", ToUTF8String(test_case_name_in_log).c_str(), " due to no test data");
        return true;
      }

      if (broken_tests) {
        BrokenTest t = {ToUTF8String(test_case_name), ""};
        auto iter = broken_tests->find(t);
        auto opset_version = model_info->GetNominalOpsetVersion();
        if (iter != broken_tests->end() &&
            (opset_version == TestModelInfo::unknown_version || iter->broken_opset_versions_.empty() ||
             iter->broken_opset_versions_.find(opset_version) != iter->broken_opset_versions_.end())) {
          fprintf(stderr, "Skip test case:: %s %s\n", ToUTF8String(test_case_name_in_log).c_str(), " due to broken_tests");
          return true;
        }
      }

      if (broken_tests_keyword_set) {
        for (auto iter2 = broken_tests_keyword_set->begin(); iter2 != broken_tests_keyword_set->end(); ++iter2) {
          std::string keyword = *iter2;
          if (ToUTF8String(test_case_name).find(keyword) != std::string::npos) {
            fprintf(stderr, "Skip test case:: %s %s\n", ToUTF8String(test_case_name_in_log).c_str(), " as it is in broken test keywords");
            return true;
          }
        }
      }

      const auto tolerance_key = ToUTF8String(my_dir_name);

      std::unique_ptr<ITestCase> l = CreateOnnxTestCase(ToUTF8String(test_case_name), std::move(model_info),
                                                        tolerances.absolute(tolerance_key),
                                                        tolerances.relative(tolerance_key));
      fprintf(stdout, "Load Test Case: %s\n", ToUTF8String(test_case_name_in_log).c_str());
      process_function(std::move(l));
      return true;
    });
  }
}

TestTolerances::TestTolerances(
    double absolute_default, double relative_default,
    const Map& absolute_overrides,
    const Map& relative_overrides) : absolute_default_(absolute_default),
                                     relative_default_(relative_default),
                                     absolute_overrides_(absolute_overrides),
                                     relative_overrides_(relative_overrides) {}

double TestTolerances::absolute(const std::string& name) const {
  const auto iter = absolute_overrides_.find(name);
  if (iter == absolute_overrides_.end()) {
    return absolute_default_;
  }
  return iter->second;
}

double TestTolerances::relative(const std::string& name) const {
  const auto iter = relative_overrides_.find(name);
  if (iter == relative_overrides_.end()) {
    return relative_default_;
  }
  return iter->second;
}

std::unique_ptr<std::set<BrokenTest>> GetBrokenTests(const std::string& provider_name) {
  auto broken_tests = std::make_unique<std::set<BrokenTest>>(std::initializer_list<BrokenTest>{
      {"slice_neg_steps",
       "Type parameter (Tind) bound to different types (tensor(int64) and tensor(int32) in node ()."},
      {"cast_BFLOAT16_to_FLOAT", "Unexpected input data type"},
      {"loop13_seq", "Creation of empty sequences is currently not supported in the test runner"},
      {"sequence_insert_at_front", "shape mismatch, expect {4} got {3}"},
      {"cast_FLOAT_to_BFLOAT16", "expect uint16 got bfloat16"},
      {"mnist", "Input data isn't in valid range"},
      {"BERT_Squad", "test data bug"},
      {"constantofshape_float_ones", "test data bug", {"opset9", "opset10"}},
      {"constantofshape_int_zeros", "test data bug", {"opset9", "opset10"}},
      {"cast_STRING_to_FLOAT", "Linux CI has old ONNX python package with bad test data", {"opset9", "opset10"}},
      // Numpy float to string has unexpected rounding for some results given numpy default precision is meant to be 8.
      // "e.g. 0.296140194 -> '0.2961402' not '0.29614019'. ORT produces the latter with precision set to 8,
      // which doesn't match the expected output that was generated with numpy.
      {"cast_FLOAT_to_STRING", "Numpy float to string has unexpected rounding for some results."},
      {"tf_nasnet_large", "disable temporarily"},
      {"tf_nasnet_mobile", "disable temporarily"},
      {"tf_pnasnet_large", "disable temporarily"},
      {"shrink", "test case is wrong", {"opset9"}},
      {"maxpool_with_argmax_2d_precomputed_strides", "ShapeInferenceError"},
      {"tf_inception_v2", "result mismatch"},
      {"tf_resnet_v1_50", "result mismatch when Conv BN Fusion is applied"},
      {"tf_resnet_v1_101", "result mismatch when Conv BN Fusion is applied"},
      {"tf_resnet_v1_152", "result mismatch when Conv BN Fusion is applied"},
      {"mxnet_arcface", "Model is an invalid ONNX model"},
      {"unique_not_sorted_without_axis", "Expected data for 'Y' is incorrect and in sorted order."},
      {"cumsum_1d_reverse_exclusive", "only failing linux GPU CI. Likely build error."},
      {"resize_downsample_scales_cubic_align_corners", "results mismatch with onnx tests"},
      {"resize_downsample_scales_linear_align_corners", "results mismatch with onnx tests"},
      {"resize_tf_crop_and_resize", "Bad onnx test output. Needs test fix."},
      {"resize_upsample_sizes_nearest_ceil_half_pixel", "Bad onnx test output. Needs test fix."},
      {"resize_upsample_sizes_nearest_floor_align_corners", "Bad onnx test output. Needs test fix."},
      {"resize_upsample_sizes_nearest_round_prefer_ceil_asymmetric", "Bad onnx test output. Needs test fix."},
      {"bitshift_right_uint16", "BitShift(11) uint16 support not enabled currently"},
      {"bitshift_left_uint16", "BitShift(11) uint16 support not enabled currently"},
      {"maxunpool_export_with_output_shape",
       "Invalid output in ONNX test. See https://github.com/onnx/onnx/issues/2398"},
      {"cntk_simple_seg", "Bad onnx test output caused by wrong SAME_UPPER/SAME_LOWER for ConvTranspose"},
      {"training_dropout", "result differs", {}},               // Temporary, subsequent PR will remove this.
      {"training_dropout_default", "result differs", {}},       // Temporary, subsequent PR will remove this.
      {"training_dropout_default_mask", "result differs", {}},  // Temporary, subsequent PR will remove this.
      {"training_dropout_mask", "result differs", {}},          // Temporary, subsequent PR will remove this.
      {"batchnorm_epsilon_training_mode", "training only", {}},
      {"batchnorm_example_training_mode", "training only", {}},
      {"bernoulli", "type error", {}},
      {"bernoulli_double", "type error", {}},
      {"bernoulli_double_expanded", "type error", {}},
      {"bernoulli_expanded", "type error", {}},
      {"bernoulli_seed", "type error", {}},
      {"bernoulli_seed_expanded", "type error", {}},
      {"castlike_BFLOAT16_to_FLOAT", "type error", {}},
      {"castlike_BFLOAT16_to_FLOAT_expanded", "type error", {}},
      {"castlike_FLOAT_to_BFLOAT16", "type error", {}},
      {"castlike_FLOAT_to_BFLOAT16_expanded", "type error", {}},
      {"castlike_FLOAT_to_STRING", "type error", {}},
      {"castlike_FLOAT_to_STRING_expanded", "type error", {}},
      {"convtranspose_autopad_same", "Test data has been corrected in ONNX 1.10.", {"opset13", "opset14"}},
      {"gru_batchwise", "type error", {}},
      {"lstm_batchwise", "type error", {}},
      {"optional_get_element", "type error", {}},
      {"optional_get_element_sequence", "type error", {}},
      {"optional_has_element", "type error", {}},
      {"optional_has_element_empty", "type error", {}},
      {"shape_end_1", "type error", {}},
      {"shape_end_negative_1", "type error", {}},
      {"shape_start_1", "type error", {}},
      {"shape_start_1_end_2", "type error", {}},
      {"shape_start_1_end_negative_1", "type error", {}},
      {"shape_start_negative_1", "type error", {}},
      {"simple_rnn_batchwise", "type error", {}},
      {"mod_float_mixed_sign_example", "fmod attribute must be true for floating point types", {}},
      {"col2im_pads", "result mismatch", {"opset18"}},
      {"reduce_l1_empty_set", "unknown version", {}},
      {"reduce_l1_empty_set_expanded", "unknown version", {}},
      {"reduce_l2_empty_set", "unknown version", {}},
      {"reduce_l2_empty_set_expanded", "unknown version", {}},
      {"reduce_log_sum_empty_set", "unknown version", {}},
      {"reduce_log_sum_empty_set_expanded", "unknown version", {}},
      {"reduce_log_sum_exp_empty_set", "unknown version", {}},
      {"reduce_log_sum_exp_empty_set_expanded", "unknown version", {}},
      {"reduce_prod_empty_set", "unknown version", {}},
      {"reduce_sum_empty_set", "unknown version", {}},
      {"reduce_sum_square_empty_set", "unknown version", {}},
      {"reduce_sum_square_empty_set_expanded", "unknown version", {}},
#ifdef ENABLE_TRAINING_CORE
      {"adagrad", "not a registered function/op", {}},                  // Op not registered.
      {"adagrad_multiple", "not a registered function/op", {}},         // Op not registered.
      {"adam", "not a registered function/op", {}},                     // Op not registered.
      {"adam_multiple", "not a registered function/op", {}},            // Op not registered.
      {"gradient_of_add", "not a registered function/op", {}},          // Op not registered.
      {"gradient_of_add_and_mul", "not a registered function/op", {}},  // Op not registered.
      {"momentum", "not a registered function/op", {}},                 // Op not registered.
      {"momentum_multiple", "not a registered function/op", {}},        // Op not registered.
      {"nesterov_momentum", "not a registered function/op", {}},        // Op not registered.
      {"softmax_cross_entropy_input_shape_is_NCd1d2d3d4d5_none_no_weight", "type error", {"opset12"}},
      {"softmax_cross_entropy_mean_weight_ignore_index_log_prob", "type error", {"opset12"}},
      {"softmax_cross_entropy_input_shape_is_NCd1_mean_weight_negative_ignore_index_log_prob",
       "type error",
       {"opset12"}},
      {"softmax_cross_entropy_mean_weight_log_prob", "type error", {"opset12"}},
      {"softmax_cross_entropy_input_shape_is_NCd1d2d3d4d5_mean_weight_log_prob", "type error", {"opset12"}},
      {"softmax_cross_entropy_mean_weight_ignore_index_3d", "type error", {"opset12"}},
      {"softmax_cross_entropy_mean_weight_ignore_index_4d_log_prob", "type error", {"opset12"}},
      {"softmax_cross_entropy_input_shape_is_NCd1d2d3d4d5_none_no_weight_log_prob", "type error", {"opset12"}},
      {"softmax_cross_entropy_mean_weight_ignore_index_4d", "type error", {"opset12"}},
      {"softmax_cross_entropy_mean", "type error", {"opset12"}},
      {"softmax_cross_entropy_mean_log_prob", "type error", {"opset12"}},
      {"softmax_cross_entropy_mean_no_weight_ignore_index", "type error", {"opset12"}},
      {"softmax_cross_entropy_input_shape_is_NCd1d2d3_sum_weight_high_ignore_index_log_prob",
       "type error",
       {"opset12"}},
      {"softmax_cross_entropy_mean_3d_log_prob", "type error", {"opset12"}},
      {"softmax_cross_entropy_none_log_prob", "type error", {"opset12"}},
      {"softmax_cross_entropy_mean_3d", "type error", {"opset12"}},
      {"softmax_cross_entropy_input_shape_is_NCd1d2d3_none_no_weight_negative_ignore_index", "type error", {"opset12"}},
      {"softmax_cross_entropy_mean_weight_ignore_index_3d_log_prob", "type error", {"opset12"}},
      {"softmax_cross_entropy_mean_no_weight_ignore_index_3d_log_prob", "type error", {"opset12"}},
      {"softmax_cross_entropy_none_weights_log_prob", "type error", {"opset12"}},
      {"softmax_cross_entropy_sum_log_prob", "type error", {"opset12"}},
      {"softmax_cross_entropy_mean_weight_ignore_index", "type error", {"opset12"}},
      {"softmax_cross_entropy_mean_no_weight_ignore_index_log_prob", "type error", {"opset12"}},
      {"softmax_cross_entropy_mean_no_weight_ignore_index_3d", "type error", {"opset12"}},
      {"softmax_cross_entropy_input_shape_is_NCd1d2d3_sum_weight_high_ignore_index", "type error", {"opset12"}},
      {"softmax_cross_entropy_sum", "type error", {"opset12"}},
      {"softmax_cross_entropy_input_shape_is_NCd1d2d3_none_no_weight_negative_ignore_index_log_prob",
       "type error",
       {"opset12"}},
      {"softmax_cross_entropy_none_weights", "type error", {"opset12"}},
      {"softmax_cross_entropy_mean_no_weight_ignore_index_4d_log_prob", "type error", {"opset12"}},
      {"softmax_cross_entropy_none", "type error", {"opset12"}},
      {"softmax_cross_entropy_input_shape_is_NCd1_mean_weight_negative_ignore_index", "type error", {"opset12"}},
      {"softmax_cross_entropy_input_shape_is_NCd1d2d3d4d5_mean_weight", "type error", {"opset12"}},
      {"softmax_cross_entropy_mean_weight", "type error", {"opset12"}},
      {"softmax_cross_entropy_mean_no_weight_ignore_index_4d", "type error", {"opset12"}},
#endif
      {"mask_rcnn_keras", "this model currently has an invalid contrib op version set to 10", {}}});

  // Some EPs may fail to pass some specific testcases.
  // For example TenosrRT EP may fail on FLOAT16 related testcases if GPU doesn't support float16.
  // Instead of list all these testcases, we can use following keyword set to filter out testcases wchich contain
  // specific keyword.
  // std::set<std::string> broken_tests_keyword_set = {};

  if (provider_name == "cuda") {
#ifdef _WIN32
    broken_tests->insert({"LSTM_Seq_lens_unpacked", "this test fails with new image since Aug 25."});
    broken_tests->insert({"bidaf", "this test fails with new image since Aug 25."});
    broken_tests->insert({"Candy", "Flaky test, need to investigate", {"opset9"}});
#else
    broken_tests->insert({"bidaf", "this test should be recovered when multi-gpu pipeline deprecates NV12", {"opset9"}});
#endif
  }

  if (provider_name == "nnapi") {
    broken_tests->insert({"scan9_sum", "Error with the extra graph"});
    broken_tests->insert({"scan_sum", "Error with the extra graph"});
    broken_tests->insert({"mvn_expanded", "Failed to find kernel for MemcpyFromHost(1) (node Memcpy_1)"});
    broken_tests->insert({"dynamicquantizelinear_expanded", "Temporarily disabled pending investigation"});
    broken_tests->insert({"dynamicquantizelinear_max_adjusted_expanded", "Temporarily disabled pending investigation"});
    broken_tests->insert({"dynamicquantizelinear_min_adjusted_expanded", "Temporarily disabled pending investigation"});
    broken_tests->insert({"gemm_transposeB", "Temporarily disabled pending investigation"});
    broken_tests->insert({"range_float_type_positive_delta_expanded", "Temporarily disabled pending investigation"});
    broken_tests->insert({"range_int32_type_negative_delta_expanded", "Temporarily disabled pending investigation"});
    broken_tests->insert({"convtranspose_1d", "1d convtranspose not supported yet"});
    broken_tests->insert({"convtranspose_3d", "3d convtranspose not supported yet"});
    broken_tests->insert({"maxpool_2d_uint8", "result mismatch"});
    broken_tests->insert({"negative_log_likelihood_loss_input_shape_is_NC_expanded", "shape mismatch"});
    broken_tests->insert({"negative_log_likelihood_loss_input_shape_is_NCd1d2_expanded", "shape mismatch"});
    broken_tests->insert(
        {"negative_log_likelihood_loss_input_shape_is_NCd1d2_reduction_mean_expanded", "shape mismatch"});
    broken_tests->insert(
        {"negative_log_likelihood_loss_input_shape_is_NCd1d2_reduction_sum_expanded", "shape mismatch"});
    broken_tests->insert({"negative_log_likelihood_loss_input_shape_is_NCd1d2_with_weight_expanded", "shape mismatch"});
    broken_tests->insert(
        {"negative_log_likelihood_loss_input_shape_is_NCd1d2_with_weight_reduction_mean_expanded", "shape mismatch"});
    broken_tests->insert(
        {"negative_log_likelihood_loss_input_shape_is_NCd1d2_with_weight_reduction_sum_expanded", "shape mismatch"});
    // Disable based on George Wu's recommendation.
    broken_tests->insert(
        {"negative_log_likelihood_loss_input_shape_is_NCd1d2_with_weight_reduction_sum_ignore_index_expanded",
         "shape mismatch"});
    broken_tests->insert({"negative_log_likelihood_loss_iinput_shape_is_NCd1_weight_ignore_index", "Shape mismatch"});
    broken_tests->insert(
        {"negative_log_likelihood_loss_iinput_shape_is_NCd1_weight_ignore_index_expanded", "Shape mismatch"});
    broken_tests->insert({"negative_log_likelihood_loss_input_shape_is_NC", "Shape mismatch"});
    broken_tests->insert({"negative_log_likelihood_loss_input_shape_is_NCd1", "Shape mismatch"});
    broken_tests->insert({"negative_log_likelihood_loss_input_shape_is_NCd1_expanded", "Shape mismatch"});
    broken_tests->insert({"negative_log_likelihood_loss_input_shape_is_NCd1_ignore_index", "Shape mismatch"});
    broken_tests->insert({"negative_log_likelihood_loss_input_shape_is_NCd1_ignore_index_expanded", "Shape mismatch"});
    broken_tests->insert(
        {"negative_log_likelihood_loss_input_shape_is_NCd1_mean_weight_negative_ignore_index", "Shape mismatch"});
    broken_tests->insert({"negative_log_likelihood_loss_input_shape_is_NCd1_mean_weight_negative_ignore_index_expanded",
                          "Shape mismatch"});
    broken_tests->insert({"negative_log_likelihood_loss_input_shape_is_NCd1_weight", "Shape mismatch"});
    broken_tests->insert({"negative_log_likelihood_loss_input_shape_is_NCd1_weight_expanded", "Shape mismatch"});
    broken_tests->insert({"negative_log_likelihood_loss_input_shape_is_NCd1d2", "Shape mismatch"});
    broken_tests->insert(
        {"negative_log_likelihood_loss_input_shape_is_NCd1d2_no_weight_reduction_mean_ignore_index", "Shape mismatch"});
    broken_tests->insert(
        {"negative_log_likelihood_loss_input_shape_is_NCd1d2_no_weight_reduction_mean_ignore_index_expanded",
         "Shape mismatch"});
    broken_tests->insert({"negative_log_likelihood_loss_input_shape_is_NCd1d2_reduction_mean", "Shape mismatch"});
    broken_tests->insert({"negative_log_likelihood_loss_input_shape_is_NCd1d2_reduction_sum", "Shape mismatch"});
    broken_tests->insert({"negative_log_likelihood_loss_input_shape_is_NCd1d2_with_weight", "Shape mismatch"});
    broken_tests->insert(
        {"negative_log_likelihood_loss_input_shape_is_NCd1d2_with_weight_reduction_mean", "Shape mismatch"});
    broken_tests->insert(
        {"negative_log_likelihood_loss_input_shape_is_NCd1d2_with_weight_reduction_sum", "Shape mismatch"});
    broken_tests->insert({"negative_log_likelihood_loss_input_shape_is_NCd1d2_with_weight_reduction_sum_ignore_index",
                          "Shape mismatch"});
    broken_tests->insert({"negative_log_likelihood_loss_input_shape_is_NCd1d2d3_none_no_weight_negative_ignore_index",
                          "Shape mismatch"});
    broken_tests->insert(
        {"negative_log_likelihood_loss_input_shape_is_NCd1d2d3_none_no_weight_negative_ignore_index_expanded",
         "Shape mismatch"});
    broken_tests->insert(
        {"negative_log_likelihood_loss_input_shape_is_NCd1d2d3_sum_weight_high_ignore_index", "Shape mismatch"});
    broken_tests->insert({"negative_log_likelihood_loss_input_shape_is_NCd1d2d3_sum_weight_high_ignore_index_expanded",
                          "Shape mismatch"});
    broken_tests->insert({"negative_log_likelihood_loss_input_shape_is_NCd1d2d3d4d5_mean_weight", "Shape mismatch"});
    broken_tests->insert(
        {"negative_log_likelihood_loss_input_shape_is_NCd1d2d3d4d5_mean_weight_expanded", "Shape mismatch"});
    broken_tests->insert({"negative_log_likelihood_loss_input_shape_is_NCd1d2d3d4d5_none_no_weight", "Shape mismatch"});
    broken_tests->insert(
        {"negative_log_likelihood_loss_input_shape_is_NCd1d2d3d4d5_none_no_weight_expanded", "Shape mismatch"});
    broken_tests->insert(
        {"softmax_cross_entropy_input_shape_is_NCd1_mean_weight_negative_ignore_index", "Shape mismatch"});
    broken_tests->insert(
        {"softmax_cross_entropy_input_shape_is_NCd1_mean_weight_negative_ignore_index_expanded", "Shape mismatch"});
    broken_tests->insert(
        {"softmax_cross_entropy_input_shape_is_NCd1_mean_weight_negative_ignore_index_log_prob", "Shape mismatch"});
    broken_tests->insert(
        {"softmax_cross_entropy_input_shape_is_NCd1_mean_weight_negative_ignore_index_log_prob_expanded",
         "Shape mismatch"});
    broken_tests->insert(
        {"softmax_cross_entropy_input_shape_is_NCd1d2d3_none_no_weight_negative_ignore_index", "Shape mismatch"});
    broken_tests->insert({"softmax_cross_entropy_input_shape_is_NCd1d2d3_none_no_weight_negative_ignore_index_expanded",
                          "Shape mismatch"});
    broken_tests->insert({"softmax_cross_entropy_input_shape_is_NCd1d2d3_none_no_weight_negative_ignore_index_log_prob",
                          "Shape mismatch"});
    broken_tests->insert(
        {"softmax_cross_entropy_input_shape_is_NCd1d2d3_none_no_weight_negative_ignore_index_log_prob_expanded",
         "Shape mismatch"});
    broken_tests->insert(
        {"softmax_cross_entropy_input_shape_is_NCd1d2d3_sum_weight_high_ignore_index", "Shape mismatch"});
    broken_tests->insert(
        {"softmax_cross_entropy_input_shape_is_NCd1d2d3_sum_weight_high_ignore_index_expanded", "Shape mismatch"});
    broken_tests->insert(
        {"softmax_cross_entropy_input_shape_is_NCd1d2d3_sum_weight_high_ignore_index_log_prob", "Shape mismatch"});
    broken_tests->insert({"softmax_cross_entropy_input_shape_is_NCd1d2d3_sum_weight_high_ignore_index_log_prob_expanded",
                          "Shape mismatch"});
    broken_tests->insert({"softmax_cross_entropy_input_shape_is_NCd1d2d3d4d5_mean_weight", "Shape mismatch"});
    broken_tests->insert({"softmax_cross_entropy_input_shape_is_NCd1d2d3d4d5_mean_weight_expanded", "Shape mismatch"});
    broken_tests->insert({"softmax_cross_entropy_input_shape_is_NCd1d2d3d4d5_mean_weight_log_prob", "Shape mismatch"});
    broken_tests->insert(
        {"softmax_cross_entropy_input_shape_is_NCd1d2d3d4d5_mean_weight_log_prob_expanded", "Shape mismatch"});
    broken_tests->insert({"softmax_cross_entropy_input_shape_is_NCd1d2d3d4d5_none_no_weight", "Shape mismatch"});
    broken_tests->insert(
        {"softmax_cross_entropy_input_shape_is_NCd1d2d3d4d5_none_no_weight_expanded", "Shape mismatch"});
    broken_tests->insert(
        {"softmax_cross_entropy_input_shape_is_NCd1d2d3d4d5_none_no_weight_log_prob", "Shape mismatch"});
    broken_tests->insert(
        {"softmax_cross_entropy_input_shape_is_NCd1d2d3d4d5_none_no_weight_log_prob_expanded", "Shape mismatch"});
    broken_tests->insert({"softmax_cross_entropy_mean", "Shape mismatch"});
    broken_tests->insert({"softmax_cross_entropy_mean_3d", "Shape mismatch"});
    broken_tests->insert({"softmax_cross_entropy_mean_3d_expanded", "Shape mismatch"});
    broken_tests->insert({"softmax_cross_entropy_mean_3d_log_prob", "Shape mismatch"});
    broken_tests->insert({"softmax_cross_entropy_mean_3d_log_prob_expanded", "Shape mismatch"});
    broken_tests->insert({"softmax_cross_entropy_mean_expanded", "Shape mismatch"});
    broken_tests->insert({"softmax_cross_entropy_mean_log_prob", "Shape mismatch"});
    broken_tests->insert({"softmax_cross_entropy_mean_log_prob_expanded", "Shape mismatch"});
    broken_tests->insert({"softmax_cross_entropy_mean_no_weight_ignore_index", "Shape mismatch"});
    broken_tests->insert({"softmax_cross_entropy_mean_no_weight_ignore_index_3d", "Shape mismatch"});
    broken_tests->insert({"softmax_cross_entropy_mean_no_weight_ignore_index_3d_expanded", "Shape mismatch"});
    broken_tests->insert({"softmax_cross_entropy_mean_no_weight_ignore_index_3d_log_prob", "Shape mismatch"});
    broken_tests->insert({"softmax_cross_entropy_mean_no_weight_ignore_index_3d_log_prob_expanded", "Shape mismatch"});
    broken_tests->insert({"softmax_cross_entropy_mean_no_weight_ignore_index_4d", "Shape mismatch"});
    broken_tests->insert({"softmax_cross_entropy_mean_no_weight_ignore_index_4d_expanded", "Shape mismatch"});
    broken_tests->insert({"softmax_cross_entropy_mean_no_weight_ignore_index_4d_log_prob", "Shape mismatch"});
    broken_tests->insert({"softmax_cross_entropy_mean_no_weight_ignore_index_4d_log_prob_expanded", "Shape mismatch"});
    broken_tests->insert({"softmax_cross_entropy_mean_no_weight_ignore_index_expanded", "Shape mismatch"});
    broken_tests->insert({"softmax_cross_entropy_mean_no_weight_ignore_index_log_prob", "Shape mismatch"});
    broken_tests->insert({"softmax_cross_entropy_mean_no_weight_ignore_index_log_prob_expanded", "Shape mismatch"});
    broken_tests->insert({"softmax_cross_entropy_mean_weight", "Shape mismatch"});
    broken_tests->insert({"softmax_cross_entropy_mean_weight_expanded", "Shape mismatch"});
    broken_tests->insert({"softmax_cross_entropy_mean_weight_ignore_index", "Shape mismatch"});
    broken_tests->insert({"softmax_cross_entropy_mean_weight_ignore_index_3d", "Shape mismatch"});
    broken_tests->insert({"softmax_cross_entropy_mean_weight_ignore_index_3d_expanded", "Shape mismatch"});
    broken_tests->insert({"softmax_cross_entropy_mean_weight_ignore_index_3d_log_prob", "Shape mismatch"});
    broken_tests->insert({"softmax_cross_entropy_mean_weight_ignore_index_3d_log_prob_expanded", "Shape mismatch"});
    broken_tests->insert({"softmax_cross_entropy_mean_weight_ignore_index_4d", "Shape mismatch"});
    broken_tests->insert({"softmax_cross_entropy_mean_weight_ignore_index_4d_expanded", "Shape mismatch"});
    broken_tests->insert({"softmax_cross_entropy_mean_weight_ignore_index_4d_log_prob", "Shape mismatch"});
    broken_tests->insert({"softmax_cross_entropy_mean_weight_ignore_index_4d_log_prob_expanded", "Shape mismatch"});
    broken_tests->insert({"softmax_cross_entropy_mean_weight_ignore_index_expanded", "Shape mismatch"});
    broken_tests->insert({"softmax_cross_entropy_mean_weight_ignore_index_log_prob", "Shape mismatch"});
    broken_tests->insert({"softmax_cross_entropy_mean_weight_ignore_index_log_prob_expanded", "Shape mismatch"});
    broken_tests->insert({"softmax_cross_entropy_mean_weight_log_prob", "Shape mismatch"});
    broken_tests->insert({"softmax_cross_entropy_mean_weight_log_prob_expanded", "Shape mismatch"});
    broken_tests->insert({"softmax_cross_entropy_none", "Shape mismatch"});
    broken_tests->insert({"softmax_cross_entropy_none_expanded", "Shape mismatch"});
    broken_tests->insert({"softmax_cross_entropy_none_log_prob", "Shape mismatch"});
    broken_tests->insert({"softmax_cross_entropy_none_log_prob_expanded", "Shape mismatch"});
    broken_tests->insert({"softmax_cross_entropy_none_weights", "Shape mismatch"});
    broken_tests->insert({"softmax_cross_entropy_none_weights_expanded", "Shape mismatch"});
    broken_tests->insert({"softmax_cross_entropy_none_weights_log_prob", "Shape mismatch"});
    broken_tests->insert({"softmax_cross_entropy_none_weights_log_prob_expanded", "Shape mismatch"});
    broken_tests->insert({"softmax_cross_entropy_sum", "Shape mismatch"});
    broken_tests->insert({"softmax_cross_entropy_sum_expanded", "Shape mismatch"});
    broken_tests->insert({"softmax_cross_entropy_sum_log_prob", "Shape mismatch"});
    broken_tests->insert({"softmax_cross_entropy_sum_log_prob_expanded", "Shape mismatch"});
  }

  if (provider_name == "tensorrt") {
    broken_tests->insert({"convtranspose_with_kernel", "It causes segmentation fault"});
    broken_tests->insert({"convtranspose_pad", "It causes segmentation fault"});
    broken_tests->insert({"convtranspose_kernel_shape", "It causes segmentation fault"});
    broken_tests->insert({"dynamicquantizelinear_expanded", "It causes segmentation fault"});
    broken_tests->insert({"dynamicquantizelinear_min_adjusted_expanded", "It causes segmentation fault"});
    broken_tests->insert({"dynamicquantizelinear_max_adjusted_expanded", "It causes segmentation fault"});

    broken_tests->insert({"basic_conv_with_padding",
                          "Cannot set more than one input unless network has Q/DQ layers. TensorRT EP could not build "
                          "engine for fused node"});
    broken_tests->insert({"basic_conv_without_padding",
                          "Cannot set more than one input unless network has Q/DQ layers. TensorRT EP could not build "
                          "engine for fused node"});
    broken_tests->insert({"conv_with_strides_no_padding",
                          "Cannot set more than one input unless network has Q/DQ layers. TensorRT EP could not build "
                          "engine for fused node"});

    broken_tests->insert({"conv_with_autopad_same",
                          "Internal Error (node_of_y: Cannot set more than one input unless network has Q/DQ layers.)"});

    // unsupported tests since opset16
    broken_tests->insert({"sequence_map_add_2_sequences", "not supported by TensorRT EP"});
    broken_tests->insert({"sequence_map_extract_shapes", "not supported by TensorRT EP."});
    broken_tests->insert({"sequence_map_add_1_sequence_1_tensor", "not supported by TensorRT EP."});
    broken_tests->insert({"sequence_map_identity_1_sequence", "not supported by TensorRT EP."});
    broken_tests->insert({"sequence_map_identity_2_sequences", "not supported by TensorRT EP."});
    broken_tests->insert({"sequence_map_identity_1_sequence_1_tensor", "not supported by TensorRT EP."});
    broken_tests->insert({"leakyrelu_expanded", "not supported by TensorRT EP."});
    broken_tests->insert({"leakyrelu_default_expanded", "not supported by TensorRT EP."});
    broken_tests->insert({"leakyrelu_example_expanded", "not supported by TensorRT EP."});
    broken_tests->insert({"prelu_broadcast_expanded", "not supported by TensorRT EP."});
    broken_tests->insert({"prelu_example_expanded", "not supported by TensorRT EP."});
  }

  if (provider_name == "dml") {
    broken_tests->insert({"tinyyolov3", "The parameter is incorrect"});
    broken_tests->insert({"PixelShuffle", "Test requires 6D Reshape, which isn't supported by DirectML"});
    broken_tests->insert({"operator_permute2", "Test requires 6D Transpose, which isn't supported by DirectML"});
    broken_tests->insert({"resize_downsample_linear",
                          "ORT 0.4 uses asymmetric but will conform to half_pixel in the next ONNX version."});
    broken_tests->insert(
        {"resize_upsample_linear", "ORT 0.4 uses asymmetric but will conform to half_pixel in the next ONNX version."});
    broken_tests->insert(
        {"resize_upsample_linear", "ORT 0.4 uses asymmetric but will conform to half_pixel in the next ONNX version."});

    // These tests are temporarily disabled pending investigation
    broken_tests->insert({"dynamicquantizelinear_expanded", "Temporarily disabled pending investigation"});
    broken_tests->insert({"dynamicquantizelinear_max_adjusted_expanded", "Temporarily disabled pending investigation"});
    broken_tests->insert({"dynamicquantizelinear_min_adjusted_expanded", "Temporarily disabled pending investigation"});
    broken_tests->insert({"mxnet_arcface", "Temporarily disabled pending investigation"});
    broken_tests->insert({"yolov3", "Temporarily disabled pending investigation"});
    broken_tests->insert({"tf_inception_v2", "Temporarily disabled pending investigation"});
    broken_tests->insert({"fp16_inception_v1", "Temporarily disabled pending investigation"});
    broken_tests->insert({"candy", "Temporarily disabled pending investigation"});
    broken_tests->insert({"BERT_Squad", "Temporarily disabled pending investigation"});
    broken_tests->insert({"LSTM_Seq_lens_unpacked", "The parameter is incorrect"});
    broken_tests->insert({"mlperf_ssd_resnet34_1200", "The parameter is incorrect"});

    broken_tests->insert({"resize_downsample_scales_linear",
                          "DML uses half_pixel and this test assumed \"asymmetric\" but does not include \"mode\""});
    broken_tests->insert({"resize_downsample_sizes_linear_pytorch_half_pixel",
                          "DML does not support downsampling by such a large factor - skips input pixels"});
    broken_tests->insert({"resize_downsample_sizes_nearest",
                          "DML uses pixel centers for nearest, rounding 1 value off for the middle column"});
    broken_tests->insert({"resize_upsample_sizes_nearest",
                          "DML uses pixel centers for nearest, which makes more sense (the 3rd row mismatches)"});
    broken_tests->insert({"unsqueeze_three_axes", "DML does not support 6D tensors"});
    broken_tests->insert({"unsqueeze_unsorted_axes", "DMLdoes not support 6D tensors"});

    broken_tests->insert({"negative_log_likelihood_loss_input_shape_is_NCd1d2d3_none_no_weight_negative_ignore_index",
                          "DML does not support 5D+ tensors"});
    broken_tests->insert(
        {"negative_log_likelihood_loss_input_shape_is_NCd1d2d3_none_no_weight_negative_ignore_index_expanded",
         "DML does not support 5D+ tensors"});
    broken_tests->insert(
        {"negative_log_likelihood_loss_input_shape_is_NCd1d2d3d4d5_mean_weight", "DML does not support 5D+ tensors"});
    broken_tests->insert({"negative_log_likelihood_loss_input_shape_is_NCd1d2d3d4d5_mean_weight_expanded",
                          "DML does not support 5D+ tensors"});
    broken_tests->insert({"negative_log_likelihood_loss_input_shape_is_NCd1d2d3d4d5_none_no_weight",
                          "DML does not support 5D+ tensors"});
    broken_tests->insert({"negative_log_likelihood_loss_input_shape_is_NCd1d2d3d4d5_none_no_weight_expanded",
                          "DML does not support 5D+ tensors"});
    broken_tests->insert({"softmax_cross_entropy_input_shape_is_NCd1d2d3_none_no_weight_negative_ignore_index",
                          "DML does not support 5D+ tensors"});
    broken_tests->insert({"softmax_cross_entropy_input_shape_is_NCd1d2d3_none_no_weight_negative_ignore_index_expanded",
                          "DML does not support 5D+ tensors"});
    broken_tests->insert({"softmax_cross_entropy_input_shape_is_NCd1d2d3_none_no_weight_negative_ignore_index_log_prob",
                          "DML does not support 5D+ tensors"});
    broken_tests->insert(
        {"softmax_cross_entropy_input_shape_is_NCd1d2d3_none_no_weight_negative_ignore_index_log_prob_expanded",
         "DML does not support 5D+ tensors"});
    broken_tests->insert(
        {"softmax_cross_entropy_input_shape_is_NCd1d2d3d4d5_mean_weight", "DML does not support 5D+ tensors"});
    broken_tests->insert(
        {"softmax_cross_entropy_input_shape_is_NCd1d2d3d4d5_mean_weight_expanded", "DML does not support 5D+ tensors"});
    broken_tests->insert(
        {"softmax_cross_entropy_input_shape_is_NCd1d2d3d4d5_mean_weight_log_prob", "DML does not support 5D+ tensors"});
    broken_tests->insert({"softmax_cross_entropy_input_shape_is_NCd1d2d3d4d5_mean_weight_log_prob_expanded",
                          "DML does not support 5D+ tensors"});
    broken_tests->insert(
        {"softmax_cross_entropy_input_shape_is_NCd1d2d3d4d5_none_no_weight", "DML does not support 5D+ tensors"});
    broken_tests->insert({"softmax_cross_entropy_input_shape_is_NCd1d2d3d4d5_none_no_weight_expanded",
                          "DML does not support 5D+ tensors"});
    broken_tests->insert({"softmax_cross_entropy_input_shape_is_NCd1d2d3d4d5_none_no_weight_log_prob",
                          "DML does not support 5D+ tensors"});
    broken_tests->insert({"softmax_cross_entropy_input_shape_is_NCd1d2d3d4d5_none_no_weight_log_prob_expanded",
                          "DML does not support 5D+ tensors"});
  }

  if (provider_name == "qnn") {
    broken_tests->insert({"gemm_default_no_bias", "result differs"});
    broken_tests->insert({"resize_downsample_scales_linear", "result differs"});
    broken_tests->insert({"resize_downsample_scales_linear_antialias", "result differs"});
    broken_tests->insert({"resize_downsample_sizes_linear_antialias", "result differs"});
    broken_tests->insert({"sce_NCd1_mean_weight_negative_ii", "result differs"});
    broken_tests->insert({"sce_NCd1_mean_weight_negative_ii_expanded", "result differs"});
    broken_tests->insert({"sce_NCd1_mean_weight_negative_ii_log_prob", "result differs"});
    broken_tests->insert({"sce_NCd1_mean_weight_negative_ii_log_prob_expanded", "result differs"});
    broken_tests->insert({"sce_mean", "result differs"});
    broken_tests->insert({"sce_mean_3d", "result differs"});
    broken_tests->insert({"sce_mean_3d_expanded", "result differs"});
    broken_tests->insert({"sce_mean_3d_log_prob", "result differs"});
    broken_tests->insert({"sce_mean_3d_log_prob_expanded", "result differs"});
    broken_tests->insert({"sce_mean_expanded", "result differs"});
    broken_tests->insert({"sce_mean_log_prob", "result differs"});
    broken_tests->insert({"sce_mean_log_prob_expanded", "result differs"});
    broken_tests->insert({"sce_mean_no_weight_ii", "result differs"});
    broken_tests->insert({"sce_mean_no_weight_ii_3d", "result differs"});
    broken_tests->insert({"sce_mean_no_weight_ii_3d_expanded", "result differs"});
    broken_tests->insert({"sce_mean_no_weight_ii_3d_log_prob", "result differs"});
    broken_tests->insert({"sce_mean_no_weight_ii_3d_log_prob_expanded", "result differs"});
    broken_tests->insert({"sce_mean_no_weight_ii_4d", "result differs"});
    broken_tests->insert({"sce_mean_no_weight_ii_4d_expanded", "result differs"});
    broken_tests->insert({"sce_mean_no_weight_ii_4d_log_prob", "result differs"});
    broken_tests->insert({"sce_mean_no_weight_ii_4d_log_prob_expanded", "result differs"});
    broken_tests->insert({"sce_mean_no_weight_ii_expanded", "result differs"});
    broken_tests->insert({"sce_mean_no_weight_ii_log_prob", "result differs"});
    broken_tests->insert({"sce_mean_no_weight_ii_log_prob_expanded", "result differs"});
    broken_tests->insert({"sce_mean_weight", "result differs"});
    broken_tests->insert({"sce_mean_weight_expanded", "result differs"});
    broken_tests->insert({"sce_mean_weight_ii", "result differs"});
    broken_tests->insert({"sce_mean_weight_ii_3d", "result differs"});
    broken_tests->insert({"sce_mean_weight_ii_3d_expanded", "result differs"});
    broken_tests->insert({"sce_mean_weight_ii_3d_log_prob", "result differs"});
    broken_tests->insert({"sce_mean_weight_ii_3d_log_prob_expanded", "result differs"});
    broken_tests->insert({"sce_mean_weight_ii_4d", "result differs"});
    broken_tests->insert({"sce_mean_weight_ii_4d_expanded", "result differs"});
    broken_tests->insert({"sce_mean_weight_ii_4d_log_prob", "result differs"});
    broken_tests->insert({"sce_mean_weight_ii_4d_log_prob_expanded", "result differs"});
    broken_tests->insert({"sce_mean_weight_ii_expanded", "result differs"});
    broken_tests->insert({"sce_mean_weight_ii_log_prob", "result differs"});
    broken_tests->insert({"sce_mean_weight_ii_log_prob_expanded", "result differs"});
    broken_tests->insert({"sce_mean_weight_log_prob", "result differs"});
    broken_tests->insert({"sce_mean_weight_log_prob_expanded", "result differs"});
    broken_tests->insert({"sce_none", "result differs"});
    broken_tests->insert({"sce_none_expanded", "result differs"});
    broken_tests->insert({"sce_none_log_prob", "result differs"});
    broken_tests->insert({"sce_none_log_prob_expanded", "result differs"});
    broken_tests->insert({"sce_sum", "result differs"});
    broken_tests->insert({"sce_sum_expanded", "result differs"});
    broken_tests->insert({"sce_sum_log_prob", "result differs"});
    broken_tests->insert({"sce_sum_log_prob_expanded", "result differs"});
    broken_tests->insert({"gridsample_reflection_padding", "result differs"});
    broken_tests->insert({"gridsample_volumetric_nearest_align_corners_0", "unknown version"});
    broken_tests->insert({"gridsample_volumetric_nearest_align_corners_1", "unknown version"});
    broken_tests->insert({"spacetodepth", "result differs"});
    // Fails with QNN SDK 2.17.0:
    // expected 7.70947 (40f6b3f3), got 7.84096 (40fae920), diff: 0.131491, tol=0.00870947 idx=419. 100 of 1715 differ
    broken_tests->insert({"facedetection_op8_qdq", "result differs"});
  }

#ifdef DISABLE_CONTRIB_OPS
  broken_tests->insert({"coreml_SqueezeNet_ImageNet", "This model uses contrib ops."});
  broken_tests->insert({"keras2coreml_Permute_ImageNet", "This model uses contrib ops."});
  broken_tests->insert({"keras2coreml_ReLU_ImageNet", "This model uses contrib ops."});
  broken_tests->insert({"keras2coreml_Padding-Upsampling-Normalizer_ImageNet", "This model uses contrib ops."});
  broken_tests->insert({"tiny_yolov2", "This model uses contrib ops."});
  broken_tests->insert({"fp16_tiny_yolov2", "This model uses contrib ops."});
  broken_tests->insert({"keras2coreml_Pooling_ImageNet", "This model uses contrib ops."});
  broken_tests->insert({"keras2coreml_Padding_ImageNet", "This model uses contrib ops."});
  broken_tests->insert({"keras2coreml_Normalizer_ImageNet", "This model uses contrib ops."});
  broken_tests->insert({"keras2coreml_linear_sklearn_load_breast_cancer", "This model uses contrib ops."});
  broken_tests->insert({"keras2coreml_linear_ImageNet_small", "This model uses contrib ops."});
  broken_tests->insert({"keras2coreml_linear_ImageNet_large", "This model uses contrib ops."});
  broken_tests->insert({"keras2coreml_linear_ImageNet", "This model uses contrib ops."});
  broken_tests->insert({"keras2coreml_leakyrelu_ImageNet", "This model uses contrib ops."});
  broken_tests->insert({"keras2coreml_hard_sigmoid_ImageNet", "This model uses contrib ops."});
  broken_tests->insert({"keras2coreml_elu_ImageNet", "This model uses contrib ops."});
  broken_tests->insert({"keras2coreml_Dense_ImageNet", "This model uses contrib ops."});
  broken_tests->insert({"keras2coreml_Conv2D_ImageNet", "This model uses contrib ops."});
  broken_tests->insert({"coreml_VGG16_ImageNet", "This model uses contrib ops."});
  broken_tests->insert({"coreml_Resnet50_ImageNet", "This model uses contrib ops."});
  broken_tests->insert({"coreml_Inceptionv3_ImageNet", "This model uses contrib ops."});
  broken_tests->insert({"coreml_FNS-Candy_ImageNet", "This model uses contrib ops."});
  broken_tests->insert({"coreml_AgeNet_ImageNet", "This model uses contrib ops."});
  broken_tests->insert({"keras2coreml_thresholdedrelu_ImageNet_large", "This model uses contrib ops."});
  broken_tests->insert({"keras2coreml_thresholdedrelu_ImageNet_small", "This model uses contrib ops."});
  broken_tests->insert({"keras2coreml_thresholdedrelu_sklearn_load_breast_cancer", "This model uses contrib ops."});
  broken_tests->insert({"thresholdedrelu", "This model uses contrib ops."});
  broken_tests->insert({"thresholdedrelu_default", "This model uses contrib ops."});
  broken_tests->insert({"dynamic_slice_default_axes", "This model uses contrib ops."});
  broken_tests->insert({"thresholdedrelu_example", "This model uses contrib ops."});
  broken_tests->insert({"dynamic_slice_neg failed", "This model uses contrib ops."});
  broken_tests->insert({"dynamic_slice_start_out_of_bounds", "This model uses contrib ops."});
  broken_tests->insert({"dynamic_slice", "This model uses contrib ops."});
  broken_tests->insert({"dynamic_slice_end_out_of_bounds", "This model uses contrib ops."});
  broken_tests->insert({"dynamic_slice_neg", "This model uses contrib ops."});
  broken_tests->insert({"mvn", "This model uses contrib ops.", {"onnx130"}});
  broken_tests->insert({"cdist_float32_euclidean_1000_2000_1", "This model uses contrib ops."});
  broken_tests->insert({"cdist_float32_euclidean_1000_2000_500", "This model uses contrib ops."});
  broken_tests->insert({"cdist_float32_euclidean_1_1_1", "This model uses contrib ops."});
  broken_tests->insert({"cdist_float32_sqeuclidean_1000_2000_1", "This model uses contrib ops."});
  broken_tests->insert({"cdist_float32_sqeuclidean_1000_2000_500", "This model uses contrib ops."});
  broken_tests->insert({"cdist_float32_sqeuclidean_1_1_1", "This model uses contrib ops."});
  broken_tests->insert({"cdist_float64_euclidean_1000_2000_1", "This model uses contrib ops."});
  broken_tests->insert({"cdist_float64_euclidean_1000_2000_500", "This model uses contrib ops."});
  broken_tests->insert({"cdist_float64_euclidean_1_1_1", "This model uses contrib ops."});
  broken_tests->insert({"cdist_float64_sqeuclidean_1000_2000_1", "This model uses contrib ops."});
  broken_tests->insert({"cdist_float64_sqeuclidean_1000_2000_500", "This model uses contrib ops."});
  broken_tests->insert({"cdist_float64_sqeuclidean_1_1_1", "This model uses contrib ops."});
  broken_tests->insert({"keras2coreml_Average_ImageNet", "This model uses contrib ops."});
  broken_tests->insert({"bidaf", "This model uses contrib ops."});
  broken_tests->insert({"fp16_test_tiny_yolov2", "This model uses contrib ops."});
  broken_tests->insert({"fp16_coreml_FNS-Candy", "This model uses contrib ops."});
  broken_tests->insert({"keras2coreml_Repeat_ImageNet", "This model uses contrib ops."});
  broken_tests->insert({"keras2coreml_BiDirectional_ImageNet", "This model uses contrib ops."});
  broken_tests->insert({"fp16_coreml_LinearRegression_NYCTaxi", "This model uses contrib ops."});
  broken_tests->insert({"keras2coreml_Average_ImageNet", "This model uses contrib ops."});
  broken_tests->insert({"keras2coreml_GRU_ImageNet", "This model uses contrib ops."});
  broken_tests->insert({"keras2coreml_SimpleRNN_ImageNet", "This model uses contrib ops."});
  broken_tests->insert({"keras2coreml_Dot_imageNet", "This model uses contrib ops."});
#endif
  return broken_tests;
}

// Some EPs may fail to pass some specific testcases.
// For example TenosrRT EP may fail on FLOAT16 related testcases if GPU doesn't support float16.
// Instead of list all these testcases, we can use following keyword set to filter out testcases wchich contain
// specific keyword.
std::unique_ptr<std::set<std::string>> GetBrokenTestsKeyWordSet(const std::string& provider_name) {
  auto broken_tests_keyword_set = std::make_unique<std::set<std::string>>();
  if (provider_name == "tensorrt") {
    broken_tests_keyword_set->insert({"scatternd_add"});
    broken_tests_keyword_set->insert({"scatternd_multiply"});
    broken_tests_keyword_set->insert({"scatter_elements_with_duplicate_indices"});

    // sce op is not supported
    broken_tests_keyword_set->insert({"sce"});

    // TensorRT EP CI uses Nvidia Tesla M60 which doesn't support fp16.
    broken_tests_keyword_set->insert({"FLOAT16"});
  }
  return broken_tests_keyword_set;
}
