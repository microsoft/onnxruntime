// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// needs to be included first to get around onnxruntime\cmake\external\onnx\onnx/common/constants.h(14): error C2513: 'bool': no variable declared before '='
#include "tensorprotoutils.h"

#include "TestCase.h"
#include <cctype>
#include <fstream>
#include <memory>
#include "core/common/logging/logging.h"
#include "core/common/common.h"
#include "core/platform/env.h"
#include "core/platform/ort_mutex.h"
#include "core/platform/path_lib.h"
#include "core/session/onnxruntime_cxx_api.h"
#include "core/framework/allocator.h"
#include "re2/re2.h"
#include <sstream>
#include <map>
#include <regex>
#include "OrtValueList.h"
#include "onnx_model_info.h"

#include "pb_helper.h"

using namespace onnxruntime;
using namespace onnxruntime::common;
using google::protobuf::RepeatedPtrField;

static constexpr int protobuf_block_size_in_bytes = 4 * 1024 * 1024;

using ORT_VALUE_HOLDER = std::unique_ptr<OrtValue, decltype(Ort::GetApi().ReleaseValue)>;

const std::string TestModelInfo::unknown_version = "unknown version";

namespace {
template <typename T>
ONNXTensorElementDataType NumericTypeToONNXType();
template <>
ONNXTensorElementDataType NumericTypeToONNXType<float>() {
  return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
}

template <>
ONNXTensorElementDataType NumericTypeToONNXType<double>() {
  return ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE;
}

template <>
ONNXTensorElementDataType NumericTypeToONNXType<int64_t>() {
  return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
}

template <>
ONNXTensorElementDataType NumericTypeToONNXType<std::string>() {
  return ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING;
}

template <typename T>
OrtValue* CreateTensorWithDataAsOrtValue(OrtMemoryInfo* info, std::vector<T>& input) {
  std::vector<int64_t> dims(1, input.size());
  OrtValue* ret = nullptr;
  Ort::ThrowOnError(Ort::GetApi().CreateTensorWithDataAsOrtValue(info, input.data(), input.size() * sizeof(T), dims.data(),
                                                                 dims.size(), NumericTypeToONNXType<T>(), &ret));
  return ret;
}

template <typename key_type, typename value_type>
OrtValue* PbMapToOrtValue(const google::protobuf::Map<key_type, value_type>& map) {
  OrtMemoryInfo* info;
  Ort::ThrowOnError(Ort::GetApi().CreateCpuMemoryInfo(OrtDeviceAllocator, OrtMemTypeDefault, &info));
  std::unique_ptr<OrtMemoryInfo, decltype(Ort::GetApi().ReleaseMemoryInfo)> rel_info(info, Ort::GetApi().ReleaseMemoryInfo);
  const size_t ele_count = map.size();
  std::vector<int64_t> dims(1, ele_count);
  std::vector<key_type> keys(ele_count);
  std::vector<value_type> values(ele_count);
  size_t i = 0;
  for (auto& kvp : map) {
    keys[i] = kvp.first;
    values[i] = kvp.second;
    ++i;
  }
  OrtValueArray map_in(2);
  OrtValue* p = CreateTensorWithDataAsOrtValue(info, keys);
  if (p == nullptr) ORT_THROW("Create keys tensor failed");
  map_in.Set(0, p);

  p = CreateTensorWithDataAsOrtValue(info, values);
  if (p == nullptr) ORT_THROW("Create values tensor failed");
  map_in.Set(1, p);

  // create map ort value
  OrtValue* map_ort = nullptr;
  Ort::ThrowOnError(Ort::GetApi().CreateValue(map_in.Data(), map_in.Length(), ONNX_TYPE_MAP, &map_ort));
  return map_ort;
}

template <typename T>
void VectorProtoToOrtValue(const RepeatedPtrField<T>& input, ORT_VALUE_HOLDER& output) {
  OrtMemoryInfo* info;
  Ort::ThrowOnError(Ort::GetApi().CreateCpuMemoryInfo(OrtDeviceAllocator, OrtMemTypeDefault, &info));
  std::unique_ptr<OrtMemoryInfo, decltype(Ort::GetApi().ReleaseMemoryInfo)> rel_info(info, Ort::GetApi().ReleaseMemoryInfo);
  OrtValueArray in(input.size());
  size_t j = 0;
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
    OrtValueArray map_in(2);
    OrtValue* p = CreateTensorWithDataAsOrtValue(info, keys);
    if (p == nullptr) ORT_THROW("Create keys tensor failed");
    map_in.Set(0, p);

    p = CreateTensorWithDataAsOrtValue(info, values);
    if (p == nullptr) ORT_THROW("Create values tensor failed");
    map_in.Set(1, p);

    // create map ort value
    OrtValue* map_ort = nullptr;
    Ort::ThrowOnError(Ort::GetApi().CreateValue(map_in.Data(), map_in.Length(), ONNX_TYPE_MAP, &map_ort));
    in.Set(j++, map_ort);
  }
  OrtValue* seq_ort = nullptr;
  Ort::ThrowOnError(Ort::GetApi().CreateValue(in.Data(), in.Length(), ONNX_TYPE_SEQUENCE, &seq_ort));
  output.reset(seq_ort);
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



static void SortTensorFileNames(std::vector<std::basic_string<PATH_CHAR_TYPE>>& input_pb_files) {
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
      ORT_THROW("illegal input file name:", ToMBString(oss.str()));
    }
  }
}

OrtValue* TensorToOrtValue(const ONNX_NAMESPACE::TensorProto& t, onnxruntime::test::HeapBuffer& b) {
  size_t len = 0;
  auto status = onnxruntime::test::GetSizeInBytesFromTensorProto<0>(t, &len);
  if (!status.IsOK()) {
    ORT_THROW(status.ToString());
  }
  void* p = len == 0 ? nullptr : b.AllocMemory(len);
  Ort::Value temp_value{nullptr};
  auto d = onnxruntime::make_unique<onnxruntime::test::OrtCallback>();
  OrtMemoryInfo cpu_memory_info(onnxruntime::CPU, OrtDeviceAllocator, OrtDevice(), 0, OrtMemTypeDefault);
  status = onnxruntime::test::TensorProtoToMLValue(t, onnxruntime::test::MemBuffer(p, len, cpu_memory_info),
                                                   temp_value, *d);
  if (!status.IsOK()) {
    ORT_THROW(status.ToString());
  }
  if (d->f) {
    b.AddDeleter(d.release());
  }
  return temp_value.release();
}

void LoopDataFile(int test_data_pb_fd, bool is_input, const TestModelInfo& modelinfo,
                  std::unordered_map<std::string, OrtValue*>& name_data_map, onnxruntime::test::HeapBuffer& b,
                  std::ostringstream& oss) {
  google::protobuf::io::FileInputStream f(test_data_pb_fd, protobuf_block_size_in_bytes);
  f.SetCloseOnDelete(true);
  google::protobuf::io::CodedInputStream coded_input(&f);
  bool clean_eof = false;
  int item_id = 1;
  for (proto::TraditionalMLData data;
       ParseDelimitedFromCodedStream(&data, &coded_input, &clean_eof);
       ++item_id, data.Clear()) {
    try {
      ORT_VALUE_HOLDER gvalue(nullptr, Ort::GetApi().ReleaseValue);
      switch (data.values_case()) {
        case proto::TraditionalMLData::kVectorMapStringToFloat:
          VectorProtoToOrtValue(data.vector_map_string_to_float().v(), gvalue);
          break;
        case proto::TraditionalMLData::kVectorMapInt64ToFloat:
          VectorProtoToOrtValue(data.vector_map_int64_to_float().v(), gvalue);
          break;
        case proto::TraditionalMLData::kMapStringToString:
          gvalue.reset(PbMapToOrtValue(data.map_string_to_string().v()));
          break;
        case proto::TraditionalMLData::kMapStringToInt64:
          gvalue.reset(PbMapToOrtValue(data.map_string_to_int64().v()));
          break;
        case proto::TraditionalMLData::kMapStringToFloat:
          gvalue.reset(PbMapToOrtValue(data.map_string_to_float().v()));
          break;
        case proto::TraditionalMLData::kMapStringToDouble:
          gvalue.reset(PbMapToOrtValue(data.map_string_to_double().v()));
          break;
        case proto::TraditionalMLData::kMapInt64ToString:
          gvalue.reset(PbMapToOrtValue(data.map_int64_to_string().v()));
          break;
        case proto::TraditionalMLData::kMapInt64ToInt64:
          gvalue.reset(PbMapToOrtValue(data.map_int64_to_int64().v()));
          break;
        case proto::TraditionalMLData::kMapInt64ToFloat:
          gvalue.reset(PbMapToOrtValue(data.map_int64_to_float().v()));
          break;
        case proto::TraditionalMLData::kMapInt64ToDouble:
          gvalue.reset(PbMapToOrtValue(data.map_int64_to_double().v()));
          break;
        case proto::TraditionalMLData::kTensor: {
          gvalue.reset(TensorToOrtValue(data.tensor(), b));
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

      auto pv = name_data_map.insert(std::make_pair(value_name, gvalue.release()));
      if (!pv.second) {
        ORT_THROW("duplicated test data name");
        break;
      }
    } catch (onnxruntime::NotImplementedException& ex) {
      std::ostringstream oss2;
      oss2 << "load the " << item_id << "-th item failed," << ex.what();
      ORT_NOT_IMPLEMENTED(oss2.str());
    } catch (std::exception& ex) {
      std::ostringstream oss2;
      oss2 << "load the " << item_id << "-th item failed," << ex.what();
      ORT_THROW(oss2.str());
    }
  }
  if (!clean_eof) {
    ORT_THROW("parse input file failed, has extra unparsed data");
  }
}

}  // namespace

std::unique_ptr<TestModelInfo> TestModelInfo::LoadOnnxModel(_In_ const PATH_CHAR_TYPE* model_url) {
  return std::unique_ptr<TestModelInfo>(new OnnxModelInfo(model_url));
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

  void ConvertTestData(const std::vector<ONNX_NAMESPACE::TensorProto>& test_data_pbs,
                       onnxruntime::test::HeapBuffer& b, bool is_input,
                       std::unordered_map<std::string, OrtValue*>& out) const;

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
  Status GetPerSampleTolerance(double* value) const override;
  Status GetRelativePerSampleTolerance(double* value) const override;
  Status GetPostProcessing(bool* value) const override;

  const ONNX_NAMESPACE::ValueInfoProto* GetOutputInfoFromModel(size_t i) const override {
    return model_info_->GetOutputInfoFromModel(i);
  }

  size_t GetDataCount() const override { return test_data_dirs_.size(); }
  const std::string& GetNodeName() const override { return model_info_->GetNodeName(); }
  const PATH_CHAR_TYPE* GetModelUrl() const override { return model_info_->GetModelUrl(); }
  const std::string& GetTestCaseName() const override { return test_case_name_; }
  std::string GetTestCaseVersion() const override { return model_info_->GetModelVersion(); }

  void LoadTestData(size_t id, onnxruntime::test::HeapBuffer& b, std::unordered_map<std::string, OrtValue*>&,
                    bool is_input) const override;
};

std::unique_ptr<ITestCase> CreateOnnxTestCase(const std::string& test_case_name,
                                              std::unique_ptr<TestModelInfo> model,
                                              double default_per_sample_tolerance,
                                              double default_relative_per_sample_tolerance) {
  return std::unique_ptr<ITestCase>(new OnnxTestCase(test_case_name, std::move(model),
                                                     default_per_sample_tolerance,
                                                     default_relative_per_sample_tolerance));
}

Status OnnxTestCase::GetPerSampleTolerance(double* value) const {
  *value = per_sample_tolerance_;
  return Status::OK();
}

Status OnnxTestCase::GetRelativePerSampleTolerance(double* value) const {
  *value = relative_per_sample_tolerance_;
  return Status::OK();
}

Status OnnxTestCase::GetPostProcessing(bool* value) const {
  *value = post_processing_;
  return Status::OK();
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

//load tensors from disk
template <typename PATH_STRING_TYPE>
static void LoadTensors(const std::vector<PATH_STRING_TYPE>& pb_files,
                        std::vector<ONNX_NAMESPACE::TensorProto>* input_pbs) {
  for (size_t i = 0; i != pb_files.size(); ++i) {
    int tensor_fd;
    auto st = Env::Default().FileOpenRd(pb_files.at(i), tensor_fd);
    if (!st.IsOK()) {
      ORT_THROW("open file '", ToMBString(pb_files.at(i)), "' failed:", st.ErrorMessage());
    }
    google::protobuf::io::FileInputStream f(tensor_fd, protobuf_block_size_in_bytes);
    f.SetCloseOnDelete(true);
    ONNX_NAMESPACE::TensorProto tensor;
    if (!tensor.ParseFromZeroCopyStream(&f)) {
      ORT_THROW("parse file '", ToMBString(pb_files.at(i)), "' failed");
    }
    input_pbs->emplace_back(tensor);
  }
}

void OnnxTestCase::LoadTestData(size_t id, onnxruntime::test::HeapBuffer& b,
                                std::unordered_map<std::string, OrtValue*>& name_data_map,
                                bool is_input) const {
  if (id >= test_data_dirs_.size()) {
    ORT_THROW("index out of bound");
  }

  PATH_STRING_TYPE test_data_pb = ConcatPathComponent<PATH_CHAR_TYPE>(
      test_data_dirs_[id], (is_input ? ORT_TSTR("inputs.pb") : ORT_TSTR("outputs.pb")));
  int test_data_pb_fd;
  auto st = Env::Default().FileOpenRd(test_data_pb, test_data_pb_fd);
  if (st.IsOK()) {  //has an all-in-one input file
    std::ostringstream oss;
    {
      std::lock_guard<OrtMutex> l(m_);
      oss << debuginfo_strings_[id];
    }
    try {
      LoopDataFile(test_data_pb_fd, is_input, *model_info_, name_data_map, b, oss);
    } catch (std::exception& ex) {
      std::ostringstream oss2;
      oss2 << "parse data file \"" << ToMBString(test_data_pb) << "\" failed:" << ex.what();
      ORT_THROW(oss.str());
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
              std::basic_string<PATH_CHAR_TYPE> p = ConcatPathComponent<PATH_CHAR_TYPE>(dir_path, filename_str);
              test_data_pb_files.push_back(p);
            }
            return true;
          });

  SortTensorFileNames(test_data_pb_files);

  std::vector<ONNX_NAMESPACE::TensorProto> test_data_pbs;
  LoadTensors(test_data_pb_files, &test_data_pbs);
  ConvertTestData(test_data_pbs, b, is_input, name_data_map);
}

void OnnxTestCase::ConvertTestData(const std::vector<ONNX_NAMESPACE::TensorProto>& test_data_pbs,
                                   onnxruntime::test::HeapBuffer& b,
                                   bool is_input, std::unordered_map<std::string, OrtValue*>& out) const {
  bool has_valid_names = true;
  std::vector<std::string> var_names(test_data_pbs.size());
  for (size_t input_index = 0; input_index != test_data_pbs.size(); ++input_index) {
    std::string name = test_data_pbs[input_index].name();
    if (name.empty()) {
      has_valid_names = false;
      break;
    }
    var_names[input_index] = name;
  }
  if (!has_valid_names) {
    size_t count = static_cast<size_t>(is_input ? model_info_->GetInputCount() : model_info_->GetOutputCount());
    if (count != test_data_pbs.size()) {
      ORT_THROW("data count mismatch, expect ", count, ", got ", test_data_pbs.size());
    }
    for (size_t i = 0; i != count; ++i) {
      var_names[i] = is_input ? model_info_->GetInputName(i) : model_info_->GetOutputName(i);
    }
  }
  for (size_t input_index = 0; input_index != test_data_pbs.size(); ++input_index) {
    std::string name = var_names[input_index];
    const ONNX_NAMESPACE::TensorProto& input = test_data_pbs[input_index];
    size_t len = 0;

    auto status = onnxruntime::test::GetSizeInBytesFromTensorProto<0>(input, &len);
    if (!status.IsOK()) {
      ORT_THROW(status.ToString());
    }
    void* p = len == 0 ? nullptr : b.AllocMemory(len);
    Ort::Value v1{nullptr};
    auto d = onnxruntime::make_unique<onnxruntime::test::OrtCallback>();
    OrtMemoryInfo cpu_memory_info(onnxruntime::CPU, OrtDeviceAllocator, OrtDevice(), 0, OrtMemTypeDefault);
    status = onnxruntime::test::TensorProtoToMLValue(input, onnxruntime::test::MemBuffer(p, len, cpu_memory_info),
                                                     v1, *d);
    if (!status.IsOK()) {
      ORT_THROW(status.ToString());
    }
    if (d->f) {
      b.AddDeleter(d.release());
    }
    out.insert(std::make_pair(name, v1.release()));
  }
}

OnnxTestCase::OnnxTestCase(const std::string& test_case_name, _In_ std::unique_ptr<TestModelInfo> model,
                           double default_per_sample_tolerance, double default_relative_per_sample_tolerance)
    : test_case_name_(test_case_name), model_info_(std::move(model)) {
  std::basic_string<PATH_CHAR_TYPE> test_case_dir = model_info_->GetDir();

  // parse config
  std::basic_string<PATH_CHAR_TYPE> config_path =
      ConcatPathComponent<PATH_CHAR_TYPE>(test_case_dir, ORT_TSTR("config.txt"));
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
      std::basic_string<PATH_CHAR_TYPE> p = ConcatPathComponent<PATH_CHAR_TYPE>(test_case_dir, filename);
      test_data_dirs_.push_back(p);
      debuginfo_strings_.push_back(ToMBString(p));
    }
    return true;
  });
}
