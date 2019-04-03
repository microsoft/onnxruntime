// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

//TODO: switch to use onnxruntime public api

#include "TestCase.h"
#include <fstream>
#include <memory>
#include "core/common/logging/logging.h"
#include "core/common/common.h"
#include "core/platform/env.h"
#include "core/platform/ort_mutex.h"
#include "core/session/onnxruntime_cxx_api.h"
#include "core/framework/path_lib.h"
#include <sstream>
#include <map>
#include <regex>
#include "OrtValueList.h"

#ifdef __GNUC__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wignored-qualifiers"
#pragma GCC diagnostic ignored "-Wunused-parameter"
#else
#pragma warning(push)
#pragma warning(disable : 4018) /*'expression' : signed/unsigned mismatch */
#pragma warning(disable : 4065) /*switch statement contains 'default' but no 'case' labels*/
#pragma warning(disable : 4100)
#pragma warning(disable : 4505)
#pragma warning(disable : 4146) /*unary minus operator applied to unsigned type, result still unsigned*/
#pragma warning(disable : 4244) /*'conversion' conversion from 'type1' to 'type2', possible loss of data*/
#pragma warning(disable : 4251) /*'identifier' : class 'type' needs to have dll-interface to be used by clients of class 'type2'*/
#pragma warning(disable : 4267) /*'var' : conversion from 'size_t' to 'type', possible loss of data*/
#pragma warning(disable : 4305) /*'identifier' : truncation from 'type1' to 'type2'*/
#pragma warning(disable : 4307) /*'operator' : integral constant overflow*/
#pragma warning(disable : 4309) /*'conversion' : truncation of constant value*/
#pragma warning(disable : 4334) /*'operator' : result of 32-bit shift implicitly converted to 64 bits (was 64-bit shift intended?)*/
#pragma warning(disable : 4355) /*'this' : used in base member initializer list*/
#pragma warning(disable : 4506) /*no definition for inline function 'function'*/
#pragma warning(disable : 4800) /*'type' : forcing value to bool 'true' or 'false' (performance warning)*/
#pragma warning(disable : 4996) /*The compiler encountered a deprecated declaration.*/
#endif
#include <google/protobuf/util/delimited_message_util.h>
#include "tml.pb.h"
#ifdef __GNUC__
#pragma GCC diagnostic pop
#else
#pragma warning(pop)
#endif

using namespace onnxruntime;
using namespace onnxruntime::common;
using google::protobuf::RepeatedPtrField;

using ORT_VALUE_HOLDER = std::unique_ptr<OrtValue, decltype(&OrtReleaseValue)>;

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
OrtValue* CreateTensorWithDataAsOrtValue(OrtAllocatorInfo* info, std::vector<T>& input) {
  std::vector<int64_t> dims(1, input.size());
  OrtValue* ret = nullptr;
  ORT_THROW_ON_ERROR(::OrtCreateTensorWithDataAsOrtValue(info, input.data(), input.size() * sizeof(T), dims.data(),
                                                         dims.size(), NumericTypeToONNXType<T>(), &ret));
  return ret;
}

template <typename key_type, typename value_type>
OrtValue* PbMapToOrtValue(const google::protobuf::Map<key_type, value_type>& map) {
  OrtAllocatorInfo* info;
  ORT_THROW_ON_ERROR(OrtCreateAllocatorInfo("Cpu", OrtDeviceAllocator, 0, OrtMemTypeDefault, &info));
  std::unique_ptr<OrtAllocatorInfo, decltype(&OrtReleaseAllocatorInfo)> rel_info(info, OrtReleaseAllocatorInfo);
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
  ORT_THROW_ON_ERROR(OrtCreateValue(map_in.Data(), map_in.Length(), ONNX_TYPE_MAP, &map_ort));
  return map_ort;
}

template <typename T>
void VectorProtoToOrtValue(const RepeatedPtrField<T>& input, ORT_VALUE_HOLDER& output) {
  OrtAllocatorInfo* info;
  ORT_THROW_ON_ERROR(OrtCreateAllocatorInfo("Cpu", OrtDeviceAllocator, 0, OrtMemTypeDefault, &info));
  std::unique_ptr<OrtAllocatorInfo, decltype(&OrtReleaseAllocatorInfo)> rel_info(info, OrtReleaseAllocatorInfo);
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
    ORT_THROW_ON_ERROR(OrtCreateValue(map_in.Data(), map_in.Length(), ONNX_TYPE_MAP, &map_ort));
    in.Set(j++, map_ort);
  }
  OrtValue* seq_ort = nullptr;
  ORT_THROW_ON_ERROR(OrtCreateValue(in.Data(), in.Length(), ONNX_TYPE_SEQUENCE, &seq_ort));
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

class OnnxModelInfo : public TestModelInfo {
 private:
  std::string node_name_;
  std::vector<ONNX_NAMESPACE::ValueInfoProto> input_value_info_;
  std::vector<ONNX_NAMESPACE::ValueInfoProto> output_value_info_;

  template <typename T>
  static void RepeatedPtrFieldToVector(const ::google::protobuf::RepeatedPtrField<T>& input_value_info,
                                       std::vector<T>& out) {
    for (int i = 0; i != input_value_info.size(); ++i) {
      out.push_back(input_value_info[i]);
    }
  }
  const std::basic_string<PATH_CHAR_TYPE> model_url_;

 public:
  OnnxModelInfo(_In_ const PATH_CHAR_TYPE* model_url) : model_url_(model_url) {
    // parse model
    int model_fd;
    auto st = Env::Default().FileOpenRd(model_url, model_fd);
    if (!st.IsOK()) {
      ORT_THROW(st.ErrorMessage());
    }
    google::protobuf::io::FileInputStream f(model_fd);
    f.SetCloseOnDelete(true);
    ONNX_NAMESPACE::ModelProto model_pb;
    if (!model_pb.ParseFromZeroCopyStream(&f)) {
      ORT_THROW("Failed to load model because protobuf parsing failed.");
    }

    const ONNX_NAMESPACE::GraphProto& graph = model_pb.graph();
    if (graph.node().size() == 1) {
      node_name_ = graph.node()[0].op_type();
    }
    std::unordered_set<std::string> initializer_names;
    for (const auto& init : graph.initializer()) {
      if (!init.has_name()) continue;
      initializer_names.insert(init.name());
    }
    for (const auto& p : graph.input()) {
      if (!p.has_name()) ORT_THROW("input without name??");
      if (initializer_names.find(p.name()) == initializer_names.end()) input_value_info_.push_back(p);
    }
    RepeatedPtrFieldToVector(graph.output(), output_value_info_);
  }

  const PATH_CHAR_TYPE* GetModelUrl() const override { return model_url_.c_str(); }

  std::basic_string<PATH_CHAR_TYPE> GetDir() const override {
    std::basic_string<PATH_CHAR_TYPE> test_case_dir;
    auto st = GetDirNameFromFilePath(model_url_, test_case_dir);
    if (!st.IsOK()) {
      ORT_THROW("GetDirNameFromFilePath failed");
    }
    return test_case_dir;
  }
  const std::string& GetNodeName() const override { return node_name_; }
  const ONNX_NAMESPACE::ValueInfoProto* GetOutputInfoFromModel(size_t i) const override {
    return &output_value_info_[i];
  }
  int GetInputCount() const override { return static_cast<int>(input_value_info_.size()); }
  int GetOutputCount() const override { return static_cast<int>(output_value_info_.size()); }
  const std::string& GetInputName(size_t i) const override { return input_value_info_[i].name(); }

  const std::string& GetOutputName(size_t i) const override { return output_value_info_[i].name(); }
};

template <typename PATH_CHAR_TYPE>
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
      ORT_THROW("illegal input file name:", ToMBString(input_pb_files[i]));
    }
  }
}

OrtValue* TensorToOrtValue(const ONNX_NAMESPACE::TensorProto& t, HeapBuffer& b) {
  std::string s = t.SerializeAsString();
  size_t len;
  ORT_THROW_ON_ERROR(OrtGetTensorMemSizeInBytesFromTensorProto(s.data(), static_cast<int>(s.size()), 0, &len));
  void* p = len == 0 ? nullptr : b.AllocMemory(len);
  OrtCallback* d;
  OrtValue* temp_value = nullptr;
  ORT_THROW_ON_ERROR(OrtTensorProtoToOrtValue(s.data(), static_cast<int>(s.size()), nullptr, p, len, &temp_value, &d));
  if (d != nullptr) {
    b.AddDeleter(d);
  }
  return temp_value;
}

void LoopDataFile(int test_data_pb_fd, bool is_input, const TestModelInfo* modelinfo,
                  std::unordered_map<std::string, OrtValue*>& name_data_map, HeapBuffer& b, std::ostringstream& oss) {
  google::protobuf::io::FileInputStream f(test_data_pb_fd);
  f.SetCloseOnDelete(true);
  google::protobuf::io::CodedInputStream coded_input(&f);
  bool clean_eof = false;
  int item_id = 1;
  for (proto::TraditionalMLData data;
       google::protobuf::util::ParseDelimitedFromCodedStream(&data, &coded_input, &clean_eof);
       ++item_id, data.Clear()) {
    try {
      ORT_VALUE_HOLDER gvalue(nullptr, OrtReleaseValue);
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
        value_name = is_input ? modelinfo->GetInputName(c) : modelinfo->GetOutputName(c);
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

TestModelInfo* TestModelInfo::LoadOnnxModel(_In_ const PATH_CHAR_TYPE* model_url) {
  return new OnnxModelInfo(model_url);
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
  std::vector<std::string> debuginfo_strings;
  onnxruntime::OrtMutex m_;

  std::vector<std::basic_string<PATH_CHAR_TYPE>> test_data_dirs_;

  std::string GetDatasetDebugInfoString(size_t dataset_id) override {
    std::lock_guard<OrtMutex> l(m_);
    if (dataset_id < debuginfo_strings.size()) {
      return debuginfo_strings[dataset_id];
    }
    // return empty string
    return std::string();
  }

  void ConvertTestData(const std::vector<ONNX_NAMESPACE::TensorProto>& test_data_pbs, HeapBuffer& b, bool is_input,
                       std::unordered_map<std::string, OrtValue*>& out);

  std::once_flag model_parsed_;
  std::once_flag config_parsed_;
  double per_sample_tolerance_;
  double relative_per_sample_tolerance_;
  bool post_processing_;
  TestModelInfo* model_info_;
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(OnnxTestCase);

 public:
  OnnxTestCase(const std::string& test_case_name, TestModelInfo* model, double default_per_sample_tolerance,
               double default_relative_per_sample_tolerance);
  ~OnnxTestCase() override { delete model_info_; }
  Status GetPerSampleTolerance(double* value) override;
  Status GetRelativePerSampleTolerance(double* value) override;
  Status GetPostProcessing(bool* value) override;

  const ONNX_NAMESPACE::ValueInfoProto* GetOutputInfoFromModel(size_t i) const override {
    return model_info_->GetOutputInfoFromModel(i);
  }

  size_t GetDataCount() const override {
    return test_data_dirs_.size();
  }
  const std::string& GetNodeName() const override { return model_info_->GetNodeName(); }

  const PATH_CHAR_TYPE* GetModelUrl() const override { return model_info_->GetModelUrl(); }
  const std::string& GetTestCaseName() const override {
    return test_case_name_;
  }
  void LoadTestData(size_t id, HeapBuffer& b, std::unordered_map<std::string, OrtValue*>&, bool is_input) override;
};

ITestCase* CreateOnnxTestCase(const std::string& test_case_name, TestModelInfo* model,
                              double default_per_sample_tolerance, double default_relative_per_sample_tolerance) {
  return new OnnxTestCase(test_case_name, model, default_per_sample_tolerance, default_relative_per_sample_tolerance);
}

Status OnnxTestCase::GetPerSampleTolerance(double* value) {
  *value = per_sample_tolerance_;
  return Status::OK();
}

Status OnnxTestCase::GetRelativePerSampleTolerance(double* value) {
  *value = relative_per_sample_tolerance_;
  return Status::OK();
}

Status OnnxTestCase::GetPostProcessing(bool* value) {
  *value = post_processing_;
  return Status::OK();
}

static std::string trim_str(const std::string& s) {
  std::string ltrim = std::regex_replace(s, std::regex("^\\s+"), std::string(""));
  std::string result = std::regex_replace(ltrim, std::regex("\\s+$"), std::string(""));
  return result;
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
    google::protobuf::io::FileInputStream f(tensor_fd);
    f.SetCloseOnDelete(true);
    ONNX_NAMESPACE::TensorProto tensor;
    if (!tensor.ParseFromZeroCopyStream(&f)) {
      ORT_THROW("parse file '", ToMBString(pb_files.at(i)), "' failed");
    }
    input_pbs->emplace_back(tensor);
  }
}

void OnnxTestCase::LoadTestData(size_t id, HeapBuffer& b, std::unordered_map<std::string, OrtValue*>& name_data_map,
                                bool is_input) {
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
      oss << debuginfo_strings[id];
    }
    try {
      LoopDataFile(test_data_pb_fd, is_input, model_info_, name_data_map, b, oss);
    } catch (std::exception& ex) {
      std::ostringstream oss2;
      oss2 << "parse data file \"" << ToMBString(test_data_pb) << "\" failed:" << ex.what();
      ORT_THROW(oss.str());
    }
    {
      std::lock_guard<OrtMutex> l(m_);
      debuginfo_strings[id] = oss.str();
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
            if (!filename_str.compare(0, file_prefix.length(), file_prefix.c_str())) {
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

void OnnxTestCase::ConvertTestData(const std::vector<ONNX_NAMESPACE::TensorProto>& test_data_pbs, HeapBuffer& b,
                                   bool is_input, std::unordered_map<std::string, OrtValue*>& out) {
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
    std::string s = input.SerializeAsString();
    OrtValue* v1;
    size_t len;
    ORT_THROW_ON_ERROR(OrtGetTensorMemSizeInBytesFromTensorProto(s.data(), (int)s.size(), 0, &len));
    void* p = len == 0 ? nullptr : b.AllocMemory(len);
    OrtCallback* d;
    ORT_THROW_ON_ERROR(OrtTensorProtoToOrtValue(s.data(), (int)s.size(), nullptr, p, len, &v1, &d));
    if (d != nullptr) b.AddDeleter(d);
    out.insert(std::make_pair(name, v1));
  }
}

OnnxTestCase::OnnxTestCase(const std::string& test_case_name, _In_ TestModelInfo* model,
                           double default_per_sample_tolerance, double default_relative_per_sample_tolerance)
    : test_case_name_(test_case_name), model_info_(model) {
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
      debuginfo_strings.push_back(ToMBString(p));
    }
    return true;
  });
}
