// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

//TODO: switch to use onnxruntime public api

#include "TestCase.h"
#include <fstream>
#include <memory>
#include "core/common/logging/logging.h"
#include "core/common/common.h"
#include "core/platform/env.h"
#include "core/session/onnxruntime_cxx_api.h"
#include <core/framework/path_lib.h>
//TODO: delete this
#include <core/platform/ort_mutex.h>
#include <core/framework/data_types.h>
#include <core/framework/ml_value.h>
#include <sstream>
#include <map>
#include <regex>

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

namespace {
template <typename InputType, typename OutputType>
Status ConvertVector(const InputType& data, OutputType** vec) {
  OutputType* v = new OutputType();
  for (const auto& i : data) {
    typename OutputType::value_type new_value;
    for (const auto& j : i.v()) {
      new_value[j.first] = j.second;
    }
    v->push_back(new_value);
  }
  *vec = v;
  return Status::OK();
}

template <typename InputType, typename OutputType>
Status Convert(const InputType& tensor_proto, OutputType** p_tensor);

template <>
Status Convert(const google::protobuf::RepeatedPtrField<proto::MapInt64ToFloat>& data, VectorMapInt64ToFloat** vec) {
  return ConvertVector<google::protobuf::RepeatedPtrField<proto::MapInt64ToFloat>, VectorMapInt64ToFloat>(data, vec);
}

template <>
Status Convert(const google::protobuf::RepeatedPtrField<proto::MapStringToFloat>& data, VectorMapStringToFloat** vec) {
  return ConvertVector<google::protobuf::RepeatedPtrField<proto::MapStringToFloat>, VectorMapStringToFloat>(data, vec);
}

template <typename InputType, typename OutputType>
void ConvertMap(const InputType& data, OutputType** out) {
  OutputType* ret = new OutputType();
  for (const auto& pv : data) {
    (*ret)[pv.first] = pv.second;
  }
  *out = ret;
}

template <>
Status Convert(const google::protobuf::Map<std::string, std::string>& data, MapStringToString** out) {
  ConvertMap(data, out);
  return Status::OK();
}

template <>
Status Convert(const google::protobuf::Map<std::string, int64_t>& data, MapStringToInt64** out) {
  ConvertMap(data, out);
  return Status::OK();
}

template <>
Status Convert(const google::protobuf::Map<std::string, float>& data, MapStringToFloat** out) {
  ConvertMap(data, out);
  return Status::OK();
}

template <>
Status Convert(const google::protobuf::Map<std::string, double>& data, MapStringToDouble** out) {
  ConvertMap(data, out);
  return Status::OK();
}

template <>
Status Convert(const google::protobuf::Map<int64_t, std::string>& data, MapInt64ToString** out) {
  ConvertMap(data, out);
  return Status::OK();
}

template <>
Status Convert(const google::protobuf::Map<int64_t, int64_t>& data, MapInt64ToInt64** out) {
  ConvertMap(data, out);
  return Status::OK();
}

template <>
Status Convert(const google::protobuf::Map<int64_t, float>& data, MapInt64ToFloat** out) {
  ConvertMap(data, out);
  return Status::OK();
}

template <>
Status Convert(const google::protobuf::Map<int64_t, double>& data, MapInt64ToDouble** out) {
  ConvertMap(data, out);
  return Status::OK();
}
template <typename InputType, typename OutputType>
Status RichTypeProtoToMLValue(const InputType& input, MLValue& value) {
  OutputType* tensor = nullptr;
  Status st = Convert(input, &tensor);
  if (!st.IsOK()) return st;
  value.Init(tensor,
             DataTypeImpl::GetType<OutputType>(),
             DataTypeImpl::GetType<OutputType>()->GetDeleteFunc());
  return Status::OK();
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
typedef std::basic_string<PATH_CHAR_TYPE> PATH_STRING_TYPE;

template <typename PATH_CHAR_TYPE>
static Status SortTensorFileNames(std::vector<std::basic_string<PATH_CHAR_TYPE>>& input_pb_files) {
  if (input_pb_files.size() <= 1) return Status::OK();
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
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "illegal input file name:", ToMBString(input_pb_files[i]));
    }
  }
  return Status::OK();
}

Status LoopDataFile(int test_data_pb_fd, const std::vector<ONNX_NAMESPACE::ValueInfoProto> value_info,
                    std::unordered_map<std::string, OrtValue*>& name_data_map, HeapBuffer& b, std::ostringstream& oss) {
  google::protobuf::io::FileInputStream f(test_data_pb_fd);
  f.SetCloseOnDelete(true);
  google::protobuf::io::CodedInputStream coded_input(&f);
  bool clean_eof = false;
  Status st;
  int item_id = 1;
  for (proto::TraditionalMLData data; google::protobuf::util::ParseDelimitedFromCodedStream(&data, &coded_input, &clean_eof); ++item_id, data.Clear()) {
    std::unique_ptr<OrtValue, decltype(&OrtReleaseValue)> gvalue(nullptr, OrtReleaseValue);
    MLValue value;
    bool is_tensor = false;
    switch (data.values_case()) {
      case proto::TraditionalMLData::kVectorMapStringToFloat:
        st = RichTypeProtoToMLValue<decltype(data.vector_map_string_to_float().v()), VectorMapStringToFloat>(data.vector_map_string_to_float().v(), value);
        break;
      case proto::TraditionalMLData::kVectorMapInt64ToFloat:
        st = RichTypeProtoToMLValue<decltype(data.vector_map_int64_to_float().v()), VectorMapInt64ToFloat>(data.vector_map_int64_to_float().v(), value);
        break;
      case proto::TraditionalMLData::kMapStringToString:
        st = RichTypeProtoToMLValue<decltype(data.map_string_to_string().v()), MapStringToString>(data.map_string_to_string().v(), value);
        break;
      case proto::TraditionalMLData::kMapStringToInt64:
        st = RichTypeProtoToMLValue<decltype(data.map_string_to_int64().v()), MapStringToInt64>(data.map_string_to_int64().v(), value);
        break;
      case proto::TraditionalMLData::kMapStringToFloat:
        st = RichTypeProtoToMLValue<decltype(data.map_string_to_float().v()), MapStringToFloat>(data.map_string_to_float().v(), value);
        break;
      case proto::TraditionalMLData::kMapStringToDouble:
        st = RichTypeProtoToMLValue<decltype(data.map_string_to_double().v()), MapStringToDouble>(data.map_string_to_double().v(), value);
        break;
      case proto::TraditionalMLData::kMapInt64ToString:
        st = RichTypeProtoToMLValue<decltype(data.map_int64_to_string().v()), MapInt64ToString>(data.map_int64_to_string().v(), value);
        break;
      case proto::TraditionalMLData::kMapInt64ToInt64:
        st = RichTypeProtoToMLValue<decltype(data.map_int64_to_int64().v()), MapInt64ToInt64>(data.map_int64_to_int64().v(), value);
        break;
      case proto::TraditionalMLData::kMapInt64ToFloat:
        st = RichTypeProtoToMLValue<decltype(data.map_int64_to_float().v()), MapInt64ToFloat>(data.map_int64_to_float().v(), value);
        break;
      case proto::TraditionalMLData::kMapInt64ToDouble:
        st = RichTypeProtoToMLValue<decltype(data.map_int64_to_double().v()), MapInt64ToDouble>(data.map_int64_to_double().v(), value);
        break;
      case proto::TraditionalMLData::kTensor: {
        OrtValue* temp_value;
        std::string s = data.tensor().SerializeAsString();
        size_t len;
        ORT_THROW_ON_ERROR(OrtGetTensorMemSizeInBytesFromTensorProto(s.data(), (int)s.size(), 0, &len));
        char* p = len == 0 ? nullptr : (char*)b.AllocMemory(len);
        OrtCallback* d;
        ORT_THROW_ON_ERROR(OrtTensorProtoToOrtValue(s.data(), (int)s.size(), nullptr, p, len, &temp_value, &d));
        if (d != nullptr) {
          b.AddDeleter(d);
        }
        gvalue.reset(temp_value);
        is_tensor = true;
      } break;
      default:
        st = Status(ONNXRUNTIME, NOT_IMPLEMENTED, "unknown data type inside TraditionalMLData");
    }
    if (!st.IsOK()) break;
    if (!data.debug_info().empty()) {
      oss << ":" << data.debug_info();
    }
    std::string value_name = data.name();
    if (value_name.empty())
      value_name = value_info[name_data_map.size()].name();

    auto pv = name_data_map.insert(std::make_pair(value_name, is_tensor ? gvalue.release() : (OrtValue*)new MLValue(value)));
    if (!pv.second) {
      st = Status(ONNXRUNTIME, FAIL, "duplicated test data name");
      break;
    }
  }
  if (!st.IsOK()) return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "load the ", item_id, "-th item failed,", st.ErrorMessage());
  if (!clean_eof) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "parse input file failed, has extra unparsed data");
  }
  return Status::OK();
}

template <typename T>
static void RepeatedPtrFieldToVector(const ::google::protobuf::RepeatedPtrField<T>& input_value_info, std::vector<T>& out) {
  for (int i = 0; i != input_value_info.size(); ++i) {
    out.push_back(input_value_info[i]);
  }
}
}  // namespace

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
  std::basic_string<PATH_CHAR_TYPE> model_url_;
  std::vector<std::string> debuginfo_strings;
  onnxruntime::OrtMutex m_;
  std::vector<ONNX_NAMESPACE::ValueInfoProto> input_value_info_;
  std::vector<ONNX_NAMESPACE::ValueInfoProto> output_value_info_;
  std::unique_ptr<OrtAllocator> allocator_;

  std::vector<std::basic_string<PATH_CHAR_TYPE>> test_data_dirs_;
  Status loadModelFile(const PATH_CHAR_TYPE* model_url, ONNX_NAMESPACE::ModelProto** model_pb);

  std::string GetDatasetDebugInfoString(size_t dataset_id) override {
    std::lock_guard<OrtMutex> l(m_);
    if (dataset_id < debuginfo_strings.size()) {
      return debuginfo_strings[dataset_id];
    }
    // return empty string
    return std::string();
  }
  //If we cannot get input name from input_pbs, we'll use names like "data_0","data_1",... It's dirty hack
  // for https://github.com/onnx/onnx/issues/679
  ::onnxruntime::common::Status ConvertTestData(OrtSession* session,
                                                const std::vector<ONNX_NAMESPACE::TensorProto>& test_data_pbs,
                                                HeapBuffer& b, bool is_input,
                                                std::unordered_map<std::string, OrtValue*>& out);
  std::string node_name_;
  std::once_flag model_parsed_;
  std::once_flag config_parsed_;
  double per_sample_tolerance_;
  double relative_per_sample_tolerance_;
  bool post_processing_;
  Status ParseModel();
  Status ParseConfig();
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(OnnxTestCase);

 public:
  explicit OnnxTestCase(const std::string& test_case_name);
  Status GetPerSampleTolerance(double* value) override;
  Status GetRelativePerSampleTolerance(double* value) override;
  Status GetPostProcessing(bool* value) override;

  const ONNX_NAMESPACE::ValueInfoProto& GetOutputInfoFromModel(size_t i) const override {
    return output_value_info_[i];
  }

  size_t GetDataCount() const override {
    return test_data_dirs_.size();
  }
  Status GetNodeName(std::string* out) override {
    Status st = ParseModel();
    if (st.IsOK()) *out = node_name_;
    return st;
  }
#ifdef _WIN32
  Status SetModelPath(const wchar_t* path) override;
#else
  Status SetModelPath(const char* path) override;
#endif

  const PATH_CHAR_TYPE* GetModelUrl() const override { return model_url_.c_str(); }
  const std::string& GetTestCaseName() const override {
    return test_case_name_;
  }
  ::onnxruntime::common::Status LoadTestData(OrtSession* session, size_t id, HeapBuffer& b,
                                             std::unordered_map<std::string, OrtValue*>&, bool is_input) override;
};

Status OnnxTestCase::loadModelFile(const PATH_CHAR_TYPE* model_url, ONNX_NAMESPACE::ModelProto** model_pb) {
  int model_fd;
  ORT_RETURN_IF_ERROR(Env::Default().FileOpenRd(model_url, model_fd));
  google::protobuf::io::FileInputStream f(model_fd);
  f.SetCloseOnDelete(true);
  ONNX_NAMESPACE::ModelProto* ret = new ONNX_NAMESPACE::ModelProto();
  if (!ret->ParseFromZeroCopyStream(&f)) {
    return Status(ONNXRUNTIME, INVALID_PROTOBUF, "Failed to load model because protobuf parsing failed.");
  }
  *model_pb = ret;
  return Status::OK();
}
ITestCase* CreateOnnxTestCase(const std::string& test_case_name) { return new OnnxTestCase(test_case_name); }

Status OnnxTestCase::GetPerSampleTolerance(double* value) {
  Status st = ParseConfig();
  if (!st.IsOK())
    return ORT_MAKE_STATUS(ONNXRUNTIME, MODEL_LOADED, "parse test config failed:", st.ErrorMessage());

  *value = per_sample_tolerance_;
  return Status::OK();
}

Status OnnxTestCase::GetRelativePerSampleTolerance(double* value) {
  Status st = ParseConfig();
  if (!st.IsOK())
    return ORT_MAKE_STATUS(ONNXRUNTIME, MODEL_LOADED, "parse test config failed:", st.ErrorMessage());
  *value = relative_per_sample_tolerance_;
  return Status::OK();
}

Status OnnxTestCase::GetPostProcessing(bool* value) {
  Status st = ParseConfig();
  if (!st.IsOK()) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, MODEL_LOADED, "parse test config failed:", st.ErrorMessage());
  }
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

Status OnnxTestCase::ParseConfig() {
  std::call_once(config_parsed_, [this]() {
    std::basic_string<PATH_CHAR_TYPE> config_path =
        ReplaceFilename<std::basic_string<PATH_CHAR_TYPE>>(model_url_, ORT_TSTR("config.txt"));
    /* Note: protobuf-lite doesn't support reading protobuf files as text-format. Config.txt is exactly that.
       That's the reason I've to parse the file in a different way to read the configs. Currently
       this affects 2 tests - fp16_tiny_yolov2 and fp16_inception_v1. It's not clear why we've to use protobuf
       to represent simple config files that have only key-value pairs.
     */
    std::map<std::string, std::string> fc;
    if (read_config_file(config_path, fc)) {
      if (fc.count("per_sample_tolerance")) {
        per_sample_tolerance_ = stod(fc["per_sample_tolerance"]);
      }
      if (fc.count("relative_per_sample_tolerance")) {
        relative_per_sample_tolerance_ = stod(fc["relative_per_sample_tolerance"]);
      }
      if (fc.count("post_processing")) {
        post_processing_ = fc["post_processing"] == "true" ? true : false;
      }
      return;
    } else {
      per_sample_tolerance_ = 1e-3;
      relative_per_sample_tolerance_ = 1e-5;
#ifdef USE_CUDA
      relative_per_sample_tolerance_ = 0.017;  // to resolve random MNIST test failure
#endif
      post_processing_ = false;
      return;
    }
  });
  return Status::OK();
}

Status OnnxTestCase::ParseModel() {
  Status st = Status::OK();
  std::call_once(model_parsed_, [this, &st]() {
    //parse model
    ONNX_NAMESPACE::ModelProto* model_pb = nullptr;
    st = loadModelFile(model_url_.c_str(), &model_pb);
    if (!st.IsOK() || model_pb == nullptr) return;
    const ONNX_NAMESPACE::GraphProto& graph = model_pb->graph();
    if (graph.node().size() == 1) {
      node_name_ = graph.node()[0].op_type();
    }
    RepeatedPtrFieldToVector(graph.input(), input_value_info_);
    RepeatedPtrFieldToVector(graph.output(), output_value_info_);
    st = Status::OK();
    delete model_pb;
  });
  return st;
}

Status OnnxTestCase::SetModelPath(const PATH_CHAR_TYPE* m) {
  model_url_ = m;
  std::basic_string<PATH_CHAR_TYPE> test_case_dir;
  ORT_RETURN_IF_ERROR(GetDirNameFromFilePath(model_url_, test_case_dir));
  LoopDir(test_case_dir, [&test_case_dir, this](const PATH_CHAR_TYPE* filename, OrtFileType f_type) -> bool {
    if (filename[0] == '.') return true;
    if (f_type == OrtFileType::TYPE_DIR) {
      std::basic_string<PATH_CHAR_TYPE> p = ConcatPathComponent<PATH_CHAR_TYPE>(test_case_dir, filename);
      test_data_dirs_.push_back(p);
      debuginfo_strings.push_back(ToMBString(p));
    }
    return true;
  });
  return Status::OK();
}

//load tensors from disk
template <typename PATH_STRING_TYPE>
static Status LoadTensors(const std::vector<PATH_STRING_TYPE>& pb_files,
                          std::vector<ONNX_NAMESPACE::TensorProto>* input_pbs) {
  for (size_t i = 0; i != pb_files.size(); ++i) {
    int tensor_fd;
    auto st = Env::Default().FileOpenRd(pb_files.at(i), tensor_fd);
    ORT_RETURN_IF_ERROR(st);
    google::protobuf::io::FileInputStream f(tensor_fd);
    f.SetCloseOnDelete(true);
    ONNX_NAMESPACE::TensorProto tensor;
    if (!tensor.ParseFromZeroCopyStream(&f)) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "parse file '", ToMBString(pb_files.at(i)), "' failed");
    }
    input_pbs->emplace_back(tensor);
  }
  return Status::OK();
}

Status OnnxTestCase::LoadTestData(OrtSession* session, size_t id, HeapBuffer& b,
                                  std::unordered_map<std::string, OrtValue*>& name_data_map, bool is_input) {
  if (id >= test_data_dirs_.size())
    return Status(ONNXRUNTIME, INVALID_ARGUMENT, "out of bound");

  Status st = ParseModel();
  if (!st.IsOK())
    return ORT_MAKE_STATUS(ONNXRUNTIME, MODEL_LOADED, "parse model failed:", st.ErrorMessage());

  PATH_STRING_TYPE test_data_pb = ConcatPathComponent<PATH_CHAR_TYPE>(
      test_data_dirs_[id], (is_input ? ORT_TSTR("inputs.pb") : ORT_TSTR("outputs.pb")));
  int test_data_pb_fd;
  st = Env::Default().FileOpenRd(test_data_pb, test_data_pb_fd);
  if (st.IsOK()) {  //has an all-in-one input file
    std::ostringstream oss;
    {
      std::lock_guard<OrtMutex> l(m_);
      oss << debuginfo_strings[id];
    }
    st = LoopDataFile(test_data_pb_fd, is_input ? input_value_info_ : output_value_info_, name_data_map, b, oss);
    {
      std::lock_guard<OrtMutex> l(m_);
      debuginfo_strings[id] = oss.str();
    }
    if (!st.IsOK())
      return ORT_MAKE_STATUS(ONNXRUNTIME, MODEL_LOADED, "parse data file \"", ToMBString(test_data_pb),
                             "\" failed:", st.ErrorMessage());
    return Status::OK();
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
  ORT_RETURN_IF_ERROR(SortTensorFileNames(test_data_pb_files));

  std::vector<ONNX_NAMESPACE::TensorProto> test_data_pbs;
  ORT_RETURN_IF_ERROR(LoadTensors(test_data_pb_files, &test_data_pbs));
  ORT_RETURN_IF_ERROR(ConvertTestData(session, test_data_pbs, b, is_input, name_data_map));
  return Status::OK();
}

Status OnnxTestCase::ConvertTestData(OrtSession* session, const std::vector<ONNX_NAMESPACE::TensorProto>& test_data_pbs,
                                     HeapBuffer& b, bool is_input, std::unordered_map<std::string, OrtValue*>& out) {
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
    size_t count;
    if (is_input) {
      ORT_THROW_ON_ERROR(OrtSessionGetInputCount(session, &count));
    } else {
      ORT_THROW_ON_ERROR(OrtSessionGetOutputCount(session, &count));
    }
    if (count != test_data_pbs.size())
      ORT_THROW("data count mismatch");
    for (size_t i = 0; i != count; ++i) {
      char* temp_name;
      if (is_input) {
        ORT_THROW_ON_ERROR(OrtSessionGetInputName(session, i, allocator_.get(), &temp_name));
      } else {
        ORT_THROW_ON_ERROR(OrtSessionGetOutputName(session, i, allocator_.get(), &temp_name));
      }
      var_names[i] = temp_name;
      allocator_->Free(allocator_.get(), temp_name);
    }
  }
  for (size_t input_index = 0; input_index != test_data_pbs.size(); ++input_index) {
    std::string name = var_names[input_index];
    const ONNX_NAMESPACE::TensorProto& input = test_data_pbs[input_index];
    std::string s = input.SerializeAsString();
    MLValue* v1;
    size_t len;
    ORT_THROW_ON_ERROR(OrtGetTensorMemSizeInBytesFromTensorProto(s.data(), (int)s.size(), 0, &len));
    char* p = len == 0 ? nullptr : (char*)b.AllocMemory(len);
    OrtCallback* d;
    ORT_THROW_ON_ERROR(OrtTensorProtoToOrtValue(s.data(), (int)s.size(), nullptr, p, len, (OrtValue**)&v1, &d));
    if (d != nullptr) b.AddDeleter(d);
    out.insert(std::make_pair(name, (OrtValue*)v1));
  }
  return Status::OK();
}

OnnxTestCase::OnnxTestCase(const std::string& test_case_name) : test_case_name_(test_case_name) {
  OrtAllocator* p;
  ORT_THROW_ON_ERROR(OrtCreateDefaultAllocator(&p));
  allocator_.reset(p);
}
