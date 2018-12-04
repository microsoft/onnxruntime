//
// Copyright (c) Microsoft Corporation.  All rights reserved.
// Licensed under the MIT License.
//

#pragma once

#include "TestDataReader.h"

#include <algorithm>
#include <codecvt>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "core/graph/onnx_protobuf.h"
#include "core/common/logging/sinks/clog_sink.h"
#include "core/common/logging/logging.h"
#include "core/framework/environment.h"
#include "core/framework/data_types.h"
#include "core/session/inference_session.h"
#include "core/providers/cpu/cpu_execution_provider.h"

#include "test/compare_mlvalue.h"

#if !defined(_MSC_VER)
#include <sys/stat.h>

#define ERROR_FILE_NOT_FOUND 2L
#define ERROR_BAD_FORMAT 11L

#define O_BINARY 0x0000
#endif

class WinMLRuntime {
 public:
  WinMLRuntime() {
    using namespace onnxruntime;
    using namespace ::onnxruntime::logging;

    static std::unique_ptr<::onnxruntime::Environment> onnxruntime_env = nullptr;
    static std::once_flag env_flag;
    std::call_once(env_flag, []() { ::onnxruntime::Environment::Create(onnxruntime_env); });

    static LoggingManager& s_default_logging_manager = DefaultLoggingManager();
    SessionOptions so;
    so.session_logid = "WinMLRuntime";

    inference_session_ = std::make_unique<::onnxruntime::InferenceSession>(so, &s_default_logging_manager);
  }

  ::onnxruntime::common::Status LoadModel(const std::wstring& model_path) {
    ::onnxruntime::common::Status result = inference_session_->Load(wstr2str(model_path));
    if (result.IsOK())
      result = inference_session_->Initialize();

    return result;
  }

  void FillInBatchSize(std::vector<int64_t>& shape, int input_size, int feature_size) {
    if ((input_size % feature_size != 0) && (feature_size != -1))
      throw DataValidationException("Input count is not a multiple of dimension.");

    int batch_size = feature_size == -1 ? 1 : input_size / feature_size;
    shape.insert(shape.begin(), batch_size);
  }

  ::onnxruntime::MLValue ReadTensorStrings(::onnxruntime::AllocatorPtr alloc, TestDataReader& inputs_reader,
                                           int feature_size, std::vector<int64_t> dims, bool variable_batch_size) {
    using namespace onnxruntime;

    auto vec = inputs_reader.GetSampleStrings(feature_size, variable_batch_size);

    std::vector<std::string> vec2;
    for (int i = 0; i < vec.size(); i++) {
      std::string str(vec[i].begin(), vec[i].end());
      vec2.push_back(str);
    }

    if (variable_batch_size)
      FillInBatchSize(dims, gsl::narrow_cast<int>(vec.size()), feature_size);

    TensorShape shape(dims);
    auto element_type = DataTypeImpl::GetType<std::string>();

    void* buffer = alloc->Alloc(sizeof(std::string) * shape.Size());
    std::unique_ptr<Tensor> p_tensor = std::make_unique<Tensor>(element_type,
                                                                shape,
                                                                buffer,
                                                                alloc->Info(), alloc);

    std::string* p = p_tensor->template MutableData<std::string>();
    for (int i = 0; i < vec.size(); i++) {
      p[i] = std::string(vec[i].begin(), vec[i].end());
    }

    ::onnxruntime::MLValue result;
    result.Init(p_tensor.release(),
                DataTypeImpl::GetType<Tensor>(),
                DataTypeImpl::GetType<Tensor>()->GetDeleteFunc());

    return result;
  }

  template <typename T>
  ::onnxruntime::MLValue ReadTensor(::onnxruntime::AllocatorPtr alloc, TestDataReader& inputs_reader,
                                    int feature_size, std::vector<int64_t> dims, bool variable_batch_size) {
    using namespace onnxruntime;

    auto vec = inputs_reader.GetSample<T>(feature_size, variable_batch_size);

    if (variable_batch_size)
      FillInBatchSize(dims, gsl::narrow_cast<int>(vec.size()), feature_size);

    ::onnxruntime::TensorShape shape(dims);
    auto location = alloc->Info();
    auto element_type = ::onnxruntime::DataTypeImpl::GetType<T>();
    void* buffer = alloc->Alloc(element_type->Size() * shape.Size());

    if (vec.size() > 0) {
      memcpy(buffer, &vec[0], element_type->Size() * shape.Size());
    }

    std::unique_ptr<Tensor> p_tensor = std::make_unique<::onnxruntime::Tensor>(element_type,
                                                                               shape,
                                                                               buffer,
                                                                               location,
                                                                               alloc);

    ::onnxruntime::MLValue result;
    result.Init(p_tensor.release(),
                ::onnxruntime::DataTypeImpl::GetType<::onnxruntime::Tensor>(),
                ::onnxruntime::DataTypeImpl::GetType<::onnxruntime::Tensor>()->GetDeleteFunc());

    return result;
  }

  template <typename V>
  onnxruntime::MLValue ReadTensorForMapStringToScalar(TestDataReader& inputs_reader) {
    auto vec = inputs_reader.GetSample<V>(-1);

    auto data = std::make_unique<std::map<std::string, V>>();
    for (int i = 0; i < vec.size(); i++) {
      // keys start at "1" so convert index to string key based on that
      data->insert({std::to_string(i + 1), vec[i]});
    }

    ::onnxruntime::MLValue result;
    result.Init(data.release(),
                ::onnxruntime::DataTypeImpl::GetType<std::map<std::string, V>>(),
                ::onnxruntime::DataTypeImpl::GetType<std::map<std::string, V>>()->GetDeleteFunc());

    return result;
  }

  int Run(TestDataReader& inputs_reader) {
    using namespace onnxruntime;
    int hr = 0;
    std::wstring_convert<std::codecvt_utf8<wchar_t>> converter;

    // Create CPU input tensors
    ::onnxruntime::NameMLValMap feed;
    inputs_reader.BufferNextSample();
    if (inputs_reader.Eof())
      return 0;

    bool variable_batch_size = false;
    auto inputs_pairs = inference_session_->GetModelInputs();
    if (!inputs_pairs.first.IsOK()) {
      auto error = inputs_pairs.first.ErrorMessage();
      return inputs_pairs.first.Code();
    }

    auto& inputs = *(inputs_pairs.second);
    for (size_t index = 0, end = inputs.size(); index < end; ++index) {
      MLValue mlvalue;
      const onnxruntime::NodeArg& input = *(inputs[index]);
      const ONNX_NAMESPACE::TensorShapeProto* input_shape = input.Shape();
      if (input.Name().empty())
        continue;

      auto type = input.Type();

      std::vector<int64_t> shape;
      int feature_size = -1;

      //Previous graph input was variable length that consumed entire input line_ so fetch new input line_.
      if (variable_batch_size)
        inputs_reader.BufferNextSample();

      //This graph input may or may not be variable length.
      //REVIEW mzs: this can cause issues if we had variable-input followed by fixed input followed by variable-input where
      //fixed-input consumed all of the input line_. *Ideally each graph input should be on its own line_*.
      variable_batch_size = false;

      //If the shape is not available then read everything into the input tensor.
      //feature_size = -1 indicates this condition.
      if (input_shape) {
        feature_size = 0;
        auto dims = input_shape->dim();
        for (auto dim : dims) {
          if (dim.has_dim_param())
            variable_batch_size = true;
          else {
            auto dim_value = dim.dim_value();
            shape.push_back(dim_value);
            feature_size = gsl::narrow_cast<int>(feature_size ? feature_size * dim_value : dim_value);
          }
        }
      }

      //REVIEW mzs: Here an assumption is made that all the input columns are for the map.
      //The supported map types in onnxruntime seen so far are <string, string> or <string, int64>.
      if (*type == "map(string,tensor(int64))") {
        // check if really map(string, int64), which is all we currently support
        bool is_map_value_scalar = input.TypeAsProto()->map_type().value_type().tensor_type().shape().dim_size() == 0;

        if (is_map_value_scalar) {
          mlvalue = ReadTensorForMapStringToScalar<int64_t>(inputs_reader);
          feed.insert(std::make_pair(input.Name(), mlvalue));
        } else {
          throw DataValidationException("Unsupported input type: " + std::string(*type));
        }
      } else if (*type == "map(string,tensor(float))" || *type == "map(string,tensor(double))") {
        // check if really map(string, float) or map(string, double), which is all we currently support
        bool is_map_value_scalar = input.TypeAsProto()->map_type().value_type().tensor_type().shape().dim_size() == 0;

        if (is_map_value_scalar) {
          mlvalue = ReadTensorForMapStringToScalar<float>(inputs_reader);
          feed.insert({input.Name(), mlvalue});
        } else {
          throw DataValidationException("Unsupported input type: " + std::string(*type));
        }
      } else {
        if (*type == "tensor(double)" || *type == "tensor(float)") {
          // If double is used in the following statement, following error occurs.
          // Tensor type mismatch, caller expects elements to be float while tensor contains double Error from operator
          mlvalue = ReadTensor<float>(TestCPUExecutionProvider().GetAllocator(0, ONNXRuntimeMemTypeDefault), inputs_reader, feature_size, shape, variable_batch_size);
        } else if (*type == "tensor(int64)")
          mlvalue = ReadTensor<int64_t>(TestCPUExecutionProvider().GetAllocator(0, ONNXRuntimeMemTypeDefault), inputs_reader, feature_size, shape, variable_batch_size);
        else if (*type == "tensor(string)")
          mlvalue = ReadTensorStrings(TestCPUExecutionProvider().GetAllocator(0, ONNXRuntimeMemTypeDefault), inputs_reader, feature_size, shape, variable_batch_size);
        else
          throw DataValidationException("Unsupported input type: " + std::string(*type));

        feed.insert(std::make_pair(input.Name(), mlvalue));
      }
    }

    // Create output feed
    std::vector<std::string> output_names;
    for (auto const& outp : *(inference_session_->GetModelOutputs().second)) {
      output_names.push_back(outp->Name());
    }

    std::cout.precision(12);
    std::string separator = "";
    // Invoke the net
    std::vector<::onnxruntime::MLValue> outputMLValue;
    RunOptions run_options;
    ::onnxruntime::common::Status result = inference_session_->Run(run_options, feed, output_names, &outputMLValue);
    if (result.IsOK()) {
      auto outputMeta = inference_session_->GetModelOutputs().second;
      // Peel the data off the CPU
      for (unsigned int i = 0; i < output_names.size(); i++) {
        ::onnxruntime::MLValue& output = outputMLValue[i];
        const ::onnxruntime::Tensor* ctensor = nullptr;

        if (output.IsTensor()) {
          ctensor = &output.Get<Tensor>();

          ONNX_NAMESPACE::ValueInfoProto expected_output_info = (*outputMeta)[i]->ToProto();
          std::pair<COMPARE_RESULT, std::string> ret = VerifyValueInfo(expected_output_info, (ONNXValue*)&output);
          COMPARE_RESULT compare_result = ret.first;
          compare_result = ret.first;
          if (compare_result != COMPARE_RESULT::SUCCESS) {
            switch (compare_result) {
              case COMPARE_RESULT::NOT_SUPPORT:
                throw std::runtime_error("Unsupported output type in onnxruntime model: " + std::string((*outputMeta)[i]->Name()));
                break;
              case COMPARE_RESULT::SHAPE_MISMATCH:
                throw std::runtime_error("Output shape mismatch in onnxruntime model: " + std::string((*outputMeta)[i]->Name()));
                break;
              case COMPARE_RESULT::TYPE_MISMATCH:
                throw std::runtime_error("Output type mismatch in onnxruntime model: " + std::string((*outputMeta)[i]->Name()));
                break;
              default:
                throw std::runtime_error("Unknown error in onnxruntime model: " + std::string((*outputMeta)[i]->Name()));
            }
          }

          //REVIEW mzs: Map output types are not tested because I couldn't find any tests for that.
          if (ctensor->DataType() == ::onnxruntime::DataTypeImpl::GetType<std::map<int64_t, float>>()) {
            const std::map<int64_t, float>* ci = &output.Get<std::map<int64_t, float>>();
            for (const auto& p : *ci) {
              std::cout << separator << p.second;
              separator = ",";
            }
          } else if (ctensor->DataType() == ::onnxruntime::DataTypeImpl::GetType<std::map<std::string, float>>()) {
            const std::map<std::string, float>* ci = &output.Get<std::map<std::string, float>>();
            for (const auto& p : *ci) {
              std::cout << separator << p.second;
              separator = ",";
            }
          } else if (ctensor->DataType() == ::onnxruntime::DataTypeImpl::GetType<float>()) {
            const float* cdata = ctensor->template Data<float>();
            for (int ci = 0; ci < ctensor->Shape().Size(); ci++) {
              std::cout << separator << cdata[ci];
              separator = ",";
            }
          } else if (ctensor->DataType() == ::onnxruntime::DataTypeImpl::GetType<int64_t>()) {
            const int64_t* cdata = ctensor->template Data<int64_t>();
            for (int ci = 0; ci < ctensor->Shape().Size(); ci++) {
              std::cout << separator << cdata[ci];
              separator = ",";
            }
          } else if (ctensor->DataType() == ::onnxruntime::DataTypeImpl::GetType<std::string>()) {
            const std::string* cdata = ctensor->template Data<std::string>();
            for (int ci = 0; ci < ctensor->Shape().Size(); ci++) {
              std::cout << separator << cdata[ci];
              separator = ",";
            }
          } else {
            throw DataValidationException("Unsupported output type in onnxruntime model: " + std::string((*outputMeta)[i]->Name()));
          }
        } else if (output.Type() == ::onnxruntime::DataTypeImpl::GetType<::onnxruntime::VectorMapStringToFloat>()) {
          auto& cdata = output.Get<::onnxruntime::VectorMapStringToFloat>();
          for (int ci = 0; ci < cdata.size(); ci++) {
            for (const auto& p : cdata[ci]) {
              std::cout << separator << p.second;
              separator = ",";
            }
          }
        } else if (output.Type() == ::onnxruntime::DataTypeImpl::GetType<::onnxruntime::VectorMapInt64ToFloat>()) {
          auto& cdata = output.Get<::onnxruntime::VectorMapInt64ToFloat>();
          for (int ci = 0; ci < cdata.size(); ci++) {
            for (const auto& p : cdata[ci]) {
              std::cout << separator << p.second;
              separator = ",";
            }
          }
        }
      }

      std::cout << std::endl;
    } else {
      std::cerr << result.ErrorMessage() << std::endl;
      hr = result.Code();
    }

    return hr;
  }

 private:
  std::unique_ptr<::onnxruntime::InferenceSession> inference_session_;

  static ::onnxruntime::logging::LoggingManager& DefaultLoggingManager() {
    using namespace onnxruntime;
    std::string default_logger_id{"Default"};

    static logging::LoggingManager default_logging_manager{
        std::unique_ptr<logging::ISink>{new ::onnxruntime::logging::CLogSink{}},
        logging::Severity::kWARNING, false,
        logging::LoggingManager::InstanceType::Default,
        &default_logger_id};

    return default_logging_manager;
  }

  static ::onnxruntime::IExecutionProvider& TestCPUExecutionProvider() {
    static ::onnxruntime::CPUExecutionProviderInfo info;
    static ::onnxruntime::CPUExecutionProvider cpu_provider(info);
    return cpu_provider;
  }
};
