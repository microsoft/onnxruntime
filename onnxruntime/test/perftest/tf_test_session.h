// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include <core/session/onnxruntime_cxx_api.h>
#include <core/platform/env.h>
#include "test_configuration.h"
#include "tensorflow/c/c_api.h"
#include "test_session.h"

namespace onnxruntime {
namespace perftest {
class TensorflowTestSession : public TestSession {
 private:
  OrtCallback model_deleter;
  std::vector<TF_Output> feed_;
  std::vector<TF_Output> fetches_;
  TF_Session* sess_;
  TF_Graph* tf_graph_;
  // This function is for both graph inputs and outputs
  static TF_Output GetOutputFromGraph(const char* tensor_name, TF_Graph* tf_graph) {
    TF_Output ret;
    const char* start = tensor_name;
    const char* sep = strchr(start, ':');
    if (sep == nullptr) {
      ORT_THROW("invalid name:", tensor_name);
    }
    size_t name_len = sep - start;
    std::string name(name_len, '\0');
    memcpy(const_cast<char*>(name.data()), start, name_len);
    ret.oper = TF_GraphOperationByName(tf_graph, name.c_str());
    if (ret.oper == nullptr) ORT_THROW("input name: \"", name, "\" can not be find in the graph");
    start = sep + 1;
    char* end;
    ret.index = static_cast<int>(strtol(start, &end, 10));
    if (start == end) {
      ORT_THROW("invalid name:", tensor_name);
    }
    return ret;
  }

  TF_Tensor* AllocateTFTensor(const OrtTensorTypeAndShapeInfo* shape, size_t& buffer_length) const {
    size_t dim_count = OrtGetNumOfDimensions(shape);
    std::vector<int64_t> dims(dim_count);
    OrtGetDimensions(shape, dims.data(), dim_count);
    int64_t ele_count = OrtGetTensorShapeElementCount(shape);
    TF_DataType d;
    switch (OrtGetTensorElementType(shape)) {
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:  // maps to c type float
        buffer_length = ele_count * sizeof(float);
        d = TF_FLOAT;
        break;
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:  // maps to c type uint8_t
        buffer_length = ele_count * sizeof(uint8_t);
        d = TF_UINT8;
        break;
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:  // maps to c type int8_t
        buffer_length = ele_count * sizeof(int8_t);
        d = TF_INT8;
        break;
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16:  // maps to c type uint16_t
        buffer_length = ele_count * sizeof(uint16_t);
        d = TF_UINT16;
        break;
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16:  // maps to c type int16_t
        buffer_length = ele_count * sizeof(int16_t);
        d = TF_INT16;
        break;
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:  // maps to c type int32_t
        buffer_length = ele_count * sizeof(int32_t);
        d = TF_INT32;
        break;
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:  // maps to c type int64_t
        buffer_length = ele_count * sizeof(int64_t);
        d = TF_INT64;
        break;
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL:
        buffer_length = ele_count * sizeof(bool);
        d = TF_BOOL;
        break;
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:  // maps to c type double
        buffer_length = ele_count * sizeof(double);
        d = TF_DOUBLE;
        break;
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32:  // maps to c type uint32_t
        buffer_length = ele_count * sizeof(uint32_t);
        d = TF_UINT32;
        break;
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64:  // maps to c type uint64_t
        buffer_length = ele_count * sizeof(uint64_t);
        d = TF_UINT64;
        break;
      default:
        ORT_NOT_IMPLEMENTED("unexpected input data type");
    }
    return TF_AllocateTensor(d, dims.data(), static_cast<int>(dims.size()), buffer_length);
  }

 public:
  TensorflowTestSession(const PerformanceTestConfig& performance_test_config, const TestModelInfo* m) {
    TF_Status* s = TF_NewStatus();
    tf_graph_ = TF_NewGraph();
    TF_ImportGraphDefOptions* opts = TF_NewImportGraphDefOptions();
    TF_ImportGraphDefOptionsSetPrefix(opts, "");
    TF_Buffer* graph_def = TF_NewBuffer();
    void* model_data;
    auto st = Env::Default().ReadFileAsString(performance_test_config.model_info.model_file_path.c_str(), 0, model_data,
                                              graph_def->length, model_deleter);
    if (!st.IsOK())
      ORT_THROW("read file ", performance_test_config.model_info.model_file_path, " failed:", st.ErrorMessage());
    graph_def->data = model_data;
    TF_GraphImportGraphDef(tf_graph_, graph_def, opts, s);
    if (TF_GetCode(s) != TF_OK) ORT_THROW("load TF model failed:", TF_Message(s));
    TF_SessionOptions* session_opts = TF_NewSessionOptions();
    sess_ = TF_NewSession(tf_graph_, session_opts, s);
    if (TF_GetCode(s) != TF_OK) ORT_THROW("load TF model failed:", TF_Message(s));
    feed_.resize(static_cast<size_t>(m->GetInputCount()));
    for (size_t i = 0; i != feed_.size(); ++i) {
      feed_[i] = GetOutputFromGraph(m->GetInputName(i).c_str(), tf_graph_);
    }
    fetches_.resize(static_cast<size_t>(m->GetOutputCount()));
    for (size_t i = 0; i != fetches_.size(); ++i) {
      fetches_[i] = GetOutputFromGraph(m->GetOutputName(i).c_str(), tf_graph_);
    }
  }
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(TensorflowTestSession);
  std::chrono::duration<double> Run(const OrtValue* const* input) override {
    size_t input_len = feed_.size();
    std::vector<TF_Tensor*> feed_tensors(input_len);
    for (size_t i = 0; i != input_len; ++i) {
      void* input_buffer = nullptr;
      ORT_THROW_ON_ERROR(OrtGetTensorMutableData(const_cast<OrtValue*>(input[i]), &input_buffer));
      assert(input_buffer != nullptr);
      OrtTensorTypeAndShapeInfo* shape;
      ORT_THROW_ON_ERROR(OrtGetTensorShapeAndType(input[i], &shape));
      size_t buffer_length = 0;
      TF_Tensor* t = AllocateTFTensor(shape, buffer_length);
      assert(t != nullptr);
      feed_tensors[i] = t;
      assert(TF_TensorByteSize(t) == buffer_length);
      memcpy(TF_TensorData(t), input_buffer, buffer_length);
    }
    std::vector<TF_Tensor*> output_tensors(fetches_.size());
    TF_Status* s = TF_NewStatus();
    auto start = std::chrono::high_resolution_clock::now();
    TF_SessionRun(sess_, nullptr, feed_.data(), feed_tensors.data(), static_cast<int>(feed_.size()), fetches_.data(),
                  output_tensors.data(), static_cast<int>(fetches_.size()), nullptr, 0, nullptr, s);
    auto end = std::chrono::high_resolution_clock::now();
    if (TF_GetCode(s) != TF_OK) ORT_THROW("run TF model failed:", TF_Message(s));
    TF_DeleteStatus(s);
    return end - start;
  }

  ~TensorflowTestSession() override {
    if (model_deleter.f != nullptr) {
      model_deleter.f(model_deleter.param);
    }
    TF_Status* s = TF_NewStatus();
    TF_DeleteSession(sess_, s);
    TF_DeleteStatus(s);
  }
};

}  // namespace perftest
}  // namespace onnxruntime