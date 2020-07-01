// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include <core/session/onnxruntime_cxx_api.h>
#include <core/platform/env.h>
#include "test_configuration.h"
#include "tensorflow/c/c_api.h"
#include "test_session.h"
extern const OrtApi* g_ort;

namespace onnxruntime {
namespace perftest {
class TensorflowTestSession : public TestSession {
 private:
  std::mt19937 rand_engine_;
  std::uniform_int_distribution<int> dist_;
  std::vector<char> model_data_;
  std::vector<TF_Output> feed_;
  std::vector<TF_Output> fetches_;
  std::vector<std::vector<TF_Tensor*>> feed_tensors_;
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

 public:
  TensorflowTestSession(std::random_device& rd, const PerformanceTestConfig& performance_test_config,
                        const TestModelInfo* m)
      : rand_engine_(rd()) {
    const auto& model_file_path = performance_test_config.model_info.model_file_path;
    size_t model_file_length;
    auto status = Env::Default().GetFileLength(model_file_path.c_str(), model_file_length);
    if (!status.IsOK()) ORT_THROW(status.ErrorMessage());
    model_data_.resize(model_file_length);
    auto model_data_span = gsl::make_span(model_data_);
    status = Env::Default().ReadFileIntoBuffer(model_file_path.c_str(), 0, model_file_length, model_data_span);
    if (!status.IsOK()) {
      ORT_THROW("read file ", model_file_path, " failed: ", status.ErrorMessage());
    }
    // TODO some of this doesn't look exception safe...
    TF_Status* s = TF_NewStatus();
    tf_graph_ = TF_NewGraph();
    TF_ImportGraphDefOptions* opts = TF_NewImportGraphDefOptions();
    TF_ImportGraphDefOptionsSetPrefix(opts, "");
    TF_Buffer* graph_def = TF_NewBuffer();
    graph_def->length = model_file_length;
    graph_def->data = model_data_.data();
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

  bool isDimMatches(const std::vector<int64_t>& dims1, const std::vector<int64_t>& dims2) {
    if (dims1.size() != dims2.size()) return false;
    size_t len = dims1.size();
    for (size_t i = 0; i != len; ++i) {
      if (dims1[i] > 0 && dims2[i] > 0 && dims1[i] != dims2[i]) return false;
    }
    return true;
  }

  /**
   * convert input from CHW format to HWC format
   * \param input A single image. This float array has length of 3*h*w
   * \param h image height
   * \param w image width
   * \param output A float array. should be freed by caller after use
   */
  template <typename T>
  static void chw_to_hwc(const T* input, int64_t h, int64_t w, T* output_data) {
    int64_t stride = h * w;
    for (int c = 0; c != 3; ++c) {
      int64_t t = c * stride;
      for (int64_t i = 0; i != stride; ++i) {
        output_data[i * 3 + c] = input[t + i];
      }
    }
  }

  void PreLoadTestData(size_t test_data_id, size_t input_id, OrtValue* value) override {
    if (feed_tensors_.size() < test_data_id + 1) {
      feed_tensors_.resize(test_data_id + 1);
    }
    if (feed_tensors_.at(test_data_id).size() < input_id + 1) {
      feed_tensors_.at(test_data_id).resize(input_id + 1);
    }

    TF_Status* s = TF_NewStatus();
    void* input_buffer = nullptr;
    Ort::ThrowOnError(g_ort->GetTensorMutableData(value, &input_buffer));
    assert(input_buffer != nullptr);
    OrtTensorTypeAndShapeInfo* shape = nullptr;
    Ort::ThrowOnError(g_ort->GetTensorTypeAndShape(value, &shape));
    size_t buffer_length = 0;
    std::vector<int64_t> dims;
    size_t dim_count;
    Ort::ThrowOnError(g_ort->GetDimensionsCount(shape, &dim_count));
    dims.resize(dim_count);
    Ort::ThrowOnError(g_ort->GetDimensions(shape, dims.data(), dim_count));
    size_t ele_count;
    Ort::ThrowOnError(g_ort->GetTensorShapeElementCount(shape, &ele_count));
    TF_DataType tf_datatype;
    ONNXTensorElementDataType element_type;
    Ort::ThrowOnError(g_ort->GetTensorElementType(shape, &element_type));
    switch (element_type) {
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:  // maps to c type float
        buffer_length = ele_count * sizeof(float);
        tf_datatype = TF_FLOAT;
        break;
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:  // maps to c type uint8_t
        buffer_length = ele_count * sizeof(uint8_t);
        tf_datatype = TF_UINT8;
        break;
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:  // maps to c type int8_t
        buffer_length = ele_count * sizeof(int8_t);
        tf_datatype = TF_INT8;
        break;
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16:  // maps to c type uint16_t
        buffer_length = ele_count * sizeof(uint16_t);
        tf_datatype = TF_UINT16;
        break;
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16:  // maps to c type int16_t
        buffer_length = ele_count * sizeof(int16_t);
        tf_datatype = TF_INT16;
        break;
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:  // maps to c type int32_t
        buffer_length = ele_count * sizeof(int32_t);
        tf_datatype = TF_INT32;
        break;
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:  // maps to c type int64_t
        buffer_length = ele_count * sizeof(int64_t);
        tf_datatype = TF_INT64;
        break;
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL:
        buffer_length = ele_count * sizeof(bool);
        tf_datatype = TF_BOOL;
        break;
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:  // maps to c type double
        buffer_length = ele_count * sizeof(double);
        tf_datatype = TF_DOUBLE;
        break;
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32:  // maps to c type uint32_t
        buffer_length = ele_count * sizeof(uint32_t);
        tf_datatype = TF_UINT32;
        break;
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64:  // maps to c type uint64_t
        buffer_length = ele_count * sizeof(uint64_t);
        tf_datatype = TF_UINT64;
        break;
      default:
        ORT_NOT_IMPLEMENTED("unexpected input data type");
    }
    TF_Tensor* t = nullptr;
    int tf_dims_count = TF_GraphGetTensorNumDims(tf_graph_, feed_[input_id], s);
    if (TF_GetCode(s) != TF_OK || tf_dims_count < 0) ORT_THROW("run TF model failed:", TF_Message(s));
    std::vector<int64_t> tf_dims(static_cast<size_t>(tf_dims_count));
    TF_GraphGetTensorShape(tf_graph_, feed_[input_id], tf_dims.data(), tf_dims_count, s);
    if (TF_GetCode(s) != TF_OK || tf_dims_count < 0) ORT_THROW("run TF model failed:", TF_Message(s));
    if (!isDimMatches(dims, tf_dims)) {
      // detect if it's NCHW, if it is, switch it to NHWC
      // TODO: make this code more generic
      if (dims.size() == 4 && tf_dims.size() == 4 && dims[0] == 1 && dims[1] == 3 && dims[2] == dims[3] &&
          (tf_dims[0] == 1 || tf_dims[0] == -1) && tf_dims[3] == 3 && tf_dims[1] == tf_dims[2] && tf_dims[1] > 0) {
        tf_dims[0] = 1;
        t = TF_AllocateTensor(tf_datatype, tf_dims.data(), static_cast<int>(tf_dims.size()), buffer_length);
        chw_to_hwc<float>((const float*)input_buffer, tf_dims[1], tf_dims[2], (float*)TF_TensorData(t));
      } else
        ORT_THROW("dimension doesn't match");
    } else {
      t = TF_AllocateTensor(tf_datatype, dims.data(), static_cast<int>(dims.size()), buffer_length);
      memcpy(TF_TensorData(t), input_buffer, buffer_length);
    }
    assert(TF_TensorByteSize(t) == buffer_length);
    assert(t != nullptr);
    feed_tensors_[test_data_id][input_id] = t;
  }
  std::chrono::duration<double> Run() override {
    //Randomly pick one OrtValueArray from feed_tensors_. (NOT ThreadSafe)
    const std::uniform_int_distribution<int>::param_type p(0, static_cast<int>(feed_tensors_.size() - 1));
    const size_t id = static_cast<size_t>(dist_(rand_engine_, p));
    std::vector<TF_Tensor*>& feed_tensors = feed_tensors_.at(id);

    TF_Status* s = TF_NewStatus();
    std::vector<TF_Tensor*> output_tensors(fetches_.size());
    auto start = std::chrono::high_resolution_clock::now();
    TF_SessionRun(sess_, nullptr, feed_.data(), feed_tensors.data(), static_cast<int>(feed_.size()), fetches_.data(),
                  output_tensors.data(), static_cast<int>(fetches_.size()), nullptr, 0, nullptr, s);
    auto end = std::chrono::high_resolution_clock::now();
    if (TF_GetCode(s) != TF_OK) ORT_THROW("run TF model failed:", TF_Message(s));
    for (TF_Tensor* f : output_tensors) {
      TF_DeleteTensor(f);
    }
    TF_DeleteStatus(s);
    return end - start;
  }

  ~TensorflowTestSession() override {
    TF_Status* s = TF_NewStatus();
    TF_DeleteSession(sess_, s);
    TF_DeleteStatus(s);
  }
};

}  // namespace perftest
}  // namespace onnxruntime
