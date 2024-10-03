// Copyright (c) Microsoft Corporation. All rights reserved.
// Copyright (c) Intel Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/inlined_containers.h"
#include "core/common/status.h"
#include "core/platform/ort_mutex.h"

#include <emscripten.h>
#include <emscripten/val.h>

namespace onnxruntime {
namespace webnn {

struct OnnxTensorInfo {
  const int32_t data_type;  // Uses TensorProto::DataType.
  const std::vector<int64_t> shape;
};

struct OnnxTensorData {
  OnnxTensorInfo tensor_info;
  void* buffer{nullptr};
};

class Model {
  friend class ModelBuilder;

 public:
  ~Model();
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(Model);

  onnxruntime::common::Status Predict(const InlinedHashMap<std::string, OnnxTensorData>& inputs,
                                      const InlinedHashMap<std::string, OnnxTensorData>& outputs);

  // Mutex for exclusive lock to this model object.
  OrtMutex& GetMutex() { return mutex_; }

  // Input and output names in the onnx model's order.
  const std::vector<std::string>& GetInputs() const { return inputs_; }
  void SetInputs(std::vector<std::string>&& inputs) { inputs_ = std::move(inputs); }

  const std::vector<std::string>& GetOutputs() const { return outputs_; }
  void SetOutputs(std::vector<std::string>&& outputs) { outputs_ = std::move(outputs); }

  const OnnxTensorInfo& GetInputOutputInfo(const std::string& name) const;

  // Set the mapping between input/output name and ORT kernel context
  // input/output index, at execution time.
  void SetInputMap(InlinedHashMap<std::string, size_t>&& input_map);
  void SetOutputMap(InlinedHashMap<std::string, size_t>&& output_map);

  // Get the ORT kernel context input/output index with given name.
  size_t GetMappedInputIdx(const std::string& name) const;
  size_t GetMappedOutputIdx(const std::string& name) const;

 private:
  onnxruntime::common::Status Dispatch(const InlinedHashMap<std::string, OnnxTensorData>& inputs,
                                       const InlinedHashMap<std::string, OnnxTensorData>& outputs);

  onnxruntime::common::Status Compute(const InlinedHashMap<std::string, OnnxTensorData>& inputs,
                                      const InlinedHashMap<std::string, OnnxTensorData>& outputs);

  emscripten::val wnn_context_ = emscripten::val::object();
  emscripten::val wnn_graph_ = emscripten::val::object();
  const logging::Logger& logger_;

  emscripten::val wnn_inputs_ = emscripten::val::object();
  emscripten::val wnn_outputs_ = emscripten::val::object();

  std::vector<std::string> inputs_;
  std::vector<std::string> outputs_;

  InlinedHashMap<std::string, OnnxTensorInfo> input_output_info_;

  InlinedHashMap<std::string, size_t> input_map_;
  InlinedHashMap<std::string, size_t> output_map_;

  OrtMutex mutex_;

  bool use_dispatch_;

  Model(const emscripten::val& context, const emscripten::val& path, const logging::Logger& logger, bool use_dispatch);

  void SetInputOutputInfo(InlinedHashMap<std::string, OnnxTensorInfo>&& input_output_info) {
    input_output_info_ = std::move(input_output_info);
  }

  void AllocateInputOutputBuffers();
};

}  // namespace webnn
}  // namespace onnxruntime
