// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/status.h"
#include "core/platform/ort_mutex.h"

namespace onnxruntime {
namespace coreml {

class Execution;

struct OnnxTensorInfo {
  const int32_t data_type;  // Uses TensorProto::DataType
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

  onnxruntime::common::Status Predict(const std::unordered_map<std::string, OnnxTensorData>& inputs,
                                      const std::unordered_map<std::string, OnnxTensorData>& outputs);

  bool IsScalarOutput(const std::string& output_name) const;

  // Mutex for exclusive lock to this model object
  OrtMutex& GetMutex() { return mutex_; }

  // Input and output names in the onnx model's order
  const std::vector<std::string>& GetInputs() const { return inputs_; }
  void SetInputs(std::vector<std::string>&& inputs) { inputs_ = std::move(inputs); }

  const std::vector<std::string>& GetOutputs() const { return outputs_; }
  void SetOutputs(std::vector<std::string>&& outputs) { outputs_ = std::move(outputs); }

  const OnnxTensorInfo& GetInputOutputInfo(const std::string& name) const;

 private:
  std::unique_ptr<Execution> execution_;
  std::unordered_set<std::string> scalar_outputs_;

  std::vector<std::string> inputs_;
  std::vector<std::string> outputs_;

  std::unordered_map<std::string, OnnxTensorInfo> input_output_info_;

  OrtMutex mutex_;

  Model(const std::string& path, const logging::Logger& logger, uint32_t coreml_flags);
  onnxruntime::common::Status LoadModel();

  void SetInputOutputInfo(std::unordered_map<std::string, OnnxTensorInfo>&& input_output_info) {
    input_output_info_ = std::move(input_output_info);
  }

  void SetScalarOutputs(std::unordered_set<std::string>&& scalar_outputs) {
    scalar_outputs_ = std::move(scalar_outputs);
  }
};

}  // namespace coreml
}  // namespace onnxruntime