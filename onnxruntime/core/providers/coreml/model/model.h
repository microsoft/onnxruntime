// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cstddef>
#include <functional>
#include <unordered_set>

#include "core/common/common.h"
#include "core/common/gsl.h"
#include "core/common/logging/logging.h"
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

using GetOutputTensorMutableRawDataFn = std::function<void*(const std::string& name,
                                                            int32_t requested_onnx_tensor_element_type,
                                                            gsl::span<const int64_t> static_shape)>;

class Model {
  friend class ModelBuilder;

 public:
  ~Model();
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(Model);

  Status Predict(const std::unordered_map<std::string, OnnxTensorData>& inputs,
                 const std::unordered_map<std::string, OnnxTensorInfo>& outputs,
                 const GetOutputTensorMutableRawDataFn& get_output_tensor_mutable_raw_data_fn);

  bool IsScalarOutput(const std::string& output_name) const;

  bool IsInt64Output(const std::string& output_name) const;

  // Mutex for exclusive lock to this model object
  OrtMutex& GetMutex() { return mutex_; }

  // Input and output names in the onnx model's order
  const std::vector<std::string>& GetOnnxInputs() const { return onnx_inputs_; }
  void SetOnnxInputs(std::vector<std::string>&& inputs) { onnx_inputs_ = std::move(inputs); }

  const std::vector<std::string>& GetOnnxOutputs() const { return onnx_outputs_; }
  void SetOnnxOutputs(std::vector<std::string>&& outputs) { onnx_outputs_ = std::move(outputs); }

  const OnnxTensorInfo* TryGetInputOutputInfo(const std::string& name) const;
  const OnnxTensorInfo& GetInputOutputInfo(const std::string& name) const;

 private:
  std::unique_ptr<Execution> execution_;
  std::unordered_set<std::string> scalar_outputs_;
  std::unordered_set<std::string> int64_outputs_;

  std::vector<std::string> onnx_inputs_;
  std::vector<std::string> onnx_outputs_;

  std::unordered_map<std::string, OnnxTensorInfo> input_output_info_;

  OrtMutex mutex_;

  Model(const std::string& path, const logging::Logger& logger, uint32_t coreml_flags);
  Status LoadModel();

  void SetInputOutputInfo(std::unordered_map<std::string, OnnxTensorInfo>&& input_output_info) {
    input_output_info_ = std::move(input_output_info);
  }

  void SetScalarOutputs(std::unordered_set<std::string>&& scalar_outputs) {
    scalar_outputs_ = std::move(scalar_outputs);
  }

  void SetInt64Outputs(std::unordered_set<std::string>&& int64_outputs) {
    int64_outputs_ = std::move(int64_outputs);
  }
};

}  // namespace coreml
}  // namespace onnxruntime
