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
 public:
  Model(const std::string& path,
        std::vector<std::string>&& model_input_names,
        std::vector<std::string>&& model_output_names,
        std::unordered_map<std::string, OnnxTensorInfo>&& input_output_info,
        std::unordered_set<std::string>&& scalar_outputs,
        std::unordered_set<std::string>&& int64_outputs,
        const logging::Logger& logger, uint32_t coreml_flags);

  ~Model();
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(Model);

  Status LoadModel();

  Status Predict(const std::unordered_map<std::string, OnnxTensorData>& inputs,
                 const std::unordered_map<std::string, OnnxTensorInfo>& outputs,
                 const GetOutputTensorMutableRawDataFn& get_output_tensor_mutable_raw_data_fn);

  bool IsScalarOutput(const std::string& output_name) const {
    return Contains(scalar_outputs_, output_name);
  }

  bool IsInt64Output(const std::string& output_name) const {
    return Contains(int64_outputs_, output_name);
  }

  // Mutex for exclusive lock to this model object
  OrtMutex& GetMutex() { return mutex_; }

  // Input and output names in the ORT fused node's order.
  // Names may have been adjusted from the originals due to CoreML naming rules.
  // We do inputs/outputs based on order at the ONNX level so this doesn't matter.
  const std::vector<std::string>& GetOrderedInputs() const { return model_input_names_; }
  const std::vector<std::string>& GetOrderedOutputs() const { return model_output_names_; }

  const OnnxTensorInfo* TryGetInputOutputInfo(const std::string& name) const {
    const auto info_it = input_output_info_.find(name);
    return info_it != input_output_info_.end() ? &info_it->second : nullptr;
  }

  const OnnxTensorInfo& GetInputOutputInfo(const std::string& name) const {
    const auto* info = TryGetInputOutputInfo(name);
    ORT_ENFORCE(info != nullptr, "Failed to get info for input/output: ", name);
    return *info;
  }

 private:
  std::unique_ptr<Execution> execution_;
  std::vector<std::string> model_input_names_;   // input names in the order of the ORT fused node's inputs
  std::vector<std::string> model_output_names_;  // output names in the order of the ORT fused node's outputs

  std::unordered_map<std::string, OnnxTensorInfo> input_output_info_;
  std::unordered_set<std::string> scalar_outputs_;
  std::unordered_set<std::string> int64_outputs_;

  OrtMutex mutex_;
};

}  // namespace coreml
}  // namespace onnxruntime
