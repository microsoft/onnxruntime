// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/coreml/model/model.h"

namespace onnxruntime {
namespace coreml {

class Execution {};

Model::Model(const std::string& /*path*/,
             std::vector<std::string>&& model_input_names,
             std::vector<std::string>&& model_output_names,
             std::unordered_map<std::string, OnnxTensorInfo>&& input_output_info,
             std::unordered_set<std::string>&& scalar_outputs,
             std::unordered_set<std::string>&& int64_outputs,
             const logging::Logger& /*logger*/,
             uint32_t /*coreml_flags*/)
    : execution_(std::make_unique<Execution>()),
      model_input_names_(std::move(model_input_names)),
      model_output_names_(std::move(model_output_names)),
      input_output_info_(std::move(input_output_info)),
      scalar_outputs_(std::move(scalar_outputs)),
      int64_outputs_(std::move(int64_outputs)) {
}

Model::~Model() {
}

Status Model::LoadModel() {
  // return OK so we hit more CoreML EP code.
  return Status::OK();
}

Status Model::Predict(const std::unordered_map<std::string, OnnxTensorData>& /*inputs*/,
                      const std::unordered_map<std::string, OnnxTensorInfo>& /*outputs*/,
                      const GetOutputTensorMutableRawDataFn& /*get_output_tensor_mutable_raw_data_fn*/) {
  return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Executing a CoreML model is not supported on this platform.");
}

}  // namespace coreml
}  // namespace onnxruntime
