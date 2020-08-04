// Copyright(C) 2019 Intel Corporation
// Licensed under the MIT License

#pragma once

#include <inference_engine.hpp>

#include "core/session/onnxruntime_cxx_api.h"
#include "contexts.h"

namespace onnxruntime {
namespace openvino_ep {
namespace backend_utils {
const std::string log_tag = "[OpenVINO-EP] ";

#ifndef NDEBUG
bool IsDebugEnabled();
#endif

void SetIODefs(const ONNX_NAMESPACE::ModelProto& model_proto,
               std::shared_ptr<InferenceEngine::CNNNetwork> network);

  std::shared_ptr<InferenceEngine::CNNNetwork>
  CreateCNNNetwork(const ONNX_NAMESPACE::ModelProto& model_proto, std::string device_id,
                   InferenceEngine::Precision precision, std::map<std::string, std::shared_ptr<ngraph::Node>>& const_outputs_map);

int GetFirstAvailableDevice(GlobalContext& global_context);

void FillOutputsWithConstantData(Ort::CustomOpApi& ort, std::map<std::string, std::shared_ptr<ngraph::Node>> const_out_map, OrtValue* out_tensor);

template <typename T>
void FillOutputHelper(Ort::CustomOpApi& ort, OrtValue* out_tensor, std::shared_ptr<ngraph::Node> node);

InferenceEngine::Precision
ConvertPrecisionONNXToOpenVINO(const ONNX_NAMESPACE::TypeProto& onnx_type);

std::vector<OrtValue*> GetOutputTensors(Ort::CustomOpApi& ort,
                                        OrtKernelContext* context, size_t batch_size,
                                        InferenceEngine::InferRequest::Ptr infer_request,
                                        std::shared_ptr<InferenceEngine::CNNNetwork> ie_cnn_network,
                                        std::unordered_map<std::string, int> output_names, std::map<std::string, std::shared_ptr<ngraph::Node>> const_output_map);

}  // namespace backend_utils
}  // namespace openvino_ep
}  // namespace onnxruntime