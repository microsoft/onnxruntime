// Copyright(C) 2019 Intel Corporation
// Licensed under the MIT License

#pragma once

#include <inference_engine.hpp>
#include "core/session/onnxruntime_cxx_api.h"

namespace onnxruntime {
namespace intel_ep {
namespace backend_utils {
  constexpr const char* log_tag = "[Intel-EP] ";
  bool IsDebugEnabled();
  void SetIODefs(const ONNX_NAMESPACE::ModelProto& model_proto, std::shared_ptr<InferenceEngine::CNNNetwork> network);
  std::shared_ptr<InferenceEngine::CNNNetwork> CreateCNNNetwork(const ONNX_NAMESPACE::ModelProto& model_proto, InferenceEngine::Precision precision);
  InferenceEngine::Precision ConvertPrecisionONNXToIntel(const ONNX_NAMESPACE::TypeProto& onnx_type);
  std::vector<const OrtValue*> GetInputTensors(Ort::CustomOpApi& ort, OrtKernelContext* context, std::shared_ptr<InferenceEngine::CNNNetwork> ie_cnn_network, std::vector<int> input_indexes);
  std::vector<OrtValue*> GetOutputTensors(Ort::CustomOpApi& ort, OrtKernelContext* context, size_t batch_size, InferenceEngine::InferRequest::Ptr infer_request, std::shared_ptr<InferenceEngine::CNNNetwork> ie_cnn_network);

} // namespace backend_utils
} // namespace intel_ep
} // namespace onnxruntime