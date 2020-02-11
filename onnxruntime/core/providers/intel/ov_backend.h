// Copyright(C) 2019 Intel Corporation
// Licensed under the MIT License

#pragma once

#include "core/session/onnxruntime_cxx_api.h"

namespace onnxruntime {
namespace intel_ep {

class OVBackend{
  public:
  virtual void Infer(Ort::CustomOpApi& ort, OrtKernelContext* context) = 0;

  protected: 
  void SetIODefs(const ONNX_NAMESPACE::ModelProto& model_proto, std::shared_ptr<InferenceEngine::CNNNetwork> network);
  std::shared_ptr<InferenceEngine::CNNNetwork> CreateCNNNetwork(const ONNX_NAMESPACE::ModelProto& model_proto);
  InferenceEngine::Precision ConvertPrecisionONNXToIntel(const ONNX_NAMESPACE::TypeProto& onnx_type);
  void GetInputTensors(Ort::CustomOpApi& ort, OrtKernelContext* context, const OrtValue* input_tensors[], std::shared_ptr<InferenceEngine::CNNNetwork> ie_cnn_network);
  void GetOutputTensors(Ort::CustomOpApi& ort, OrtKernelContext* context, OrtValue* output_tensors[], size_t batch_size, std::vector<InferenceEngine::InferRequest::Ptr>& infer_requests, std::shared_ptr<InferenceEngine::CNNNetwork> ie_cnn_network);

  std::vector<int> input_indexes_;
  InferenceEngine::Precision precision_;
};

} // namespace intel_ep
} // namespace onnxruntime
