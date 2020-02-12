// Copyright(C) 2019 Intel Corporation
// Licensed under the MIT License

#pragma once

#include "core/session/onnxruntime_cxx_api.h"

namespace onnxruntime {
namespace intel_ep {

class OVBackend{
  public:
  virtual void Infer(Ort::CustomOpApi& ort, OrtKernelContext* context) = 0;
  static const std::string log_tag;

  protected:
  OVBackend (InferenceEngine::Precision precision, std::vector<int> input_indexes) : precision_{precision}, input_indexes_{input_indexes} {};
  void SetIODefs(const ONNX_NAMESPACE::ModelProto& model_proto, std::shared_ptr<InferenceEngine::CNNNetwork> network);
  std::shared_ptr<InferenceEngine::CNNNetwork> CreateCNNNetwork(const ONNX_NAMESPACE::ModelProto& model_proto);
  InferenceEngine::Precision ConvertPrecisionONNXToIntel(const ONNX_NAMESPACE::TypeProto& onnx_type);
  std::vector<const OrtValue*> GetInputTensors(Ort::CustomOpApi& ort, OrtKernelContext* context, std::shared_ptr<InferenceEngine::CNNNetwork> ie_cnn_network);
  std::vector<OrtValue*> GetOutputTensors(Ort::CustomOpApi& ort, OrtKernelContext* context, size_t batch_size, InferenceEngine::InferRequest::Ptr infer_request, std::shared_ptr<InferenceEngine::CNNNetwork> ie_cnn_network);

  InferenceEngine::Precision precision_;
  std::vector<int> input_indexes_;
  mutable std::mutex compute_lock_;
};

} // namespace intel_ep
} // namespace onnxruntime
