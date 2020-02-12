// Copyright(C) 2019 Intel Corporation
// Licensed under the MIT License

#pragma once

#include <memory>
#include <inference_engine.hpp>
#include "core/session/onnxruntime_cxx_api.h"
#include "ov_backend.h"

namespace onnxruntime {
namespace intel_ep {

bool IsDebugEnabled();

class CPUBackend : public OVBackend {
 public:
  CPUBackend(const ONNX_NAMESPACE::ModelProto& model_proto, std::vector<int> input_indexes, std::string device_id, InferenceEngine::Precision precision);

  void Infer(Ort::CustomOpApi& ort, OrtKernelContext* context) override;


 private:


  void StartAsyncInference(Ort::CustomOpApi& ort, std::vector<const OrtValue*> input_tensors, InferenceEngine::InferRequest::Ptr infer_request, std::shared_ptr<InferenceEngine::CNNNetwork> ie_cnn_network);

  void CompleteAsyncInference(Ort::CustomOpApi& ort, std::vector<OrtValue*> output_tensors, InferenceEngine::InferRequest::Ptr infer_request, std::shared_ptr<InferenceEngine::CNNNetwork> ie_cnn_network);

  std::vector<int> input_indexes_;
  mutable std::mutex compute_lock_;
  std::shared_ptr<InferenceEngine::CNNNetwork> ie_cnn_network_;
  InferenceEngine::InferRequest::Ptr infer_request_;
};
}  // namespace intel_ep
}  // namespace onnxruntime
