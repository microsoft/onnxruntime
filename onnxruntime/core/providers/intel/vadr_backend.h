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

class VADRBackend : public OVBackend {
 public:
  VADRBackend(const ONNX_NAMESPACE::ModelProto& model_proto, std::vector<int> input_indexes, std::string device_id, InferenceEngine::Precision precision);

  void Infer(Ort::CustomOpApi& ort, OrtKernelContext* context) override;


 private:


  void StartAsyncInference(Ort::CustomOpApi& ort, const OrtValue* input_tensors[], size_t batch_slice_idx, size_t infer_req_idx, std::vector<InferenceEngine::InferRequest::Ptr>& infer_requests, std::shared_ptr<InferenceEngine::CNNNetwork> ie_cnn_network);

  void CompleteAsyncInference(Ort::CustomOpApi& ort, OrtValue* output_tensors[], size_t batch_slice_idx, size_t infer_req_idx, std::vector<InferenceEngine::InferRequest::Ptr>& infer_requests, std::shared_ptr<InferenceEngine::CNNNetwork> ie_cnn_network);

  std::shared_ptr<InferenceEngine::CNNNetwork> ie_cnn_network_;
  std::vector<InferenceEngine::InferRequest::Ptr> infer_requests_;
  size_t num_inf_reqs_;
};
}  // namespace intel_ep
}  // namespace onnxruntime
