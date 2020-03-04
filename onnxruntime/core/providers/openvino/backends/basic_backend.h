// Copyright(C) 2020 Intel Corporation
// Licensed under the MIT License

#pragma once

#include <memory>
#include <inference_engine.hpp>
#include "core/session/onnxruntime_cxx_api.h"
#include "core/providers/openvino/ibackend.h"

namespace onnxruntime {
namespace openvino_ep {

bool IsDebugEnabled();

class BasicBackend : public IBackend {
 public:
  BasicBackend(const ONNX_NAMESPACE::ModelProto& model_proto, std::vector<int> input_indexes,std::unordered_map<std::string, int> output_names, std::string device_id, InferenceEngine::Precision precision);

  void Infer(Ort::CustomOpApi& ort, OrtKernelContext* context) override;


 private:


  void StartAsyncInference(Ort::CustomOpApi& ort, std::vector<const OrtValue*> input_tensors, InferenceEngine::InferRequest::Ptr infer_request, std::shared_ptr<InferenceEngine::CNNNetwork> ie_cnn_network);

  void CompleteAsyncInference(Ort::CustomOpApi& ort, std::vector<OrtValue*> output_tensors, InferenceEngine::InferRequest::Ptr infer_request, std::shared_ptr<InferenceEngine::CNNNetwork> ie_cnn_network);

  std::vector<int> input_indexes_;
  std::unordered_map<std::string, int> output_names_;
  mutable std::mutex compute_lock_;
  std::shared_ptr<InferenceEngine::CNNNetwork> ie_cnn_network_;
  InferenceEngine::InferRequest::Ptr infer_request_;
};
}  // namespace openvino_ep
}  // namespace onnxruntime
