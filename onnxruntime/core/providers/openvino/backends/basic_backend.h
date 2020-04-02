// Copyright(C) 2020 Intel Corporation
// Licensed under the MIT License

#pragma once

#include <memory>
#include <inference_engine.hpp>
// IE defines a macro 'OPTIONAL' that conflicts the remaining headers using MSVC
#if defined(_MSC_VER)
#undef OPTIONAL
#endif

#include "core/session/onnxruntime_cxx_api.h"
#include "core/providers/openvino/ibackend.h"

namespace onnxruntime {
namespace openvino_ep {

class BasicBackend : public IBackend {
 public:
  BasicBackend(const ONNX_NAMESPACE::ModelProto& model_proto, const std::vector<int>& input_indexes,
               const std::unordered_map<std::string, int>& output_names, std::string device_id,
               InferenceEngine::Precision precision);

  void Infer(Ort::CustomOpApi& ort, OrtKernelContext* context) override;

 private:
  void StartAsyncInference(Ort::CustomOpApi& ort, std::vector<const OrtValue*> input_tensors,
                           InferenceEngine::InferRequest::Ptr infer_request,
                           std::shared_ptr<InferenceEngine::CNNNetwork> ie_cnn_network);

  void CompleteAsyncInference(Ort::CustomOpApi& ort, std::vector<OrtValue*> output_tensors,
                              InferenceEngine::InferRequest::Ptr infer_request,
                              std::shared_ptr<InferenceEngine::CNNNetwork> ie_cnn_network);

  const std::vector<int>& input_indexes_;
  const std::unordered_map<std::string, int>& output_names_;
  mutable std::mutex compute_lock_;
  std::shared_ptr<InferenceEngine::CNNNetwork> ie_cnn_network_;
  InferenceEngine::InferRequest::Ptr infer_request_;
};
}  // namespace openvino_ep
}  // namespace onnxruntime
