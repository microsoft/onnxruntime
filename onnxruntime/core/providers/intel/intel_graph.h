// Copyright(C) 2019 Intel Corporation
// Licensed under the MIT License

#pragma once

#include <memory>

#include <inference_engine.hpp>

#include "core/session/onnxruntime_cxx_api.h"

namespace onnxruntime {
namespace intel_ep {

bool IsDebugEnabled();

class IntelGraph {
 public:
  IntelGraph(const onnxruntime::Node* fused_node);

  void Infer(const ONNX_NAMESPACE::ModelProto& model_proto, Ort::CustomOpApi ort, OrtKernelContext* context);

  static const std::string log_tag;

 private:
  std::shared_ptr<InferenceEngine::CNNNetwork> CreateCNNNetwork(const ONNX_NAMESPACE::ModelProto& model_proto);

  InferenceEngine::Precision ConvertPrecisionONNXToIntel(ONNX_NAMESPACE::DataType onnx_type);

  void SetIODefs(std::shared_ptr<InferenceEngine::CNNNetwork> network);

  size_t DeduceBatchSize(Ort::CustomOpApi ort, const OrtValue* input_tensor, InferenceEngine::SizeVector graph_dims);

  void GetInputTensors(Ort::CustomOpApi ort, OrtKernelContext* context, const OrtValue* input_tensors[], std::shared_ptr<InferenceEngine::CNNNetwork> ie_cnn_network);

  void GetOutputTensors(Ort::CustomOpApi ort, OrtKernelContext* context, OrtValue* output_tensors[], size_t batch_size, std::vector<InferenceEngine::InferRequest::Ptr>& infer_requests, std::shared_ptr<InferenceEngine::CNNNetwork> ie_cnn_network);

  void StartAsyncInference(Ort::CustomOpApi ort, const OrtValue* input_tensors[], size_t batch_slice_idx, size_t infer_req_idx, std::vector<InferenceEngine::InferRequest::Ptr>& infer_requests, std::shared_ptr<InferenceEngine::CNNNetwork> ie_cnn_network);

  void CompleteAsyncInference(Ort::CustomOpApi ort, OrtValue* output_tensors[], size_t batch_slice_idx, size_t infer_req_idx, std::vector<InferenceEngine::InferRequest::Ptr>& infer_requests, std::shared_ptr<InferenceEngine::CNNNetwork> ie_cnn_network);

  std::string name_ = "test";
  const onnxruntime::Node* fused_node_;
  size_t num_inf_reqs_;
  mutable std::mutex compute_lock_;
  std::string device_id_;
  std::vector<int> input_indexes_;
  InferenceEngine::Precision precision_;
};
}  // namespace intel_ep
}  // namespace onnxruntime
