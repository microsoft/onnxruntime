// Copyright(C) 2019 Intel Corporation
// Licensed under the MIT License

#include <map>
#include <memory>

#include <inference_engine.hpp>
#include <ie_utils.hpp>
#include <ie_builders.hpp>
#include <cpp/ie_infer_request.hpp>

#include "core/framework/func_api.h"

#include "core/graph/graph.h"

#ifndef OPENVINO_EP__OPENVINO_GRAPH_H
#define OPENVINO_EP__OPENVINO_GRAPH_H

namespace openvino_ep {

class OpenVINOGraph {
public:
  InferenceEngine::Precision precision_;
  const onnxruntime::Graph* onnx_graph_;

  OpenVINOGraph(onnxruntime::Node* fused_node, std::string device_info);

  std::shared_ptr<InferenceEngine::CNNNetwork> GetCNNNetwork();


  void Infer(onnxruntime::ONNXRunTimeTensor* input_tensors,
      size_t num_inputs, onnxruntime::ONNXRunTimeTensor* output_tensors,
      size_t num_outputs, onnxruntime::AllocateFunc& output_allocator_func,
      onnxruntime::AllocatorHandle& output_allocator_handle);


private:

  std::shared_ptr<InferenceEngine::CNNNetwork> BuildCNNNetworkWithMO();

  std::vector<InferenceEngine::InferRequest::Ptr> GetExecutableHandle(
      std::shared_ptr<InferenceEngine::CNNNetwork> network,
      const std::string& device, InferenceEngine::Precision precision);

  std::vector<std::string> GetEnvLdLibraryPath();

  onnxruntime::Node* fused_node_;
  std::shared_ptr<InferenceEngine::CNNNetwork> cnn_network_;
  size_t num_inf_reqs_;
  std::vector<InferenceEngine::InferRequest::Ptr> infer_requests_;
  std::string device_id_;
  mutable std::mutex compute_lock_;

};
}

#endif
