// Copyright(C) 2019 Intel Corporation
// Licensed under the MIT License

#pragma once

#include <memory>
#include <inference_engine.hpp>

#define ORT_API_MANUAL_INIT
#include "core/session/onnxruntime_cxx_api.h"
#include "core/providers/openvino/contexts.h"
#include "core/providers/openvino/ibackend.h"

#include <vector>
#include <iostream>
#include <string>
#include <condition_variable>
#include <mutex>

namespace onnxruntime {
namespace openvino_ep {

class InferRequestsQueue;
class BasicBackend : public IBackend {
 public:
  BasicBackend(const ONNX_NAMESPACE::ModelProto& model_proto,
               GlobalContext& global_context,
               const SubGraphContext& subgraph_context);

  void Infer(Ort::CustomOpApi& ort, OrtKernelContext* context) override;

 private:
  void StartAsyncInference(Ort::CustomOpApi& ort, OrtKernelContext* context, std::shared_ptr<InferenceEngine::InferRequest> infer_request);

  void CompleteAsyncInference(Ort::CustomOpApi& ort, OrtKernelContext* context, std::shared_ptr<InferenceEngine::InferRequest> infer_request);

  GlobalContext& global_context_;
  SubGraphContext subgraph_context_;
  mutable std::mutex compute_lock_;
  std::shared_ptr<InferenceEngine::CNNNetwork> ie_cnn_network_;
  InferenceEngine::ExecutableNetwork exe_network_;
  std::map<std::string, std::shared_ptr<ngraph::Node>> const_outputs_map_;
  std::unique_ptr<InferRequestsQueue> inferRequestsQueue_;
};

class InferRequestsQueue {
 public:
  InferRequestsQueue(InferenceEngine::ExecutableNetwork& net, size_t nireq) {
    InferenceEngine::InferRequest::Ptr infer_request;
    for (size_t id = 0; id < nireq; id++) {
      infer_request = std::make_shared<InferenceEngine::InferRequest>(net.CreateInferRequest());
      infer_requests_.push_back(infer_request);
    }
  }

  ~InferRequestsQueue() {
    // clearing out the infer_requests_ vector pool in the class's destructor
    for (auto& pointer : infer_requests_) {
      pointer = nullptr;
    }
    infer_requests_.erase(std::remove(infer_requests_.begin(), infer_requests_.end(), nullptr), infer_requests_.end());
  }

  void printstatus() {
    std::cout << "printing elements of the vector (infer_requests_): " << std::endl;
    for (auto i = infer_requests_.begin(); i != infer_requests_.end(); ++i) {
      std::cout << *i << " ";
    }
    std::cout << '\n';
  }

  void putIdleRequest(InferenceEngine::InferRequest::Ptr infer_request) {
    std::unique_lock<std::mutex> lock(_mutex);
    infer_requests_.push_back(infer_request);
    _cv.notify_one();
  }

  InferenceEngine::InferRequest::Ptr getIdleRequest() {
    std::unique_lock<std::mutex> lock(_mutex);
    _cv.wait(lock, [this] { return infer_requests_.size() > 0; });
    auto request = infer_requests_.at(0);
    infer_requests_.erase(infer_requests_.begin());
    return request;
  }

 private:
  std::mutex _mutex;
  std::condition_variable _cv;
  std::vector<InferenceEngine::InferRequest::Ptr> infer_requests_;
};

}  // namespace openvino_ep
}  // namespace onnxruntime
