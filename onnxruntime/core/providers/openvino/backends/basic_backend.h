// Copyright(C) 2019 Intel Corporation
// Licensed under the MIT License

#pragma once

#include <memory>
#include <inference_engine.hpp>

#include "core/session/onnxruntime_cxx_api.h"
#include "core/providers/openvino/contexts.h"
#include "core/providers/openvino/ibackend.h"

#include <vector>
#include <queue>
#include <string>
#include <condition_variable>
#include <mutex>

namespace onnxruntime {
namespace openvino_ep {

class BasicBackend : public IBackend {
 public:
  BasicBackend(const ONNX_NAMESPACE::ModelProto& model_proto,
               GlobalContext& global_context,
               const SubGraphContext& subgraph_context);

  void Infer(Ort::CustomOpApi& ort, OrtKernelContext* context) override;

 private:
  void StartAsyncInference(Ort::CustomOpApi& ort, OrtKernelContext* context);

  void CompleteAsyncInference(Ort::CustomOpApi& ort, OrtKernelContext* context);

  GlobalContext& global_context_;
  SubGraphContext subgraph_context_;
  mutable std::mutex compute_lock_;
  std::shared_ptr<InferenceEngine::CNNNetwork> ie_cnn_network_;
  std::map<std::string, std::shared_ptr<ngraph::Node>> const_outputs_map_;
  InferenceEngine::InferRequest::Ptr infer_request_;
  InferenceEngine::ExecutableNetwork exe_network;
};

class InferRequestsQueue {
public:
InferRequestsQueue(InferenceEngine::ExecutableNetwork& net, size_t nireq) {
  for (size_t id = 0; id < nireq; id++) {
      infer_request_ = net.CreateInferRequestPtr();
      infer_requests_.push_back(infer_request_);
    }
  }

~InferRequestsQueue() {
  std::cout << "calling out the ~InferRequestsQueue() Destructor " << std::endl;
}

void printstatus() {
    std::cout << "printing elements of the vector (infer_requests_): " << std::endl;
    for (auto i = infer_requests_.begin(); i != infer_requests_.end(); ++i)
    {
        std::cout << *i << " ";
    }
        std::cout << '\n';
}

void putIdleRequest(InferenceEngine::InferRequest::Ptr infer_request_) {
    std::unique_lock<std::mutex> lock(_mutex);
    infer_requests_.push_back(infer_request_);
    _cv.notify_one();
}

InferenceEngine::InferRequest::Ptr getIdleRequest() {
    std::unique_lock<std::mutex> lock(_mutex);
    _cv.wait(lock, [this]{ return infer_requests_.size() > 0; });
    auto request = infer_requests_.at(0);
    infer_requests_.erase(infer_requests_.begin());
    return request;
}

private:
std::mutex _mutex;
std::condition_variable _cv;
std::vector<InferenceEngine::InferRequest::Ptr> infer_requests_;
InferenceEngine::InferRequest::Ptr infer_request_;
};

}  // namespace openvino_ep
}  // namespace onnxruntime
