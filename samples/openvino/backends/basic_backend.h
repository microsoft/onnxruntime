// Copyright (C) 2019-2022 Intel Corporation
// Licensed under the MIT License

#pragma once

#include <memory>

#define ORT_API_MANUAL_INIT
#include "core/session/onnxruntime_cxx_api.h"
#include "../contexts.h"
#include "../ibackend.h"
#include "../ov_interface.h"
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

  void Infer(OrtKernelContext* context) override;

 private:
  bool ImportBlob(std::string hw_target, bool vpu_status);
  void PopulateCompiledDirectory(std::string, std::string&, std::string&, bool&);
  bool ValidateSubgraph(std::map<std::string, std::shared_ptr<ov::Node>>& const_outputs_map);
  void PopulateConfigValue(ov::AnyMap& device_config);
  void EnableCaching();
  void EnableGPUThrottling(ov::AnyMap& device_config);
  void EnableStreams();
  void StartAsyncInference(Ort::KernelContext& context, std::shared_ptr<OVInferRequest> infer_request);

#ifdef IO_BUFFER_ENABLED
  void StartRemoteAsyncInference(Ort::KernelContext& context, std::shared_ptr<OVInferRequest> infer_request);
#endif

  void CompleteAsyncInference(Ort::KernelContext& context, std::shared_ptr<OVInferRequest> infer_request);

  GlobalContext& global_context_;
  SubGraphContext subgraph_context_;
  mutable std::mutex compute_lock_;
  std::shared_ptr<OVNetwork> ie_cnn_network_;
  OVExeNetwork exe_network_;
  std::map<std::string, std::shared_ptr<ov::Node>> const_outputs_map_;
  std::unique_ptr<InferRequestsQueue> inferRequestsQueue_;
#if defined IO_BUFFER_ENABLED
  OVRemoteContextPtr remote_context_;
#endif
};

class InferRequestsQueue {
 public:
  InferRequestsQueue(OVExeNetwork& net, size_t nireq) {
    OVInferRequestPtr infer_request;
    for (size_t id = 0; id < nireq; id++) {
      infer_request = std::make_shared<OVInferRequest>(net.CreateInferRequest());
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
      i->get()->QueryStatus();
    }
    std::cout << '\n';
  }

  void putIdleRequest(OVInferRequestPtr infer_request) {
    std::unique_lock<std::mutex> lock(_mutex);
    infer_requests_.push_back(infer_request);
    _cv.notify_one();
  }

  OVInferRequestPtr getIdleRequest() {
    std::unique_lock<std::mutex> lock(_mutex);
    _cv.wait(lock, [this] { return infer_requests_.size() > 0; });
    auto request = infer_requests_.at(0);
    infer_requests_.erase(infer_requests_.begin());
    return request;
  }

 private:
  std::mutex _mutex;
  std::condition_variable _cv;
  std::vector<OVInferRequestPtr> infer_requests_;
};

}  // namespace openvino_ep
}  // namespace onnxruntime
