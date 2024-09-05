// Copyright (C) Intel Corporation
// Licensed under the MIT License

#pragma once

#include <memory>

#define ORT_API_MANUAL_INIT
#include <vector>
#include <iostream>
#include <string>
#include <condition_variable>
#include <mutex>
#include <map>

#include "core/session/onnxruntime_cxx_api.h"
#include "core/providers/openvino/contexts.h"
#include "core/providers/openvino/ibackend.h"
#include "core/providers/openvino/ov_interface.h"

namespace onnxruntime {
namespace openvino_ep {

struct ov_tensor_data_t {
  OVTensorPtr tensor_ptr;
  bool copy_needed;
};

class InferRequestsQueue;
class BasicBackend : public IBackend {
 public:
  BasicBackend(std::unique_ptr<ONNX_NAMESPACE::ModelProto>& model_proto,
               GlobalContext& global_context,
               const SubGraphContext& subgraph_context,
               EPCtxHandler& ep_ctx_handle);

  void Infer(OrtKernelContext* context) override;
  ov::CompiledModel& GetOVCompiledModel() override {
    return exe_network_.Get();
  }

 private:
  void PopulateCompiledDirectory(std::string, std::string&, std::string&, bool&);
  bool ValidateSubgraph(std::map<std::string, std::shared_ptr<ov::Node>>& const_outputs_map);
  void PopulateConfigValue(ov::AnyMap& device_config);
  void EnableCaching(ov::AnyMap& device_config);
  void EnableGPUThrottling(ov::AnyMap& device_config);
  void EnableStreams();
  void SetNumThreads(ov::AnyMap& device_config);
  void StartAsyncInference(Ort::KernelContext& context, std::shared_ptr<OVInferRequest> infer_request);

#ifdef IO_BUFFER_ENABLED
  void StartRemoteAsyncInference(Ort::KernelContext& context, std::shared_ptr<OVInferRequest> infer_request);
#endif

  void CompleteAsyncInference(Ort::KernelContext& context, std::shared_ptr<OVInferRequest> infer_request);

  GlobalContext& global_context_;
  SubGraphContext subgraph_context_;
  mutable std::mutex compute_lock_;
  std::shared_ptr<const OVNetwork> ie_cnn_network_;
  OVExeNetwork exe_network_;
  std::map<std::string, std::shared_ptr<ov::Node>> const_outputs_map_;
  std::unique_ptr<InferRequestsQueue> inferRequestsQueue_;
  bool is_ep_ctx_graph_{false};
#if defined IO_BUFFER_ENABLED
  OVRemoteContextPtr remote_context_;
#endif

  using ort_tensor_key_t = std::pair<const void *, const std::string>;
  std::map<ort_tensor_key_t, ov_tensor_data_t> ort_ov_tensor_map;
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
