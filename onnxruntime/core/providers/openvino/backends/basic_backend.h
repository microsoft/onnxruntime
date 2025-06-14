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
#include <functional>
#include <algorithm>
#include <utility>

#include "core/session/onnxruntime_cxx_api.h"
#include "core/providers/openvino/contexts.h"
#include "core/providers/openvino/ibackend.h"
#include "core/providers/openvino/ov_interface.h"
#include "core/providers/openvino/backend_utils.h"

namespace onnxruntime {
namespace openvino_ep {

struct ov_tensor_data_t {
  OVTensorPtr tensor_ptr;
  const void* ort_ptr;
};

struct OnnxToOvNetworkBindings {
  struct ParameterInfo {
    std::string name;
    uint32_t ov_index;
    uint32_t onnx_index;
    ov::element::Type type;
    ov::PartialShape ov_shape;
    std::vector<int64_t> onnx_shape;
    uint8_t dynamic_flags = 0;  // bit 0: fully_dynamic, bit 1: bounded_dynamic

    // Query methods
    bool IsStatic() const { return dynamic_flags == 0; }
    bool IsFullyDynamic() const { return dynamic_flags & 1; }
    bool IsBoundedDynamic() const { return dynamic_flags & 2; }
    bool IsMixed() const { return (dynamic_flags & 3) == 3; }

    // Setter methods
    void SetFullyDynamic(bool value) {
      dynamic_flags = value ? (dynamic_flags | 1) : (dynamic_flags & ~1);
    }
    void SetBoundedDynamic(bool value) {
      dynamic_flags = value ? (dynamic_flags | 2) : (dynamic_flags & ~2);
    }
  };

  std::vector<ParameterInfo> network_outputs_;
  std::vector<ParameterInfo> network_inputs_;

  OnnxToOvNetworkBindings(OVExeNetwork& exec_network, SubGraphContext& subgraph_context, SessionContext& session_context) {
    auto populate = [&](auto& input_output_map, const SubGraphContext::string_index_map_t& onnx_input_map, const auto& ov_parameters) {
      for (const auto& [onnx_name, onnx_param_index] : onnx_input_map) {
        auto it = std::find_if(ov_parameters.begin(), ov_parameters.end(),
                               [&onnx_name](const auto& ov_parameter_info) { return ov_parameter_info.get_names().contains(onnx_name); });

        // For Stateful Model Compilation, the ONNX model includes KV cache (past/present) tensors.
        // However, these tensors are internally converted to a stateful representation, which removes them.
        // To prevent runtime exceptions, we simply continue processing here.
        if ((onnx_name.empty() || onnx_name == "beam_idx" ||
             onnx_name.find("past_key_values") != std::string::npos ||
             onnx_name.find("present") != std::string::npos) &&
            session_context.enable_causallm) {
          continue;
        }

        ORT_ENFORCE(it != ov_parameters.end(), backend_utils::log_tag,
                    "Input names mismatch between OpenVINO and ONNX. ", onnx_name,
                    " doesn't exist in the list of OpenVINO input tensor names");

        auto ov_param_index = std::distance(ov_parameters.begin(), it);

        auto shape = ov_parameters[ov_param_index].get_partial_shape();
        auto type = ov_parameters[ov_param_index].get_element_type();
        ParameterInfo info{onnx_name, ov_param_index, onnx_param_index, type, shape};

        // Analyze shape dynamism and set flags
        if (shape.is_static()) {
          // dynamic_flags remains 0 (static)
          auto static_shape = shape.get_shape();
          std::transform(static_shape.begin(), static_shape.end(), std::back_inserter(info.onnx_shape),
                         [](const auto& dim) { return static_cast<int64_t>(dim); });
        } else {
          // Analyze dynamic dimensions
          bool has_fully_dynamic = false;
          bool has_bounded_dynamic = false;

          for (const auto& dim : shape) {
            if (dim.is_dynamic()) {
              if (dim.get_interval().has_upper_bound()) {
                has_bounded_dynamic = true;
              } else {
                has_fully_dynamic = true;
              }
            }
          }

          info.SetFullyDynamic(has_fully_dynamic);
          info.SetBoundedDynamic(has_bounded_dynamic);
        }

        input_output_map.push_back(std::move(info));
      }
    };

    // Populate inputs and outputs
    populate(network_inputs_, subgraph_context.input_names, exec_network.Get().inputs());
    populate(network_outputs_, subgraph_context.output_names, exec_network.Get().outputs());
  }
};
class InferRequestsQueue;
class BasicBackend : public IBackend {
 public:
  BasicBackend(std::unique_ptr<ONNX_NAMESPACE::ModelProto>& model_proto,
               SessionContext& session_context,
               const SubGraphContext& subgraph_context,
               SharedContext& shared_context,
               ptr_stream_t& model_stream);

  void Infer(OrtKernelContext* context) override;
  ~BasicBackend() override = default;
  ov::CompiledModel GetOVCompiledModel() override {
    return exe_network_.Get();
  }
  void RewindKVCache(size_t index) override;

 private:
  bool ValidateSubgraph(std::map<std::string, std::shared_ptr<ov::Node>>& const_outputs_map);
  void PopulateConfigValue(ov::AnyMap& device_config);
  void EnableCaching();
  void EnableGPUThrottling(ov::AnyMap& device_config);
  void EnableStreams();
  void SetNumThreads(ov::AnyMap& device_config);
  void StartAsyncInference(Ort::KernelContext& context, std::shared_ptr<OVInferRequest> infer_request);
  void ValidateOrtDimsAgainstPartialShape(const std::vector<int64_t>& ort_dims,
                                          const ov::PartialShape& partial_shape) const;
  void CompleteAsyncInference(Ort::KernelContext& context, std::shared_ptr<OVInferRequest> infer_request);

  SessionContext& session_context_;
  SubGraphContext subgraph_context_;
  SharedContext& shared_context_;
  mutable std::mutex compute_lock_;
  OVExeNetwork exe_network_;
  std::map<std::string, std::shared_ptr<ov::Node>> const_outputs_map_;
  std::unique_ptr<InferRequestsQueue> inferRequestsQueue_;
  using ort_tensor_key_t = const std::string;
  std::map<ort_tensor_key_t, ov_tensor_data_t> ort_ov_tensor_map;
  std::unique_ptr<OnnxToOvNetworkBindings> bindings_;
};

class InferRequestsQueue {
 public:
  InferRequestsQueue(OVExeNetwork& net, size_t nireq, std::function<void(OVInferRequestPtr)> initializer) {
    OVInferRequestPtr infer_request;
    live_threads = nireq;
    for (size_t id = 0; id < nireq; id++) {
      infer_request = net.CreateInferRequest();
      initializer(infer_request);
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
    if (live_threads == 0) {
      return nullptr;
    }

    _cv.wait(lock, [this] { return infer_requests_.size() > 0; });
    auto request = infer_requests_.at(0);
    infer_requests_.erase(infer_requests_.begin());
    return request;
  }

  void deleteRequest() {
    std::unique_lock<std::mutex> lock(_mutex);
    live_threads = live_threads - 1;
  }

 private:
  std::mutex _mutex;
  std::condition_variable _cv;
  std::vector<OVInferRequestPtr> infer_requests_;
  int live_threads;
};

}  // namespace openvino_ep
}  // namespace onnxruntime
