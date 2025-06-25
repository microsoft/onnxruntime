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

struct ParameterInfo {
  std::string name;
  uint32_t ov_index;
  uint32_t onnx_index;
  ov::element::Type type;
  ParameterShape shape;
  uint8_t dynamic_flags = 0;

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

struct OnnxToOvNetworkBindings {
  std::vector<ParameterInfo> network_outputs_;
  std::vector<ParameterInfo> network_inputs_;
  bool has_dynamic_io_ = false;

  inline static const std::array special_io_names_{
      "beam_idx",
      "past_key_values",
      "present",
  };

  OnnxToOvNetworkBindings(OVExeNetwork& exec_network, SubGraphContext& subgraph_context, SessionContext& session_context) {
    auto populate = [&](auto& input_output_map, const SubGraphContext::string_index_map_t& onnx_input_map, const auto& ov_parameters) {
      for (const auto& [onnx_name, onnx_param_index] : onnx_input_map) {
        auto it = std::find_if(ov_parameters.begin(), ov_parameters.end(),
                               [&onnx_name](const auto& ov_parameter_info) { return ov_parameter_info.get_names().contains(onnx_name); });
        bool matched_names = it != ov_parameters.end();

        // For Stateful Model Compilation, the ONNX model includes KV cache (past/present) tensors.
        // However, these tensors are internally converted to a stateful representation, which removes them.
        // It's also possible that the onnx model does not contain tensors such as beam_idx, whereas our converted
        // stateful representation has introduced these new tensors, creating a name mismatch (matched_names=false).
        // So, if there is a name mismatch, or the name matches our special io list, we simply continue processing
        // here to prevent runtime exceptions.
        if (session_context.enable_causallm) {
          if (!matched_names ||
              std::any_of(special_io_names_.begin(), special_io_names_.end(),
                          [&onnx_name](const std::string& name) { return onnx_name.find(name) != std::string::npos; })) {
            // This case also requires dynamic shape inference, so we'll mark the bindings as dynamic.
            has_dynamic_io_ = true;
            continue;
          }
        }

        ORT_ENFORCE(matched_names, log_tag,
                    "Input names mismatch between OpenVINO and ONNX. ", onnx_name,
                    " doesn't exist in the list of OpenVINO input tensor names");

        auto ov_param_index = std::distance(ov_parameters.begin(), it);

        auto shape = ov_parameters[ov_param_index].get_partial_shape();
        auto type = ov_parameters[ov_param_index].get_element_type();
        ParameterInfo info{onnx_name, ov_param_index, onnx_param_index, type, ParameterShape{shape}};

        // Analyze shape dynamism and set flags
        if (!shape.is_static()) {
          has_dynamic_io_ = true;
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

class InferRequestPool;
class BasicBackend : public IBackend {
 public:
  BasicBackend(std::unique_ptr<ONNX_NAMESPACE::ModelProto>& model_proto,
               SessionContext& session_context,
               const SubGraphContext& subgraph_context,
               SharedContext& shared_context,
               ptr_stream_t& model_stream);

  void Infer(OrtKernelContext* context) const override;
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
  void ValidateOrtDimsAgainstPartialShape(const std::vector<int64_t>& ort_dims,
                                          const ov::PartialShape& partial_shape) const;

  SessionContext& session_context_;
  SubGraphContext subgraph_context_;
  SharedContext& shared_context_;
  OVExeNetwork exe_network_;
  std::map<std::string, std::shared_ptr<ov::Node>> const_outputs_map_;
  std::unique_ptr<InferRequestPool> infer_req_pool_;

  using ort_tensor_key_t = const std::string;
  std::unique_ptr<const OnnxToOvNetworkBindings> bindings_;
};

class InferRequestPool {
 public:
  struct GuardedInferReq {
    OVInferRequestPtr infer_request_;
    GuardedInferReq(InferRequestPool& queue, OVInferRequestPtr&& infer_req) : queue_(queue), infer_request_(std::move(infer_req)) {}
    ~GuardedInferReq() { queue_.putIdleRequest(std::move(infer_request_)); }

    // Movable but not copyable
    ORT_DISALLOW_COPY_AND_ASSIGNMENT(GuardedInferReq);
    GuardedInferReq(GuardedInferReq&&) = default;
    GuardedInferReq& operator=(GuardedInferReq&&) = default;

   private:
    InferRequestPool& queue_;
    friend class InferRequestPool;
  };

  InferRequestPool(OVExeNetwork& net, size_t initial_size, std::function<void(OVInferRequestPtr)> initializer) : exe_network_(net), initializer_(std::move(initializer)) {
    for (size_t id = 0; id < initial_size; id++) {
      infer_requests_.emplace_back(createInferRequest());
    }
  }
  ~InferRequestPool() = default;

  GuardedInferReq getRequest() {
    std::unique_lock<std::mutex> lock(_mutex);
    if (infer_requests_.empty()) {
      infer_requests_.emplace_back(createInferRequest());
    }
    auto request = std::move(infer_requests_.back());
    infer_requests_.pop_back();
    return GuardedInferReq(*this, std::move(request));
  }

  template <typename Func>
  void forEachIdleRequest(Func&& func) {
    std::unique_lock<std::mutex> lock(_mutex);
    for (auto& infer_request : infer_requests_) {
      func(infer_request);
    }
  }

 private:
  void putIdleRequest(OVInferRequestPtr&& infer_request) {
    if (infer_request) {
      std::unique_lock<std::mutex> lock(_mutex);
      infer_requests_.emplace_back(std::move(infer_request));
    }
  }

  OVInferRequestPtr createInferRequest() {
    auto infer_request = exe_network_.CreateInferRequest();
    initializer_(infer_request);
    return infer_request;
  }

 private:
  std::mutex _mutex;
  std::vector<OVInferRequestPtr> infer_requests_;
  OVExeNetwork& exe_network_;
  std::function<void(OVInferRequestPtr)> initializer_;
};

}  // namespace openvino_ep
}  // namespace onnxruntime
