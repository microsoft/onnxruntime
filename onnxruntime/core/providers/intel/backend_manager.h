// Copyright(C) 2019 Intel Corporation
// Licensed under the MIT License

#pragma once

#include "core/common/common.h"
#include "core/graph/graph_viewer.h"
#include "core/framework/execution_provider.h"
#include "core/framework/kernel_registry.h"
#include "core/framework/allocatormgr.h"
#include "core/session/onnxruntime_cxx_api.h"
#include <inference_engine.hpp>
#include "ov_backend.h"
#include "cpu_backend.h"

namespace onnxruntime {
namespace intel_ep {

// Singleton class that manages all the backends
class BackendManager {
 public:
  BackendManager(const onnxruntime::Node* fused_node, const logging::Logger& logger);
  void Compute(Ort::CustomOpApi api, OrtKernelContext* context);
  void ShutdownBackendManager();

 private:
  ONNX_NAMESPACE::ModelProto GetModelProtoFromFusedNode(const onnxruntime::Node* fused_node, const logging::Logger& logger) const;
  bool ModelHasSymbolicInputDims(const ONNX_NAMESPACE::ModelProto& model_proto) const;

  std::string device_id_;
  InferenceEngine::Precision precision_;
  std::string precision_str_;
  ONNX_NAMESPACE::ModelProto model_proto_;
  bool has_dynamic_input_shape_ = false;
  std::shared_ptr<OVBackend> concrete_backend_;
  std::map<std::string, std::shared_ptr<OVBackend>> backend_map_;
  std::vector<int> input_indexes_;
};

}  // namespace intel_ep
}  // namespace onnxruntime