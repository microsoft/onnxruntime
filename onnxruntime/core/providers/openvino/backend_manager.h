// Copyright(C) 2020 Intel Corporation
// Licensed under the MIT License

#pragma once

#include "core/common/common.h"
#include "core/graph/graph_viewer.h"
#include "core/framework/execution_provider.h"
#include "core/framework/kernel_registry.h"
#include "core/framework/allocatormgr.h"
#include "core/session/onnxruntime_cxx_api.h"
#include <inference_engine.hpp>
// IE defines a macro 'OPTIONAL' that conflicts the remaining headers using MSVC
#if defined(_MSC_VER)
#undef OPTIONAL
#endif

#include "ibackend.h"

namespace onnxruntime {
namespace openvino_ep {

// Singleton class that manages all the backends
class BackendManager {
 public:
  BackendManager(const onnxruntime::Node* fused_node, const logging::Logger& logger,
                 std::string dev_id, std::string prec_str);
  static InferenceEngine::Core ie_core_;
  void Compute(Ort::CustomOpApi api, OrtKernelContext* context);
  void ShutdownBackendManager();

 private:
  ONNX_NAMESPACE::ModelProto GetModelProtoFromFusedNode(
    const onnxruntime::Node* fused_node, const logging::Logger& logger) const;
  bool ModelHasSymbolicInputDims(const onnxruntime::Node* fused_node) const;

  std::string device_id_;
  InferenceEngine::Precision precision_;
  std::string precision_str_;
  ONNX_NAMESPACE::ModelProto model_proto_;
  bool has_dynamic_input_shape_ = false;
  std::shared_ptr<IBackend> concrete_backend_;
  std::map<std::string, std::shared_ptr<IBackend>> backend_map_;
  std::vector<int> input_indexes_;
  std::string subgraph_name_;
  bool set_vpu_config_;
  std::unordered_map<std::string, int> output_names_;
};

}  // namespace openvino_ep
}  // namespace onnxruntime