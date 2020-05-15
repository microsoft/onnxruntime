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

#include "contexts.h"
#include "ibackend.h"

namespace onnxruntime {
namespace openvino_ep {

// Singleton class that manages all the backends
class BackendManager {
 public:
  BackendManager(const onnxruntime::Node* fused_node, const logging::Logger& logger,
                 std::string dev_id, std::string prec_str);
  void Compute(Ort::CustomOpApi api, OrtKernelContext* context);
  void ShutdownBackendManager();
  static GlobalContext& GetGlobalContext();

 private:
  ONNX_NAMESPACE::ModelProto GetModelProtoFromFusedNode(
      const onnxruntime::Node* fused_node, const logging::Logger& logger) const;
  bool ModelHasSymbolicInputDims(const onnxruntime::Node* fused_node) const;
  bool ModelHasBatchedInputs(const ONNX_NAMESPACE::ModelProto& model_proto) const;

  std::shared_ptr<ONNX_NAMESPACE::ModelProto>
  ReWriteBatchDimWithOne(const ONNX_NAMESPACE::ModelProto& model_proto);

  std::shared_ptr<ONNX_NAMESPACE::ModelProto>
  ReWriteInputShapeInfo(const ONNX_NAMESPACE::ModelProto& model_proto,
                        std::vector<std::vector<int64_t>> input_shapes);

  ONNX_NAMESPACE::ModelProto model_proto_;
  std::shared_ptr<IBackend> concrete_backend_;
  std::map<std::string, std::shared_ptr<IBackend>> backend_map_;
  SubGraphContext subgraph_context_;
};

}  // namespace openvino_ep
}  // namespace onnxruntime