// Copyright (C) Intel Corporation
// Licensed under the MIT License

#pragma once

#include <vector>
#include <map>
#include <memory>
#include <string>

#include "core/providers/openvino/ov_interface.h"
#include "core/providers/openvino/contexts.h"
#include "core/providers/openvino/onnx_ctx_model_helper.h"
#include "core/providers/openvino/ibackend.h"

namespace onnxruntime {
namespace openvino_ep {

// Singleton class that manages all the backends
class BackendManager {
 public:
  BackendManager(const GlobalContext& global_context,
                 const onnxruntime::Node& fused_node,
                 const onnxruntime::GraphViewer& subgraph,
                 const logging::Logger& logger,
                 EPCtxHandler& ctx_handle);
  void Compute(OrtKernelContext* context);
  void ShutdownBackendManager();
  void SetGlobalCotext(const GlobalContext& global_context);
  GlobalContext& GetGlobalContext();
  Status ExportCompiledBlobAsEPCtxNode(const onnxruntime::GraphViewer& subgraph,
                                       const logging::Logger& logger);
  ov::CompiledModel& GetOVCompiledModel();

 private:
  std::unique_ptr<ONNX_NAMESPACE::ModelProto> GetModelProtoFromFusedNode(
      const onnxruntime::Node& fused_node,
      const onnxruntime::GraphViewer& subgraph,
      const logging::Logger& logger) const;

  bool ModelHasSymbolicInputDims(const onnxruntime::GraphViewer& subgraph) const;
  bool ModelHasBatchedInputs(const ONNX_NAMESPACE::ModelProto& model_proto) const;

  std::shared_ptr<ONNX_NAMESPACE::ModelProto>
  ReWriteBatchDimWithOne(const ONNX_NAMESPACE::ModelProto& model_proto);

  std::unique_ptr<ONNX_NAMESPACE::ModelProto>
  ReWriteInputShapeInfo(const ONNX_NAMESPACE::ModelProto& model_proto,
                        const std::vector<std::vector<int64_t>>& input_shapes);

  std::unique_ptr<ONNX_NAMESPACE::ModelProto> model_proto_;
  std::shared_ptr<IBackend> concrete_backend_;
  std::map<std::string, std::shared_ptr<IBackend>> backend_map_;
  SubGraphContext subgraph_context_;
  GlobalContext global_context_;
  EPCtxHandler ep_ctx_handle_{};
  std::string openvino_sdk_version_{};
};

}  // namespace openvino_ep
}  // namespace onnxruntime
