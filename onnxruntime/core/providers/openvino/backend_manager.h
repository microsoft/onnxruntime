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
  BackendManager(SessionContext& session_context,
                 SharedContext& shared_context,
                 const onnxruntime::Node& fused_node,
                 const onnxruntime::GraphViewer& subgraph,
                 const logging::Logger& logger,
                 EPCtxHandler& ctx_handle);
  void Compute(OrtKernelContext* context);
  void ShutdownBackendManager();
  SessionContext& GetSessionContext();
  Status ExportCompiledBlobAsEPCtxNode(const onnxruntime::GraphViewer& subgraph);
  ov::CompiledModel GetOVCompiledModel();
  void RewindKVCache(size_t index);

 private:
  std::unique_ptr<ONNX_NAMESPACE::ModelProto> GetModelProtoFromFusedNode(
      const onnxruntime::Node& fused_node,
      const onnxruntime::GraphViewer& subgraph,
      const logging::Logger& logger) const;

  bool ModelHasSymbolicInputDims(const onnxruntime::GraphViewer& subgraph) const;
  std::unordered_set<std::string> IdentifyDynamicInputs(const onnxruntime::GraphViewer& subgraph,
                                                        const std::vector<const NodeArg*>& graph_inputs) const;
  bool ModelHasBatchedInputs(const ONNX_NAMESPACE::ModelProto& model_proto) const;
  void ValidateInputShapes(const reshape_t& shapes,
                           const std::vector<const NodeArg*>& graph_inputs) const;

  std::shared_ptr<ONNX_NAMESPACE::ModelProto>
  ReWriteBatchDimWithOne(const ONNX_NAMESPACE::ModelProto& model_proto);

  std::unique_ptr<ONNX_NAMESPACE::ModelProto>
  ReWriteInputShapeInfo(const ONNX_NAMESPACE::ModelProto& model_proto,
                        const std::vector<std::vector<int64_t>>& input_shapes);

  std::unique_ptr<ONNX_NAMESPACE::ModelProto> model_proto_;
  std::shared_ptr<IBackend> concrete_backend_;
  std::map<std::string, std::shared_ptr<IBackend>> backend_map_;
  SubGraphContext subgraph_context_;
  EPCtxHandler& ep_ctx_handle_;
  SessionContext& session_context_;
  SharedContext& shared_context_;
};

}  // namespace openvino_ep
}  // namespace onnxruntime
