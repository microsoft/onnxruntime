// Copyright (C) 2019-2022 Intel Corporation
// Licensed under the MIT License

#pragma once

#include "contexts.h"
#include "ibackend.h"

namespace onnxruntime {
namespace interface {
  class NodeViewRef;
  class GraphViewRef;
}
namespace openvino_ep {

// Singleton class that manages all the backends
class BackendManager {
 public:
  BackendManager(const onnxruntime::interface::NodeViewRef& fused_node, const onnxruntime::interface::GraphViewRef& subgraph);
  void Compute(OrtKernelContext* context);
  void ShutdownBackendManager();
  static GlobalContext& GetGlobalContext();
  static void ReleaseGlobalContext();

 private:
  std::unique_ptr<ONNX_NAMESPACE::ModelProto> GetModelProtoFromFusedNode(
      const onnxruntime::interface::NodeViewRef& fused_node, const onnxruntime::interface::GraphViewRef& subgraph) const;
  bool ModelHasSymbolicInputDims(const onnxruntime::interface::GraphViewRef& subgraph) const;
  bool ModelHasBatchedInputs(const ONNX_NAMESPACE::ModelProto& model_proto) const;

  std::shared_ptr<ONNX_NAMESPACE::ModelProto>
  ReWriteBatchDimWithOne(const ONNX_NAMESPACE::ModelProto& model_proto);

  std::shared_ptr<ONNX_NAMESPACE::ModelProto>
  ReWriteInputShapeInfo(const ONNX_NAMESPACE::ModelProto& model_proto,
                        const std::vector<std::vector<int64_t>>& input_shapes);

  std::unique_ptr<ONNX_NAMESPACE::ModelProto> model_proto_;
  std::shared_ptr<IBackend> concrete_backend_;
  std::map<std::string, std::shared_ptr<IBackend>> backend_map_;
  SubGraphContext subgraph_context_;
};

}  // namespace openvino_ep
}  // namespace onnxruntime
