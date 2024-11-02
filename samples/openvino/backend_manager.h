// Copyright (C) Intel Corporation
// Licensed under the MIT License

#pragma once

#include <vector>
#include <map>
#include <memory>
#include <string>

#include "ov_interface.h"
#include "contexts.h"
#include "onnx_ctx_model_helper.h"
#include "ibackend.h"
#include "onnx/onnx_pb.h"

namespace onnxruntime {
namespace openvino_ep {

// Singleton class that manages all the backends
class BackendManager {
 public:
  BackendManager(const GlobalContext& global_context,
                 const OrtNode* fused_node,
                 const OrtGraphViewer* subgraph,
                 EPCtxHandler& ctx_handle);
  void Compute(OrtKernelContext* context);
  void ShutdownBackendManager();
//  void SetGlobalCotext(const GlobalContext& global_context);
  GlobalContext& GetGlobalContext();
  OrtStatus* ExportCompiledBlobAsEPCtxNode(const OrtGraphViewer* subgraph);

 private:
  void* GetModelProtoFromFusedNode(
      const OrtNode* fused_node,
      const OrtGraphViewer* subgraph, size_t* model_proto_len) const;

 bool ModelHasSymbolicInputDims(const OrtGraphViewer* subgraph) const;
//  bool ModelHasBatchedInputs(const ONNX_NAMESPACE::ModelProto& model_proto) const;
//
//  std::shared_ptr<ONNX_NAMESPACE::ModelProto>
//  ReWriteBatchDimWithOne(const ONNX_NAMESPACE::ModelProto& model_proto);
//
  std::unique_ptr<ONNX_NAMESPACE::ModelProto>
  ReWriteInputShapeInfo(void* model_proto, size_t model_proto_len,
                        const std::vector<std::vector<int64_t>>& input_shapes);

  void* model_proto_;   // TODO(leca): release
  size_t model_proto_len_;
  std::shared_ptr<IBackend> concrete_backend_;
  std::map<std::string, std::shared_ptr<IBackend>> backend_map_;
  SubGraphContext subgraph_context_;
  GlobalContext global_context_;
  EPCtxHandler ep_ctx_handle_{};
  std::string openvino_sdk_version_{};
  static const OrtGraphApi* graph_api_;
};

}  // namespace openvino_ep
}  // namespace onnxruntime
//
