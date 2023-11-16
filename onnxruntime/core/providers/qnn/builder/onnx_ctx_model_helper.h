// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <string>
#include <vector>

#include "qnn_def.h"
#include "core/common/logging/logging.h"
#include "core/graph/graph_viewer.h"
#include "core/providers/shared/utils/utils.h"
#include "core/graph/model.h"
#include "core/framework/execution_provider.h"

namespace onnxruntime {

namespace qnn {

class QnnModel;
class QnnBackendManager;

static const std::string EPCONTEXT_OP = "EPContext";
static const std::string MAIN_CONTEXT = "main_context";
static const std::string EMBED_MODE = "embed_mode";
static const std::string EP_CACHE_CONTEXT = "ep_cache_context";
static const std::string EP_SDK_VER = "ep_sdk_version";
static const std::string PARTITION_NAME = "partition_name";
static const std::string SOURCE = "source";

Status IsFusedGraphHasCtxNode(const std::vector<IExecutionProvider::FusedNodeAndGraph>& fused_nodes_and_graphs,
                              bool& is_qnn_ctx_model);

bool IsQnnCtxModel(const onnxruntime::GraphViewer& graph_viewer);

Status CreateNodeArgs(const std::vector<std::string>& names,
                      const std::unordered_map<std::string, OnnxTensorInfo>& tensor_info_table,
                      std::vector<NodeArg*>& node_args,
                      onnxruntime::Graph& graph);

class QnnCacheModelHandler {
 public:
  QnnCacheModelHandler(bool qnn_context_embed_mode) : qnn_context_embed_mode_(qnn_context_embed_mode) {
  }
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(QnnCacheModelHandler);

  Status LoadQnnCtxFromOnnxModel(const onnxruntime::GraphViewer& graph_viewer,
                                 const std::string& ctx_onnx_model_path,
                                 bool is_qnn_ctx_model,
                                 bool is_ctx_cache_file_exist,
                                 QnnBackendManager* qnn_backend_manager,
                                 QnnModel& qnn_model,
                                 const logging::Logger& logger) {
    if (is_qnn_ctx_model) {
      return GetEpContextFromGraph(graph_viewer, ctx_onnx_model_path, qnn_backend_manager, qnn_model);
    } else if (is_ctx_cache_file_exist) {
      return GetEpContextFromModel(ctx_onnx_model_path, qnn_backend_manager, qnn_model, logger);
    }
    return Status::OK();
  }

  bool IsContextCacheFileExists(const std::string& customer_context_cache_path,
                                const std::string& model_name,
                                const std::string& model_description,
                                const onnxruntime::PathString& model_pathstring);

  bool GetIsContextCacheFileExists() const {
    return ctx_file_exists_;
  }

  Status ValidateWithContextFile(const std::string& model_name,
                                 const std::string& graph_name,
                                 const logging::Logger& logger);

  Status GetMetadataFromEpContextModel(const std::string& ctx_onnx_model_path,
                                       std::string& model_name,
                                       std::string& model_description,
                                       std::string& graph_partition_name,
                                       std::string& cache_source,
                                       const logging::Logger& logger);

  Status GenerateCtxCacheOnnxModel(unsigned char* buffer,
                                   uint64_t buffer_size,
                                   const std::string& sdk_build_version,
                                   const std::vector<IExecutionProvider::FusedNodeAndGraph>& fused_nodes_and_graphs,
                                   const std::unordered_map<std::string, std::unique_ptr<QnnModel>>& qnn_models,
                                   const logging::Logger& logger);

 private:
  Status GetEpContextFromModel(const std::string& ctx_onnx_model_path,
                               QnnBackendManager* qnn_backend_manager,
                               QnnModel& qnn_model,
                               const logging::Logger& logger);

  Status GetEpContextFromGraph(const onnxruntime::GraphViewer& graph_viewer,
                               const std::string& ctx_onnx_model_path,
                               QnnBackendManager* qnn_backend_manager,
                               QnnModel& qnn_model);

 private:
  bool is_metadata_ready_ = false;
  // model_name_ to cache_source_ -- metadata get from generated Qnn context binary Onnx model
  std::string model_name_ = "";
  std::string model_description_ = "";
  std::string graph_partition_name_ = "";
  std::string cache_source_ = "";

  std::string context_cache_path_ = "";
  bool ctx_file_exists_ = false;
  bool get_capability_round_2_ = false;
  bool qnn_context_embed_mode_ = true;
};  // QnnCacheModelHandler

}  // namespace qnn
}  // namespace onnxruntime
