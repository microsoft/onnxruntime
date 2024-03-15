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

bool GraphHasEpContextNode(const onnxruntime::GraphViewer& graph_viewer);

bool IsFusedGraphHasCtxNode(const std::vector<IExecutionProvider::FusedNodeAndGraph>& fused_nodes_and_graphs);

Status GetMainContextNode(const std::vector<IExecutionProvider::FusedNodeAndGraph>& fused_nodes_and_graphs,
                          QnnBackendManager* qnn_backend_manager,
                          const logging::Logger& logger,
                          int& main_context_pos,
                          std::unordered_map<std::string, std::unique_ptr<qnn::QnnModel>>& qnn_models);

Status CreateNodeArgs(const std::vector<std::string>& names,
                      const std::unordered_map<std::string, OnnxTensorInfo>& tensor_info_table,
                      std::vector<NodeArg*>& node_args,
                      onnxruntime::Graph& graph);

bool ValidateContextCacheFilePath(bool is_qnn_ctx_model,
                                  const std::string& customer_context_cache_path,
                                  const onnxruntime::PathString& model_pathstring,
                                  onnxruntime::PathString& context_cache_path);

Status GetEpContextFromMainNode(const onnxruntime::Node& main_context_node,
                                const onnxruntime::PathString& ctx_onnx_model_path,
                                QnnBackendManager* qnn_backend_manager,
                                std::unordered_map<std::string, std::unique_ptr<qnn::QnnModel>>& qnn_models);

Status LoadQnnCtxFromOnnxGraph(const onnxruntime::GraphViewer& graph_viewer,
                               const onnxruntime::PathString& ctx_onnx_model_path,
                               QnnBackendManager* qnn_backend_manager,
                               std::unordered_map<std::string, std::unique_ptr<qnn::QnnModel>>& qnn_models,
                               const logging::Logger& logger);

Status CreateEPContextNodes(Model* model,
                            unsigned char* buffer,
                            uint64_t buffer_size,
                            const std::string& sdk_build_version,
                            const std::vector<IExecutionProvider::FusedNodeAndGraph>& fused_nodes_and_graphs,
                            const std::unordered_map<std::string, std::unique_ptr<QnnModel>>& qnn_models,
                            const onnxruntime::PathString& context_cache_path,
                            bool qnn_context_embed_mode,
                            const logging::Logger& logger);
}  // namespace qnn
}  // namespace onnxruntime
