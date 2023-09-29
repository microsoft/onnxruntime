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

namespace onnxruntime {
namespace qnn {

Status CreateNodeArgs(const std::vector<std::string>& names,
                      const std::unordered_map<std::string, OnnxTensorInfo>& tensor_info_table,
                      std::vector<NodeArg*>& node_args,
                      onnxruntime::Graph& graph);

Status GenerateCtxCacheOnnxModle(const std::string& model_name, const std::string& graph_name,
                                 const std::vector<std::string>& input_names,
                                 const std::unordered_map<std::string, OnnxTensorInfo>& inputs_info,
                                 const std::vector<std::string>& output_names,
                                 const std::unordered_map<std::string, OnnxTensorInfo>& outputs_info,
                                 const std::string& model_description,
                                 const std::string& sdk_build_version,
                                 const std::string file_path,
                                 unsigned char* buffer,
                                 uint64_t buffer_size,
                                 const logging::Logger& logger);

class QnnCacheModelHandler {
 public:
  QnnCacheModelHandler() {
  }
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(QnnCacheModelHandler);

  bool IsQnnCtxModel(const onnxruntime::GraphViewer& graph_viewer) {
    if (is_qnn_ctx_model_) {
      return is_qnn_ctx_model_;
    }
    // It's a Onnx model with Qnn context cache binary if it only has a node with EPCache type
    if (graph_viewer.NumberOfNodes() == 1 && graph_viewer.Nodes().begin()->OpType() == "EPCache") {
      is_qnn_ctx_model_ = true;
    }
    return is_qnn_ctx_model_;
  }

  bool GetIsQnnCtxModel() const {
    return is_qnn_ctx_model_;
  }

  Status GetEpEngineCache(const onnxruntime::GraphViewer& graph_viewer,
                          const std::string& onnx_model_path,
                          bool is_ctx_cache_file_exist,
                          std::string& ep_engine_cache,
                          const logging::Logger& logger) const {
    if (is_qnn_ctx_model_) {
      ORT_RETURN_IF_ERROR(GetEpEngineCacheFromGraph(graph_viewer, ep_engine_cache));
    } else if (is_ctx_cache_file_exist) {
      ORT_RETURN_IF_ERROR(GetEpEngineCacheFromModel(onnx_model_path, ep_engine_cache, logger));
    }

    return Status::OK();
  }

 private:
  Status GetEpEngineCacheFromModel(const std::string& onnx_model_path,
                                   std::string& ep_engine_cache,
                                   const logging::Logger& logger) const {
    using namespace onnxruntime;
    std::shared_ptr<Model> model;
    ORT_RETURN_IF_ERROR(Model::Load(ToPathString(onnx_model_path), model, {}, logger));
    const auto& graph = model->MainGraph();
    ORT_RETURN_IF_ERROR(GetEpEngineCacheFromGraph(GraphViewer(graph), ep_engine_cache));

    return Status::OK();
  }

  Status GetEpEngineCacheFromGraph(const onnxruntime::GraphViewer& graph_viewer, std::string& ep_engine_cache) const {
    const auto& node = graph_viewer.Nodes().begin();
    NodeAttrHelper node_helper(*node);
    ep_engine_cache = node_helper.Get("ep_engine_cache", ep_engine_cache);
    return Status::OK();
  }

 private:
  bool is_qnn_ctx_model_ = false;
}; // QnnCacheModelHandler

}  // namespace qnn
}  // namespace onnxruntime
