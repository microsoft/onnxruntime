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
static const std::string EPCONTEXT_OP = "EPContext";

bool IsQnnCtxModel(const onnxruntime::GraphViewer& graph_viewer);

Status CreateNodeArgs(const std::vector<std::string>& names,
                      const std::unordered_map<std::string, OnnxTensorInfo>& tensor_info_table,
                      std::vector<NodeArg*>& node_args,
                      onnxruntime::Graph& graph);

Status GenerateCtxCacheOnnxModel(const std::string& model_name, const std::string& graph_name,
                                 const std::vector<std::string>& input_names,
                                 const std::unordered_map<std::string, OnnxTensorInfo>& inputs_info,
                                 const std::vector<std::string>& output_names,
                                 const std::unordered_map<std::string, OnnxTensorInfo>& outputs_info,
                                 const std::string& model_description,
                                 const std::string& sdk_build_version,
                                 const std::string& file_path,
                                 unsigned char* buffer,
                                 uint64_t buffer_size,
                                 const logging::Logger& logger);

Status GetEpEngineCacheFromModel(const std::string& onnx_model_path,
                                 std::string& ep_engine_cache,
                                 const logging::Logger& logger);

std::string GetEpEngineCacheFromGraph(const onnxruntime::GraphViewer& graph_viewer);

class QnnCacheModelHandler {
 public:
  QnnCacheModelHandler() {
  }
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(QnnCacheModelHandler);

  Status GetEpEngineCache(const onnxruntime::GraphViewer& graph_viewer,
                          const std::string& onnx_model_path,
                          bool is_qnn_ctx_model,
                          bool is_ctx_cache_file_exist,
                          std::string& ep_engine_cache,
                          const logging::Logger& logger) const {
    if (is_qnn_ctx_model) {
      ep_engine_cache = GetEpEngineCacheFromGraph(graph_viewer);
    } else if (is_ctx_cache_file_exist) {
      ORT_RETURN_IF_ERROR(GetEpEngineCacheFromModel(onnx_model_path, ep_engine_cache, logger));
    }

    return Status::OK();
  }

  Status GetMetadataFromEpEngineCacheModel(const std::string& onnx_model_path,
                                           std::string& model_name,
                                           std::string& model_description,
                                           std::string& graph_partition_name,
                                           std::string& cache_source,
                                           const logging::Logger& logger);

 private:
  bool is_metadata_ready_ = false;
  std::string model_name_ = "";
  std::string model_description_ = "";
  std::string graph_partition_name_ = "";
  std::string cache_source_ = "";
};  // QnnCacheModelHandler

}  // namespace qnn
}  // namespace onnxruntime
