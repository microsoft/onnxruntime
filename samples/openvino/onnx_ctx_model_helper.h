// Copyright (C) Intel Corporation
// Licensed under the MIT License

#pragma once

#include <sstream>
#include <string>
#include <memory>
#include "core/session/onnxruntime_c_api_ep.h"

namespace onnxruntime {
namespace openvino_ep {

class EPCtxHandler {
 public:
  EPCtxHandler() = default;
  EPCtxHandler(const EPCtxHandler&) = default;
//  Status ExportEPCtxModel(const GraphViewer& graph_viewer,
//                          const std::string& graph_name,
//                          const logging::Logger& logger,
//                          const bool& ep_context_embed_mode,
//                          const std::string& model_blob_str,
//                          const std::string& openvino_sdk_version,
//                          const std::string& device_type) const;
//  Status ImportBlobFromEPCtxModel(const GraphViewer& graph_viewer);
  bool CheckForOVEPCtxNode(const OrtGraphViewer* graph_viewer, std::string openvino_sdk_version) const;
  bool IsValidOVEPCtxGraph() const { return is_valid_ep_ctx_graph_; }
  [[nodiscard]] const std::shared_ptr<std::istringstream> GetModelBlobStream() const { return model_stream_; }

 private:
  bool is_valid_ep_ctx_graph_{false};
  std::shared_ptr<std::istringstream> model_stream_;
  static const OrtGraphApi* graph_api_;
};

}  // namespace openvino_ep
}  // namespace onnxruntime
