// Copyright (C) Intel Corporation
// Licensed under the MIT License

#pragma once

#include <sstream>

#include "core/providers/shared_library/provider_api.h"

namespace onnxruntime {
namespace openvino_ep {

// Utilities to handle EPContext node export and parsing of an EPContext node
// to create the compiled_model object to infer on
static const std::string EPCONTEXT_OP = "EPContext";
static const std::string EMBED_MODE = "embed_mode";
static const std::string EP_CACHE_CONTEXT = "ep_cache_context";
static const std::string EP_SDK_VER = "ep_sdk_version";
static const std::string SOURCE = "source";

class EPCtxHandler {
 public:
  EPCtxHandler() = default;
  EPCtxHandler(const EPCtxHandler&) = default;
  Status ExportEPCtxModel(const GraphViewer& graph_viewer,
                          const onnxruntime::Node& fused_node,
                          const logging::Logger& logger,
                          const bool& ep_context_embed_mode,
                          const std::string& model_blob_str,
                          const std::string& openvino_sdk_version) const;
  Status ImportBlobFromEPCtxModel(const GraphViewer& graph_viewer);
  bool CheckForOVEPCtxNode(const GraphViewer& graph_viewer, std::string openvino_sdk_version) const;
  bool IsValidOVEPCtxGraph() const {return is_valid_ep_ctx_graph_;};
  [[nodiscard]] const std::string& GetModelBlobString() const { return blob_serialized_; }

 private:
  bool is_valid_ep_ctx_graph_{false};
  std::string blob_serialized_{};
};

}  // namespace openvino_ep
}  // namespace onnxruntime
