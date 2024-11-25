// Copyright (C) Intel Corporation
// Licensed under the MIT License

#pragma once

#include <sstream>
#include <string>
#include <memory>

#include "core/providers/shared_library/provider_api.h"

namespace onnxruntime {
namespace openvino_ep {

// Utilities to handle EPContext node export and parsing of an EPContext node
// to create the compiled_model object to infer on
static const char EPCONTEXT_OP[] = "EPContext";
static const char EMBED_MODE[] = "embed_mode";
static const char EP_CACHE_CONTEXT[] = "ep_cache_context";
static const char EP_SDK_VER[] = "ep_sdk_version";
static const char SOURCE[] = "source";

class EPCtxHandler {
 public:
  EPCtxHandler() = default;
  EPCtxHandler(const EPCtxHandler&) = default;
  Status ExportEPCtxModel(const GraphViewer& graph_viewer,
                          const std::string& graph_name,
                          const logging::Logger& logger,
                          const bool& ep_context_embed_mode,
                          std::string&& model_blob_str,
                          const std::string& openvino_sdk_version) const;
  Status ImportBlobFromEPCtxModel(const GraphViewer& graph_viewer, bool& ep_context_embed_mode);
  bool CheckForOVEPCtxNode(const GraphViewer& graph_viewer, std::string openvino_sdk_version) const;
  bool IsValidOVEPCtxGraph() const { return is_valid_ep_ctx_graph_; }
  [[nodiscard]] const std::shared_ptr<std::istringstream> GetModelBlobStream() const { return model_stream_; }

 private:
  bool is_valid_ep_ctx_graph_{false};
  std::shared_ptr<std::istringstream> model_stream_;
};

}  // namespace openvino_ep
}  // namespace onnxruntime
