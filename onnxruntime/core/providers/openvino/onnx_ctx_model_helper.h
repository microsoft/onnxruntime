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
  EPCtxHandler(std::string ov_sdk_version, const logging::Logger& logger);
  EPCtxHandler(const EPCtxHandler&) = delete;  // No copy constructor
  Status ExportEPCtxModel(const std::string& model_name);
  bool CheckForOVEPCtxNodeInGraph(const GraphViewer& graph_viewer) const;
  bool CheckForOVEPCtxNode(const Node& node) const;
  Status AddOVEPCtxNodeToGraph(const GraphViewer& graph_viewer,
                               const std::string& graph_name,
                               const bool embed_mode,
                               std::string&& model_blob_str) const;
  std::unique_ptr<std::istream> GetModelBlobStream(const std::filesystem::path& so_context_file_path, const GraphViewer& graph_viewer) const;
  InlinedVector<const Node*> GetEPCtxNodes() const;

 private:
  const std::string openvino_sdk_version_;
  std::unique_ptr<Model> epctx_model_;
  const logging::Logger& logger_;
};

}  // namespace openvino_ep
}  // namespace onnxruntime
