// Copyright (C) Intel Corporation
// Licensed under the MIT License

#pragma once

#include <sstream>
#include <string>
#include <memory>

#include "core/providers/shared_library/provider_api.h"
#include "core/framework/execution_provider.h"
#include "ov_bin_manager.h"
#include "ov_shared_context.h"

namespace onnxruntime {
namespace openvino_ep {

class SharedBinManager;

struct ModelBlobWrapper {
  ModelBlobWrapper(std::unique_ptr<std::istream> stream, const ov::Tensor& tensor) : stream_(std::move(stream)), tensor_(tensor) {}
  std::unique_ptr<std::istream> stream_;
  ov::Tensor tensor_;  // May be empty if model blob is provided as stream only.
};

// Utilities to handle EPContext node export and parsing of an EPContext node
// to create the compiled_model object to infer on
static const char EPCONTEXT_OP[] = "EPContext";
static const char EMBED_MODE[] = "embed_mode";
static const char MAIN_CONTEXT[] = "main_context";
static const char PARTITION_NAME[] = "partition_name";
static const char EP_CACHE_CONTEXT[] = "ep_cache_context";
static const char EP_SDK_VER[] = "ep_sdk_version";
static const char SOURCE[] = "source";

class EPCtxHandler {
 public:
  EPCtxHandler(std::string ov_sdk_version, const logging::Logger& logger, std::shared_ptr<SharedContextManager> shared_context_manager);
  EPCtxHandler(const EPCtxHandler&) = delete;  // No copy constructor
  bool CheckForOVEPCtxNodeInGraph(const GraphViewer& subgraph_view) const;
  std::shared_ptr<SharedContext> GetSharedContextForEpContextSubgraph(const GraphViewer& subgraph_view, const std::filesystem::path& ep_context_path) const;
  bool CheckForOVEPCtxNode(const Node& node) const;
  Status AddOVEPCtxNodeToGraph(const GraphViewer& subgraph_view,
                               const std::string& graph_name,
                               const bool embed_mode,
                               std::string&& model_blob_str) const;
  std::unique_ptr<ModelBlobWrapper> GetModelBlobStream(const std::filesystem::path& so_context_file_path, const GraphViewer& subgraph_view) const;
  InlinedVector<const Node*> GetEPCtxNodes() const;
  bool CheckEPCacheContextAttribute(const GraphViewer& subgraph_view, const std::string& target_attr_extn) const;
  void Initialize(const std::vector<IExecutionProvider::FusedNodeAndGraph>& fused_nodes, const std::filesystem::path& ep_context_path);

 private:
  const std::string openvino_sdk_version_;
  std::unique_ptr<Model> epctx_model_;
  const logging::Logger& logger_;
  std::shared_ptr<SharedContextManager> shared_context_manager_;
};

}  // namespace openvino_ep
}  // namespace onnxruntime
