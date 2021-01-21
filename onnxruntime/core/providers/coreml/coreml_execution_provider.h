// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/framework/execution_provider.h"
#include "core/providers/coreml/coreml_provider_factory.h"

namespace onnxruntime {
namespace coreml {
class Model;
}

class CoreMLExecutionProvider : public IExecutionProvider {
 public:
  CoreMLExecutionProvider();
  virtual ~CoreMLExecutionProvider();

  std::vector<std::unique_ptr<ComputeCapability>>
  GetCapability(const onnxruntime::GraphViewer& graph_view,
                const std::vector<const KernelRegistry*>& /*kernel_registries*/) const override;

  // we implement the Compile that takes FusedNodeAndGraph instances
  FusionStyle GetFusionStyle() const override { return FusionStyle::FilteredGraphViewer; }

#if !defined(ORT_MINIMAL_BUILD) || defined(ORT_EXTENDED_MINIMAL_BUILD)
  common::Status Compile(const std::vector<FusedNodeAndGraph>& fused_nodes,
                         std::vector<NodeComputeInfo>& node_compute_funcs) override;
#endif

 private:
  // unique counter to name each fused kernel across the entire model
  mutable int metadef_id_{0};

  // <fused_node_name, <coreml_model_file_path, compiled_coreml_model>>
  std::unordered_map<std::string, std::pair<std::string, std::unique_ptr<onnxruntime::coreml::Model>>> coreml_models_;
};
}  // namespace onnxruntime
