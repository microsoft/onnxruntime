// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/coreml/coreml_options.h"
#include "core/framework/execution_provider.h"
#include "core/framework/model_metadef_id_generator.h"
#include "core/framework/compute_capability.h"

namespace onnxruntime {
namespace coreml {
class Model;
}

class CoreMLExecutionProvider : public IExecutionProvider {
 public:
  CoreMLExecutionProvider(const CoreMLOptions& options);
  virtual ~CoreMLExecutionProvider();

  std::vector<std::unique_ptr<ComputeCapability>>
  GetCapability(const onnxruntime::GraphViewer& graph_viewer,
                const IKernelLookup& /*kernel_lookup*/,
                const GraphOptimizerRegistry& /* graph_optimizer_registry */,
                IResourceAccountant* resource_accountant) const override;

#if !defined(ORT_MINIMAL_BUILD) || defined(ORT_EXTENDED_MINIMAL_BUILD)
  common::Status Compile(const std::vector<FusedNodeAndGraph>& fused_nodes,
                         std::vector<NodeComputeInfo>& node_compute_funcs) override;
#endif

 private:
  // The bit flags which define bool options for COREML EP, bits are defined as
  // COREMLFlags in include/onnxruntime/core/providers/coreml/coreml_provider_factory.h
  CoreMLOptions coreml_options_;
  const int32_t coreml_version_;
  ModelMetadefIdGenerator metadef_id_generator_;

  // map of fused_node_name to compiled_coreml_model
  InlinedHashMap<std::string, std::unique_ptr<onnxruntime::coreml::Model>> coreml_models_;

  // Helper function to check if a type is supported by CoreML
  bool IsSupportedInputOrOutputType(int32_t type) const;

  bool ProcessIncompatibleNodes(const onnxruntime::GraphViewer& graph_viewer,
                                std::unordered_set<NodeIndex>& partition_nodes,
                                IndexedSubGraph::MetaDef* meta_def,
                                const bool is_input,
                                const logging::Logger& logger) const;

  void FilterIncompatibleEdgeNodesFromPartition(IndexedSubGraph& partition,
                                                const onnxruntime::GraphViewer& graph_viewer,
                                                const logging::Logger& logger) const;

  std::vector<std::unique_ptr<ComputeCapability>> FilterIncompatibleEdgeNodesFromPartitions(
      std::vector<std::unique_ptr<ComputeCapability>>&& capabilities,
      const onnxruntime::GraphViewer& graph_viewer,
      const logging::Logger& logger) const;
};
}  // namespace onnxruntime
