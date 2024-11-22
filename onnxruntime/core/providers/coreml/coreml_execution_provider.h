// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/inlined_containers.h"
#include "core/framework/execution_provider.h"
#include "core/framework/model_metadef_id_generator.h"

namespace onnxruntime {
namespace coreml {
class Model;
}

class CoreMLOptions {
 private:
  bool require_static_shape_{false};
  bool create_mlprogram_{false};
  bool enable_on_subgraph_{false};
  uint32_t compute_units_{0};

 public:
  explicit CoreMLOptions(uint32_t coreml_flags);

  CoreMLOptions(const ProviderOptions& options) {
    ValidateAndParseProviderOption(options);
  }
  bool RequireStaticShape() const { return require_static_shape_; }
  bool CreateMLProgram() const { return create_mlprogram_; }
  bool EnableOnSubgraph() const { return enable_on_subgraph_; }
  uint32_t ComputeUnits(uint32_t specific_flag = 0xffffffff) const { return compute_units_ & specific_flag; }

 private:
  void ValidateAndParseProviderOption(const ProviderOptions& options);
};

class CoreMLExecutionProvider : public IExecutionProvider {
 public:
  CoreMLExecutionProvider(const CoreMLOptions& options);
  virtual ~CoreMLExecutionProvider();

  std::vector<std::unique_ptr<ComputeCapability>>
  GetCapability(const onnxruntime::GraphViewer& graph_viewer,
                const IKernelLookup& /*kernel_lookup*/) const override;

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
};
}  // namespace onnxruntime
