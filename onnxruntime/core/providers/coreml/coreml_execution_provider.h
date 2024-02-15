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

class CoreMLExecutionProvider : public IExecutionProvider {
 public:
  CoreMLExecutionProvider(uint32_t coreml_flags);
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
  uint32_t coreml_flags_;
  const int32_t coreml_version_;
  ModelMetadefIdGenerator metadef_id_generator_;

  // map of fused_node_name to compiled_coreml_model
  InlinedHashMap<std::string, std::unique_ptr<onnxruntime::coreml::Model>> coreml_models_;
};
}  // namespace onnxruntime
