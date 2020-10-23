// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/framework/execution_provider.h"
#include "core/providers/nnapi/nnapi_builtin/model.h"
#include "core/providers/nnapi/nnapi_provider_factory.h"

namespace onnxruntime {

class NnapiExecutionProvider : public IExecutionProvider {
 public:
  NnapiExecutionProvider(unsigned long nnapi_flags);
  virtual ~NnapiExecutionProvider();

  std::vector<std::unique_ptr<ComputeCapability>>
  GetCapability(const onnxruntime::GraphViewer& graph_view,
                const std::vector<const KernelRegistry*>& /*kernel_registries*/) const override;
  common::Status Compile(const std::vector<onnxruntime::Node*>& fused_nodes,
                         std::vector<NodeComputeInfo>& node_compute_funcs) override;
  const std::bitset<NNAPI_FLAG_MAX>& GetNNAPIFlags() const { return nnapi_flags_; }

 private:
  std::unordered_map<std::string, std::unique_ptr<onnxruntime::nnapi::Model>> nnapi_models_;

  // The bit set which defined bool options for NNAPI EP, bits are defined as
  // NNAPIFlag in include/onnxruntime/core/providers/nnapi/nnapi_provider_factory.h
  std::bitset<NNAPI_FLAG_MAX> nnapi_flags_;
};
}  // namespace onnxruntime
