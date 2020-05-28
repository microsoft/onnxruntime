// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/framework/execution_provider.h"
#include "core/graph/onnx_protobuf.h"
#include "core/providers/nnapi/nnapi_builtin/model.h"
#include "core/providers/nnapi/nnapi_builtin/nnapi_lib/nnapi_implementation.h"

namespace onnxruntime {
class NnapiExecutionProvider : public IExecutionProvider {
 public:
  NnapiExecutionProvider();
  virtual ~NnapiExecutionProvider();

  std::vector<std::unique_ptr<ComputeCapability>>
  GetCapability(const onnxruntime::GraphViewer& graph,
                const std::vector<const KernelRegistry*>& /*kernel_registries*/) const override;
  common::Status Compile(const std::vector<onnxruntime::Node*>& fused_nodes,
                         std::vector<NodeComputeInfo>& node_compute_funcs) override;

 private:
  std::unordered_map<std::string, std::unique_ptr<onnxruntime::nnapi::Model>> nnapi_models_;
  const NnApi* nnapi_{nullptr};

  //std::vector<std::vector<int>> GetSupportedNodes(const ONNX_NAMESPACE::ModelProto& model_proto) const;
};
}  // namespace onnxruntime
