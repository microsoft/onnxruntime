// Copyright(C) 2019 Intel Corporation
// Licensed under the MIT License

#pragma once

#include "core/framework/execution_provider.h"
#include <map>

namespace ngraph {
  namespace runtime {
    class Backend;
  }
}

namespace ONNX_NAMESPACE {
  class AttributeProto;
}

namespace onnxruntime {

// Information needed to construct nGraph execution providers.
struct NGRAPHExecutionProviderInfo {
  const std::string ng_backend_type;
};

// Logical device representation.
class NGRAPHExecutionProvider : public IExecutionProvider {
 public:
  explicit NGRAPHExecutionProvider(const NGRAPHExecutionProviderInfo& info);
  ~NGRAPHExecutionProvider() = default;

  Status CopyTensor(const Tensor& src, Tensor& dst) const override;

  std::vector<std::unique_ptr<ComputeCapability>>
  GetCapability(const onnxruntime::GraphViewer& graph_viewer,
                const std::vector<const KernelRegistry*>& kernel_registries) const override;

  Status Compile(const std::vector<onnxruntime::Node*>& fused_nodes,
                 std::vector<NodeComputeInfo>& node_compute_funcs) override;

 private:
  std::shared_ptr<ngraph::runtime::Backend> ng_backend_;
  mutable std::unordered_map<std::string, std::shared_ptr<ONNX_NAMESPACE::AttributeProto>> graph_initializers_;
};

}
