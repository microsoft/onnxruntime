#pragma once

#include "core/framework/execution_provider.h"
#include "dnnlibrary/Model.h"

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

  Status CopyTensor(const Tensor& src, Tensor& dst) const override {
    ORT_UNUSED_PARAMETER(src);
    ORT_UNUSED_PARAMETER(dst);
    return Status(common::ONNXRUNTIME, common::FAIL, "Shouldn't reach here. CPUExecutionProvider doesn't support CopyTensor");
  }

  const void* GetExecutionHandle() const noexcept override {
    return nullptr;
  }

  std::shared_ptr<KernelRegistry> GetKernelRegistry() const override;
 private:
  std::unordered_map<std::string, std::unique_ptr<dnn::Model>> dnn_models_;
};
}  // namespace onnxruntime
