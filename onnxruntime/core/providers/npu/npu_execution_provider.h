

#pragma once

#include "core/framework/execution_provider.h"
#include "core/graph/constants.h"

namespace onnxruntime {

// Information needed to construct NPU execution providers.
struct NPUExecutionProviderInfo {
  bool create_arena{true};

  explicit NPUExecutionProviderInfo(bool use_arena)
      : create_arena(use_arena) {}

  NPUExecutionProviderInfo() = default;
};

using FuseRuleFn = std::function<void(const onnxruntime::GraphViewer&,
                                      std::vector<std::unique_ptr<ComputeCapability>>&)>;

// Logical device representation.
class NPUExecutionProvider : public IExecutionProvider {
 public:
  explicit NPUExecutionProvider(const NPUExecutionProviderInfo& info);

  //std::shared_ptr<KernelRegistry> GetKernelRegistry() const override;
  //std::unique_ptr<IDataTransfer> GetDataTransfer() const override;
  //std::vector<AllocatorPtr> CreatePreferredAllocators() override;
  std::vector<std::unique_ptr<ComputeCapability>>
  GetCapability(const onnxruntime::GraphViewer& graph,
                const IKernelLookup& kernel_lookup) const override;

 private:
  NPUExecutionProviderInfo info_;
  std::vector<FuseRuleFn> fuse_rules_;
};

// Registers all available NPU kernels
//Status RegisterNPUKernels(KernelRegistry& kernel_registry);

}  // namespace onnxruntime
