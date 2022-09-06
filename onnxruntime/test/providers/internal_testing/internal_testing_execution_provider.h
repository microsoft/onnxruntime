// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include <set>
#include "core/framework/execution_provider.h"

namespace onnxruntime {
namespace internal_testing_ep {

class InternalTestingExecutionProvider : public IExecutionProvider {
 public:
  InternalTestingExecutionProvider(const std::unordered_set<std::string>& ops,
                                   const std::unordered_set<std::string>& stop_ops = {},
                                   DataLayout preferred_layout = static_cast<DataLayout>(0));
  virtual ~InternalTestingExecutionProvider();

  std::vector<std::unique_ptr<ComputeCapability>>
  GetCapability(const onnxruntime::GraphViewer& graph_view,
                const std::vector<const KernelRegistry*>& /*kernel_registries*/) const override;

  common::Status Compile(const std::vector<FusedNodeAndGraph>& fused_nodes,
                         std::vector<NodeComputeInfo>& node_compute_funcs) override;

  DataLayout GetPreferredLayout() const override;
  std::shared_ptr<KernelRegistry> GetKernelRegistry() const override;

  AllocatorPtr GetAllocator(int device_id, OrtMemType mem_type) const override;
  void RegisterAllocator(AllocatorManager& /*allocator_manager*/) override;

  InternalTestingExecutionProvider& SetDebugOutput(bool debug_output) {
    debug_output_ = debug_output;
    return *this;
  }

  InternalTestingExecutionProvider& EnableStaticKernels() {
    enable_static_kernels_ = true;
    return *this;
  }

 private:
  const std::string ep_name_;

  // List of operators that the EP will claim nodes for
  const std::unordered_set<std::string> ops_;

  // operators that we stop processing at.
  // all nodes of an operator in this list and all their downstream nodes will be skipped
  // e.g. NonMaxSuppression is the beginning of post-processing in an SSD model. It's unsupported for NNAPI,
  //      so from the NMS node on we want to use the CPU EP as the remaining work to do is far cheaper than
  //      the cost of going back to NNAPI.
  const std::unordered_set<std::string> stop_ops_;

  bool debug_output_{false};
  bool enable_static_kernels_{false};
  DataLayout preferred_layout_;

  // used for testing allocator sharing as a few EPs (e.g. CUDA, TRT, TVM) override GetAllocator and have a local
  // AllocatorPtr that can get out of sync with the allocator lists in the base IExecutionProvider
  AllocatorPtr local_allocator_;
};

}  // namespace internal_testing_ep
}  // namespace onnxruntime
