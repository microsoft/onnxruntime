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
                                   DataLayout preferred_layout = DataLayout::NCHW);

  virtual ~InternalTestingExecutionProvider();

  std::vector<std::unique_ptr<ComputeCapability>>
  GetCapability(const onnxruntime::GraphViewer& graph_view,
                const IKernelLookup& /*kernel_lookup*/) const override;

  common::Status Compile(const std::vector<FusedNodeAndGraph>& fused_nodes,
                         std::vector<NodeComputeInfo>& node_compute_funcs) override;

  DataLayout GetPreferredLayout() const override;
  std::shared_ptr<KernelRegistry> GetKernelRegistry() const override;

  InternalTestingExecutionProvider& SetDebugOutput(bool debug_output) {
    debug_output_ = debug_output;
    return *this;
  }

  InternalTestingExecutionProvider& EnableStaticKernels() {
#if defined(ORT_MINIMAL_BUILD)
    ORT_THROW("Static kernels are not currently supported in a minimal build");
#else
    enable_static_kernels_ = true;
    return *this;

#endif
  }

  /// <summary>
  /// Request all nodes in GetCapability.
  /// If EnableStaticKernels has been called, use static kernels for all nodes.
  /// Otherwise compile the requested nodes.
  ///
  /// NOTE: If using static kernels the graph will not be executable as we don't have the kernel implementations
  /// so this is for testing model initialization components such as optimizers and layout transformation.
  /// </summary>
  InternalTestingExecutionProvider& TakeAllNodes() {
    take_all_nodes_ = true;
    return *this;
  }

  std::vector<AllocatorPtr> CreatePreferredAllocators() override;

 private:
  const std::string ep_name_;

  // List of operators that the EP will claim nodes for.
  // If take_all_nodes_ is true this is ignored.
  const std::unordered_set<std::string> ops_;

  // operators that we stop processing at.
  // all nodes of an operator in this list and all their downstream nodes will be skipped
  // e.g. NonMaxSuppression is the beginning of post-processing in an SSD model. It's unsupported for NNAPI,
  //      so from the NMS node on we want to use the CPU EP as the remaining work to do is far cheaper than
  //      the cost of going back to NNAPI.
  const std::unordered_set<std::string> stop_ops_;

  bool debug_output_{false};
  bool enable_static_kernels_{false};

  // request all nodes ignoring ops_ and stop_ops_.
  // if enabled_static_kernels_ use static kernels for them, otherwise compile.
  bool take_all_nodes_{false};

  DataLayout preferred_layout_;  // request all nodes

  // per-instance kernel registry so tests using static kernels don't clash.
  // shared_ptr as required by IExecutionProvider::GetKernelRegistry
  std::shared_ptr<KernelRegistry> kernel_registry_;
};

}  // namespace internal_testing_ep
}  // namespace onnxruntime
