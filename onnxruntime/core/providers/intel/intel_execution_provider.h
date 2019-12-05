// Copyright(C) 2019 Intel Corporation
// Licensed under the MIT License

#pragma once

#include "core/common/common.h"
#include "core/graph/graph_viewer.h"
#include "core/framework/execution_provider.h"
#include "core/framework/kernel_registry.h"
#include "core/framework/allocatormgr.h"
#include "backend_manager.h"
// #include "intel_graph.h"
//#include "intel_custom_op.h"
#include <map>

namespace ngraph {
namespace runtime {
class Backend;
}
}  // namespace ngraph

namespace onnxruntime {

constexpr const char* INTEL = "Intel";

// Information needed to construct OpenVINO execution providers.
struct IntelExecutionProviderInfo {
  const char* device{"CPU_FP32"};

  explicit IntelExecutionProviderInfo(const char* dev) : device(dev) {
  }
  IntelExecutionProviderInfo() {
  }
};

struct IntelEPFunctionState {
  AllocateFunc allocate_func = nullptr;
  DestroyFunc destroy_func = nullptr;
  AllocatorHandle allocator_handle = nullptr;
  std::shared_ptr<intel_ep::BackendManager> backend_manager;
};

// Logical device representation.
class IntelExecutionProvider : public IExecutionProvider {
 public:
  explicit IntelExecutionProvider(const IntelExecutionProviderInfo& info);
  ~IntelExecutionProvider() = default;

  std::vector<std::unique_ptr<ComputeCapability>>
  GetCapability(const onnxruntime::GraphViewer& graph_viewer,
                const std::vector<const KernelRegistry*>& kernel_registries) const override;

  Status Compile(const std::vector<onnxruntime::Node*>& fused_nodes,
                 std::vector<NodeComputeInfo>& node_compute_funcs) override;

  std::shared_ptr<KernelRegistry> GetKernelRegistry() const override {
    return std::make_shared<KernelRegistry>();
  }

  const void* GetExecutionHandle() const noexcept override {
    return nullptr;
  }
  //const onnxruntime::Node* fused_node_copy;
 private:
  IntelExecutionProviderInfo info_;
};

}  // namespace onnxruntime
