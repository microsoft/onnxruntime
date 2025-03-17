// Copyright (C) Intel Corporation
// Licensed under the MIT License

#pragma once

#include <map>
#include <algorithm>
#include <iostream>
#include <string>
#include <memory>
#include <vector>
#include <set>
#include <utility>

#include "core/providers/openvino/backend_manager.h"
#include "core/providers/openvino/contexts.h"

namespace onnxruntime {
namespace openvino_ep {

static void print_build_options() {
  std::cout << "[ERROR] INVALID DEVICE BUILD TYPE SPECIFIED" << std::endl;
  std::cout << "Specify the keyword HETERO (or) MULTI (or) AUTO followed by the devices in the order of priority "
            << "you want to build"
            << std::endl;
  std::cout << "The different hardware devices that can be added with HETERO/MULTI/AUTO build "
            << "are ['CPU','GPU','NPU','GPU.x'] where x = 0,1,2 and so on"
            << std::endl;
  std::cout << "An example of how to specify the HETERO or MULTI or AUTO build type. "
            << "Ex: HETERO:GPU,CPU  Ex: MULTI:GPU,CPU Ex: AUTO:GPU,CPU Ex: AUTO:GPU.0,CPU Ex: AUTO:GPU.1,CPU"
            << std::endl;
}

static std::vector<std::string> split(const std::string& s, char delim) {
  std::vector<std::string> result;
  std::stringstream ss(s);
  std::string item;

  while (getline(ss, item, delim)) {
    result.push_back(item);
  }
  return result;
}

// Logical device representation.
class OpenVINOExecutionProvider : public IExecutionProvider {
 public:
  explicit OpenVINOExecutionProvider(const ProviderInfo& info, std::shared_ptr<SharedContext> shared_context);
  ~OpenVINOExecutionProvider();

  std::vector<std::unique_ptr<ComputeCapability>>
  GetCapability(const GraphViewer& graph_viewer,
                const IKernelLookup& /*kernel_lookup*/,
                IResourceAccountant* /* resource_accountant */) const override;

  Status Compile(const std::vector<FusedNodeAndGraph>& fused_nodes,
                 std::vector<NodeComputeInfo>& node_compute_funcs) override;

  Status SetEpDynamicOptions(gsl::span<const char* const> /*keys*/,
                             gsl::span<const char* const> /*values*/) override;

  const void* GetExecutionHandle() const noexcept override {
    return nullptr;
  }

  const InlinedVector<const Node*> GetEpContextNodes() const override;

#ifdef USE_OVEP_NPU_MEMORY
  std::vector<AllocatorPtr> CreatePreferredAllocators() override;
#endif
 private:
  SessionContext session_context_;
  std::shared_ptr<SharedContext> shared_context_;
  std::list<BackendManager> backend_managers_;  // EP session owns the backend objects
  EPCtxHandler ep_ctx_handle_;
};

}  // namespace openvino_ep
}  // namespace onnxruntime
