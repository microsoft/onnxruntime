// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License

#pragma once

#include "core/framework/execution_provider.h"
#include "core/platform/ort_mutex.h"
#include <map>
#include "migraphx_inc.h"

namespace onnxruntime {

// Information needed to construct amdmigraphx execution providers.
struct MIGraphXExecutionProviderInfo {
  std::string target_device;
  int device_id {0};
};

// Information to construct kernel function state.
struct MIGraphXFuncState {
  AllocateFunc allocate_func = nullptr;
  DestroyFunc release_func = nullptr;
  AllocatorHandle allocate_handle = nullptr;
  migraphx::program prog{};
  migraphx::target t{};
  std::unordered_map<std::size_t, std::size_t> input_indexes;
  std::unordered_map<std::size_t, std::size_t> output_indexes;
  OrtMutex* mgx_mu_ptr = nullptr;
};

// Logical device representation.
class MIGraphXExecutionProvider : public IExecutionProvider {
 public:
  explicit MIGraphXExecutionProvider(const MIGraphXExecutionProviderInfo& info);
  ~MIGraphXExecutionProvider() = default;

  std::vector<std::unique_ptr<ComputeCapability>>
  GetCapability(const onnxruntime::GraphViewer& graph_viewer,
                const std::vector<const KernelRegistry*>& kernel_registries) const override;

  Status Compile(const std::vector<onnxruntime::Node*>& fused_nodes,
                 std::vector<NodeComputeInfo>& node_compute_funcs) override;

  virtual std::shared_ptr<KernelRegistry> GetKernelRegistry() const override;
  std::unique_ptr<onnxruntime::IDataTransfer> GetDataTransfer() const override;
  AllocatorPtr GetAllocator(int id, OrtMemType mem_type) const override;

private:
  int device_id_;
  migraphx::target t_; 
  OrtMutex mgx_mu_;

  std::unordered_map<std::string, migraphx::program> map_progs_;
  std::unordered_map<std::string, std::unordered_map<std::size_t, std::size_t>> map_input_index_;
  std::unordered_map<std::string, std::unordered_map<std::size_t, std::size_t>> map_output_index_;

  AllocatorPtr allocator_;
};

}
