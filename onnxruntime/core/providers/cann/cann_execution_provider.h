// Copyright (c) Microsoft Corporation. All rights reserved.
// Copyright (c) Huawei. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <memory>
#include <vector>
#include <string>
#include <unordered_map>

#include "core/providers/shared_library/provider_api.h"
#include "core/framework/arena_extend_strategy.h"
#include "core/framework/execution_provider.h"
#include "core/platform/ort_mutex.h"
#include "core/providers/cann/cann_execution_provider_info.h"
#include "core/providers/cann/cann_inc.h"
#include "core/providers/cann/cann_utils.h"
#include "core/providers/cann/cann_graph.h"

namespace onnxruntime {

// Information to construct kernel function state.
struct CannFuncState {
  AllocateFunc allocate_func = nullptr;
  DestroyFunc release_func = nullptr;
  AllocatorHandle allocate_handle = nullptr;
  std::string node_name;
};

class CANNExecutionProvider : public IExecutionProvider {
 public:
  explicit CANNExecutionProvider(const CANNExecutionProviderInfo& info);
  virtual ~CANNExecutionProvider();

  Status OnRunStart() override;

  template <typename T>
  Status Fill(Tensor* y, void* addr, aclrtStream stream) const {
    return cann::Fill<T>(y, addr, stream);
  }

  template <typename T>
  Status Broadcast(const Tensor* x, Tensor* y, void* addr, aclrtStream stream) const {
    return cann::Broadcast<T>(x, y, addr, stream);
  }

  int GetDeviceId() const override { return info_.device_id; }
  std::shared_ptr<KernelRegistry> GetKernelRegistry() const override;
  std::unique_ptr<onnxruntime::IDataTransfer> GetDataTransfer() const override;

  std::unique_ptr<IndexedSubGraph> GetSubGraph(
      const std::vector<std::size_t>& graph_nodes_index,
      const GraphViewer& graph_viewer) const;

  std::vector<std::unique_ptr<ComputeCapability>> GetCapability(
      const onnxruntime::GraphViewer& graph_viewer,
      const IKernelLookup& kernel_lookup) const override;

  Status Compile(const std::vector<FusedNodeAndGraph>& fused_nodes_and_graphs,
                 std::vector<NodeComputeInfo>& node_compute_funcs) override;

  ProviderOptions GetProviderOptions() const override {
    return CANNExecutionProviderInfo::ToProviderOptions(info_);
  }

  static AllocatorPtr CreateCannAllocator(OrtDevice::DeviceId device_id, size_t npu_mem_limit,
                                          ArenaExtendStrategy arena_extend_strategy,
                                          OrtArenaCfg* default_memory_arena_cfg);

  void RegisterStreamHandlers(IStreamCommandHandleRegistry& stream_handle_registry, AllocatorMap&) const override;

  OrtDevice GetOrtDeviceByMemType(OrtMemType mem_type) const override;

  std::vector<AllocatorPtr> CreatePreferredAllocators() override;

 private:
  CANNExecutionProviderInfo info_;
  const char* soc_name_ = nullptr;

  std::unordered_map<std::string, uint32_t> modelIDs_;
  std::unordered_map<std::string, std::string> models_;
  std::unordered_map<std::string, std::unordered_map<std::size_t, std::string>> names_;
};

}  // namespace onnxruntime
