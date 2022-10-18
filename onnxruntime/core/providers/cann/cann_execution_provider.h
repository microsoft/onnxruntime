// Copyright (c) Microsoft Corporation. All rights reserved.
// Copyright (c) Huawei. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <memory>
#include <set>
#include <vector>

#include "core/providers/shared_library/provider_api.h"
#include "core/framework/allocatormgr.h"
#include "core/framework/arena_extend_strategy.h"
#include "core/framework/execution_provider.h"
#include "core/platform/ort_mutex.h"
#include "core/providers/cann/cann_execution_provider_info.h"
#include "core/providers/cann/cann_inc.h"

namespace onnxruntime {

class CANNExecutionProvider : public IExecutionProvider {
 public:
  explicit CANNExecutionProvider(const CANNExecutionProviderInfo& info);
  virtual ~CANNExecutionProvider();

  Status OnRunStart() override;

  Status OnRunEnd(bool sync_stream) override;

  void* GetComputeStream() const override { return static_cast<void*>(stream_); }

  template <typename T>
  IAllocatorUniquePtr<T> GetScratchBuffer(size_t count_or_bytes) const {
    if (count_or_bytes == 0)
      return nullptr;

    return IAllocator::MakeUniquePtr<T>(GetAllocator(info_.device_id, OrtMemTypeDefault), count_or_bytes);
  }

  std::shared_ptr<KernelRegistry> GetKernelRegistry() const override;
  std::unique_ptr<onnxruntime::IDataTransfer> GetDataTransfer() const override;

  std::vector<std::unique_ptr<ComputeCapability>> GetCapability(
      const onnxruntime::GraphViewer& graph,
      const IKernelLookup& kernel_lookup) const override;

  ProviderOptions GetProviderOptions() const override {
    return CANNExecutionProviderInfo::ToProviderOptions(info_);
  }

  AllocatorPtr GetAllocator(int id, OrtMemType mem_type) const override;
  void RegisterAllocator(AllocatorManager& allocator_manager) override;

 private:
  CANNExecutionProviderInfo info_;
  aclrtStream stream_ = nullptr;
};

}  // namespace onnxruntime
