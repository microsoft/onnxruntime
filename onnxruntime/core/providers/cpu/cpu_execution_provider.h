// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/framework/allocatormgr.h"
#include "core/framework/execution_provider.h"
#include "core/graph/constants.h"

namespace onnxruntime {

// Information needed to construct CPU execution providers.
struct CPUExecutionProviderInfo {
  bool create_arena{true};

  explicit CPUExecutionProviderInfo(bool use_arena)
      : create_arena(use_arena) {}

  CPUExecutionProviderInfo() = default;
};

using FuseRuleFn = std::function<void(const onnxruntime::GraphViewer&,
                                      std::vector<std::unique_ptr<ComputeCapability>>&)>;

// Logical device representation.
class CPUExecutionProvider : public IExecutionProvider {
 public:
  explicit CPUExecutionProvider(const CPUExecutionProviderInfo& info)
      : IExecutionProvider{onnxruntime::kCpuExecutionProvider} {
    DeviceAllocatorRegistrationInfo device_info{OrtMemTypeDefault,
                                                [](int) { return onnxruntime::make_unique<TAllocator>(); },
                                                std::numeric_limits<size_t>::max()};

    bool create_arena = info.create_arena;

#ifdef USE_JEMALLOC
#if defined(USE_MIMALLOC_ARENA_ALLOCATOR) || defined(USE_MIMALLOC_STL_ALLOCATOR)
#error jemalloc and mimalloc should not both be enabled
#endif
    //JEMalloc already has memory pool, so just use device allocator.
    create_arena = false;
#elif !(defined(__amd64__) || defined(_M_AMD64))
    //Disable Arena allocator for x86_32 build because it may run into infinite loop when integer overflow happens
    create_arena = false;
#endif

    InsertAllocator(CreateAllocator(device_info, 0, create_arena));
  }

  std::shared_ptr<KernelRegistry> GetKernelRegistry() const override;
  std::unique_ptr<IDataTransfer> GetDataTransfer() const override;

 private:
  std::vector<FuseRuleFn> fuse_rules_;
};
}  // namespace onnxruntime
