// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/framework/allocatormgr.h"
#include "core/framework/execution_provider.h"
#include "core/graph/constants.h"

namespace onnxruntime {

// Information needed to construct CPU execution providers.
struct WasmExecutionProviderInfo {
  bool create_arena{false};

  explicit WasmExecutionProviderInfo(bool use_arena)
      : create_arena(use_arena) {}

  WasmExecutionProviderInfo() = default;
};

using FuseRuleFn = std::function<void(const onnxruntime::GraphViewer&,
                                      std::vector<std::unique_ptr<ComputeCapability>>&)>;

// Logical device representation.
class WasmExecutionProvider : public IExecutionProvider {
 public:
  explicit WasmExecutionProvider(const WasmExecutionProviderInfo& info)
      : IExecutionProvider{onnxruntime::kCpuExecutionProvider} {
    bool create_arena = info.create_arena;

#ifdef USE_JEMALLOC
#if defined(USE_MIMALLOC_ARENA_ALLOCATOR) || defined(USE_MIMALLOC_STL_ALLOCATOR)
#error jemalloc and mimalloc should not both be enabled
#endif
    //JEMalloc already has memory pool, so just use device allocator.
    create_arena = false;
#elif !(defined(__amd64__) || defined(_M_AMD64) || defined(__aarch64__) || defined(_M_ARM64))
    //Disable Arena allocator for x86_32 build because it may run into infinite loop when integer overflow happens
    create_arena = false;
#endif

    AllocatorCreationInfo device_info{[](int) { return onnxruntime::make_unique<TAllocator>(); },
                                      0, create_arena};

    InsertAllocator(CreateAllocator(device_info));
  }

  std::shared_ptr<KernelRegistry> GetKernelRegistry() const override;
  std::unique_ptr<IDataTransfer> GetDataTransfer() const override;

 private:
  std::vector<FuseRuleFn> fuse_rules_;
};
}  // namespace onnxruntime
