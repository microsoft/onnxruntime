// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/graph/onnx_protobuf.h"
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

#ifdef USE_JEMALLOC
#if defined(USE_MIMALLOC_ARENA_ALLOCATOR) || defined(USE_MIMALLOC_STL_ALLOCATOR)
#error jemalloc and mimalloc should not both be enabled
#endif

    ORT_UNUSED_PARAMETER(info);
    //JEMalloc already has memory pool, so just use device allocator.
    InsertAllocator(device_info.factory(0));
#else
//Disable Arena allocator for x86_32 build because it may run into infinite loop when integer overflow happens
#if defined(__amd64__) || defined(_M_AMD64)
    if (info.create_arena)
      InsertAllocator(CreateAllocator(device_info));
    else
#else
    ORT_UNUSED_PARAMETER(info);
#endif
      InsertAllocator(device_info.factory(0));
#endif
  }

  std::shared_ptr<KernelRegistry> GetKernelRegistry() const override;
  std::unique_ptr<IDataTransfer> GetDataTransfer() const override;

 private:
  std::vector<FuseRuleFn> fuse_rules_;
};
}  // namespace onnxruntime
