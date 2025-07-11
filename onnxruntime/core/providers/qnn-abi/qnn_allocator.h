// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <memory>
#include <mutex>

#include "core/common/common.h"
#include "core/common/inlined_containers.h"
#include "core/common/logging/logging.h"
#include "core/providers/qnn-abi/rpcmem_library.h"
#include "core/session/onnxruntime_c_api.h"

namespace onnxruntime::qnn {

class HtpSharedMemoryAllocator : public OrtAllocator {
 public:
  // Gets the OrtMemoryInfo value that is associated with this allocator type.
  static OrtMemoryInfo AssociatedMemoryInfo();

  HtpSharedMemoryAllocator(const OrtMemoryInfo* mem_info,
                           std::shared_ptr<RpcMemLibrary> rpcmem_lib,
                           const logging::Logger* logger = nullptr)
    : memory_info_{mem_info},
      rpcmem_lib_{std::move(rpcmem_lib)},
      logger_(logger != nullptr ? *logger : logging::LoggingManager::DefaultLogger()) {
  ORT_ENFORCE(rpcmem_lib_ != nullptr);

  Alloc = AllocImpl;
  Free = FreeImpl;
  Info = InfoImpl;
  Reserve = AllocImpl;
}

  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(HtpSharedMemoryAllocator);

  // OrtAllocator implementations.
  static void* ORT_API_CALL AllocImpl(struct OrtAllocator* this_, size_t size);

  static void ORT_API_CALL FreeImpl(struct OrtAllocator* this_, void* p);

  static const struct OrtMemoryInfo* ORT_API_CALL InfoImpl(const struct OrtAllocator* this_) {
    const HtpSharedMemoryAllocator& impl = *static_cast<const HtpSharedMemoryAllocator*>(this_);
    return impl.memory_info_;
  }

  struct SharedMemoryInfo {
    int fd;
    uint64_t offset;
    uint64_t total_size;
  };

  // Gets an allocation's shared memory info.
  // `address_within_allocation` identifies the allocation. It must be an address within an allocation returned by
  // Alloc() which has not yet been freed.
  static Status GetAllocationSharedMemoryInfo(void* address_within_allocation,
                                              SharedMemoryInfo& allocation_info);

  // Allocation clean up callback signature.
  // For a given allocation, any added clean up callbacks will be called with the allocation's base address when the
  // allocation is freed.
  using AllocationCleanUpFn = std::function<void(void* allocation_base_address)>;

  // Adds allocation clean up callback to call when the allocation is freed.
  // `address_within_allocation` identifies the allocation. It must be an address within an allocation returned by
  // Alloc() which has not yet been freed.
  // `allocation_clean_up` is the clean up callback. The associated allocator takes ownership of the callback.
  static Status AddAllocationCleanUp(void* address_within_allocation, AllocationCleanUpFn&& allocation_clean_up);

 private:
  Status GetAllocationSharedMemoryInfoForThisAllocator(void* allocation_base_address,
                                                       SharedMemoryInfo& allocation_info);

  Status AddAllocationCleanUpForThisAllocator(void* allocation_base_address, AllocationCleanUpFn&& allocation_clean_up);

  struct AllocationRecord {
    SharedMemoryInfo shared_memory_info;
    InlinedVector<AllocationCleanUpFn, 1> clean_up_fns;
  };

  // allocation address -> corresponding allocation record
  InlinedHashMap<const void*, AllocationRecord> allocations_;
  std::mutex allocations_mutex_;  // synchronize access to allocations_

  const OrtMemoryInfo* memory_info_;
  std::shared_ptr<RpcMemLibrary> rpcmem_lib_;
  const logging::Logger& logger_;
};

}  // namespace onnxruntime::qnn
