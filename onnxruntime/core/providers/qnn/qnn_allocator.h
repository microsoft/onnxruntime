// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <memory>
#include <mutex>

#include "core/common/common.h"
#include "core/common/inlined_containers.h"
#include "core/common/status.h"
#include "core/framework/allocator.h"
#include "core/providers/qnn/rpcmem_library.h"

namespace onnxruntime::qnn {

class HtpSharedMemoryAllocator : public IAllocator {
 public:
  // Gets the OrtMemoryInfo value that is associated with this allocator type.
  static OrtMemoryInfo AssociatedMemoryInfo();

  HtpSharedMemoryAllocator(std::shared_ptr<RpcMemLibrary> rpcmem_lib);

  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(HtpSharedMemoryAllocator);

  // IAllocator overrides

  void* Alloc(size_t size) override;
  void Free(void* p) override;
  // void GetStats(AllocatorStats* stats) override;

  struct SharedMemoryInfo {
    int fd;
    uint64_t offset;
    uint64_t total_size;
  };

  // Get an allocation's shared memory info.
  // `allocation_address` must be an address returned by Alloc() which has not yet been freed.
  static Status GetAllocationSharedMemoryInfo(void* allocation_address,
                                              SharedMemoryInfo& allocation_info);

  using AllocationCleanUpFn = std::function<void(void* allocation_address)>;

  // Add allocation clean up callback to call when the allocation is freed.
  // `allocation_address` identifies the allocation. It must be an address returned by Alloc() which has not yet been freed.
  // `allocation_clean_up` is the clean up callback. This call takes ownership.
  static Status AddAllocationCleanUp(void* allocation_address, AllocationCleanUpFn&& allocation_clean_up);

 private:
  Status GetAllocationSharedMemoryInfoForThisAllocator(void* allocation_address,
                                                       SharedMemoryInfo& allocation_info);

  Status AddAllocationCleanUpForThisAllocator(void* allocation_address, AllocationCleanUpFn&& allocation_clean_up);

  struct AllocationRecord {
    SharedMemoryInfo shared_memory_info;
    InlinedVector<AllocationCleanUpFn, 1> clean_up_fns;
  };

  // allocation address -> corresponding allocation record
  InlinedHashMap<const void*, AllocationRecord> allocations_;
  std::mutex allocations_mutex_;  // synchronize access to allocation_

  std::shared_ptr<RpcMemLibrary> rpcmem_lib_;
};

}  // namespace onnxruntime::qnn
