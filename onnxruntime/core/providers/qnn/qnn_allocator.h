// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <memory>
#include <mutex>

#include "core/providers/qnn/ort_api.h"
#include "core/providers/qnn/rpcmem_library.h"

namespace onnxruntime::qnn {

class OriginalHtpSharedMemoryAllocator : public IAllocator {
 public:
  // Gets the OrtMemoryInfo value that is associated with this allocator type.
  static OrtMemoryInfo AssociatedMemoryInfo();

  OriginalHtpSharedMemoryAllocator(std::shared_ptr<RpcMemLibrary> rpcmem_lib,
                                   const logging::Logger* logger = nullptr);

  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(OriginalHtpSharedMemoryAllocator);

  ~OriginalHtpSharedMemoryAllocator();

  // IAllocator overrides

  void* Alloc(size_t size) override;
  void Free(void* p) override;
  void GetStats(AllocatorStats* stats) override;

  struct SharedMemoryInfo {
    int fd;
    uint64_t offset;
    uint64_t total_size;
  };

  // Gets an allocation's shared memory info.
  // `allocation_address` identifies the allocation. It must be an address returned by Alloc() which has not yet been
  // freed.
  static Status GetAllocationSharedMemoryInfo(void* allocation_address,
                                              SharedMemoryInfo& allocation_info);

  using AllocationCleanUpFn = std::function<void(void* allocation_address)>;

  // Adds allocation clean up callback to call when the allocation is freed.
  // `allocation_address` identifies the allocation. It must be an address returned by Alloc() which has not yet been
  // freed.
  // `allocation_clean_up` is the clean up callback. The associated allocator takes ownership of the callback.
  static Status AddAllocationCleanUp(void* allocation_address, AllocationCleanUpFn&& allocation_clean_up);

 private:
  Status GetAllocationSharedMemoryInfoForThisAllocator(void* allocation_address,
                                                       SharedMemoryInfo& allocation_info);

  Status AddAllocationCleanUpForThisAllocator(void* allocation_address, AllocationCleanUpFn&& allocation_clean_up);

  struct AllocationRecord {
    size_t requested_size;
    SharedMemoryInfo shared_memory_info;
    InlinedVector<AllocationCleanUpFn, 1> clean_up_fns;
  };

  // allocation address -> corresponding allocation record
  InlinedHashMap<const void*, AllocationRecord> allocations_;
  std::mutex allocations_mutex_;  // synchronize access to allocations_

  std::shared_ptr<RpcMemLibrary> rpcmem_lib_;

  const logging::Logger& logger_;

  AllocatorStats stats_;
  std::mutex stats_mutex_;  // synchronize access to stats_
};

class DumbHtpSharedMemoryAllocator : public IAllocator {
 public:
  // Gets the OrtMemoryInfo value that is associated with this allocator type.
  static OrtMemoryInfo AssociatedMemoryInfo();

  DumbHtpSharedMemoryAllocator(std::shared_ptr<RpcMemLibrary> rpcmem_lib,
                               const logging::Logger* logger = nullptr);

  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(DumbHtpSharedMemoryAllocator);

  ~DumbHtpSharedMemoryAllocator();

  // IAllocator overrides

  void* Alloc(size_t size) override;
  void Free(void* p) override;
  // void GetStats(AllocatorStats* stats) override;

  struct SharedMemoryInfo {
    int fd;
    uint64_t offset;
    uint64_t total_size;
  };

  // Gets an allocation's shared memory info.
  // `allocation_address` identifies the allocation. It must be an address returned by Alloc() which has not yet been
  // freed.
  static Status GetAllocationSharedMemoryInfo(void* allocation_address,
                                              SharedMemoryInfo& allocation_info);

  using AllocationCleanUpFn = std::function<void(void* allocation_address)>;

  // Adds allocation clean up callback to call when the allocation is freed.
  // `allocation_address` identifies the allocation. It must be an address returned by Alloc() which has not yet been
  // freed.
  // `allocation_clean_up` is the clean up callback. The associated allocator takes ownership of the callback.
  static Status AddAllocationCleanUp(void* allocation_address, AllocationCleanUpFn&& allocation_clean_up);

 private:
  Status GetAllocationSharedMemoryInfoForThisAllocator(void* allocation_address,
                                                       SharedMemoryInfo& allocation_info);

  Status AddAllocationCleanUpForThisAllocator(void* allocation_address, AllocationCleanUpFn&& allocation_clean_up);

  void InitializeRegion();

  struct AllocationRecord {
    size_t requested_size;
    SharedMemoryInfo shared_memory_info;
    InlinedVector<AllocationCleanUpFn, 1> clean_up_fns;
  };

  std::mutex mutex_;  // synchronize access to allocator state

  // allocation address -> corresponding allocation record
  InlinedHashMap<const void*, AllocationRecord> allocations_;

  std::shared_ptr<RpcMemLibrary> rpcmem_lib_;

  const logging::Logger& logger_;

  // one way incremental allocation across a region

  size_t region_size_in_bytes_{0};
  void* region_base_address_{nullptr};
  int region_fd_{-1};

  size_t current_region_offset_{0};
};

using HtpSharedMemoryAllocator = DumbHtpSharedMemoryAllocator;

}  // namespace onnxruntime::qnn
