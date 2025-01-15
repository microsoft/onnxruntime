// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/qnn/qnn_allocator.h"

#include <cassert>
#include <cstddef>
#include <algorithm>

#include "core/common/common.h"
#include "core/common/safeint.h"
#include "core/mlas/inc/mlas.h"  // for MlasGetPreferredBufferAlignment()

namespace onnxruntime::qnn {

/**
 * HtpSharedMemoryAllocator allocation details
 *
 * The HTP shared memory allocator will allocate a block of shared memory larger than the amount requested in order to
 * hold some additional info.
 * Each allocation returned by HtpSharedMemoryAllocator::Alloc() is preceded by an AllocationHeader structure.
 *
 * For example, if Alloc(num_requested_bytes) is called, this is what the memory layout looks like:
 *   | AllocationHeader bytes | num_requested_bytes bytes |
 *                              ^- address returned by Alloc()
 *
 * The AllocationHeader can be used to obtain the owning allocator instance, which in turn can be used to do other
 * operations with that allocation, such as retrieving more info about the allocation.
 */

namespace {

struct AllocationHeader {
  static constexpr std::array<char, 8> kAllocationHeaderMarker{'o', 'r', 't', 'a', 'l', 'l', 'o', 'c'};

  // Marker bytes to verify as a sanity check.
  std::array<char, 8> marker;

  // Pointer to the allocating allocator instance.
  // Note: A critical assumption here is that the allocating allocator is not destroyed before the allocation is freed.
  HtpSharedMemoryAllocator* allocator_ptr;

  AllocationHeader(HtpSharedMemoryAllocator* allocator_ptr)
      : marker{kAllocationHeaderMarker},
        allocator_ptr{allocator_ptr} {
  }

  ~AllocationHeader() {
    marker.fill('\0');
    allocator_ptr = nullptr;
  }
};

size_t AllocationAlignment() {
  return std::max(alignof(AllocationHeader), MlasGetPreferredBufferAlignment());
}

size_t DivRoundUp(size_t a, size_t b) {  // TODO is there already a helper function somewhere for this?
  return (a + b - 1) / b;
}

bool IsAligned(const void* address, size_t alignment) {
  assert((alignment & alignment - 1) == 0);  // alignment must be a power of two
  return (reinterpret_cast<uintptr_t>(address) & (alignment - 1)) == 0;
}

size_t AllocationOffsetFromStartOfHeader() {
  const size_t allocation_alignment = AllocationAlignment();
  const size_t offset = DivRoundUp(sizeof(AllocationHeader), allocation_alignment) * allocation_alignment;
  return offset;
}

std::byte* GetAllocationHeaderAddress(void* allocation_address) {
  auto* allocation_header_address = reinterpret_cast<std::byte*>(allocation_address) - sizeof(AllocationHeader);
  return allocation_header_address;
}

AllocationHeader& ValidateAllocationAddressAndGetHeader(void* allocation_address) {
  const size_t allocation_alignment = AllocationAlignment();
  ORT_ENFORCE(IsAligned(allocation_address, allocation_alignment),
              "Allocation address (", allocation_address, ") does not have required alignment (",
              allocation_alignment, " bytes).");

  auto* allocation_header = reinterpret_cast<AllocationHeader*>(GetAllocationHeaderAddress(allocation_address));
  ORT_ENFORCE(allocation_header->marker == AllocationHeader::kAllocationHeaderMarker,
              "AllocationHeader for allocation address (", allocation_address,
              ") does not have the expected marker bytes.");

  return *allocation_header;
}

std::unique_ptr<void, void (*)(void*)> WrapSharedMemoryWithUniquePtr(void* shared_memory_raw,
                                                                     const RpcMemApi& rpcmem_api) {
  return {shared_memory_raw, rpcmem_api.free};
}

}  // namespace

OrtMemoryInfo HtpSharedMemoryAllocator::AssociatedMemoryInfo() {
  return OrtMemoryInfo{QNN_HTP_SHARED, OrtAllocatorType::OrtDeviceAllocator,
                       OrtDevice{OrtDevice::CPU, OrtDevice::MemType::QNN_HTP_SHARED, /* device_id */ 0},
                       /* id */ 0, OrtMemTypeDefault};
}

HtpSharedMemoryAllocator::HtpSharedMemoryAllocator(std::shared_ptr<RpcMemLibrary> rpcmem_lib,
                                                   const logging::Logger* logger)
    : IAllocator{AssociatedMemoryInfo()},
      rpcmem_lib_{std::move(rpcmem_lib)},
      logger_(logger != nullptr ? *logger : logging::LoggingManager::DefaultLogger()) {
  ORT_ENFORCE(rpcmem_lib_ != nullptr);
}

void* HtpSharedMemoryAllocator::Alloc(size_t requested_size) {
  const size_t allocation_offset = AllocationOffsetFromStartOfHeader();
  const size_t shared_memory_block_size_in_bytes = allocation_offset + requested_size;

  // rpcmem_alloc() has an int size parameter. make sure we don't overflow.
  // TODO switch to rpcmem_alloc2() which has size_t size parameter.
  // need to verify that rpcmem_alloc2() is available in all environments we care about.
  const SafeInt<int> shared_memory_block_size_in_bytes_int = shared_memory_block_size_in_bytes;

  // allocate shared memory
  void* shared_memory_raw = rpcmem_lib_->Api().alloc(rpcmem::RPCMEM_HEAP_ID_SYSTEM, rpcmem::RPCMEM_DEFAULT_FLAGS,
                                                     shared_memory_block_size_in_bytes_int);
  ORT_ENFORCE(shared_memory_raw != nullptr, "rpcmem_alloc() failed to allocate and returned nullptr.");
  auto shared_memory = WrapSharedMemoryWithUniquePtr(shared_memory_raw, rpcmem_lib_->Api());

  const size_t allocation_alignment = AllocationAlignment();
  ORT_ENFORCE(IsAligned(shared_memory_raw, allocation_alignment),
              "Shared memory address (", shared_memory_raw, ") does not have required alignment (",
              allocation_alignment, " bytes).");

  // get shared memory fd
  const auto shared_memory_fd = rpcmem_lib_->Api().to_fd(shared_memory.get());
  ORT_ENFORCE(shared_memory_fd != -1, "rpcmem_to_fd() returned invalid file descriptor.");

  std::byte* allocation_address = reinterpret_cast<std::byte*>(shared_memory_raw) + allocation_offset;

  // store allocation record
  {
    SharedMemoryInfo shared_memory_info{};
    shared_memory_info.fd = shared_memory_fd;
    shared_memory_info.offset = allocation_offset;
    shared_memory_info.total_size = shared_memory_block_size_in_bytes;

    AllocationRecord allocation_record{};
    allocation_record.shared_memory_info = std::move(shared_memory_info);

    std::scoped_lock g{allocations_mutex_};
    const bool inserted = allocations_.emplace(allocation_address, std::move(allocation_record)).second;
    ORT_ENFORCE(inserted, "Allocation record already exists for address (", allocation_address, ").");
  }

  // initialize header
  {
    std::byte* allocation_header_address = GetAllocationHeaderAddress(allocation_address);
    new (allocation_header_address) AllocationHeader(this);
  }

  shared_memory.release();
  return allocation_address;
}

void HtpSharedMemoryAllocator::Free(void* allocation_address) {
  if (allocation_address == nullptr) {
    return;
  }

  auto& allocation_header = ValidateAllocationAddressAndGetHeader(allocation_address);
  ORT_ENFORCE(allocation_header.allocator_ptr == this,
              "AllocationHeader points to a different allocator (", allocation_header.allocator_ptr,
              ") than this one (", this, ").");

  const auto allocation_node = [this, allocation_address]() {
    std::scoped_lock g{allocations_mutex_};
    return allocations_.extract(allocation_address);
  }();

  ORT_ENFORCE(!allocation_node.empty(), "Failed to get allocation info for address (", allocation_address, ").");

  // At this point, we have a valid allocation to free.
  // Avoid throwing exceptions as this may be running from a destructor.
  try {
    // take ownership of shared memory and free at end of scope
    auto shared_memory = WrapSharedMemoryWithUniquePtr(allocation_address, rpcmem_lib_->Api());

    // destroy header
    allocation_header.~AllocationHeader();

    // clean up allocation record
    const auto& allocation_record = allocation_node.mapped();
    for (auto& clean_up_fn : allocation_record.clean_up_fns) {
      // attempt to run each clean_up_fn even if exceptions are thrown
      try {
        clean_up_fn(allocation_address);
      } catch (const std::exception& e) {
        LOGS(logger_, ERROR) << "Caught exception while running clean up callback for address (" << allocation_address
                             << "): " << e.what();
      }
    }
  } catch (const std::exception& e) {
    LOGS(logger_, ERROR) << "Caught exception while freeing address (" << allocation_address << "): " << e.what();
  }
}

Status HtpSharedMemoryAllocator::GetAllocationSharedMemoryInfo(void* allocation_address,
                                                               SharedMemoryInfo& allocation_info) {
  auto& allocation_header = ValidateAllocationAddressAndGetHeader(allocation_address);
  return allocation_header.allocator_ptr->GetAllocationSharedMemoryInfoForThisAllocator(allocation_address,
                                                                                        allocation_info);
}

Status HtpSharedMemoryAllocator::AddAllocationCleanUp(void* allocation_address,
                                                      AllocationCleanUpFn&& allocation_clean_up) {
  auto& allocation_header = ValidateAllocationAddressAndGetHeader(allocation_address);
  return allocation_header.allocator_ptr->AddAllocationCleanUpForThisAllocator(allocation_address,
                                                                               std::move(allocation_clean_up));
}

Status HtpSharedMemoryAllocator::GetAllocationSharedMemoryInfoForThisAllocator(void* allocation_address,
                                                                               SharedMemoryInfo& allocation_info) {
  std::scoped_lock g{allocations_mutex_};
  const auto allocation_it = allocations_.find(allocation_address);
  ORT_RETURN_IF(allocation_it == allocations_.end(),
                "Failed to get allocation info for address (", allocation_address, ").");

  allocation_info = allocation_it->second.shared_memory_info;
  return Status::OK();
}

Status HtpSharedMemoryAllocator::AddAllocationCleanUpForThisAllocator(void* allocation_address,
                                                                      AllocationCleanUpFn&& allocation_clean_up) {
  ORT_RETURN_IF(allocation_clean_up == nullptr, "allocation_clean_up should not be empty.");

  std::scoped_lock g{allocations_mutex_};
  const auto allocation_it = allocations_.find(allocation_address);
  ORT_RETURN_IF(allocation_it == allocations_.end(),
                "Failed to get allocation info for address (", allocation_address, ").");

  auto& clean_up_fns = allocation_it->second.clean_up_fns;
  clean_up_fns.emplace_back(std::move(allocation_clean_up));
  return Status::OK();
}

}  // namespace onnxruntime::qnn
