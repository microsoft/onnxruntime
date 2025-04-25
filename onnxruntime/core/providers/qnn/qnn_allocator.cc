// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/qnn/qnn_allocator.h"

#include <cassert>
#include <cstddef>
#include <algorithm>
#include <shared_mutex>

#include "core/providers/qnn/ort_api.h"

namespace onnxruntime::qnn {

namespace {

size_t AllocationAlignment() {
  constexpr size_t min_allocation_alignment = 64;  // Equal to MlasGetPreferredBufferAlignment()
  return min_allocation_alignment;
}

bool IsAligned(const void* address, size_t alignment) {
  assert((alignment & (alignment - 1)) == 0);  // alignment must be a power of two
  return (reinterpret_cast<uintptr_t>(address) & (alignment - 1)) == 0;
}

std::unique_ptr<void, void (*)(void*)> WrapSharedMemoryWithUniquePtr(void* shared_memory_raw,
                                                                     const RpcMemApi& rpcmem_api) {
  return {shared_memory_raw, rpcmem_api.free};
}

class AllocationTracker {
 public:
  struct Record {
    void* base_address;
    size_t size_in_bytes;
    gsl::not_null<HtpSharedMemoryAllocator*> allocator;
  };

  bool RegisterAllocation(void* base_address, size_t size, HtpSharedMemoryAllocator& allocator);
  bool UnregisterAllocation(void* base_address);
  std::optional<Record> LookUp(void* address_within_allocation);

 private:
  std::map<const void*, Record> records_;
  std::shared_mutex records_mutex_;
};

bool AllocationTracker::RegisterAllocation(void* base_address, size_t size_in_bytes,
                                           HtpSharedMemoryAllocator& allocator) {
  Record record{base_address, size_in_bytes, &allocator};

  std::unique_lock write_lock{records_mutex_};
  const bool registered = records_.emplace(base_address, std::move(record)).second;
  return registered;
}

bool AllocationTracker::UnregisterAllocation(void* base_address) {
  std::unique_lock write_lock{records_mutex_};
  const bool unregistered = records_.erase(base_address) == 1;
  return unregistered;
}

std::optional<AllocationTracker::Record> AllocationTracker::LookUp(void* address_within_allocation) {
  std::shared_lock read_lock{records_mutex_};

  // Look for a record where `address_within_allocation` falls within the range:
  //   [`record.base_address`, `record.base_address` + `record.size_in_bytes`)

  // First, find the first record with a base address greater than `address_within_allocation`, or the end of the
  // container if no such record exists.
  const auto first_record_with_larger_base_address_it = records_.upper_bound(address_within_allocation);

  // The previous record should have the greatest base address that is not greater than `address_within_allocation`.
  // Make sure it exists.
  if (first_record_with_larger_base_address_it == records_.begin()) {
    return std::nullopt;
  }

  const auto record_it = std::prev(first_record_with_larger_base_address_it);

  const auto record = record_it->second;
  assert(address_within_allocation >= record.base_address);

  // Verify that `address_within_allocation` is within the upper end of the range.
  if (reinterpret_cast<std::byte*>(address_within_allocation) >=
      reinterpret_cast<std::byte*>(record.base_address) + record.size_in_bytes) {
    return std::nullopt;
  }

  return record;
}

AllocationTracker& GlobalAllocationTracker() {
  static AllocationTracker allocation_tracker{};
  return allocation_tracker;
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
  const size_t shared_memory_block_size_in_bytes = requested_size;

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

  std::byte* allocation_address = reinterpret_cast<std::byte*>(shared_memory_raw);

  // store allocation record
  {
    SharedMemoryInfo shared_memory_info{};
    shared_memory_info.fd = shared_memory_fd;
    shared_memory_info.offset = 0;
    shared_memory_info.total_size = shared_memory_block_size_in_bytes;

    AllocationRecord allocation_record{};
    allocation_record.shared_memory_info = std::move(shared_memory_info);

    std::scoped_lock g{allocations_mutex_};
    const bool inserted = allocations_.emplace(allocation_address, std::move(allocation_record)).second;
    ORT_ENFORCE(inserted, "Allocation record already exists for address (", allocation_address, ").");
  }

  // register with global allocation tracker
  {
    const bool registered = GlobalAllocationTracker().RegisterAllocation(allocation_address,
                                                                         shared_memory_block_size_in_bytes,
                                                                         *this);

    ORT_ENFORCE(registered, "Attempted to register allocation but it is already tracked for address (",
                allocation_address, ").");
  }

  shared_memory.release();
  return allocation_address;
}

void HtpSharedMemoryAllocator::Free(void* allocation_address) {
  if (allocation_address == nullptr) {
    return;
  }

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

    // unregister with global allocation tracker
    {
      const bool unregistered = GlobalAllocationTracker().UnregisterAllocation(allocation_address);
      if (!unregistered) {
        LOGS(logger_, ERROR) << "Attempted to deregister allocation but it is untracked for address ("
                             << allocation_address << ").";
      }
    }

    // clean up allocation record
    const auto& allocation_record = allocation_node.mapped();
    for (const auto& [addr, cleanup_fn] : allocation_record.address_to_cleanup_fn) {
      if (cleanup_fn) {
        try {
          cleanup_fn(allocation_address);
        } catch (const std::exception& e) {
          LOGS(logger_, ERROR) << "Caught exception while running general clean up callback for address ("
                               << allocation_address << "): " << e.what();
        }
      }
    }
  } catch (const std::exception& e) {
    LOGS(logger_, ERROR) << "Caught exception while freeing address (" << allocation_address << "): " << e.what();
  }
}

Status HtpSharedMemoryAllocator::GetAllocationSharedMemoryInfo(void* address_within_allocation,
                                                               SharedMemoryInfo& shared_memory_info_out) {
  const auto tracked_record = GlobalAllocationTracker().LookUp(address_within_allocation);
  ORT_RETURN_IF_NOT(tracked_record.has_value(), "Failed to look up tracked allocation.");

  void* const base_address = tracked_record->base_address;
  SharedMemoryInfo shared_memory_info{};
  ORT_RETURN_IF_ERROR(tracked_record->allocator->GetAllocationSharedMemoryInfoForThisAllocator(
      base_address, shared_memory_info));

  // adjust `shared_memory_info.offset` for `address_within_allocation`
  const auto offset_from_base = std::distance(reinterpret_cast<std::byte*>(base_address),
                                              reinterpret_cast<std::byte*>(address_within_allocation));

  shared_memory_info.offset += offset_from_base;

  shared_memory_info_out = std::move(shared_memory_info);
  return Status::OK();
}

Status HtpSharedMemoryAllocator::AddAllocationCleanUp(void* address_within_allocation,
                                                      AllocationCleanUpFn&& allocation_clean_up) {
  const auto tracked_record = GlobalAllocationTracker().LookUp(address_within_allocation);
  ORT_RETURN_IF_NOT(tracked_record.has_value(), "Failed to look up tracked allocation.");

  void* const base_address = tracked_record->base_address;
  return tracked_record->allocator->AddAllocationCleanUpForThisAllocator(base_address, address_within_allocation,
                                                                         std::move(allocation_clean_up));
}

Status HtpSharedMemoryAllocator::GetAllocationSharedMemoryInfoForThisAllocator(void* allocation_base_address,
                                                                               SharedMemoryInfo& allocation_info) {
  std::scoped_lock g{allocations_mutex_};
  const auto allocation_it = allocations_.find(allocation_base_address);
  ORT_RETURN_IF(allocation_it == allocations_.end(),
                "Failed to get allocation info for address (", allocation_base_address, ").");

  allocation_info = allocation_it->second.shared_memory_info;
  return Status::OK();
}

Status HtpSharedMemoryAllocator::AddAllocationCleanUpForThisAllocator(
    void* allocation_base_address,
    void* address_within_allocation,
    AllocationCleanUpFn&& allocation_clean_up) {
  ORT_RETURN_IF(allocation_clean_up == nullptr, "allocation_clean_up should not be empty.");

  std::scoped_lock g{allocations_mutex_};
  const auto allocation_it = allocations_.find(allocation_base_address);
  ORT_RETURN_IF(allocation_it == allocations_.end(),
                "Failed to get allocation info for address (", allocation_base_address, ").");

  auto& allocation_record = allocation_it->second;

  // Store the address-specific cleanup function in the map
  auto result = allocation_record.address_to_cleanup_fn.emplace(address_within_allocation,
                                                                std::move(allocation_clean_up));

  return Status::OK();
}

Status HtpSharedMemoryAllocator::RemoveAndExecuteAllocationCleanUp(void* address_within_allocation) {
  const auto tracked_record = GlobalAllocationTracker().LookUp(address_within_allocation);
  ORT_RETURN_IF_NOT(tracked_record.has_value(), "Failed to look up tracked allocation.");

  void* const base_address = tracked_record->base_address;
  return tracked_record->allocator->RemoveAndExecuteAllocationCleanUpForThisAllocator(base_address, address_within_allocation);
}

Status HtpSharedMemoryAllocator::RemoveAndExecuteAllocationCleanUpForThisAllocator(
    void* allocation_base_address, void* address_within_allocation) {
  AllocationCleanUpFn cleanup_fn_to_execute;

  {
    std::scoped_lock g{allocations_mutex_};
    const auto allocation_it = allocations_.find(allocation_base_address);
    ORT_RETURN_IF(allocation_it == allocations_.end(),
                  "Failed to get allocation info for address (", allocation_base_address, ").");

    auto& address_to_cleanup_fn = allocation_it->second.address_to_cleanup_fn;

    // Look for a cleanup function associated with the specific address
    auto cleanup_it = address_to_cleanup_fn.find(address_within_allocation);
    if (cleanup_it != address_to_cleanup_fn.end()) {
      // Found a specific cleanup function for this address
      cleanup_fn_to_execute = std::move(cleanup_it->second);
      address_to_cleanup_fn.erase(cleanup_it);
    }
  }

  // Execute the cleanup function outside of the lock to avoid potential deadlocks
  if (cleanup_fn_to_execute) {
    try {
      cleanup_fn_to_execute(allocation_base_address);
    } catch (const std::exception& e) {
      LOGS(logger_, ERROR) << "Caught exception while executing clean up callback for address ("
                           << address_within_allocation << "): " << e.what();
      // Even though the function threw an exception, we still removed it successfully,
      // so we consider this a success from the API's perspective.
    }
  }

  return Status::OK();
}

}  // namespace onnxruntime::qnn
