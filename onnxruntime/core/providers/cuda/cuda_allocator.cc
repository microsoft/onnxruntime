// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "cuda_allocator.h"
#include "cuda_common.h"
#include "gpu_data_transfer.h"

namespace onnxruntime {

std::shared_ptr<CudaDeviceMemoryBudget> GetCudaDeviceMemoryBudget(OrtDevice::DeviceId device_id, size_t limit) {
  static std::mutex registry_mutex;
  static InlinedHashMap<OrtDevice::DeviceId, std::shared_ptr<CudaDeviceMemoryBudget>> budgets;
  std::lock_guard<std::mutex> lock(registry_mutex);
  auto& budget = budgets[device_id];
  if (!budget) budget = std::make_shared<CudaDeviceMemoryBudget>();
  budget->Configure(limit);
  return budget;
}

CUDAAllocator::CUDAAllocator(OrtDevice::DeviceId device_id, const char* name,
                             std::shared_ptr<SessionBudgetAccount> account)
    : IAllocator(OrtMemoryInfo(name, OrtAllocatorType::OrtDeviceAllocator,
                               OrtDevice(OrtDevice::GPU, OrtDevice::MemType::DEFAULT,
                                         OrtDevice::VendorIds::NVIDIA, device_id),
                               OrtMemTypeDefault)),
      budget_(GetCudaDeviceMemoryBudget(device_id)),
      account_(std::move(account)) {
  if (budget_) budget_->AddOwner();
}

CUDAAllocator::~CUDAAllocator() {
  if (budget_) budget_->RemoveOwner();
}

void CUDAAllocator::CheckDevice(bool throw_when_fail) const {
#ifndef NDEBUG
  // check device to match at debug build
  // if it's expected to change, call cudaSetDevice instead of the check
  int current_device;
  auto cuda_err = cudaGetDevice(&current_device);
  if (cuda_err == cudaSuccess) {
    ORT_ENFORCE(current_device == Info().device.Id());
  } else if (throw_when_fail) {
    CUDA_CALL_THROW(cuda_err);
  }
#else
  ORT_UNUSED_PARAMETER(throw_when_fail);
#endif
}

void CUDAAllocator::SetDevice(bool throw_when_fail) const {
  int current_device;
  auto cuda_err = cudaGetDevice(&current_device);
  if (cuda_err == cudaSuccess) {
    int allocator_device_id = Info().device.Id();
    if (current_device != allocator_device_id) {
      cuda_err = cudaSetDevice(allocator_device_id);
    }
  }

  if (cuda_err != cudaSuccess && throw_when_fail) {
    CUDA_CALL_THROW(cuda_err);
  }
}

void* CUDAAllocator::Alloc(size_t size) {
  SetDevice(true);
  CheckDevice(true);
  if (size == 0) return nullptr;
  if (account_) {
    auto reservation = account_->Reserve(size);
    void* allocation = nullptr;
    CUDA_CALL_THROW(cudaMalloc(&allocation, size));
    try {
      std::lock_guard<std::mutex> lock(allocations_mutex_);
      allocations_.emplace(allocation, AllocationRecord{size, AllocationState::kCommitted});
    } catch (...) {
      if (cudaFree(allocation) != cudaSuccess) reservation.Commit();
      throw;
    }
    reservation.Commit();
    return allocation;
  }
  auto reservation = budget_->Reserve(size);
  void* allocation = nullptr;
  CUDA_CALL_THROW(cudaMalloc(&allocation, size));
  try {
    std::lock_guard<std::mutex> lock(allocations_mutex_);
    allocations_.emplace(allocation, AllocationRecord{size, AllocationState::kCommitted});
  } catch (...) {
    if (cudaFree(allocation) != cudaSuccess) reservation.Commit();
    throw;
  }
  reservation.Commit();
  return allocation;
}

void CUDAAllocator::Free(void* p) {
  if (p == nullptr) return;
  SetDevice(false);
  CheckDevice(false);  // ignore CUDA failure when free
  size_t bytes = 0;
  bool tracked = false;
  {
    std::lock_guard<std::mutex> lock(allocations_mutex_);
    const auto it = allocations_.find(p);
    if (it != allocations_.end()) {
      if (it->second.state != AllocationState::kCommitted) {
        return;  // Prevent concurrent double-free or re-free of quarantined pointer (F4).
      }
      it->second.state = AllocationState::kFreeing;
      bytes = it->second.bytes;
      tracked = true;
      if (account_) {
        account_->BeginFree(bytes);
      } else {
        budget_->BeginFree(bytes);
      }
    }
  }

  // Call driver free outside allocations_mutex_ (F4).
  const bool driver_ok = (cudaFree(p) == cudaSuccess);

  if (tracked) {
    std::lock_guard<std::mutex> lock(allocations_mutex_);
    const auto it = allocations_.find(p);
    if (account_) {
      account_->FinishFree(bytes, driver_ok);
    } else {
      budget_->FinishFree(bytes, driver_ok);
    }
    if (it != allocations_.end()) {
      if (driver_ok) {
        allocations_.erase(it);
      } else {
        it->second.state = AllocationState::kQuarantined;
      }
    }
  }
}

void* CUDAExternalAllocator::Alloc(size_t size) {
  void* p = nullptr;
  if (size > 0) {
    p = alloc_(size);

    // review(codemzs): ORT_ENFORCE does not seem appropriate.
    ORT_ENFORCE(p != nullptr);
  }

  return p;
}

void CUDAExternalAllocator::Free(void* p) {
  free_(p);
  std::lock_guard<std::mutex> lock(lock_);
  auto it = reserved_.find(p);
  if (it != reserved_.end()) {
    reserved_.erase(it);
    if (empty_cache_) empty_cache_();
  }
}

void* CUDAExternalAllocator::Reserve(size_t size) {
  void* p = Alloc(size);
  if (!p) return nullptr;
  std::lock_guard<std::mutex> lock(lock_);
  ORT_ENFORCE(reserved_.find(p) == reserved_.end());
  reserved_.insert(p);
  return p;
}

void* CUDAPinnedAllocator::Alloc(size_t size) {
  void* p = nullptr;
  if (size > 0) {
    CUDA_CALL_THROW(cudaMallocHost((void**)&p, size));
  }
  return p;
}

void CUDAPinnedAllocator::Free(void* p) {
  CUDA_CALL_THROW(cudaFreeHost(p));
}

}  // namespace onnxruntime
