// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "rocm_allocator.h"
#include "rocm_common.h"
#include "gpu_data_transfer.h"

namespace onnxruntime {

void ROCMAllocator::CheckDevice(bool throw_when_fail) const {
#ifndef NDEBUG
  // check device to match at debug build
  // if it's expected to change, call hipSetDevice instead of the check
  int current_device;
  auto hip_err = hipGetDevice(&current_device);
  if (hip_err == hipSuccess) {
    ORT_ENFORCE(current_device == Info().id);
  } else if (throw_when_fail) {
    HIP_CALL_THROW(hip_err);
  }
#else
  ORT_UNUSED_PARAMETER(throw_when_fail);
#endif
}

void ROCMAllocator::SetDevice(bool throw_when_fail) const {
  int current_device;
  auto hip_err = hipGetDevice(&current_device);
  if (hip_err == hipSuccess) {
    int allocator_device_id = Info().id;
    if (current_device != allocator_device_id) {
      hip_err = hipSetDevice(allocator_device_id);
    }
  }

  if (hip_err != hipSuccess && throw_when_fail) {
    HIP_CALL_THROW(hip_err);
  }
}

void* ROCMAllocator::Alloc(size_t size) {
  SetDevice(true);
  CheckDevice(true);
  void* p = nullptr;
  if (size > 0) {
    // BFCArena was updated recently to handle the exception and adjust the request size
    HIP_CALL_THROW(hipMalloc((void**)&p, size));
  }
  return p;
}

void ROCMAllocator::Free(void* p) {
  SetDevice(false);
  CheckDevice(false);  // ignore ROCM failure when free
  // do not throw error since it's OK for hipFree to fail during shutdown; void to silence nodiscard
  (void)hipFree(p);
}

void* ROCMExternalAllocator::Alloc(size_t size) {
  void* p = nullptr;
  if (size > 0) {
    p = alloc_(size);

    // review(codemzs): ORT_ENFORCE does not seem appropiate.
    ORT_ENFORCE(p != nullptr);
  }

  return p;
}

void ROCMExternalAllocator::Free(void* p) {
  free_(p);
  std::lock_guard<OrtMutex> lock(lock_);
  auto it = reserved_.find(p);
  if (it != reserved_.end()) {
    reserved_.erase(it);
    if (empty_cache_) empty_cache_();
  }
}

void* ROCMExternalAllocator::Reserve(size_t size) {
  void* p = Alloc(size);
  if (!p) return nullptr;
  std::lock_guard<OrtMutex> lock(lock_);
  ORT_ENFORCE(reserved_.find(p) == reserved_.end());
  reserved_.insert(p);
  return p;
}

void* ROCMPinnedAllocator::Alloc(size_t size) {
  void* p = nullptr;
  if (size > 0) {
    HIP_CALL_THROW(hipHostMalloc((void**)&p, size));
  }
  return p;
}

void ROCMPinnedAllocator::Free(void* p) {
  HIP_CALL_THROW(hipHostFree(p));
}

}  // namespace onnxruntime
