// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/shared_library/provider_api.h"
#include "migraphx_call.h"
#include "hip_allocator.h"
#include "core/common/status.h"
#include "core/framework/float16.h"
#include "core/common/status.h"
#include "gpu_data_transfer.h"

namespace onnxruntime {

void HIPAllocator::CheckDevice() const {
#ifndef NDEBUG
  // check device to match at debug build
  // if it's expected to change, call hipSetDevice instead of the check
  int current_device;
  auto hip_err = hipGetDevice(&current_device);
  if (hip_err == hipSuccess) {
    ORT_ENFORCE(current_device == Info().id);
  }
#endif
}

void* HIPAllocator::Alloc(size_t size) {
  CheckDevice();
  void* p = nullptr;
  if (size > 0) {
    HIP_CALL_THROW(hipMalloc((void**)&p, size));
  }
  return p;
}

void HIPAllocator::Free(void* p) {
  CheckDevice();
  (void)hipFree(p);  // do not throw error since it's OK for hipFree to fail during shutdown
}

void* HIPExternalAllocator::Alloc(size_t size) {
  void* p = nullptr;
  if (size > 0) {
    p = alloc_(size);

    // review(codemzs): ORT_ENFORCE does not seem appropiate.
    ORT_ENFORCE(p != nullptr);
  }

  return p;
}

void HIPExternalAllocator::Free(void* p) {
  free_(p);
  std::lock_guard<OrtMutex> lock(lock_);
  auto it = reserved_.find(p);
  if (it != reserved_.end()) {
    reserved_.erase(it);
    if (empty_cache_) empty_cache_();
  }
}

void* HIPExternalAllocator::Reserve(size_t size) {
  void* p = Alloc(size);
  if (!p) return nullptr;
  std::lock_guard<OrtMutex> lock(lock_);
  ORT_ENFORCE(reserved_.find(p) == reserved_.end());
  reserved_.insert(p);
  return p;
}

void* HIPPinnedAllocator::Alloc(size_t size) {
  void* p = nullptr;
  if (size > 0) {
    HIP_CALL_THROW(hipHostMalloc((void**)&p, size));
  }
  return p;
}

void HIPPinnedAllocator::Free(void* p) {
  HIP_CALL_THROW(hipHostFree(p));
}

}  // namespace onnxruntime
