// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/shared_library/provider_api.h"
#include "core/providers/migraphx/migraphx_call.h"
#include "core/providers/migraphx/migraphx_allocator.h"
#include "core/common/status.h"
#include "core/common/float16.h"
#include "core/providers/migraphx/gpu_data_transfer.h"

namespace onnxruntime {

#ifdef _DEBUG
void CheckDevice(const OrtDevice& device) {
  // check device to match at debug build
  // if it's expected to change, call hipSetDevice instead of the check
  int current_device;
  auto hip_err = hipGetDevice(&current_device);
  if (hip_err == hipSuccess) {
    ORT_ENFORCE(current_device == device.Id());
  }
}
#else
#define CheckDevice(...)
#endif

void* MIGraphXAllocator::Alloc(size_t size) {
  CheckDevice(Info().device);
  void* p = nullptr;
  if (size > 0) {
    HIP_CALL_THROW(hipMalloc(&p, size));
  }
  return p;
}

void MIGraphXAllocator::Free(void* p) {
  CheckDevice(Info().device);
  (void)hipFree(p);  // do not throw error since it's OK for hipFree to fail during shutdown
}

void* MIGraphXExternalAllocator::Alloc(size_t size) {
  void* p = nullptr;
  if (size > 0) {
    p = alloc_(size);
  }
  return p;
}

void MIGraphXExternalAllocator::Free(void* p) {
  free_(p);
  std::lock_guard<std::mutex> lock(lock_);
  auto it = reserved_.find(p);
  if (it != reserved_.end()) {
    reserved_.erase(it);
    if (empty_cache_ != nullptr) {
      empty_cache_();
    }
  }
}

void* MIGraphXExternalAllocator::Reserve(size_t size) {
  void* p = Alloc(size);
  if (!p) return nullptr;
  std::lock_guard<std::mutex> lock(lock_);
  ORT_ENFORCE(reserved_.find(p) == reserved_.end());
  reserved_.insert(p);
  return p;
}

void* MIGraphXPinnedAllocator::Alloc(size_t size) {
  void* p = nullptr;
  if (size > 0) {
    HIP_CALL_THROW(hipHostMalloc(&p, size));
  }
  return p;
}

void MIGraphXPinnedAllocator::Free(void* p) {
  HIP_CALL_THROW(hipHostFree(p));
}

}  // namespace onnxruntime
