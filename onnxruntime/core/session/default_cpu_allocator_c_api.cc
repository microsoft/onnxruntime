// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/framework/utils.h"
#include "core/framework/allocator.h"
#include "core/mlas/inc/mlas.h"
#include "core/session/abi_key_value_pairs.h"
#include "core/session/allocator_adapters.h"
#include "core/session/onnxruntime_cxx_api.h"
#include "core/session/ort_apis.h"
#include "core/framework/error_code_helper.h"

struct OrtDefaultCpuAllocator : onnxruntime::OrtAllocatorImpl {
  OrtDefaultCpuAllocator() {
    OrtAllocator::version = ORT_API_VERSION;
    OrtAllocator::Alloc =
        [](OrtAllocator* this_, size_t size) { return static_cast<OrtDefaultCpuAllocator*>(this_)->Alloc(size); };
    OrtAllocator::Free =
        [](OrtAllocator* this_, void* p) { static_cast<OrtDefaultCpuAllocator*>(this_)->Free(p); };
    OrtAllocator::Info =
        [](const OrtAllocator* this_) { return static_cast<const OrtDefaultCpuAllocator*>(this_)->Info(); };
    OrtAllocator::Reserve =
        [](OrtAllocator* this_, size_t size) { return static_cast<OrtDefaultCpuAllocator*>(this_)->Alloc(size); };
    OrtAllocator::GetStats =
        [](const OrtAllocator* /*this_*/, OrtKeyValuePairs** stats) noexcept -> OrtStatusPtr {
      // Default allocator does not support stats, return an empty OrtKeyValuePairs.
      auto kvp = std::make_unique<OrtKeyValuePairs>();
      *stats = reinterpret_cast<OrtKeyValuePairs*>(kvp.release());
      return nullptr;
    };
    Ort::ThrowOnError(OrtApis::CreateCpuMemoryInfo(OrtDeviceAllocator, OrtMemTypeDefault, &cpu_memory_info));
  }

  ~OrtDefaultCpuAllocator() { OrtApis::ReleaseMemoryInfo(cpu_memory_info); }

  void* Alloc(size_t size) {
    return onnxruntime::AllocatorDefaultAllocAligned(size, alignment_);
  }
  void Free(void* p) {
    return onnxruntime::AllocatorDefaultFreeAligned(p, alignment_);
  }
  const OrtMemoryInfo* Info() const {
    return cpu_memory_info;
  }

  ORT_DISALLOW_COPY_AND_ASSIGNMENT(OrtDefaultCpuAllocator);

 private:
  OrtMemoryInfo* cpu_memory_info;
  const size_t alignment_ = MlasGetPreferredBufferAlignment();
};

ORT_API_STATUS_IMPL(OrtApis::GetAllocatorWithDefaultOptions, _Outptr_ OrtAllocator** out) {
  API_IMPL_BEGIN
  static OrtDefaultCpuAllocator ort_default_cpu_allocator;
  *out = &ort_default_cpu_allocator;
  return nullptr;
  API_IMPL_END
}
