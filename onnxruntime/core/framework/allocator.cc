// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/common/safeint.h"
#include "core/framework/allocator.h"
#include "core/mlas/inc/mlas.h"
#include "core/framework/utils.h"
#include "core/session/ort_apis.h"
#include <cstdlib>
#include <sstream>

#if defined(USE_MIMALLOC)
#include <mimalloc.h>
#endif

#include "core/framework/bfc_arena.h"

namespace onnxruntime {

// private helper for calculation so SafeInt usage doesn't bleed into the public allocator.h header
bool IAllocator::CalcMemSizeForArrayWithAlignment(size_t nmemb, size_t size, size_t alignment, size_t* out) noexcept {
  bool ok = true;

  ORT_TRY {
    SafeInt<size_t> alloc_size(size);
    if (alignment == 0) {
      *out = alloc_size * nmemb;
    } else {
      size_t alignment_mask = alignment - 1;
      *out = (alloc_size * nmemb + alignment_mask) & ~static_cast<size_t>(alignment_mask);
    }
  }
  ORT_CATCH(const OnnxRuntimeException& ex) {
    // overflow in calculating the size thrown by SafeInt.
    ORT_HANDLE_EXCEPTION([&]() {
      LOGS_DEFAULT(ERROR) << ex.what();
      ok = false;
    });
  }
  return ok;
}

#ifdef USE_MIMALLOC
void* AllocatorDefaultAlloc(size_t size) {
  const size_t alignment = MlasGetPreferredBufferAlignment();
  if (size <= 0) return nullptr;
  size += MLAS_SYMM_QGEMM_BUF_OVERRUN;
  void* p;
#if defined(_MSC_VER)
  p = mi_malloc_aligned(size, alignment);
  if (p == nullptr)
    ORT_THROW_EX(std::bad_alloc);
#elif defined(_LIBCPP_SGX_CONFIG)
  p = mi_memalign(alignment, size);
  if (p == nullptr)
    ORT_THROW_EX(std::bad_alloc);
#else
  int ret = mi_posix_memalign(&p, alignment, size);
  if (ret != 0)
    ORT_THROW_EX(std::bad_alloc);
#endif
  return p;
}

void AllocatorDefaultFree(void* p) {
#if defined(_MSC_VER)
  const size_t alignment = MlasGetPreferredBufferAlignment();
  mi_free_aligned(p, alignment);
#else
  mi_free(p);
#endif
}

#else
void* AllocatorDefaultAlloc(size_t size) {
  const size_t alignment = MlasGetPreferredBufferAlignment();
  if (size <= 0) return nullptr;
  size += MLAS_SYMM_QGEMM_BUF_OVERRUN;
  void* p;
#if _MSC_VER
  p = _aligned_malloc(size, alignment);
  if (p == nullptr)
    ORT_THROW_EX(std::bad_alloc);
#elif defined(_LIBCPP_SGX_CONFIG)
  p = memalign(alignment, size);
  if (p == nullptr)
    ORT_THROW_EX(std::bad_alloc);
#else
  int ret = posix_memalign(&p, alignment, size);
  if (ret != 0)
    ORT_THROW_EX(std::bad_alloc);
#endif
  return p;
}

void AllocatorDefaultFree(void* p) {
#if _MSC_VER
  _aligned_free(p);
#else
  free(p);
#endif
}

#endif  // USE_MIMALLOC

void* CPUAllocator::Alloc(size_t size) {
  return AllocatorDefaultAlloc(size);
}

void CPUAllocator::Free(void* p) {
  AllocatorDefaultFree(p);
}

void* AllocateBufferWithOptions(IAllocator& alloc, size_t size, bool use_reserve, Stream* stream, WaitNotificationFn wait_fn) {
  if (use_reserve)
    return alloc.Reserve(size);
  if (stream && alloc.Info().alloc_type == OrtArenaAllocator) {
#ifdef ORT_ENABLE_STREAM
    auto* stream_aware_alloc = StreamAwareArena::FromBFCArena(static_cast<BFCArena&>(alloc));
    if (stream_aware_alloc) {
      return stream_aware_alloc->AllocOnStream(size, stream, wait_fn);
    }
#else
    ORT_UNUSED_PARAMETER(wait_fn);
#endif  // ORT_ENABLE_STREAM
  }
  return alloc.Alloc(size);
}
}  // namespace onnxruntime

std::ostream& operator<<(std::ostream& out, const OrtMemoryInfo& info) { return (out << info.ToString()); }
#if defined(_MSC_VER) && !defined(__clang__)
#pragma warning(push)
#pragma warning(disable : 26409)
#endif
ORT_API_STATUS_IMPL(OrtApis::CreateMemoryInfo, _In_ const char* name1, enum OrtAllocatorType type, int id1,
                    enum OrtMemType mem_type1, _Outptr_ OrtMemoryInfo** out) {
  if (strcmp(name1, onnxruntime::CPU) == 0) {
    *out = new OrtMemoryInfo(onnxruntime::CPU, type, OrtDevice(), id1, mem_type1);
  } else if (strcmp(name1, onnxruntime::CUDA) == 0) {
    *out = new OrtMemoryInfo(
        onnxruntime::CUDA, type, OrtDevice(OrtDevice::GPU, OrtDevice::MemType::DEFAULT, static_cast<OrtDevice::DeviceId>(id1)), id1,
        mem_type1);
  } else if (strcmp(name1, onnxruntime::CUDA_PINNED) == 0) {
    *out = new OrtMemoryInfo(
        onnxruntime::CUDA_PINNED, type, OrtDevice(OrtDevice::CPU, OrtDevice::MemType::CUDA_PINNED, static_cast<OrtDevice::DeviceId>(id1)),
        id1, mem_type1);
  } else if (strcmp(name1, onnxruntime::OpenVINO_GPU) == 0) {
    *out = new OrtMemoryInfo(
        onnxruntime::OpenVINO_GPU, type, OrtDevice(OrtDevice::GPU, OrtDevice::MemType::DEFAULT, static_cast<OrtDevice::DeviceId>(id1)),
        id1, mem_type1);
  } else if (strcmp(name1, onnxruntime::DML) == 0) {
    *out = new OrtMemoryInfo(
        onnxruntime::DML, type, OrtDevice(OrtDevice::GPU, OrtDevice::MemType::DEFAULT, static_cast<OrtDevice::DeviceId>(id1)),
        id1, mem_type1);
  } else if (strcmp(name1, onnxruntime::HIP) == 0) {
    *out = new OrtMemoryInfo(
        onnxruntime::HIP, type, OrtDevice(OrtDevice::GPU, OrtDevice::MemType::DEFAULT, static_cast<OrtDevice::DeviceId>(id1)), id1,
        mem_type1);
  } else if (strcmp(name1, onnxruntime::HIP_PINNED) == 0) {
    *out = new OrtMemoryInfo(
        onnxruntime::HIP_PINNED, type, OrtDevice(OrtDevice::CPU, OrtDevice::MemType::HIP_PINNED, static_cast<OrtDevice::DeviceId>(id1)),
        id1, mem_type1);
  } else {
    return OrtApis::CreateStatus(ORT_INVALID_ARGUMENT, "Specified device is not supported.");
  }
  return nullptr;
}

ORT_API(void, OrtApis::ReleaseMemoryInfo, _Frees_ptr_opt_ OrtMemoryInfo* p) { delete p; }
#if defined(_MSC_VER) && !defined(__clang__)
#pragma warning(pop)
#endif
ORT_API_STATUS_IMPL(OrtApis::MemoryInfoGetName, _In_ const OrtMemoryInfo* ptr, _Out_ const char** out) {
  *out = ptr->name;
  return nullptr;
}

ORT_API_STATUS_IMPL(OrtApis::MemoryInfoGetId, _In_ const OrtMemoryInfo* ptr, _Out_ int* out) {
  *out = ptr->id;
  return nullptr;
}

ORT_API_STATUS_IMPL(OrtApis::MemoryInfoGetMemType, _In_ const OrtMemoryInfo* ptr, _Out_ OrtMemType* out) {
  *out = ptr->mem_type;
  return nullptr;
}

ORT_API_STATUS_IMPL(OrtApis::MemoryInfoGetType, _In_ const OrtMemoryInfo* ptr, _Out_ OrtAllocatorType* out) {
  *out = ptr->alloc_type;
  return nullptr;
}

ORT_API_STATUS_IMPL(OrtApis::CompareMemoryInfo, _In_ const OrtMemoryInfo* info1, _In_ const OrtMemoryInfo* info2,
                    _Out_ int* out) {
  *out = (*info1 == *info2) ? 0 : -1;
  return nullptr;
}

ORT_API(void, OrtApis::MemoryInfoGetDeviceType, _In_ const OrtMemoryInfo* info, _Out_ OrtMemoryInfoDeviceType* out) {
  *out = static_cast<OrtMemoryInfoDeviceType>(info->device.Type());
}
