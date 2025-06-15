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
      LOGS_DEFAULT(ERROR) << ex.what() << " nmemb=" << nmemb << " size=" << size << " alignment=" << alignment;
      ok = false;
    });
  }
  return ok;
}

#ifdef USE_MIMALLOC
void* AllocatorDefaultAllocAligned(size_t size, size_t alignment) {
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

void AllocatorDefaultFreeAligned(void* p, size_t alignment) {
#if defined(_MSC_VER)
  mi_free_aligned(p, alignment);
#else
  mi_free(p);
#endif
}

#else

void* AllocatorDefaultAllocAligned(size_t size, size_t alignment) {
  if (size == 0) return nullptr;

  size += MLAS_SYMM_QGEMM_BUF_OVERRUN;

  return ::operator new(size, std::align_val_t{alignment});
}

void AllocatorDefaultFreeAligned(void* p, size_t alignment) {
  ::operator delete(p, std::align_val_t{alignment});
}

#endif  // USE_MIMALLOC

void* AllocatorDefaultAlloc(size_t size) {
  const size_t alignment = MlasGetPreferredBufferAlignment();
  return AllocatorDefaultAllocAligned(size, alignment);
}

AllocatorPtr CPUAllocator::DefaultInstance() {
  static AllocatorPtr instance = std::make_shared<CPUAllocator>();
  return instance;
}

void* CPUAllocator::Alloc(size_t size) {
  const auto alignment = std::max(Info().device.GetAlignment(), MlasGetPreferredBufferAlignment());
  return AllocatorDefaultAllocAligned(size, alignment);
}

void CPUAllocator::Free(void* p) {
  const auto alignment = std::max(Info().device.GetAlignment(), MlasGetPreferredBufferAlignment());
  AllocatorDefaultFreeAligned(p, alignment);
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
  auto device_id = static_cast<OrtDevice::DeviceId>(id1);
  if (strcmp(name1, onnxruntime::CPU) == 0) {
    *out = new OrtMemoryInfo(onnxruntime::CPU, type, OrtDevice(), mem_type1);
  } else if (strcmp(name1, onnxruntime::CUDA) == 0) {
    *out = new OrtMemoryInfo(
        name1, type,
        OrtDevice(OrtDevice::GPU, OrtDevice::MemType::DEFAULT, OrtDevice::VendorIds::NVIDIA, device_id),
        mem_type1);
  } else if (strcmp(name1, onnxruntime::OpenVINO_GPU) == 0) {
    *out = new OrtMemoryInfo(
        name1, type,
        OrtDevice(OrtDevice::GPU, OrtDevice::MemType::DEFAULT, OrtDevice::VendorIds::INTEL, device_id),
        mem_type1);
  } else if (strcmp(name1, onnxruntime::HIP) == 0) {
    *out = new OrtMemoryInfo(
        name1, type,
        OrtDevice(OrtDevice::GPU, OrtDevice::MemType::DEFAULT, OrtDevice::VendorIds::AMD, device_id),
        mem_type1);
  } else if (strcmp(name1, onnxruntime::WEBGPU_BUFFER) == 0 ||
             strcmp(name1, onnxruntime::WEBNN_TENSOR) == 0) {
    *out = new OrtMemoryInfo(
        name1, type,
        OrtDevice(OrtDevice::GPU, OrtDevice::MemType::DEFAULT, OrtDevice::VendorIds::NONE, device_id),
        mem_type1);

  } else if (strcmp(name1, onnxruntime::DML) == 0) {
    *out = new OrtMemoryInfo(
        name1, type,
        OrtDevice(OrtDevice::GPU, OrtDevice::MemType::DEFAULT, OrtDevice::VendorIds::MICROSOFT, device_id),
        mem_type1);
  } else if (strcmp(name1, onnxruntime::OpenVINO_RT_NPU) == 0) {
    *out = new OrtMemoryInfo(
        name1, type,
        OrtDevice(OrtDevice::NPU, OrtDevice::MemType::DEFAULT, OrtDevice::VendorIds::INTEL, device_id),
        mem_type1);
  } else if (strcmp(name1, onnxruntime::CUDA_PINNED) == 0) {
    *out = new OrtMemoryInfo(
        name1, type,
        OrtDevice(OrtDevice::GPU, OrtDevice::MemType::HOST_ACCESSIBLE, OrtDevice::VendorIds::NVIDIA, device_id),
        mem_type1);
  } else if (strcmp(name1, onnxruntime::HIP_PINNED) == 0) {
    *out = new OrtMemoryInfo(
        name1, type,
        OrtDevice(OrtDevice::GPU, OrtDevice::MemType::HOST_ACCESSIBLE, OrtDevice::VendorIds::AMD, device_id),
        mem_type1);
  } else if (strcmp(name1, onnxruntime::QNN_HTP_SHARED) == 0) {
    *out = new OrtMemoryInfo(
        name1, type,
        OrtDevice(OrtDevice::NPU, OrtDevice::MemType::HOST_ACCESSIBLE, OrtDevice::VendorIds::QUALCOMM, device_id),
        mem_type1);
  } else if (strcmp(name1, onnxruntime::CPU_ALIGNED_4K) == 0) {
    *out = new OrtMemoryInfo(
        name1, type,
        OrtDevice(OrtDevice::CPU, OrtDevice::MemType::DEFAULT, OrtDevice::VendorIds::NONE, device_id,
                  onnxruntime::kAlloc4KAlignment),
        mem_type1);
  } else {
    return OrtApis::CreateStatus(ORT_INVALID_ARGUMENT, "Specified device is not supported. Try CreateMemoryInfo_V2.");
  }
  return nullptr;
}

ORT_API_STATUS_IMPL(OrtApis::CreateMemoryInfo_V2, _In_ const char* name, _In_ enum OrtMemoryInfoDeviceType device_type,
                    _In_ uint32_t vendor_id, _In_ int16_t device_id, _In_ enum OrtDeviceMemoryType mem_type,
                    _In_ size_t alignment, enum OrtAllocatorType type,
                    _Outptr_ OrtMemoryInfo** out) {
  // map the public enum values to internal OrtDevice values
  OrtDevice::MemoryType mt = mem_type == OrtDeviceMemoryType_DEFAULT ? OrtDevice::MemType::DEFAULT
                                                                     : OrtDevice::MemType::HOST_ACCESSIBLE;

  OrtDevice::DeviceType dt;
  switch (device_type) {
    case OrtMemoryInfoDeviceType::OrtMemoryInfoDeviceType_CPU:
      dt = OrtDevice::CPU;
      break;
    case OrtMemoryInfoDeviceType::OrtMemoryInfoDeviceType_GPU:
      dt = OrtDevice::GPU;
      break;
    case OrtMemoryInfoDeviceType::OrtMemoryInfoDeviceType_NPU:
      dt = OrtDevice::NPU;
      break;
    case OrtMemoryInfoDeviceType::OrtMemoryInfoDeviceType_FPGA:
      dt = OrtDevice::FPGA;
      break;
    default:
      return OrtApis::CreateStatus(ORT_INVALID_ARGUMENT, "Invalid device type specified.");
  }

  *out = new OrtMemoryInfo(name, type, OrtDevice{dt, mt, vendor_id, device_id, alignment},
                           mem_type == OrtDeviceMemoryType_DEFAULT ? OrtMemTypeDefault : OrtMemTypeCPU);
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
  *out = ptr->device.Id();
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
