// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/shared_library/provider_api.h"
#include "core/common/status.h"
#include "core/framework/float16.h"
#include "core/providers/rocm/rocm_pch.h"
#include "core/providers/rocm/shared_inc/rocm_call.h"
#include "core/providers/rocm/shared_inc/fast_divmod.h"
#include "core/util/math.h"
#include <gsl/gsl>

namespace onnxruntime {
namespace rocm {

#define HIP_RETURN_IF_ERROR(expr) ORT_RETURN_IF_ERROR(HIP_CALL(expr))
#define ROCBLAS_RETURN_IF_ERROR(expr) ORT_RETURN_IF_ERROR(ROCBLAS_CALL(expr))
#define HIPBLAS_RETURN_IF_ERROR(expr) ORT_RETURN_IF_ERROR(HIPBLAS_CALL(expr))
#define HIPSPARSE_RETURN_IF_ERROR(expr) ORT_RETURN_IF_ERROR(HIPSPARSE_CALL(expr))
#define HIPRAND_RETURN_IF_ERROR(expr) ORT_RETURN_IF_ERROR(HIPRAND_CALL(expr))
#define MIOPEN_RETURN_IF_ERROR(expr) ORT_RETURN_IF_ERROR(MIOPEN_CALL(expr))
#define MIOPEN2_RETURN_IF_ERROR(expr, m) ORT_RETURN_IF_ERROR(MIOPEN_CALL2(expr, m))
#define HIPFFT_RETURN_IF_ERROR(expr) ORT_RETURN_IF_ERROR(HIPFFT_CALL(expr))

#ifdef USE_HIPBLASLT
#define HIPBLASLT_RETURN_IF_ERROR(expr) ORT_RETURN_IF_ERROR(HIPBLASLT_CALL(expr))
#endif

// Type mapping for MLFloat16 to half
template <typename T>
class ToHipType {
 public:
  typedef T MappedType;
  static MappedType FromFloat(float f) {
    return static_cast<T>(f);
  }
};

template <>
class ToHipType<MLFloat16> {
 public:
  typedef __half MappedType;
  static MappedType FromFloat(float f) {
    uint16_t h = math::floatToHalf(f);
    return *reinterpret_cast<MappedType*>(&h);
  }
};

inline bool CalculateFdmStrides(gsl::span<fast_divmod> p, const std::vector<int64_t>& dims) {
  int stride = 1;
  if (dims.empty() || p.size() < dims.size())
    return false;
  auto rank = p.size();
  for (size_t i = 0; i < rank; i++) {
    p[rank - 1 - i] = fast_divmod(stride);
    if (i < dims.size() - 1) {
      stride *= static_cast<int>(dims[dims.size() - 1 - i]);
    }
  }
  return true;
}

inline int warpSizeDynamic() {
  hipDeviceProp_t deviceProp;
  HIP_CALL_THROW(hipGetDeviceProperties(&deviceProp, 0));
  return deviceProp.warpSize;
}

inline void hipMemGetInfoAlt(uint32_t deviceId, size_t* pFree, size_t* pTotal) {
  const auto status = hipMemGetInfo(pFree, pTotal);
  if (status != hipSuccess) {
    size_t usedMemory = 0;
    ROCMSMI_CALL_THROW(rsmi_init(0));
    ROCMSMI_CALL_THROW(rsmi_dev_memory_total_get(deviceId, RSMI_MEM_TYPE_VIS_VRAM, pTotal));
    ROCMSMI_CALL_THROW(rsmi_dev_memory_usage_get(deviceId, RSMI_MEM_TYPE_VIS_VRAM, &usedMemory));
    *pFree = *pTotal - usedMemory;
    ROCMSMI_CALL_THROW(rsmi_shut_down());
  }
}

}  // namespace rocm
}  // namespace onnxruntime
