// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/shared_library/provider_api.h"
#include "core/common/status.h"
#include "core/providers/rocm/rocm_pch.h"
#include "core/providers/rocm/shared_inc/rocm_call.h"
#include "core/providers/rocm/shared_inc/fast_divmod.h"
#include "core/util/math.h"

namespace onnxruntime {
namespace rocm {

#define HIP_RETURN_IF_ERROR(expr)               \
  ORT_RETURN_IF_ERROR(HIP_CALL(expr)            \
                          ? common::Status::OK() \
                          : ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "HIP error executing ", #expr))

#define ROCBLAS_RETURN_IF_ERROR(expr)             \
  ORT_RETURN_IF_ERROR(ROCBLAS_CALL(expr)          \
                          ? common::Status::OK() \
                          : ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "ROCBLAS error executing ", #expr))

#define HIPSPARSE_RETURN_IF_ERROR(expr)           \
  ORT_RETURN_IF_ERROR(HIPSPARSE_CALL(expr)        \
                          ? common::Status::OK() \
                          : ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "HIPSPARSE error executing ", #expr))

#define HIPRAND_RETURN_IF_ERROR(expr)             \
  ORT_RETURN_IF_ERROR(HIPRAND_CALL(expr)          \
                          ? common::Status::OK() \
                          : ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "HIPRAND error executing ", #expr))

#define MIOPEN_RETURN_IF_ERROR(expr)              \
  ORT_RETURN_IF_ERROR(MIOPEN_CALL(expr)           \
                          ? common::Status::OK() \
                          : ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "MIOPEN error executing ", #expr))

#define MIOPEN2_RETURN_IF_ERROR(expr, m)          \
  ORT_RETURN_IF_ERROR(MIOPEN_CALL2(expr, m)       \
                          ? common::Status::OK() \
                          : ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "MIOPEN2 error executing ", #expr))

#define HIPFFT_RETURN_IF_ERROR(expr)              \
  ORT_RETURN_IF_ERROR(HIPFFT_CALL(expr)           \
                          ? common::Status::OK() \
                          : ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "HIPFFT error executing ", #expr))

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

}  // namespace rocm
}  // namespace onnxruntime
