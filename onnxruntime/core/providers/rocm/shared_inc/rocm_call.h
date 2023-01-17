// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/common/common.h"
#include "core/providers/rocm/rocm_pch.h"

namespace onnxruntime {

// -----------------------------------------------------------------------
// Error handling
// -----------------------------------------------------------------------

template <typename ERRTYPE, bool THRW>
std::conditional_t<THRW, void, Status> RocmCall(
  ERRTYPE retCode, const char* exprString, const char* libName, ERRTYPE successCode, const char* msg = "");

#define HIP_CALL(expr) (RocmCall<hipError_t, false>((expr), #expr, "HIP", hipSuccess))
#define ROCBLAS_CALL(expr) (RocmCall<rocblas_status, false>((expr), #expr, "ROCBLAS", rocblas_status_success))
#define HIPSPARSE_CALL(expr) (RocmCall<hipsparseStatus_t, false>((expr), #expr, "HIPSPARSE", HIPSPARSE_STATUS_SUCCESS))
#define HIPRAND_CALL(expr) (RocmCall<hiprandStatus_t, false>((expr), #expr, "HIPRAND", HIPRAND_STATUS_SUCCESS))
#define MIOPEN_CALL(expr) (RocmCall<miopenStatus_t, false>((expr), #expr, "MIOPEN", miopenStatusSuccess))
#define MIOPEN_CALL2(expr, m) (RocmCall<miopenStatus_t, false>((expr), #expr, "MIOPEN", miopenStatusSuccess, m))
#define HIPFFT_CALL(expr) (RocmCall<hipfftResult, false>((expr), #expr, "HIPFFT", HIPFFT_SUCCESS))

#define HIP_CALL_THROW(expr) (RocmCall<hipError_t, true>((expr), #expr, "HIP", hipSuccess))
#define ROCBLAS_CALL_THROW(expr) (RocmCall<rocblas_status, true>((expr), #expr, "ROCBLAS", rocblas_status_success))
#define HIPSPARSE_CALL_THROW(expr) (RocmCall<hipsparseStatus_t, true>((expr), #expr, "HIPSPARSE", HIPSPARSE_STATUS_SUCCESS))
#define HIPRAND_CALL_THROW(expr) (RocmCall<hiprandStatus_t, true>((expr), #expr, "HIPRAND", HIPRAND_STATUS_SUCCESS))
#define MIOPEN_CALL_THROW(expr) (RocmCall<miopenStatus_t, true>((expr), #expr, "MIOPEN", miopenStatusSuccess))
#define MIOPEN_CALL_THROW2(expr, m) (RocmCall<miopenStatus_t, true>((expr), #expr, "MIOPEN", miopenStatusSuccess, m))
#define HIPFFT_CALL_THROW(expr) (RocmCall<hipfftResult, true>((expr), #expr, "HIPFFT", HIPFFT_SUCCESS))

#ifdef ORT_USE_NCCL
#define NCCL_CALL(expr) (RocmCall<ncclResult_t, false>((expr), #expr, "NCCL", ncclSuccess))
#define NCCL_CALL_THROW(expr) (RocmCall<ncclResult_t, true>((expr), #expr, "NCCL", ncclSuccess))
#endif

}  // namespace onnxruntime
