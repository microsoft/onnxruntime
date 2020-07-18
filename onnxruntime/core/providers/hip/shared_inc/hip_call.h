// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/providers/hip/hip_pch.h"

namespace onnxruntime {

// -----------------------------------------------------------------------
// Error handling
// -----------------------------------------------------------------------

template <typename ERRTYPE, bool THRW>
bool HipCall(ERRTYPE retCode, const char* exprString, const char* libName, ERRTYPE successCode, const char* msg = "");

#define HIP_CALL(expr) (HipCall<hipError_t, false>((expr), #expr, "HIP", hipSuccess))
#define HIPBLAS_CALL(expr) (HipCall<hipblasStatus_t, false>((expr), #expr, "HIPBLAS", HIPBLAS_STATUS_SUCCESS))
#define HIPSPARSE_CALL(expr) (HipCall<hipsparseStatus_t, false>((expr), #expr, "HIPSPARSE", HIPSPARSE_STATUS_SUCCESS))
#define HIPRAND_CALL(expr) (HipCall<hiprandStatus_t, false>((expr), #expr, "HIPRAND", HIPRAND_STATUS_SUCCESS))
#define MIOPEN_CALL(expr) (HipCall<miopenStatus_t, false>((expr), #expr, "MIOPEN", miopenStatusSuccess))
#define MIOPEN_CALL2(expr, m) (HipCall<miopenStatus_t, false>((expr), #expr, "MIOPEN", miopenStatusSuccess, m))
#define HIPFFT_CALL(expr) (HipCall<hipfftResult, false>((expr), #expr, "HIPFFT", HIPFFT_SUCCESS))

#define HIP_CALL_THROW(expr) (HipCall<hipError_t, true>((expr), #expr, "HIP", hipSuccess))
#define HIPBLAS_CALL_THROW(expr) (HipCall<hipblasStatus_t, true>((expr), #expr, "HIPBLAS", HIPBLAS_STATUS_SUCCESS))
#define HIPSPARSE_CALL_THROW(expr) (HipCall<hipsparseStatus_t, true>((expr), #expr, "HIPSPARSE", HIPSPARSE_STATUS_SUCCESS))
#define HIPRAND_CALL_THROW(expr) (HipCall<hiprandStatus_t, true>((expr), #expr, "HIPRAND", HIPRAND_STATUS_SUCCESS))
#define MIOPEN_CALL_THROW(expr) (HipCall<miopenStatus_t, true>((expr), #expr, "MIOPEN", miopenStatusSuccess))
#define MIOPEN_CALL_THROW2(expr, m) (HipCall<miopenStatus_t, true>((expr), #expr, "MIOPEN", miopenStatusSuccess, m))
#define HIPFFT_CALL_THROW(expr) (HipCall<hipfftResult, true>((expr), #expr, "HIPFFT", HIPFFT_SUCCESS))

#ifdef USE_NCCL
#define NCCL_CALL(expr) (HipCall<ncclResult_t, false>((expr), #expr, "NCCL", ncclSuccess))
#define NCCL_CALL_THROW(expr) (HipCall<ncclResult_t, true>((expr), #expr, "NCCL", ncclSuccess))
#endif

}  // namespace onnxruntime
