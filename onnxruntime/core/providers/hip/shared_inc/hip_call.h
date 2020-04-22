// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>
#include <hipblas.h>
#include <miopen/miopen.h>
//#include <hiprand/hiprand.h>

namespace onnxruntime {

// -----------------------------------------------------------------------
// Error handling
// -----------------------------------------------------------------------

template <typename ERRTYPE, bool THRW>
bool HipCall(ERRTYPE retCode, const char* exprString, const char* libName, ERRTYPE successCode, const char* msg = "");

#define HIP_CALL(expr) (HipCall<hipError_t, false>((expr), #expr, "HIP", hipSuccess))
#define HIPBLAS_CALL(expr) (HipCall<hipblasStatus_t, false>((expr), #expr, "HIPBLAS", HIPBLAS_STATUS_SUCCESS))
// #define CUSPARSE_CALL(expr) (HipCall<hipsparseStatus_t, false>((expr), #expr, "CUSPARSE", HIPSPARSE_STATUS_SUCCESS))
// #define CURAND_CALL(expr) (HipCall<hiprandStatus_t, false>((expr), #expr, "CURAND", HIPRAND_STATUS_SUCCESS))
#define MIOPEN_CALL(expr) (HipCall<miopenStatus_t , false>((expr), #expr, "MIOPEN",  miopenStatusSuccess))
#define MIOPEN_CALL2(expr, m) (HipCall<miopenStatus_t , false>((expr), #expr, "MIOPEN",  miopenStatusSuccess, m))

#define HIP_CALL_THROW(expr) (HipCall<hipError_t, true>((expr), #expr, "HIP", hipSuccess))
#define HIPBLAS_CALL_THROW(expr) (HipCall<hipblasStatus_t, true>((expr), #expr, "HIPBLAS", HIPBLAS_STATUS_SUCCESS))
// #define CUSPARSE_CALL_THROW(expr) (HipCall<hipsparseStatus_t, true>((expr), #expr, "CUSPARSE", HIPSPARSE_STATUS_SUCCESS))
// #define CURAND_CALL_THROW(expr) (HipCall<hiprandStatus_t, true>((expr), #expr, "CURAND", HIPRAND_STATUS_SUCCESS))
#define MIOPEN_CALL_THROW(expr) (HipCall<miopenStatus_t , true>((expr), #expr, "MIOPEN",  miopenStatusSuccess))
#define MIOPEN_CALL_THROW2(expr, m) (HipCall<miopenStatus_t , true>((expr), #expr, "MIOPEN",  miopenStatusSuccess, m))

}  // namespace onnxruntime
