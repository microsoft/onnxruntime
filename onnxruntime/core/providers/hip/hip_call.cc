// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "shared_inc/hip_call.h"
#include "core/common/common.h"
#include "core/common/status.h"
#include "core/common/logging/logging.h"

#ifdef _WIN32
#else  // POSIX
#include <unistd.h>
#include <string.h>
#endif

namespace onnxruntime {

using namespace common;

template <typename ERRTYPE>
const char* HipErrString(ERRTYPE x) {
  ORT_NOT_IMPLEMENTED();
}

#define CASE_ENUM_TO_STR(x) \
  case x:                   \
    return #x

template <>
const char* HipErrString<hipError_t>(hipError_t x) {
  hipDeviceSynchronize();
  return hipGetErrorString(x);
}

template <>
const char* HipErrString<hipblasStatus_t>(hipblasStatus_t e) {
  hipDeviceSynchronize();

  switch (e) {
    CASE_ENUM_TO_STR(HIPBLAS_STATUS_SUCCESS);
    CASE_ENUM_TO_STR(HIPBLAS_STATUS_NOT_INITIALIZED);
    CASE_ENUM_TO_STR(HIPBLAS_STATUS_ALLOC_FAILED);
    CASE_ENUM_TO_STR(HIPBLAS_STATUS_INVALID_VALUE);
    CASE_ENUM_TO_STR(HIPBLAS_STATUS_ARCH_MISMATCH);
    CASE_ENUM_TO_STR(HIPBLAS_STATUS_MAPPING_ERROR);
    CASE_ENUM_TO_STR(HIPBLAS_STATUS_EXECUTION_FAILED);
    CASE_ENUM_TO_STR(HIPBLAS_STATUS_INTERNAL_ERROR);
    CASE_ENUM_TO_STR(HIPBLAS_STATUS_NOT_SUPPORTED);
    //CASE_ENUM_TO_STR(HIPBLAS_STATUS_LICENSE_ERROR);
    default:
      return "(look for HIPBLAS_STATUS_xxx in hipblas_api.h)";
  }
}

// template <>
// const char* HipErrString<hiprandStatus_t>(hiprandStatus_t) {
//   hipDeviceSynchronize();
//   return "(see hiprand.h & look for hiprandStatus_t or HIPRAND_STATUS_xxx)";
// }

template <>
const char* HipErrString<miopenStatus_t>(miopenStatus_t e) {
  hipDeviceSynchronize();
  return miopenGetErrorString(e);
}

template <>
const char* HipErrString<hipfftResult>(hipfftResult e) {
  hipDeviceSynchronize();
  switch (e) {
    CASE_ENUM_TO_STR(HIPFFT_SUCCESS);
    CASE_ENUM_TO_STR(HIPFFT_ALLOC_FAILED);
    CASE_ENUM_TO_STR(HIPFFT_INVALID_VALUE);
    CASE_ENUM_TO_STR(HIPFFT_INTERNAL_ERROR);
    CASE_ENUM_TO_STR(HIPFFT_SETUP_FAILED);
    CASE_ENUM_TO_STR(HIPFFT_INVALID_SIZE);
    default:
      return "Unknown cufft error status";
  }
}

#ifdef USE_NCCL
template <>
const char* HipErrString<ncclResult_t>(ncclResult_t e) {
  hipDeviceSynchronize();
  return ncclGetErrorString(e);
}
#endif

template <typename ERRTYPE, bool THRW>
bool HipCall(ERRTYPE retCode, const char* exprString, const char* libName, ERRTYPE successCode, const char* msg) {
  if (retCode != successCode) {
    try {
#ifdef _WIN32
      auto del = [](char* p) { free(p); };
      std::unique_ptr<char, decltype(del)> hostname_ptr(nullptr, del);
      size_t hostname_len = 0;
      char* hostname = nullptr;
      if (-1 == _dupenv_s(&hostname, &hostname_len, "COMPUTERNAME"))
        hostname = "?";
      else
        hostname_ptr.reset(hostname);
#else
      char hostname[HOST_NAME_MAX];
      if (gethostname(hostname, HOST_NAME_MAX) != 0)
        strcpy(hostname, "?");
#endif
      int currentHipDevice;
      hipGetDevice(&currentHipDevice);
      hipGetLastError();  // clear last HIP error
      static char str[1024];
      snprintf(str, 1024, "%s failure %d: %s ; GPU=%d ; hostname=%s ; expr=%s; %s",
               libName, (int)retCode, HipErrString(retCode), currentHipDevice,
               hostname,
               exprString, msg);
      if (THRW) {
        // throw an exception with the error info
        ORT_THROW(str);
      } else {
        LOGS_DEFAULT(ERROR) << str;
      }
    } catch (const std::exception& e) {  // catch, log, and rethrow since HIP code sometimes hangs in destruction, so we'd never get to see the error
      if (THRW) {
        ORT_THROW(e.what());
      } else {
        LOGS_DEFAULT(ERROR) << e.what();
      }
    }
    return false;
  }
  return true;
}

template bool HipCall<hipError_t, false>(hipError_t retCode, const char* exprString, const char* libName, hipError_t successCode, const char* msg);
template bool HipCall<hipError_t, true>(hipError_t retCode, const char* exprString, const char* libName, hipError_t successCode, const char* msg);
template bool HipCall<hipblasStatus_t, false>(hipblasStatus_t retCode, const char* exprString, const char* libName, hipblasStatus_t successCode, const char* msg);
template bool HipCall<hipblasStatus_t, true>(hipblasStatus_t retCode, const char* exprString, const char* libName, hipblasStatus_t successCode, const char* msg);
template bool HipCall<miopenStatus_t, false>(miopenStatus_t retCode, const char* exprString, const char* libName, miopenStatus_t successCode, const char* msg);
template bool HipCall<miopenStatus_t, true>(miopenStatus_t retCode, const char* exprString, const char* libName, miopenStatus_t successCode, const char* msg);
// template bool HipCall<hiprandStatus_t, false>(hiprandStatus_t retCode, const char* exprString, const char* libName, hiprandStatus_t successCode, const char* msg);
// template bool HipCall<hiprandStatus_t, true>(hiprandStatus_t retCode, const char* exprString, const char* libName, hiprandStatus_t successCode, const char* msg);
template bool HipCall<hipfftResult, false>(hipfftResult retCode, const char* exprString, const char* libName, hipfftResult successCode, const char* msg);
template bool HipCall<hipfftResult, true>(hipfftResult retCode, const char* exprString, const char* libName, hipfftResult successCode, const char* msg);

#ifdef USE_NCCL
template bool HipCall<ncclResult_t, false>(ncclResult_t retCode, const char* exprString, const char* libName, ncclResult_t successCode, const char* msg);
#endif
}  // namespace onnxruntime
