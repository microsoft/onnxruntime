// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/shared_library/provider_api.h"
#include "shared_inc/rocm_call.h"

#ifdef _WIN32
#else  // POSIX
#include <unistd.h>
#include <string.h>
#endif

namespace onnxruntime {

using namespace common;

template <typename ERRTYPE>
const char* RocmErrString(ERRTYPE) {
  ORT_NOT_IMPLEMENTED();
}

#define CASE_ENUM_TO_STR(x) \
  case x:                   \
    return #x

template <>
const char* RocmErrString<hipError_t>(hipError_t x) {
  (void)hipDeviceSynchronize(); // void to silence nodiscard
  return hipGetErrorString(x);
}

template <>
const char* RocmErrString<rocblas_status>(rocblas_status e) {
  (void)hipDeviceSynchronize(); // void to silence nodiscard

  switch (e) {
    CASE_ENUM_TO_STR(rocblas_status_success);
    CASE_ENUM_TO_STR(rocblas_status_invalid_handle);
    CASE_ENUM_TO_STR(rocblas_status_not_implemented);
    CASE_ENUM_TO_STR(rocblas_status_invalid_pointer);
    CASE_ENUM_TO_STR(rocblas_status_invalid_size);
    CASE_ENUM_TO_STR(rocblas_status_memory_error);
    CASE_ENUM_TO_STR(rocblas_status_internal_error);
    CASE_ENUM_TO_STR(rocblas_status_perf_degraded);
    CASE_ENUM_TO_STR(rocblas_status_size_query_mismatch);
    CASE_ENUM_TO_STR(rocblas_status_size_increased);
    CASE_ENUM_TO_STR(rocblas_status_size_unchanged);
    CASE_ENUM_TO_STR(rocblas_status_invalid_value);
    CASE_ENUM_TO_STR(rocblas_status_continue);
    default:
      return "(look for rocblas_status in rocblas-types.h)";
  }
}

template <>
const char* RocmErrString<hiprandStatus_t>(hiprandStatus_t) {
  (void)hipDeviceSynchronize(); // void to silence nodiscard
  return "(see hiprand.h & look for hiprandStatus_t or HIPRAND_STATUS_xxx)";
}

template <>
const char* RocmErrString<miopenStatus_t>(miopenStatus_t e) {
  (void)hipDeviceSynchronize(); // void to silence nodiscard
  return miopenGetErrorString(e);
}

template <>
const char* RocmErrString<hipfftResult>(hipfftResult e) {
  (void)hipDeviceSynchronize(); // void to silence nodiscard
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

#ifdef ORT_USE_NCCL
template <>
const char* RocmErrString<ncclResult_t>(ncclResult_t e) {
  (void)hipDeviceSynchronize(); // void to silence nodiscard
  return ncclGetErrorString(e);
}
#endif

template <typename ERRTYPE, bool THRW>
std::conditional_t<THRW, void, Status> RocmCall(
  ERRTYPE retCode, const char* exprString, const char* libName, ERRTYPE successCode, const char* msg) {
  if (retCode != successCode) {
    try {
#ifdef _WIN32
      auto del = [](char* p) { free(p); };
      std::unique_ptr<char, decltype(del)> hostname_ptr(nullptr, del);
      size_t hostname_len = 0;
      char* hostname = nullptr;
      //TODO: avoid using const_cast
      if (-1 == _dupenv_s(&hostname, &hostname_len, "COMPUTERNAME"))
        hostname = const_cast<char*>("?");
      else
        hostname_ptr.reset(hostname);
#else
      char hostname[HOST_NAME_MAX];
      if (gethostname(hostname, HOST_NAME_MAX) != 0)
        strcpy(hostname, "?");
#endif
      int currentHipDevice;
      (void)hipGetDevice(&currentHipDevice); // void to silence nodiscard
      (void)hipGetLastError();  // clear last ROCM error; void to silence nodiscard
      static char str[1024];
      snprintf(str, 1024, "%s failure %d: %s ; GPU=%d ; hostname=%s ; expr=%s; %s",
               libName, (int)retCode, RocmErrString(retCode), currentHipDevice,
               hostname,
               exprString, msg);
      if constexpr (THRW) {
        // throw an exception with the error info
        ORT_THROW(str);
      } else {
        LOGS_DEFAULT(ERROR) << str;
        return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, str);
      }
    } catch (const std::exception& e) {  // catch, log, and rethrow since ROCM code sometimes hangs in destruction, so we'd never get to see the error
      if constexpr (THRW) {
        ORT_THROW(e.what());
      } else {
        LOGS_DEFAULT(ERROR) << e.what();
        return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, e.what());
      }
    }
  }
  if constexpr (!THRW) {
    return Status::OK();
  }
}

template Status RocmCall<hipError_t, false>(hipError_t retCode, const char* exprString, const char* libName, hipError_t successCode, const char* msg);
template void RocmCall<hipError_t, true>(hipError_t retCode, const char* exprString, const char* libName, hipError_t successCode, const char* msg);
template Status RocmCall<rocblas_status, false>(rocblas_status retCode, const char* exprString, const char* libName, rocblas_status successCode, const char* msg);
template void RocmCall<rocblas_status, true>(rocblas_status retCode, const char* exprString, const char* libName, rocblas_status successCode, const char* msg);
template Status RocmCall<miopenStatus_t, false>(miopenStatus_t retCode, const char* exprString, const char* libName, miopenStatus_t successCode, const char* msg);
template void RocmCall<miopenStatus_t, true>(miopenStatus_t retCode, const char* exprString, const char* libName, miopenStatus_t successCode, const char* msg);
template Status RocmCall<hiprandStatus_t, false>(hiprandStatus_t retCode, const char* exprString, const char* libName, hiprandStatus_t successCode, const char* msg);
template void RocmCall<hiprandStatus_t, true>(hiprandStatus_t retCode, const char* exprString, const char* libName, hiprandStatus_t successCode, const char* msg);
template Status RocmCall<hipfftResult, false>(hipfftResult retCode, const char* exprString, const char* libName, hipfftResult successCode, const char* msg);
template void RocmCall<hipfftResult, true>(hipfftResult retCode, const char* exprString, const char* libName, hipfftResult successCode, const char* msg);

#ifdef ORT_USE_NCCL
template Status RocmCall<ncclResult_t, false>(ncclResult_t retCode, const char* exprString, const char* libName, ncclResult_t successCode, const char* msg);
template void RocmCall<ncclResult_t, true>(ncclResult_t retCode, const char* exprString, const char* libName, ncclResult_t successCode, const char* msg);
#endif
}  // namespace onnxruntime
