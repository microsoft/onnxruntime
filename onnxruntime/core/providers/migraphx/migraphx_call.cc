// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <unistd.h>
#include <string.h>
#include <miopen/miopen.h>
#include <rocblas/rocblas.h>
#include "core/providers/shared_library/provider_api.h"
#include "core/common/common.h"
#include "core/common/status.h"
#include "migraphx_call.h"

namespace onnxruntime {

using namespace common;

template <typename ERRTYPE>
const char* RocmErrString(ERRTYPE x) {
  ORT_NOT_IMPLEMENTED();
}

#define CASE_ENUM_TO_STR(x) \
  case x:                   \
    return #x

template <>
const char* RocmErrString<hipError_t>(hipError_t x) {
  (void)hipDeviceSynchronize();
  return hipGetErrorString(x);
}

template <typename ERRTYPE, bool THRW>
std::conditional_t<THRW, void, Status> RocmCall(
    ERRTYPE retCode, const char* exprString, const char* libName, ERRTYPE successCode, const char* msg, const char* file, const int line) {
  if (retCode != successCode) {
    try {
      char hostname[HOST_NAME_MAX];
      if (gethostname(hostname, HOST_NAME_MAX) != 0)
        strcpy(hostname, "?");
      int currentHipDevice;
      (void)hipGetDevice(&currentHipDevice);
      (void)hipGetLastError();  // clear last HIP error
      static char str[1024];
      snprintf(str, 1024, "%s failure %d: %s ; GPU=%d ; hostname=%s ; file=%s ; line=%d ; expr=%s; %s",
               libName, (int)retCode, RocmErrString(retCode), currentHipDevice,
               hostname,
               file, line, exprString, msg);
      if constexpr (THRW) {
        // throw an exception with the error info
        ORT_THROW(str);
      } else {
        LOGS_DEFAULT(ERROR) << str;
        return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, str);
      }
    } catch (const std::exception& e) {  // catch, log, and rethrow since HIP code sometimes hangs in destruction, so we'd never get to see the error
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

template Status RocmCall<hipError_t, false>(hipError_t retCode, const char* exprString, const char* libName, hipError_t successCode, const char* msg, const char* file, const int line);
template void RocmCall<hipError_t, true>(hipError_t retCode, const char* exprString, const char* libName, hipError_t successCode, const char* msg, const char* file, const int line);
template Status RocmCall<rocblas_status, false>(rocblas_status retCode, const char* exprString, const char* libName, rocblas_status successCode, const char* msg, const char* file, const int line);
template void RocmCall<rocblas_status, true>(rocblas_status retCode, const char* exprString, const char* libName, rocblas_status successCode, const char* msg, const char* file, const int line);
template Status RocmCall<miopenStatus_t, false>(miopenStatus_t retCode, const char* exprString, const char* libName, miopenStatus_t successCode, const char* msg, const char* file, const int line);
template void RocmCall<miopenStatus_t, true>(miopenStatus_t retCode, const char* exprString, const char* libName, miopenStatus_t successCode, const char* msg, const char* file, const int line);

}  // namespace onnxruntime
