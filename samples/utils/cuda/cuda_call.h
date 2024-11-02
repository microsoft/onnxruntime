// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "../common.h"

namespace onnxruntime {

// -----------------------------------------------------------------------
// Error handling
// -----------------------------------------------------------------------
//
template <typename ERRTYPE>
const char* CudaErrString(ERRTYPE) {
  ORT_NOT_IMPLEMENTED();
}

template <typename ERRTYPE, bool THRW>
std::conditional_t<THRW, void, Status> CudaCall(
    ERRTYPE retCode, const char* exprString, const char* libName, ERRTYPE successCode, const char* msg, const char* file, const int line) {
  if (retCode != successCode) {
    try {
//#ifdef _WIN32
      //std::string hostname_str = GetEnvironmentVar("COMPUTERNAME");
      //if (hostname_str.empty()) {
        //hostname_str = "?";
      //}
      //const char* hostname = hostname_str.c_str();
//#else
      //char hostname[HOST_NAME_MAX];
      //if (gethostname(hostname, HOST_NAME_MAX) != 0)
        //strcpy(hostname, "?");
//#endif
      int currentCudaDevice = -1;
      cudaGetDevice(&currentCudaDevice);
      cudaGetLastError();  // clear last CUDA error
      static char str[1024];
      snprintf(str, 1024, "%s failure %d: %s ; GPU=%d ; hostname=? ; file=%s ; line=%d ; expr=%s; %s",
               libName, (int)retCode, CudaErrString(retCode), currentCudaDevice,
               //hostname,
               file, line, exprString, msg);
      if constexpr (THRW) {
        // throw an exception with the error info
        ORT_THROW(str);
      } else {
        //LOGS_DEFAULT(ERROR) << str;
        return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, str);
      }
    } catch (const std::exception& e) {  // catch, log, and rethrow since CUDA code sometimes hangs in destruction, so we'd never get to see the error
      if constexpr (THRW) {
        ORT_THROW(e.what());
      } else {
        //LOGS_DEFAULT(ERROR) << e.what();
        return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, e.what());
      }
    }
  }
  if constexpr (!THRW) {
    return Status::OK();
  }
}

//template <typename ERRTYPE, bool THRW>
//std::conditional_t<THRW, void, Status> CudaCall(
    //ERRTYPE retCode, const char* exprString, const char* libName, ERRTYPE successCode, const char* msg, const char* file, const int line);

#define CUDA_CALL(expr) (CudaCall<cudaError, false>((expr), #expr, "CUDA", cudaSuccess, "", __FILE__, __LINE__))

}  // namespace onnxruntime
