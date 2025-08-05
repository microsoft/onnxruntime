// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <cstdio>
#include <string>

#ifdef _WIN32
#include <winsock.h>
#else
#include <unistd.h>
#endif

#include "core/common/common.h"
#include "core/common/status.h"
#include "core/providers/shared_library/provider_api.h"
#include "core/providers/migraphx/migraphx_call.h"

namespace onnxruntime {

namespace {
template <typename ERRTYPE>
std::string_view RocmErrString(ERRTYPE x) {
  ORT_NOT_IMPLEMENTED();
}

#define CASE_ENUM_TO_STR(x) \
  case x:                   \
    return #x

template <>
std::string_view RocmErrString<hipError_t>(hipError_t x) {
  (void)hipDeviceSynchronize();
  return std::string_view{hipGetErrorString(x)};
}

}  // namespace

template <typename ERRTYPE, bool THRW>
std::conditional_t<THRW, void, Status> RocmCall(
    ERRTYPE retCode, std::string_view exprString, std::string_view libName, ERRTYPE successCode, std::string_view msg, std::string_view file, const int line) {
  if (retCode != successCode) {
    try {
#ifdef _WIN32
      // According to the POSIX spec, 255 is the safe minimum value.
      static constexpr int HOST_NAME_MAX = 255;
#endif
      std::string hostname(HOST_NAME_MAX, 0);
      if (gethostname(hostname.data(), HOST_NAME_MAX) != 0)
        hostname = "?";
      int currentHipDevice;
      (void)hipGetDevice(&currentHipDevice);
      (void)hipGetLastError();  // clear last HIP error
      std::stringstream ss;
      ss << libName << " failure " << static_cast<int>(retCode) << ": " << RocmErrString(retCode)
         << "; GPU=" << currentHipDevice << "; hostname=" << hostname << "; file=" << file << "; line=" << line
         << "; expr=" << exprString << "; " << msg;
      if constexpr (THRW) {
        // throw an exception with the error info
        ORT_THROW(ss.str());
      } else {
        LOGS_DEFAULT(ERROR) << ss.str();
        return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, ss.str());
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

template Status RocmCall<hipError_t, false>(hipError_t retCode, std::string_view exprString, std::string_view libName, hipError_t successCode, std::string_view msg, std::string_view file, int line);
template void RocmCall<hipError_t, true>(hipError_t retCode, std::string_view exprString, std::string_view libName, hipError_t successCode, std::string_view msg, std::string_view file, int line);

}  // namespace onnxruntime
