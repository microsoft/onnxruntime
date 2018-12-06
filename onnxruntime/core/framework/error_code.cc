// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/session/onnxruntime_c_api.h"
#include "core/common/status.h"
#include "core/framework/error_code_helper.h"
#include <cassert>
using onnxruntime::common::Status;

ONNXRUNTIME_API(ONNXStatus*, CreateONNXStatus, ONNXRuntimeErrorCode code, const char* msg) {
  assert(!(code == 0 && msg != nullptr));
  size_t clen = strlen(msg);
  size_t len = clen + 1 + sizeof(int);
  char* p = new char[len];
  char* ret = p;
  *reinterpret_cast<int*>(p) = static_cast<int>(code);
  p += sizeof(int);
  memcpy(p, msg, clen);
  p += clen;
  *p = '\0';
  return ret;
}
namespace onnxruntime {
ONNXStatus* ToONNXStatus(const Status& st) {
  if (st.IsOK())
    return nullptr;
  size_t clen = st.ErrorMessage().length();
  size_t len = clen + 1 + sizeof(int);
  char* p = new char[len];
  char* ret = p;
  *reinterpret_cast<int*>(p) = static_cast<int>(st.Code());
  p += sizeof(int);
  memcpy(p, st.ErrorMessage().c_str(), clen);
  p += clen;
  *p = '\0';
  return ret;
}
}  // namespace onnxruntime
ONNXRUNTIME_API(ONNXRuntimeErrorCode, ONNXRuntimeGetErrorCode, _In_ const ONNXStatus* status) {
  return *reinterpret_cast<ONNXRuntimeErrorCode*>(const_cast<ONNXStatus*>(status));
}

ONNXRUNTIME_API(const char*, ONNXRuntimeGetErrorMessage, _In_ const ONNXStatus* status) {
  return reinterpret_cast<const char*>(status) + sizeof(int);
}
