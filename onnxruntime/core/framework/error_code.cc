// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/session/onnxruntime_c_api.h"
#include "core/common/status.h"
#include "core/framework/error_code_helper.h"
#include <cassert>
using onnxruntime::common::Status;

ORT_API(ONNXStatus*, CreateONNXStatus, OrtErrorCode code, const char* msg) {
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
ORT_API(OrtErrorCode, OrtGetErrorCode, _In_ const ONNXStatus* status) {
  return *reinterpret_cast<OrtErrorCode*>(const_cast<ONNXStatus*>(status));
}

ORT_API(const char*, OrtGetErrorMessage, _In_ const ONNXStatus* status) {
  return reinterpret_cast<const char*>(status) + sizeof(int);
}
